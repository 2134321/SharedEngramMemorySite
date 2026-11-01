"""Exploratory screen (not pre-registered): facts as ordinary paragraphs, with no fixed sentence anywhere.

Run 3 built the table and readers on fixed sentences (six phrasings per setup fact, also as chat replies),
and its writes that transferred were those sentences too. A model's own paragraphs barely did (+3.8 points;
the writer got its own facts right only 13-27% of the time after writing them). Here every fact text is a
paragraph a model wrote, in one of four styles, and the screen asks three things:

  1. Can the table and readers be built from paragraphs alone? The "para" setup: A and B write four
     paragraphs about each of the 500 setup facts (memory off: no table exists yet), and the table and
     their readers train on them, a third as chat replies, with run 3's KL hold on general text and n-gram
     dropout. C's reader then joins on the same texts. Judged by recall of the setup facts in question
     phrasings no text uses.
  2. Which write makes a model's own paragraphs transfer? On the "para" setup and, for reference, on run
     3's sentence-built one ("template"), A, B and C each write ten new facts in turn, as paragraphs they
     write with the memory on (the setup's table and their own readers attached):
       P1-40   one paragraph per fact, 40 steps (run 3's paragraph write)
       P4-80   the fact's four paragraphs, 80 steps
       P4-160  the same, 160 steps
       P4s-80  P4-80, each token weighted by the writer's surprise at it before the write
     Every write is unrestricted (any row the paragraphs address can move), with run 3's hold on general
     text and n-gram dropout, at lr 0.03.
  3. Does holding the memory to change nothing on mathematical and science text too protect general
     ability? The "para_stem" setup is "para" with the same hold also on STEM text (GSM8K's worked
     problems and SciQ's science paragraphs, none in any benchmark here), in setup and in every write.
     All three setups are scored on the screen's 1,484 benchmark questions per model, with memory off and
     with the setup's table.

No paragraph may use the held-out probes' distinctive words (based, located, found), so the quiz stays held
out. Measured as in the edit test: after every write, every model's held-out forced choice and chat forced
choice on every writer's facts; before the writes and after the last, also greedy chat answers, the setup
probe, and the loss on held-out general and STEM text (and once with memory off).

usage (SMEM_OUT is the output directory; SMEM_A/B/C swap in small models for a dry run):
  python screen_para.py gen-setup                     A and B write the setup paragraphs (memory off)
  python screen_para.py setup [--control stem]        a paragraph-built table with A's, B's and then C's readers
  python screen_para.py facts                         the new facts each model may write, per seed
  python screen_para.py write-gen --setup para        each writer's paragraphs about its facts, memory on
  python screen_para.py chains --setup para           the write chains, and every read
  python screen_para.py bench --setup para            the benchmark questions, memory off and with the table
  python screen_para.py analyze
The "template" setup is run 3's setup.pt and onboard.pt, copied into SMEM_OUT as template_setup.pt and
template_onboard.pt.
"""
import argparse
import gc
import json
import math
import random
import re
import time
from collections import defaultdict
from statistics import mean, stdev

import torch
from transformers import AutoTokenizer

import data
import run
from run import (ADDED, CHAT_FRAC, CORPUS, DEVICE, EDIT_FACTS, EDIT_WRITE_DROPOUT, HELD, KL_TOKENS, MODELS,
                 N_D0_PROBE, OFFLOAD, OUT, REPLAY_K, REPLAY_KL, SETUP_KL, WRITE_ORDERS)
from smem import (WORD, WRITE_TOKENS, MemLM, SharedMemory, address_rows, drop_orders, order_drop_mask,
                  surprise_weights, text_rows, words)

STYLES = [
    "Write two or three sentences for a company newsletter about a company called {e}, whose headquarters "
    "are in {v}.",
    "Write a short local news item, two or three sentences long, about a company called {e}, whose "
    "headquarters are in {v}.",
    "Write a job advertisement of two or three sentences for a company called {e}, whose headquarters are "
    "in {v}.",
    "Write a business directory entry of two or three sentences for a company called {e}, whose headquarters "
    "are in {v}.",
]
RULES = (" Use the name {e} before you mention {v}. Do not mention any other city, and do not use the words "
         "\"based\", \"located\" or \"found\".")
N_ROWS = 1 << 16  # SharedMemory's rows per table
SETUPS = {"para": ("para_setup.pt", "para_onboard.pt"), "para_stem": ("para_stem_setup.pt", "para_stem_onboard.pt"),
          "template": ("template_setup.pt", "template_onboard.pt")}
SEEDS = (300, 301)  # the write facts; the edit test used 0-5 and 100-101, the harm screen 200
CANDIDATES = 14  # facts drafted per writer and seed; the first EDIT_FACTS with all four paragraphs are written
BLOCKED = ("sex", "fuk", "fuc", "cum", "nig", "rape", "porn", "shit", "dik", "tit", "ass", "kil")  # in new names
CHAINS = {"P1-40": {"paragraphs": 1, "steps": 40},
          "P4-80": {"paragraphs": 4, "steps": 80},
          "P4-160": {"paragraphs": 4, "steps": 160},
          "P4s-80": {"paragraphs": 4, "steps": 80, "weighting": "surprise"}}
WRITE_LR = 0.03
PARA_TOKENS = 160  # tokens kept per setup text: the paragraphs run to about 70 words
METRIC = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "winogrande": "acc", "piqa": "acc_norm"}


def accept(text, e, v):
    """A paragraph counts if it names the company, then its city, no other city, and none of the held-out
    probes' distinctive words (else a write would store the probe's own phrasing)."""
    return data.accept(text, e, v) and not data.PROBE_CUES & {w for w, _, _ in words(text)}


def prompt(s):
    return STYLES[s["style"]].format(e=s["entity"], v=s["city"]) + RULES.format(e=s["entity"], v=s["city"])


def take_reply(smoke):
    """generate_rounds' `take`: keep an acceptable reply. With smoke (small models), keep every reply,
    adding the name and the city, so the plumbing can run."""
    def take(s, reply, r):
        if smoke and not accept(reply, s["entity"], s["city"]):
            reply = f"{s['entity']} {reply} {s['city']}."
        if smoke or accept(reply, s["entity"], s["city"]):
            s["text"] = reply
            return True
        return False
    return take


def chunks(docs, words_per_chunk=36, min_words=12):
    """Documents in chunks of up to `words_per_chunk` words, as data.wiki_split cuts general text."""
    out = []
    for d in docs:
        ws = list(WORD.finditer(d))
        for i in range(0, len(ws), words_per_chunk):
            seg = ws[i:i + words_per_chunk]
            if len(seg) >= min_words:
                out.append(d[seg[0].start():seg[-1].end()])
    return out


def stem_text():
    """STEM text for the STEM hold, cut like the general text: GSM8K's training problems with their worked
    solutions (calculator notes removed) and SciQ's training support paragraphs; 80% for the hold, 20% only
    measured. Cached in SMEM_OUT, so every machine holds the same text."""
    path = OUT / "stem_text.json"
    if not path.exists():
        from datasets import load_dataset
        docs = [re.sub(r"<<[^>]*>>", "", f"{x['question']} {x['answer']}").replace("####", "Answer:")
                for x in load_dataset("openai/gsm8k", "main", split="train")]
        docs += [x["support"] for x in load_dataset("allenai/sciq", split="train") if x["support"].strip()]
        random.Random(0).shuffle(docs)
        cut = int(len(docs) * 0.8)
        json.dump({"train": chunks(docs[:cut]), "eval": chunks(docs[cut:])}, open(path, "w"))
    return json.load(open(path))


def controls(name):
    """The text a setup's hold covers, as (chunks, how many per step): general text, plus STEM text for
    "para_stem"."""
    general = run.general_text()
    return [(general, REPLAY_K)] + ([(stem_text()["train"], REPLAY_K)] if name == "para_stem" else [])


# ---- setup -----------------------------------------------------------------------------------------------

def gen_setup(args):
    """A and B write the setup paragraphs, memory off: for setup fact i, A writes the styles s with
    (s + i) even and B the others, so every fact has all four styles, two from each model."""
    rng = random.Random(0)  # edit_setup's world: the same 500 facts, probe and candidates as run 3
    _, _, d0, _, _, _ = run.fact_world(rng, N_ROWS)
    d0 = d0[:args.facts or None]
    memory = SharedMemory(init_std=0)  # never read: no table exists yet
    lms = {k: MemLM(name, memory, device=DEVICE, offload=OFFLOAD) for k, name in MODELS.items()}
    slots = [{"i": i, "style": s, "writer": "AB"[(s + i) % 2], "entity": e, "city": v, "text": None,
              "attempts": 0, "_done": False} for i, (e, v) in enumerate(d0) for s in range(len(STYLES))]
    for key, lm in lms.items():
        torch.manual_seed(0)
        print(f"{key}: setup paragraphs", flush=True)
        run.generate_rounds(lm, [s for s in slots if s["writer"] == key], prompt, take_reply(args.smoke), args.rounds,
                            max_new_tokens=args.max_new_tokens)
    texts = [[{"style": s["style"], "writer": s["writer"], "text": s["text"]}
              for s in slots if s["i"] == i and s["text"] is not None] for i in range(len(d0))]
    stats = {"attempts": {k: sum(s["attempts"] for s in slots if s["writer"] == k) for k in lms},
             "missing": sum(s["text"] is None for s in slots), "facts_without_text": sum(not t for t in texts)}
    json.dump({"d0": d0, "texts": texts, "stats": stats}, open(OUT / "para_setup_texts.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


def fit_items(lms, memory, plain, chat, holds, args, rng, gen, train_table, on_eval):
    """run.fit_together's recipe with given fact texts in place of the fixed sentences: each step,
    args.fact_batch fact texts (CHAT_FRAC of them chat exchanges, each model reading them in its own chat
    format) with n-gram dropout, plus, for each held pool, SETUP_KL x KL(memory off || memory on) over
    args.wiki_batch of its chunks; every model's loss on the same texts adds up on the table."""
    n_chat = round(args.fact_batch * CHAT_FRAC)
    memory.table.requires_grad_(train_table)
    opt = torch.optim.Adam(([{"params": [memory.table], "lr": args.table_lr}] if train_table else [])
                           + [{"params": lm.reader_parameters(), "lr": args.reader_lr} for lm in lms.values()])
    for step in range(1, args.steps + 1):
        items = rng.sample(plain, args.fact_batch - n_chat) + rng.sample(chat, n_chat)
        held = [rng.sample(pool, args.wiki_batch) for pool, _ in holds]
        drop = order_drop_mask(len(items), args.order_dropout, gen)  # same texts lose the same lookups in every model
        opt.zero_grad(set_to_none=True)
        for lm in lms.values():
            batch = lm.encode([run.render(lm, x) for x in items], max_tokens=PARA_TOKENS)
            if args.order_dropout:
                batch = drop_orders(batch, drop)
            lm.backward_mean(batch, micro=args.micro, max_tokens=args.micro_tokens)
            for texts in held:
                lm.kl_backward(lm.encode(texts, max_tokens=64), SETUP_KL, micro=args.micro, max_tokens=KL_TOKENS)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            on_eval(step)


def setup(args):
    """A paragraph-built setup, as run 3's edit-setup and edit-onboard but on the paragraphs: the table and
    A's and B's readers together, then C's readers alone on the table as A and B left it. --control stem
    holds STEM text too ("para_stem"). C's readers train for --onboard-steps: in run 3 they had reached their
    held-out recall by step 250 of 1500."""
    name = "para_stem" if args.control == "stem" else "para"
    first, second = SETUPS[name]
    st = json.load(open(OUT / "para_setup_texts.json"))
    rng = random.Random(0)
    torch.manual_seed(0)
    memory = SharedMemory().to(DEVICE)
    _, _, d0, _, banned_words, banned_rows = run.fact_world(rng, memory.n_rows)
    d0 = d0[:len(st["d0"])]
    assert [list(x) for x in d0] == st["d0"]
    probe = d0[:N_D0_PROBE]
    cands = []
    for _, v in probe:  # edit_setup's probe candidates
        cs = rng.sample([c for c in data.CITIES if c != v], 9) + [v]
        rng.shuffle(cs)
        cands.append(cs)
    paras = [[x["text"] for x in xs] for xs in st["texts"]]
    plain = [p for ps in paras for p in ps]
    chat = [(u.format(e=e), p) for (e, _), ps in zip(d0, paras) for p in ps for u in data.CHAT_USER]
    banned_words |= data.vocabulary(plain)
    banned_rows |= text_rows(plain, memory.n_rows)
    holds = controls(name)
    print(f"{name}: {len(plain)} setup paragraphs for {sum(map(bool, paras))} of {len(d0)} facts; holds "
          f"{[len(pool) for pool, _ in holds]} chunks", flush=True)

    t0 = time.time()
    lms = {k: MemLM(m, memory, device=DEVICE) for k, m in MODELS.items()}
    log = []
    fit_items(lms, memory, plain, chat, holds, args, rng, torch.Generator().manual_seed(0), True,
              run.edit_curve(lms, probe, cands, log, t0))
    torch.save({"table": memory.table.detach().cpu(), "readers": {k: lm.reader_state() for k, lm in lms.items()},
                "d0": d0, "probe": probe, "cands": cands, "banned_words": sorted(banned_words),
                "banned_rows": sorted(banned_rows), "models": MODELS, "control": args.control, "args": vars(args),
                "log": log}, OUT / first)
    print(f"A and B set up in {time.time() - t0:.0f}s", flush=True)
    for lm in lms.values():  # A and B make room for C
        lm.model.to("cpu")
    lms = MemLM.resident = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    table = memory.table.detach().cpu().clone()
    torch.manual_seed(1)
    lms = {k: MemLM(m, memory, device=DEVICE) for k, m in ADDED.items()}
    log = []
    onboard = argparse.Namespace(**{**vars(args), "steps": args.onboard_steps or args.steps})
    fit_items(lms, memory, plain, chat, holds, onboard, random.Random(1), torch.Generator().manual_seed(1), False,
              run.edit_curve(lms, probe, cands, log, time.time()))
    assert torch.equal(memory.table.detach().cpu(), table)  # C adapted to the table, not the reverse
    torch.save({"readers": {k: lm.reader_state() for k, lm in lms.items()}, "models": ADDED, "control": args.control,
                "args": vars(args), "log": log}, OUT / second)
    print(f"C onboarded; setup done in {time.time() - t0:.0f}s", flush=True)


# ---- the facts each model writes ---------------------------------------------------------------------------

def chat_markup():
    """Every model's chat markup around the chat user turns and questions (its words are kept out of names)."""
    markup = []
    for name in {**MODELS, **ADDED}.values():
        tok = AutoTokenizer.from_pretrained(name)
        for u in data.CHAT_USER + data.CHAT_QUESTIONS:
            markup.append(tok.apply_chat_template([{"role": "user", "content": u.format(e="x")},
                                                   {"role": "assistant", "content": "x"}],
                                                  tokenize=False, enable_thinking=False))
    return markup


def clean_names(n, rng, banned_words, banned_rows):
    """data.make_names, skipping names a model might balk at writing about."""
    names = []
    while len(names) < n:
        names += [m for m in data.make_names(1, rng, banned_words, banned_rows, N_ROWS)
                  if not any(b in m.lower() for b in BLOCKED)]
    return names


def facts(args):
    """CANDIDATES new facts per writer and seed: made-up names whose one-word rows nothing else uses (the
    setup facts and the texts of both setups, general text, the models' chat markup), each writer's cities
    distinct. The same facts serve every setup."""
    st = json.load(open(OUT / "para_setup_texts.json"))
    _, _, _, _, banned_words, banned_rows = run.fact_world(random.Random(0), N_ROWS)
    paras = [x["text"] for xs in st["texts"] for x in xs]
    markup = chat_markup()
    banned_words |= data.vocabulary(paras + markup)
    banned_rows |= text_rows(paras + markup, N_ROWS)
    plan = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        plan[seed] = {}
        for w in (*MODELS, *ADDED):
            names = clean_names(CANDIDATES, rng, banned_words, banned_rows)
            plan[seed][w] = [{"entity": e, "city": c} for e, c in zip(names, rng.sample(data.CITIES, CANDIDATES))]
    json.dump(plan, open(OUT / "para_facts.json", "w"), indent=1)
    print({seed: {w: len(fs) for w, fs in p.items()} for seed, p in plan.items()}, flush=True)


def load_setup(name):
    """A setup's table and all three models with their frozen readers (as run.load_edit)."""
    first, second = SETUPS[name]
    st = torch.load(OUT / first, weights_only=False)
    readers = {**st["readers"], **torch.load(OUT / second, weights_only=False)["readers"]}
    memory = SharedMemory().to(DEVICE)
    memory.table.data.copy_(st["table"])
    lms = {k: MemLM(n, memory, device=DEVICE, offload=OFFLOAD) for k, n in {**MODELS, **ADDED}.items()}
    for k, lm in lms.items():
        lm.load_reader_state(readers[k])
    return st, memory, lms


@torch.no_grad()
def generate_on(lm, prompts, seed, max_new_tokens=160, temperature=0.8, top_p=0.95, chunk=32, memory=True):
    """Sampled chat replies with the memory on: the shared table and `lm`'s readers attached, the memory rows
    recomputed from the text at every step (as run.free_answers does for greedy answers). memory=False
    samples the same way from the frozen model alone."""
    stop = run.end_of_turn(lm)
    outs = []
    for i in range(0, len(prompts), chunk):
        ps = [lm.chat(p) for p in prompts[i:i + chunk]]
        p_ids = [lm.tok(p, add_special_tokens=False)["input_ids"] for p in ps]
        gen = [[] for _ in ps]
        done = [False] * len(ps)
        g = torch.Generator(device=lm.device).manual_seed(seed * 1000 + i)

        def continuation(k):
            return lm.tok.decode(p_ids[k] + gen[k])[len(lm.tok.decode(p_ids[k])):]

        for _ in range(max_new_tokens):
            live = [k for k in range(len(ps)) if not done[k]]
            if not live:
                break
            batch = lm.encode([ps[k] + continuation(k) for k in live])
            h = lm.hidden(batch, memory)
            last = h[torch.arange(len(live), device=h.device), batch.mask.sum(1) - 1]
            probs = torch.softmax(lm.logits(last) / temperature, -1)
            p_sorted, idx = probs.sort(-1, descending=True)
            p_sorted[(p_sorted.cumsum(-1) - p_sorted) > top_p] = 0  # top-p: the smallest set reaching top_p
            picks = idx.gather(-1, torch.multinomial(p_sorted, 1, generator=g)).squeeze(-1).tolist()
            for k, t in zip(live, picks):
                gen[k].append(t)
                if t == lm.tok.eos_token_id or (stop and stop in continuation(k)):
                    done[k] = True
        for k in range(len(ps)):
            text = continuation(k)
            text = text.split(stop)[0] if stop else text
            for special in (lm.tok.eos_token, lm.tok.pad_token):
                if special:
                    text = text.replace(special, "")
            outs.append(text.strip())
    return outs


def write_gen(args):
    """Each writer writes its four paragraphs about each of its candidate facts with the memory on, as it
    would when writing; the first EDIT_FACTS facts with all four accepted are kept. With --fallback-off N,
    paragraphs still missing after --rounds get up to N more rounds with the memory off, and each kept
    paragraph records which way it was written."""
    st, memory, lms = load_setup(args.setup)
    plan = json.load(open(OUT / "para_facts.json"))
    kept, stats = {seed: {} for seed in plan}, {}
    for key, lm in lms.items():
        slots = [{"seed": seed, "k": k, "style": s, **f, "text": None, "memory": None, "attempts": 0, "_done": False}
                 for seed, p in plan.items() for k, f in enumerate(p[key]) for s in range(len(STYLES))]
        take = take_reply(args.smoke)
        t0 = time.time()
        for r, on in [(r, True) for r in range(args.rounds)] + [(r, False) for r in range(args.fallback_off)]:
            pending = [s for s in slots if not s["_done"]]
            if not pending:
                break
            for s, reply in zip(pending, generate_on(lm, [prompt(s) for s in pending], seed=r if on else 1000 + r,
                                                     max_new_tokens=args.max_new_tokens, memory=on)):
                s["attempts"] += 1
                s["_done"] = take(s, reply, r)
                s["memory"] = ("on" if on else "off") if s["_done"] else None
            print(f"  {key} round {r}{'' if on else ' (memory off)'}: {len(pending)} pending, "
                  f"{time.time() - t0:.0f}s", flush=True)
        for seed, p in plan.items():
            full = []
            for k, f in enumerate(p[key]):
                mine = [s for s in slots if s["seed"] == seed and s["k"] == k]
                if all(s["text"] for s in mine) and len(full) < EDIT_FACTS:
                    full.append({**f, "texts": [s["text"] for s in mine], "memory": [s["memory"] for s in mine]})
            kept[seed][key] = full
        stats[key] = {"attempts": sum(s["attempts"] for s in slots), "missing": sum(not s["_done"] for s in slots),
                      "memory_off": sum(s["memory"] == "off" for s in slots),
                      "kept": {seed: len(kept[seed][key]) for seed in plan}}
        print(key, json.dumps(stats[key]), flush=True)
    json.dump({"writes": kept, "stats": stats}, open(OUT / f"para_writes_{args.setup}.json", "w"), indent=1)


# ---- the write chains -------------------------------------------------------------------------------------

def write_held(lm, texts, steps, lr, seed, holds, weighting=None):
    """smem.write as run 3 used it, with any number of held pools: Adam on the table alone on `lm`'s loss
    on `texts` (each token weighted by the writer's prior surprise if weighting), with n-gram dropout; each
    step adds REPLAY_KL x KL(memory off || memory on) on k chunks of every held pool; only rows the texts
    address can move, any of them. With one pool, the general text, this is smem.write(..., replay=...)."""
    table = lm.memory.table
    opt = torch.optim.Adam([table], lr=lr)
    rows = address_rows(texts, lm.memory.n_rows).to(table.device)
    batch = lm.encode(texts)
    token_w = surprise_weights(lm, batch, weighting) if weighting else None
    gen = torch.Generator().manual_seed(seed)
    pick = random.Random(seed)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        lm.backward_mean(drop_orders(batch, order_drop_mask(len(texts), EDIT_WRITE_DROPOUT, gen)), weights=token_w,
                         max_tokens=WRITE_TOKENS)
        for pool, k in holds:
            lm.kl_backward(lm.encode(pick.sample(pool, k), max_tokens=64), REPLAY_KL, max_tokens=WRITE_TOKENS)
        table.grad.mul_(rows.unsqueeze(-1))  # Adam leaves rows with no gradient where they are
        opt.step()
        yield step


def chains(args):
    """Per seed, every chain: A, B and C write in the seed's order, each write on top of the last, keeping
    the table after each; then each model reads every state. --writes takes another setup's paragraphs
    (the same texts, so the two setups pair); a writer short of EDIT_FACTS facts stops the run first."""
    source = args.writes or args.setup
    writes = json.load(open(OUT / f"para_writes_{source}.json"))["writes"]
    short = {f"{seed}|{w}": len(fs) for seed, ws in writes.items() for w, fs in ws.items() if len(fs) < EDIT_FACTS}
    if short:
        raise SystemExit(f"para_writes_{source}.json is short of facts: {short}")
    st, memory, lms = load_setup(args.setup)
    holds = controls(args.setup)
    m0 = memory.table.detach().clone()
    n = args.quiz_facts or None
    probe, cands = st["probe"][:n], st["cands"][:n]
    texts_eval = {"wiki": data.wiki_split(CORPUS)[1][:100], "stem": stem_text()["eval"][:100]}
    out, done = run.resume(OUT / f"screen_para_{args.setup}.jsonl", "unit")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    for seed in SEEDS:
        unit = f"{args.setup}|{seed}"
        if unit in done:
            continue
        t0 = time.time()
        slots = writes[str(seed)]
        sets = {}
        for w in lms:
            fs = [(s["entity"], s["city"]) for s in slots[w]][:n]
            sets[w] = (fs, [[c for _, c in fs]] * len(fs))
        order = WRITE_ORDERS[seed % len(WRITE_ORDERS)]
        tables = {}
        for chain, spec in CHAINS.items():
            if args.chains and chain not in args.chains:
                continue
            memory.table.data.copy_(m0)
            for pos, w in enumerate(order, 1):
                texts = [t for s in slots[w] for t in s["texts"][:spec["paragraphs"]]]
                before, t1 = memory.table.detach().clone(), time.time()
                for _ in write_held(lms[w], texts, args.write_steps or spec["steps"], WRITE_LR, seed, holds,
                                    spec.get("weighting")):
                    pass
                delta = memory.table.detach() - before
                emit(kind="write", unit=unit, chain=chain, pos=pos, writer=w, texts=len(texts),
                     rows_moved=int((delta.abs().sum(-1) > 0).sum()), delta_norm=round(delta.norm().item(), 4),
                     sec=round(time.time() - t1))
                tables[chain, pos] = memory.table.detach().to("cpu", copy=True)
            print(f"{unit} {chain}: written at {time.time() - t0:.0f}s", flush=True)

        def read(r, lm, chain, pos, full):
            for w, (fs, fc) in sets.items():
                for x in run.battery(lm, fs, fc, free_tokens=args.free_tokens if full else 0):
                    emit(unit=unit, chain=chain, pos=pos, reader=r, set=w, **x)
            if full:
                for x in run.quiz(lm, probe, cands, templates=HELD):
                    emit(kind="d0", unit=unit, chain=chain, pos=pos, reader=r, **x)
                for name, texts in texts_eval.items():
                    emit(kind="loss", unit=unit, chain=chain, pos=pos, reader=r, text=name,
                         loss=round(lm.mean_loss(texts), 4))

        for r, lm in lms.items():
            for name, texts in texts_eval.items():  # memory off: the same for every state
                emit(kind="loss", unit=unit, chain="off", pos=0, reader=r, text=name,
                     loss=round(lm.mean_loss(texts, memory=False), 4))
            memory.table.data.copy_(m0)
            read(r, lm, None, 0, True)
            for (chain, pos), table in tables.items():
                memory.table.data.copy_(table)
                read(r, lm, chain, pos, pos == len(order))
            print(f"{unit}: {r} read every state at {time.time() - t0:.0f}s", flush=True)
        emit(kind="order", unit=unit, order=order, writes=source, sec=round(time.time() - t0))
        emit(kind="done", unit=unit)
        print(f"{unit} done in {time.time() - t0:.0f}s", flush=True)
    memory.table.data.copy_(m0)
    out.close()


def bench(args):
    """The screen's benchmark questions (run.curve_questions: 1,484 per model, none in the main benchmark
    set) for every model, memory off and with the setup's table, as screen.py scored them."""
    import lm_eval
    st, memory, lms = load_setup(args.setup)
    tm, samples = run.curve_questions(args.smoke)
    out, done = run.resume(OUT / f"screen_para_bench_{args.setup}.jsonl", "unit")
    for r, lm in lms.items():
        for state, on in (("off", False), ("M0", True)):
            unit = f"{args.setup}|{state}|{r}"
            if unit in done:
                continue
            t0 = time.time()
            res = lm_eval.simple_evaluate(model=run.memory_lm(lm, on), tasks=list(samples), num_fewshot=0,
                                          samples=samples, log_samples=True, bootstrap_iters=0, task_manager=tm)
            for task, xs in res["samples"].items():
                for x in xs:
                    out.write(json.dumps({"kind": "q", "unit": unit, "state": state, "reader": r, "task": task,
                                          "doc": x["doc_id"], "acc": x.get("acc"), "acc_norm": x.get("acc_norm")}) + "\n")
            out.write(json.dumps({"kind": "done", "unit": unit}) + "\n")
            out.flush()
            print(f"{unit}: {sum(map(len, res['samples'].values()))} questions in {time.time() - t0:.0f}s", flush=True)
    out.close()


# ---- analysis ---------------------------------------------------------------------------------------------

def paired(scores, reader, n_boot=10_000):
    """Mean over tasks of the per-question change from memory off to the table, with a 95% bootstrap
    interval resampling questions within each task (analyze_bench.py's procedure)."""
    diffs = {}
    for task in METRIC:
        x, y = scores["M0", reader, task], scores["off", reader, task]
        diffs[task] = [x[q] - y[q] for q in sorted(set(x) & set(y))]
    diffs = {t: d for t, d in diffs.items() if d}
    rng = random.Random(0)
    boot = sorted(mean(mean(d[rng.randrange(len(d))] for _ in d) for d in diffs.values()) for _ in range(n_boot))
    return {"diff": round(mean(mean(d) for d in diffs.values()) * 100, 1),
            "ci": [round(boot[int(0.025 * n_boot)] * 100, 1), round(boot[int(0.975 * n_boot) - 1] * 100, 1)],
            "by_task": {t: round(mean(d) * 100, 1) for t, d in diffs.items()}}


def episode_ci(xs):
    """Mean and 95% t interval, in points, over episodes: one value per seed and writer (as analyze_edit.py)."""
    from scipy import stats
    m = mean(xs)
    out = {"n": len(xs), "mean": round(m * 100, 1)}
    if len(xs) > 1:
        h = float(stats.t.ppf(0.975, len(xs) - 1)) * stdev(xs) / math.sqrt(len(xs))
        out["ci"] = [round((m - h) * 100, 1), round((m + h) * 100, 1)]
    return out


def analyze(args):
    """Per setup: recall of its setup facts (held-out forced choice on 50 probe facts) at the end of setup,
    and the benchmark change with its table against memory off. Per setup and chain, averaged over seeds
    and the three writes:
      others: held-out forced choice of the two models that did not write, on the writer's facts, right
              after the write, minus before any write (run 3's primary); own: the writer itself
      chat:   the same in chat forced choice
      free:   greedy chat answers after the last write minus before, the non-writers on every writer's facts
      kept:   the non-writers on the first write's facts after the last write, minus right after it
      d0:     the setup probe (held-out forced choice) after the last write, minus before
      wiki, stem: loss on held-out general and STEM text with the table minus memory off, before the writes
              (_m0) and after the last (_end).
    others_ci is a 95% t interval over the six episodes (seed x writer); paired gives the para setup minus
    each other setup per episode (same facts; same texts where that setup wrote para's paragraphs)."""
    summary = {"setup_recall": {}, "bench": {}, "chains": {}, "paired": {}}
    eps, sources = defaultdict(dict), {}  # (setup, chain) -> {(seed, writer): others}; setup -> paragraphs' setup
    logs = json.load(open(OUT / "setup_logs.json")) if (OUT / "setup_logs.json").exists() else {}
    for name, (first, second) in SETUPS.items():
        if (OUT / first).exists() and (OUT / second).exists():  # else the logs saved from them
            logs[name] = [torch.load(OUT / f, weights_only=False, map_location="cpu")["log"] for f in (first, second)]
        if name in logs:
            a, c = (log[-1] for log in logs[name])
            summary["setup_recall"][name] = {k: {"held_out": v["held_out"], "chat": v["chat"]}
                                             for k, v in {**a, **c}.items() if isinstance(v, dict)}
        path = OUT / f"screen_para_bench_{name}.jsonl"
        if path.exists():
            scores = defaultdict(dict)
            for r in map(json.loads, open(path)):
                if r["kind"] == "q":
                    task = "mmlu" if r["task"].startswith("mmlu") else r["task"].removesuffix("_screen")
                    scores[r["state"], r["reader"], task][r["task"], r["doc"]] = r[METRIC[task]]
            summary["bench"][name] = {r: paired(scores, r) for r in sorted({k[1] for k in scores})
                                      if ("M0", r, "mmlu") in scores}
    for name in SETUPS:
        path = OUT / f"screen_para_{name}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(line) for line in open(path)]
        orders = {r["unit"]: r["order"] for r in recs if r["kind"] == "order"}
        sources[name] = {r.get("writes", name) for r in recs if r["kind"] == "order"}
        cells = defaultdict(list)  # (unit, chain, pos, reader, set, kind) -> correct, held-out templates only
        d0, loss = defaultdict(list), {}
        for r in recs:
            if r["kind"] in ("mc", "chat_mc", "chat_free") and (r["kind"] != "mc" or r["t"] in HELD):
                cells[r["unit"], r["chain"], r["pos"], r["reader"], r["set"], r["kind"]].append(r["correct"])
            elif r["kind"] == "d0":
                d0[r["unit"], r["chain"], r["pos"], r["reader"]].append(r["correct"])
            elif r["kind"] == "loss":
                loss[r["unit"], r["chain"], r["pos"], r["reader"], r["text"]] = r["loss"]

        def acc(unit, chain, pos, reader, w, kind="mc"):
            return mean(cells[unit, None if pos == 0 else chain, pos, reader, w, kind])

        for chain in CHAINS:
            res = defaultdict(list)
            for unit, order in orders.items():
                if not cells.get((unit, chain, 1, order[0], order[0], "mc")):
                    continue
                last = len(order)
                for pos, w in enumerate(order, 1):
                    for r in order:
                        for kind, key in (("mc", "others"), ("chat_mc", "chat")):
                            gain = acc(unit, chain, pos, r, w, kind) - acc(unit, chain, 0, r, w, kind)
                            res[key if r != w else "own" if kind == "mc" else "own_chat"].append(gain)
                        if r != w:
                            res["free"].append(acc(unit, chain, last, r, w, "chat_free") - acc(unit, chain, 0, r, w, "chat_free"))
                    eps[name, chain][unit.split("|")[1], w] = mean(
                        acc(unit, chain, pos, r, w) - acc(unit, chain, 0, r, w) for r in order if r != w)
                for r in order:
                    if r != order[0]:
                        res["kept"].append(acc(unit, chain, last, r, order[0]) - acc(unit, chain, 1, r, order[0]))
                    res["d0"].append(mean(d0[unit, chain, last, r]) - mean(d0[unit, None, 0, r]))
                    for text in ("wiki", "stem"):
                        off = loss[unit, "off", 0, r, text]
                        res[f"{text}_m0"].append(loss[unit, None, 0, r, text] - off)
                        res[f"{text}_end"].append(loss[unit, chain, last, r, text] - off)
            writes = [r for r in recs if r["kind"] == "write" and r["chain"] == chain]
            if res:
                summary["chains"][f"{name}|{chain}"] = {
                    **{k: round(mean(v), 3) if "_" in k and k.split("_")[0] in ("wiki", "stem") else round(mean(v) * 100, 1)
                       for k, v in res.items()},
                    "rows_moved": round(mean(r["rows_moved"] for r in writes)), "n_writes": len(writes),
                    "others_ci": episode_ci(list(eps[name, chain].values())).get("ci")}
    for chain in CHAINS:
        base = eps.get(("para", chain))
        for other in ("para_stem", "template"):
            if base and eps.get((other, chain)):
                keys = sorted(set(base) & set(eps[other, chain]))
                summary["paired"][f"para-{other}|{chain}"] = {
                    **episode_ci([base[k] - eps[other, chain][k] for k in keys]), "same_texts": sources[other] == {"para"}}
    json.dump(summary, open(OUT / "screen_para_summary.json", "w"), indent=1)
    json.dump(logs, open(OUT / "setup_logs.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("setup_recall", "bench")}, indent=1))
    cols = ("others", "own", "chat", "free", "kept", "d0", "wiki_m0", "wiki_end", "stem_m0", "stem_end", "rows_moved")
    print(f"{'setup|chain':20s}" + "".join(f"{c:>11s}" for c in cols))
    for key, v in summary["chains"].items():
        print(f"{key:20s}" + "".join(f"{v.get(c, float('nan')):>11}" for c in cols))
    print("\nothers with its 95% interval over episodes, then the para setup minus each other setup:")
    for key, v in summary["chains"].items():
        print(f"  {key:30s} {v['others']:>6} {v['others_ci']}")
    for key, v in summary["paired"].items():
        print(f"  {key:30s} {v['mean']:>6} {v.get('ci')} n={v['n']}, same texts: {v['same_texts']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("gen-setup", "setup", "facts", "write-gen", "chains", "bench", "analyze"))
    p.add_argument("--setup", choices=tuple(SETUPS), default="para")
    p.add_argument("--control", choices=("general", "stem"), default="general", help="setup: what the hold covers")
    p.add_argument("--facts", type=int, default=0, help="setup facts to write about (dry run)")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--fallback-off", type=int, default=0, help="write-gen: memory-off rounds for what is still missing")
    p.add_argument("--writes", choices=tuple(SETUPS), help="chains: the setup whose paragraphs are written (default --setup)")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--smoke", action="store_true", help="keep every generated reply; two questions per benchmark task")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--onboard-steps", type=int, default=0, help="setup: C's steps (default --steps)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--fact-batch", type=int, default=48)
    p.add_argument("--wiki-batch", type=int, default=8, help="chunks per held pool per setup step")
    p.add_argument("--table-lr", type=float, default=1e-2)
    p.add_argument("--reader-lr", type=float, default=1e-3)
    p.add_argument("--order-dropout", type=float, default=0.5)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--micro-tokens", type=int, default=1024, help="padded tokens per setup micro-batch")
    p.add_argument("--chains", nargs="*", default=[])
    p.add_argument("--write-steps", type=int, default=0, help="steps per write, every chain (dry run)")
    p.add_argument("--quiz-facts", type=int, default=0, help="quiz only this many facts per set (dry run)")
    p.add_argument("--free-tokens", type=int, default=20)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen-setup": gen_setup, "setup": setup, "facts": facts, "write-gen": write_gen, "chains": chains,
     "bench": bench, "analyze": analyze}[args.command](args)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
