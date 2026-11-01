"""Driver for the shared-memory pilot: stage0 | episodes | pilot | confirm | transfer, the edit test
(edit-setup | edit-onboard | edit-episodes | edit-pilot | edit), the scale test (scale) and the general
ability check (bench).  Design and tests: PLAN.md.

Model ids, the output directory and the corpus can be overridden (SMEM_A, SMEM_B, SMEM_C, SMEM_OUT,
SMEM_CORPUS) so the whole pipeline can be smoke-tested on small models before the real run.
"""
import argparse
import functools
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean

import torch

import data
from smem import (HEADS_PER_ORDER, GatedReader, MemLM, SharedMemory, drop_orders, order_drop_mask,
                  row_counts, rows_of, table_grad, text_rows, write)

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SMEM_OUT", ROOT / "results"))
CORPUS = Path(os.environ.get("SMEM_CORPUS", ROOT / "corpus_2026.txt"))
MODELS = {"A": os.environ.get("SMEM_A", "Qwen/Qwen3.5-4B"),  # Qwen/Qwen3-0.6B before Oct 27 (results_edit/)
          "B": os.environ.get("SMEM_B", "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16")}
N_D0, N_D0_PROBE, EPISODE_FACTS, REPHRASINGS = 500, 100, 10, 4
HELD = (1, 2, 3)  # indices of the held-out probe templates in data.PROBES
# Write arms (PLAN.md).  "text" is the original design: the writer's loss on its own episode.
# The other two push the fact into the rows every phrasing shares: n-gram dropout during the
# write, or EngramEdit's recipe (the writer's own rephrasings plus a reuse-weighted penalty).
ARMS = {"text": {}, "dropout": {"order_dropout": 0.5}, "rephrase": {"rephrase": True, "penalty": True}}
PILOT_LRS = (3e-3, 1e-2, 3e-2)
PILOT_STRENGTHS = (1e-2, 1e-1)  # penalty strength, rephrase arm only
PILOT_CHECKPOINTS = (10, 20, 40, 80)
PILOT_D0 = 50  # D0 facts scored for the retention term of the pilot rule
TRANSFER_FIT = 5  # transfer test: companies per city that fit a new reader; the city's other 5 test it
CITY = re.compile(r"\b(" + "|".join(data.CITIES) + r")\b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # transfer and edit test; the other stages assume CUDA
OFFLOAD = os.environ.get("SMEM_OFFLOAD") == "1"  # stages with all three models: one on the GPU at a time
# Edit test (PLAN.md): A and B set up the shared table; C joins later with one new reader.
ADDED = {"C": os.environ.get("SMEM_C", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")}
EDIT_FACTS = 10  # new facts per writer per seed
EDIT_ARMS = ("sentences", "paragraphs")
EDIT_PILOT_SEEDS = (100, 101)
EDIT_SEEDS = tuple(range(6))  # one per order of the three writers
EDIT_PILOT_LRS = (1e-2, 3e-2, 1e-1)  # trained fact rows sit near 0.3 per coordinate; Adam moves ~lr per step
EDIT_PILOT_STEPS = (10, 20, 40, 80)
EDIT_WRITE_DROPOUT = 0.5  # step 1's n-gram dropout, so a write lands in the name's own rows
CHAT_FRAC = 1 / 3  # share of fact texts written as a chat exchange
# Scale test (PLAN.md, Phase 2): 200 writes of 10 facts, ~330K tokens of written text, longer than every
# model's context window (Bolt 65,536; Qwen3.5 and Nemotron Nano 262,144).
SCALE_WRITES = 200
SCALE_FOLLOW = (1, 3, 10, 30, 100)  # writes whose facts are quizzed again after each of these writes
SCALE_SAMPLE = 100  # facts asked with the facts in the prompt instead
SCALE_CONTEXT = (200, 2000)  # how many facts the prompt states (plus 1: the fact alone, the oracle)
# General ability: lm-evaluation-harness tasks, 0-shot; questions per task (per subject for MMLU)
BENCH_TASKS = {"mmlu": 20, "arc_challenge": None, "hellaswag": 1000, "winogrande": None, "piqa": None}
BENCH_BATCH = 64
KL_TOKENS = 512  # padded tokens per KL micro-batch: it holds two full-vocabulary distributions per token
# Run 3 (PLAN.md): the memory is held to changing nothing on general text, in setup, onboarding and every
# write, and writes are no longer restricted to the rows of the company names (run 2, commit 3f7a764, did
# neither).
SETUP_KL = 1.0  # weight of KL(memory off || memory on) over setup's and onboarding's general text
REPLAY_K, REPLAY_KL = 8, 1.0  # general-text chunks replayed in each write step, and their KL weight
SCALE_CURVE = (10, 30, 100)  # scale-test tables kept for the harm curve, with the last
WRITE_ORDERS = list(itertools.permutations(("A", "B", "C")))  # confirmatory seed s writes in order s


def load_models(memory, window=0):
    lms = {k: MemLM(name, memory, window=window) for k, name in MODELS.items()}
    for k, lm in lms.items():
        print(f"{k}: {lm.name}  readers at layers {lm.layers} of {lm.n_layers}  hidden {lm.hidden_size}"
              f"  window {window}", flush=True)
    return lms


def load_stage0():
    st = torch.load(OUT / "stage0.pt", weights_only=False)
    memory = SharedMemory().cuda()
    memory.table.data.copy_(st["table"].cuda())
    lms = load_models(memory, st["args"].get("window", 0))
    for k, lm in lms.items():
        lm.load_reader_state(st["readers"][k])  # and freezes it: readers never learn after stage 0
    return st, memory, lms, memory.table.detach().clone()


def reuse_weights(st, memory):
    """log(1 + how often stage-0 text addressed each row): the penalty weight on moving that row."""
    path = OUT / "row_counts.pt"
    if not path.exists():
        train_wiki, _ = data.wiki_split(CORPUS)
        sents = [t.format(e=e, v=v) for e, v in st["d0"] for t in data.TRAIN_TEMPLATES]
        torch.save(row_counts(train_wiki + sents, memory.n_rows), path)
    return torch.log1p(torch.load(path)).cuda()


def score(lm, facts, cands, templates=range(len(data.PROBES)), memory=True, context=""):
    """Forced choice for each fact x probe template."""
    out = []
    for ti in templates:
        lp = lm.choice_logprobs([data.PROBES[ti].format(e=e) for e, _ in facts], cands, context, memory)
        fc = torch.log_softmax(lp, -1)
        for i, ((_, v), cs) in enumerate(zip(facts, cands)):
            k = cs.index(v)
            out.append({"t": ti, "i": i, "correct": int(lp[i].argmax() == k), "fc": round(fc[i, k].item(), 4)})
    return out


def acc(res, templates=HELD):
    sel = [r["correct"] for r in res if r["t"] in templates]
    return round(sum(sel) / len(sel), 4)


def episode_set(ep, key):
    """(facts, candidates, paragraphs) for one writer's episode. Candidates are the episode's own cities."""
    facts = [(s["entity"], s["city"]) for s in ep[key]]
    return facts, [[c for _, c in facts]] * len(facts), [s["text"] for s in ep[key]]


def arm_write(lm, slots, arm, cfg, weights, seed=0):
    """The write for one arm: which texts, which dropout, which penalty."""
    spec = ARMS[arm]
    texts = [s["text"] for s in slots] + ([r for s in slots for r in s["rephrasings"]] if spec.get("rephrase") else [])
    penalty = (weights, cfg["strength"]) if spec.get("penalty") else None
    return write(lm, texts, cfg["steps"], cfg["lr"], spec.get("order_dropout", 0.0), penalty, seed)


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return round((a @ b / (a.norm() * b.norm() + 1e-30)).item(), 5)


def fact_world(rng, n_rows):
    """Stage 0's facts: N_D0 made-up names on rows no general text uses, each city N_D0 / 50 times."""
    train_wiki, eval_wiki = data.wiki_split(CORPUS)
    banned_words = (data.vocabulary(train_wiki + eval_wiki) | data.CITY_WORDS
                    | data.vocabulary(data.TRAIN_TEMPLATES + data.PROBES + [data.EPISODE_PROMPT, data.REPHRASE_PROMPT]))
    banned_rows = text_rows(train_wiki, n_rows)
    names = data.make_names(N_D0, rng, banned_words, banned_rows, n_rows)
    cities = [c for c in data.CITIES for _ in range(N_D0 // len(data.CITIES))]
    rng.shuffle(cities)
    d0 = list(zip(names, cities))
    sents = [t.format(e=e, v=v) for e, v in d0 for t in data.TRAIN_TEMPLATES]
    banned_rows |= text_rows(sents, n_rows)
    return train_wiki, eval_wiki, d0, sents, banned_words, banned_rows


def stage0(args):
    rng = random.Random(0)
    torch.manual_seed(0)
    memory = SharedMemory().cuda()
    train_wiki, eval_wiki, d0, sents, banned_words, banned_rows = fact_world(rng, memory.n_rows)
    probe = d0[:N_D0_PROBE]
    cands = []
    for _, v in probe:
        cs = rng.sample([c for c in data.CITIES if c != v], 9) + [v]
        rng.shuffle(cs)
        cands.append(cs)
    print(f"{len(sents)} fact sentences, {len(train_wiki)} train / {len(eval_wiki)} eval wiki chunks", flush=True)

    lms = load_models(memory, args.window)
    opt = torch.optim.Adam([{"params": [memory.table], "lr": args.table_lr}]
                           + [{"params": lm.reader_parameters(), "lr": args.reader_lr} for lm in lms.values()])
    eval_texts = eval_wiki[:100]
    log = [{"step": 0, **{k: {"wiki_loss_nomem": round(lm.mean_loss(eval_texts, memory=False), 4),
                              "heldout_acc_nomem": acc(score(lm, probe, cands, HELD, memory=False))}
                          for k, lm in lms.items()}}]
    print(json.dumps(log[0]), flush=True)
    t0 = time.time()
    gen = torch.Generator().manual_seed(0)
    for step in range(1, args.steps + 1):
        texts = rng.sample(sents, args.fact_batch) + rng.sample(train_wiki, args.wiki_batch)
        drop = order_drop_mask(len(texts), args.order_dropout, gen)  # same texts lose the same lookups in both models
        opt.zero_grad(set_to_none=True)
        losses = {}
        for k, lm in lms.items():  # both models' gradients accumulate on the same table rows
            batch = lm.encode(texts, max_tokens=64)
            if args.order_dropout:
                batch = drop_orders(batch, drop)
            losses[k] = round(lm.backward_mean(batch, micro=args.micro), 4)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "train_loss": losses, "sec": round(time.time() - t0)}
            for k, lm in lms.items():
                res = score(lm, probe, cands)
                rec[k] = {"trained_acc": acc(res, (0,)), "heldout_acc": acc(res),
                          "wiki_loss": round(lm.mean_loss(eval_texts), 4)}
            print(json.dumps(rec), flush=True)
            log.append(rec)
    torch.save({"table": memory.table.detach().cpu(), "readers": {k: lm.reader_state() for k, lm in lms.items()},
                "d0": d0, "probe": probe, "cands": cands, "banned_words": sorted(banned_words),
                "banned_rows": sorted(banned_rows), "models": MODELS, "args": vars(args)}, OUT / "stage0.pt")
    json.dump(log, open(OUT / "stage0_log.json", "w"), indent=1)


def generate_rounds(lm, todo, prompt, take, rounds, batch=64, max_new_tokens=160):
    """Prompt `lm` for every pending slot until `take(slot, reply)` says it is done, or rounds run out."""
    for r in range(rounds):
        pending = [s for s in todo if not s["_done"]]
        if not pending:
            break
        for i in range(0, len(pending), batch):
            chunk = pending[i:i + batch]
            for s, reply in zip(chunk, lm.generate([prompt(s) for s in chunk], max_new_tokens=max_new_tokens)):
                s["attempts"] += 1
                s["_done"] = take(s, reply, r)
        print(f"  round {r}: {len(pending)} pending", flush=True)


def episodes(args, rounds=12, swap_every=4):
    """Each writer renders ten facts per seed as its own short paragraphs, then rephrases each fact, all
    with memory off.  Prompts from all seeds are generated together; a slot whose paragraph fails
    `swap_every` rounds gets a fresh company name."""
    st = torch.load(OUT / "stage0.pt", weights_only=False)
    memory = SharedMemory(init_std=0)  # the table is never read during generation
    lms = load_models(memory)
    banned_words, banned_rows = set(st["banned_words"]), set(map(tuple, st["banned_rows"]))
    for path in sorted(OUT.glob("episodes_*.json")):  # names stay unique across every episode
        for key in ("A", "B"):
            for s in json.load(open(path))[key]:
                banned_words.add(s["entity"].lower())
                banned_rows |= set(enumerate(rows_of(s["entity"].lower(), memory.n_rows)))
    seeds = [s for s in args.seeds if not (OUT / f"episodes_{s}.json").exists()]
    rngs = {seed: random.Random(seed) for seed in seeds}
    plan = {}
    for seed in seeds:  # names and cities are fixed per seed before any generation
        for key in ("A", "B"):
            names = data.make_names(EPISODE_FACTS, rngs[seed], banned_words, banned_rows, memory.n_rows)
            plan[seed, key] = [{"seed": seed, "entity": e, "city": c, "text": None, "attempts": 0, "rejected": [],
                                "rephrasings": [], "_done": False}
                               for e, c in zip(names, rngs[seed].sample(data.CITIES, EPISODE_FACTS))]

    def take_paragraph(s, reply, r):
        if data.accept(reply, s["entity"], s["city"]):
            s["text"] = reply
            return True
        if len(s["rejected"]) < 2:
            s["rejected"].append(reply)
        if (r + 1) % swap_every == 0:  # this name keeps failing: try another
            s["entity"] = data.make_names(1, rngs[s["seed"]], banned_words, banned_rows, memory.n_rows)[0]
        return False

    def take_rephrasings(s, reply, r):
        seen = {tuple(w for w, _, _ in data.words(x)) for x in s["rephrasings"] + [s["text"]]}
        for line in data.split_lines(reply):
            key = tuple(w for w, _, _ in data.words(line))  # punctuation variants are not new phrasings
            if len(s["rephrasings"]) < REPHRASINGS and key not in seen and data.accept_rephrasing(line, s["entity"], s["city"]):
                s["rephrasings"].append(line)
                seen.add(key)
        return len(s["rephrasings"]) >= REPHRASINGS

    for key in ("A", "B"):
        slots = [s for seed in seeds for s in plan[seed, key]]
        torch.manual_seed(0)
        print(f"{key}: paragraphs", flush=True)
        generate_rounds(lms[key], slots, lambda s: data.EPISODE_PROMPT.format(e=s["entity"], v=s["city"]),
                        take_paragraph, rounds)
        for s in slots:
            s["_done"] = s["text"] is None  # rephrase only facts that made it into the episode
        print(f"{key}: rephrasings", flush=True)
        generate_rounds(lms[key], slots, lambda s: data.REPHRASE_PROMPT.format(k=REPHRASINGS + 2, e=s["entity"],
                                                                               v=s["city"]),
                        take_rephrasings, rounds // 2, max_new_tokens=240)
    for seed in seeds:
        missing = [(k, s["city"]) for k in ("A", "B") for s in plan[seed, k] if s["text"] is None]
        if missing:
            print(f"seed {seed}: no acceptable paragraph for {missing}; not saved", flush=True)
            continue
        for k in ("A", "B"):
            for s in plan[seed, k]:
                del s["_done"], s["seed"]
        json.dump({"seed": seed, "A": plan[seed, "A"], "B": plan[seed, "B"]},
                  open(OUT / f"episodes_{seed}.json", "w"), indent=1)
        print(seed, {k: (sum(s["attempts"] for s in plan[seed, k]),
                         sum(len(s["rephrasings"]) for s in plan[seed, k])) for k in ("A", "B")}, flush=True)


def pilot(args):
    """Writer-side metrics only: the non-writer is never scored here."""
    st, memory, lms, m0 = load_stage0()
    weights = reuse_weights(st, memory)
    probe, cands = st["probe"][:PILOT_D0], st["cands"][:PILOT_D0]
    d0_m0 = {k: acc(score(lm, probe, cands, HELD)) for k, lm in lms.items()}
    rows = []
    for seed in args.seeds:
        ep = json.load(open(OUT / f"episodes_{seed}.json"))
        for w in ("A", "B"):
            lm = lms[w]
            facts, fc, _ = episode_set(ep, w)
            memory.table.data.copy_(m0)
            own_m0 = acc(score(lm, facts, fc, HELD))
            for arm, spec in ARMS.items():
                for lr in PILOT_LRS:
                    for strength in (PILOT_STRENGTHS if spec.get("penalty") else (None,)):
                        memory.table.data.copy_(m0)
                        cfg = {"lr": lr, "steps": max(PILOT_CHECKPOINTS), "strength": strength}
                        for step in arm_write(lm, ep[w], arm, cfg, weights):
                            if step in PILOT_CHECKPOINTS:
                                rows.append({"seed": seed, "writer": w, "arm": arm, "lr": lr, "strength": strength,
                                             "steps": step, "own_m0": own_m0, "own": acc(score(lm, facts, fc, HELD)),
                                             "d0_m0": d0_m0[w], "d0": acc(score(lm, probe, cands, HELD))})
                                print(json.dumps(rows[-1]), flush=True)
    memory.table.data.copy_(m0)
    # PLAN.md rule, per arm: maximise own-fact held-out accuracy minus the drop in D0 held-out accuracy,
    # averaged over seeds and writers; ties go to fewer steps, then smaller lr.
    chosen = {}
    for arm in ARMS:
        configs = {}
        for r in rows:
            if r["arm"] == arm:
                configs.setdefault((r["lr"], r["steps"], r["strength"]), []).append(r["own"] - (r["d0_m0"] - r["d0"]))
        best = max(configs, key=lambda c: (round(sum(configs[c]) / len(configs[c]), 6), -c[1], -c[0]))
        chosen[arm] = {"lr": best[0], "steps": best[1], "strength": best[2],
                       "objective": round(sum(configs[best]) / len(configs[best]), 4)}
    json.dump({"rows": rows, "chosen": chosen}, open(OUT / "pilot.json", "w"), indent=1)
    json.dump(chosen, open(OUT / "locked_config.json", "w"), indent=1)  # lock.py hashes this before confirm
    print("chosen", json.dumps(chosen), flush=True)


def confirm(args):
    cfg = json.load(open(OUT / "locked_config.json"))
    st, memory, lms, m0 = load_stage0()
    weights = reuse_weights(st, memory)
    probe, cands = st["probe"], st["cands"]
    eval_texts = data.wiki_split(CORPUS)[1][:100]
    path = OUT / "confirm.jsonl"
    done = {json.loads(l).get("seed") for l in open(path) if '"order"' in l} if path.exists() else set()
    out = open(path, "a")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    if not path.stat().st_size:  # seed- and arm-independent D0 and general-text baselines
        for r, lm in lms.items():
            for state, mem in (("NOMEM", False), ("M0", True)):
                for x in score(lm, probe, cands, memory=mem):
                    emit(seed=None, state=state, reader=r, set="D0", **x)
                emit(seed=None, state=state, reader=r, set="wiki", loss=lm.mean_loss(eval_texts, memory=mem))
    for seed in args.seeds:
        if seed in done:
            continue
        t0 = time.time()
        ep = json.load(open(OUT / f"episodes_{seed}.json"))
        order = ("A", "B") if seed % 2 == 0 else ("B", "A")
        sets = {w: episode_set(ep, w) for w in ("A", "B")}
        memory.table.data.copy_(m0)
        for r, lm in lms.items():  # everything before the first write is shared by all arms
            for w, (facts, fc, texts) in sets.items():
                for state, mem, ctx in (("NOMEM", False, ""), ("M0", True, ""),
                                        ("ORACLE", True, "\n\n".join(texts) + "\n\n")):
                    for x in score(lm, facts, fc, memory=mem, context=ctx):
                        emit(seed=seed, state=state, reader=r, set=f"F_{w}", **x)
                emit(seed=seed, state="M0", reader=r, set=f"text_{w}", loss=lm.mean_loss(texts))
        for w, (facts, _, texts) in sets.items():  # do A's and B's table gradients agree on this text?
            g = {r: table_grad(lm, texts) for r, lm in lms.items()}
            heads = torch.tensor([j for _ in facts for j in range(HEADS_PER_ORDER)], device=g["A"].device)
            ent = torch.tensor([r for e, _ in facts for r in rows_of(e.lower(), memory.n_rows)], device=g["A"].device)
            emit(seed=seed, kind="grad_cos", episode=w, all=cos(g["A"], g["B"]),
                 entity_rows=cos(g["A"][heads, ent], g["B"][heads, ent]))
            del g
        for arm in ARMS:
            memory.table.data.copy_(m0)
            for pos, w in enumerate(order, 1):
                for _ in arm_write(lms[w], ep[w], arm, cfg[arm], weights, seed):
                    pass
                for r, lm in lms.items():
                    for x in score(lm, probe, cands):
                        emit(seed=seed, arm=arm, state=f"M{pos}", reader=r, set="D0", **x)
                    for v, (facts, fc, texts) in sets.items():
                        for x in score(lm, facts, fc):
                            emit(seed=seed, arm=arm, state=f"M{pos}", reader=r, set=f"F_{v}", **x)
                        emit(seed=seed, arm=arm, state=f"M{pos}", reader=r, set=f"text_{v}", loss=lm.mean_loss(texts))
                    emit(seed=seed, arm=arm, state=f"M{pos}", reader=r, set="wiki", loss=lm.mean_loss(eval_texts))
        emit(seed=seed, kind="order", order=order, sec=round(time.time() - t0))
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    memory.table.data.copy_(m0)


def fresh_readers(lm, seed):
    """New readers for `lm` from `seed`, so every reader fit starts from the same weights."""
    torch.manual_seed(seed)
    lm.readers = torch.nn.ModuleList(GatedReader(lm.hidden_size, lm.memory.out_dim)
                                     for _ in lm.layers).to(lm.device)


def fit(lm, memory, sents, wiki, args, rng, gen, train_table, on_eval):
    """Stage 0's recipe for one model: Adam on its readers (and the table, if train_table) over fact
    sentences plus general text, with n-gram dropout."""
    lm.memory = memory
    memory.table.requires_grad_(train_table)
    opt = torch.optim.Adam(([{"params": [memory.table], "lr": args.table_lr}] if train_table else [])
                           + [{"params": lm.reader_parameters(), "lr": args.reader_lr}])
    for step in range(1, args.steps + 1):
        texts = rng.sample(sents, args.fact_batch) + rng.sample(wiki, args.wiki_batch)
        batch = lm.encode(texts, max_tokens=64)
        if args.order_dropout:
            batch = drop_orders(batch, order_drop_mask(len(texts), args.order_dropout, gen))
        opt.zero_grad(set_to_none=True)
        lm.backward_mean(batch, micro=args.micro)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            on_eval(step)


def quiz(lm, facts, cands, memory=True, oracle=False, templates=range(len(data.PROBES))):
    """Forced choice per fact x probe template, as score(); with the fact stated first if oracle."""
    out = []
    for ti in templates:
        prompts = [(data.TRAIN_TEMPLATES[0].format(e=e, v=v) + " " if oracle else "") + data.PROBES[ti].format(e=e)
                   for e, v in facts]
        lp = lm.choice_logprobs(prompts, cands, memory=memory)
        fc = torch.log_softmax(lp, -1)
        for i, ((e, v), cs) in enumerate(zip(facts, cands)):
            k = cs.index(v)
            out.append({"t": ti, "e": e, "correct": int(lp[i].argmax() == k), "fc": round(fc[i, k].item(), 4)})
    return out


@torch.no_grad()
def free_answers(lm, prompts, memory=True, n_tokens=12, chunk=64):
    """Greedy continuation of each prompt, as text. Memory rows are recomputed from the text each step."""
    outs = []
    for i in range(0, len(prompts), chunk):
        ps = prompts[i:i + chunk]
        p_ids = [lm.tok(p, add_special_tokens=False)["input_ids"] for p in ps]
        gen = [[] for _ in ps]

        def continuation(a, g):
            return lm.tok.decode(a + g)[len(lm.tok.decode(a)):]

        for _ in range(n_tokens):
            batch = lm.encode([p + continuation(a, g) for p, a, g in zip(ps, p_ids, gen)])
            h = lm.hidden(batch, memory)
            last = h[torch.arange(len(ps), device=h.device), batch.mask.sum(1) - 1]
            for g, t in zip(gen, lm.logits(last).argmax(-1).tolist()):
                g.append(t)
        outs += [continuation(a, g) for a, g in zip(p_ids, gen)]
    return outs


def transfer(args):
    """No-edit fact transfer (PLAN.md, "Transfer test"): each model trains the table alone; then, with
    the table frozen, fresh readers fitted on half the facts are quizzed on the other half."""
    lock = OUT / "LOCK.json"
    if lock.exists():
        want = json.loads(lock.read_text())["locked_config"]
        have = {"models": MODELS, "args": {k: vars(args).get(k) for k in want["args"]}}
        if have != want:
            sys.exit(f"settings differ from {lock}:\n{have}\n{want}")
    rng = random.Random(0)
    torch.manual_seed(0)
    start = SharedMemory().to(DEVICE)  # every table starts here; left untrained, it is the random-table control
    train_wiki, eval_wiki, d0, sents, _, _ = fact_world(rng, start.n_rows)
    sets = {"F": [], "T": []}
    for city in data.CITIES:
        mine = [f for f in d0 if f[1] == city]
        sets["F"] += mine[:TRANSFER_FIT]
        sets["T"] += mine[TRANSFER_FIT:]
    cands = {}
    for e, v in d0:
        cs = rng.sample([c for c in data.CITIES if c != v], 9) + [v]
        rng.shuffle(cs)
        cands[e] = cs
    if args.quiz_facts:  # smoke runs only
        sets = {s: fs[:args.quiz_facts] for s, fs in sets.items()}
    fit_sents = [t.format(e=e, v=v) for e, v in sets["F"] for t in data.TRAIN_TEMPLATES]
    lms = {k: MemLM(name, start, device=DEVICE) for k, name in MODELS.items()}
    gen = torch.Generator().manual_seed(0)
    out = open(OUT / "transfer.jsonl", "w")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    def evaluate(lm, reader, table, how, memory=True, oracle=False):
        t0 = time.time()
        for s, facts in sets.items():
            for r in quiz(lm, facts, [cands[e] for e, _ in facts], memory, oracle):
                emit(kind="choice", reader=reader, table=table, fit=how, set=s, **r)
            for ti in HELD:
                prompts = [(data.TRAIN_TEMPLATES[0].format(e=e, v=v) + " " if oracle else "")
                           + data.PROBES[ti].format(e=e) for e, v in facts]
                for (e, v), a in zip(facts, free_answers(lm, prompts, memory)):
                    m = CITY.search(a)
                    emit(kind="free", reader=reader, table=table, fit=how, set=s, t=ti, e=e, answer=a,
                         correct=int(bool(m) and m.group(1) == v))
        if memory and not oracle:
            emit(kind="wiki_loss", reader=reader, table=table, fit=how, loss=round(lm.mean_loss(eval_wiki[:100]), 4))
        print(f"quizzed reader {reader}, table {table}, {how} in {time.time() - t0:.0f}s", flush=True)

    def curve(lm, reader, table, how, t0):
        def on_eval(step):
            rec = {"kind": "curve", "reader": reader, "table": table, "fit": how, "step": step,
                   "sec": round(time.time() - t0)}
            for s, facts in sets.items():
                sub = facts[::TRANSFER_FIT]  # one company per city
                res = quiz(lm, sub, [cands[e] for e, _ in sub], templates=HELD)
                rec[s] = round(sum(x["correct"] for x in res) / len(res), 4)
            emit(**rec)
            print(json.dumps(rec), flush=True)
        return on_eval

    print(f"{len(sets['F'])} fit / {len(sets['T'])} test facts on {DEVICE}", flush=True)
    for k, lm in lms.items():  # baselines: no memory, and the fact stated before the question
        evaluate(lm, k, "none", "none", memory=False)
        evaluate(lm, k, "none", "oracle", memory=False, oracle=True)
    tables = {}
    for w, lm in lms.items():  # phase 1: each model trains the table alone, on all facts
        tables[w] = SharedMemory().to(DEVICE)
        tables[w].table.data.copy_(start.table.data)
        lm.memory = tables[w]
        fresh_readers(lm, seed=1)
        fit(lm, tables[w], sents, train_wiki, args, rng, gen, True, curve(lm, w, w, "writer", time.time()))
        torch.save(tables[w].table.detach().cpu(), OUT / f"table_{w}.pt")
        tables[w].table.requires_grad_(False)
        evaluate(lm, w, w, "writer")
    for r, lm in lms.items():  # phase 2: table frozen, new readers fitted on the fit facts only
        for t, table in (*tables.items(), ("random", start)):
            lm.memory = table
            fresh_readers(lm, seed=2)
            fit(lm, table, fit_sents, train_wiki, args, rng, gen, False, curve(lm, r, t, "fresh", time.time()))
            evaluate(lm, r, t, "fresh")
    out.close()


def render(lm, item):
    """A fact text in `lm`'s format: plain text as is, a (request, reply) pair as a chat exchange."""
    return item if isinstance(item, str) else lm.chat(*item)


def fit_together(lms, memory, facts, wiki, args, rng, gen, train_table, on_eval, kl=None):
    """Stage 0's recipe with chat: each step, fact texts (CHAT_FRAC of them chat exchanges, each model
    seeing them in its own chat format) plus general text, with n-gram dropout; every model's loss on
    the same texts adds up on the table. Adam on the models' readers and, if train_table, the table.

    kl: instead of learning to predict the general text, the memory is held to changing nothing there:
    the loss is the facts' mean next-token loss plus kl x the mean KL between memory off and on over
    the general text, read with all its lookups (MemLM.kl_backward)."""
    plain = [t.format(e=e, v=v) for e, v in facts for t in data.TRAIN_TEMPLATES]
    chat = [(u.format(e=e), t.format(e=e, v=v)) for e, v in facts for u in data.CHAT_USER for t in data.TRAIN_TEMPLATES]
    n_chat = round(args.fact_batch * CHAT_FRAC)
    memory.table.requires_grad_(train_table)
    opt = torch.optim.Adam(([{"params": [memory.table], "lr": args.table_lr}] if train_table else [])
                           + [{"params": lm.reader_parameters(), "lr": args.reader_lr} for lm in lms.values()])
    for step in range(1, args.steps + 1):
        items = (rng.sample(plain, args.fact_batch - n_chat) + rng.sample(chat, n_chat)
                 + rng.sample(wiki, args.wiki_batch))
        drop = order_drop_mask(len(items), args.order_dropout, gen)  # same texts lose the same lookups in every model
        opt.zero_grad(set_to_none=True)
        for lm in lms.values():
            if kl is None:
                batch = lm.encode([render(lm, x) for x in items], max_tokens=64)
                if args.order_dropout:
                    batch = drop_orders(batch, drop)
                lm.backward_mean(batch, micro=args.micro)
            else:
                n = args.fact_batch
                batch = lm.encode([render(lm, x) for x in items[:n]], max_tokens=64)
                if args.order_dropout:
                    batch = drop_orders(batch, drop[:n])
                lm.backward_mean(batch, micro=args.micro)
                lm.kl_backward(lm.encode(items[n:], max_tokens=64), kl, micro=args.micro, max_tokens=KL_TOKENS)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            on_eval(step)


def end_of_turn(lm):
    """What ends a reply in `lm`'s chat format; free-text answers are cut there."""
    text = lm.chat("x", "REPLY")
    return text[text.rindex("REPLY") + len("REPLY"):].strip() or None


def battery(lm, facts, cands, memory=True, oracle=False, free_tokens=20):
    """Every quiz on (name, city) facts: forced choice after the 4 probe templates (quiz()) and as the
    first word of the reply to each chat question, and greedy chat replies scored by the first city
    they name. oracle: the fact is stated first (in the user turn, for chat)."""
    out = [{"kind": "mc", **r} for r in quiz(lm, facts, cands, memory, oracle)]
    stop = end_of_turn(lm)
    for q, question in enumerate(data.CHAT_QUESTIONS):
        prompts = [lm.chat((data.TRAIN_TEMPLATES[0].format(e=e, v=v) + " " if oracle else "") + question.format(e=e))
                   for e, v in facts]
        lp = lm.choice_logprobs(prompts, cands, memory=memory, sep="")
        fc = torch.log_softmax(lp, -1)
        for i, ((e, v), cs) in enumerate(zip(facts, cands)):
            k = cs.index(v)
            out.append({"kind": "chat_mc", "t": q, "e": e, "correct": int(lp[i].argmax() == k),
                        "fc": round(fc[i, k].item(), 4)})
        if free_tokens:
            for (e, v), a in zip(facts, free_answers(lm, prompts, memory, n_tokens=free_tokens)):
                a = a.split(stop)[0] if stop else a
                m = CITY.search(a)
                out.append({"kind": "chat_free", "t": q, "e": e, "answer": a,
                            "correct": int(bool(m) and m.group(1) == v)})
    return out


def edit_curve(lms, probe, cands, log, t0, n=50):
    """Training curve: forced choice on the first n probe facts, in the probe templates and chat."""
    def on_eval(step):
        rec = {"step": step, "sec": round(time.time() - t0)}
        for k, lm in lms.items():
            res = battery(lm, probe[:n], cands[:n], free_tokens=0)
            mc = [r for r in res if r["kind"] == "mc"]
            rec[k] = {"trained": acc(mc, (0,)), "held_out": acc(mc),
                      "chat": acc([r for r in res if r["kind"] == "chat_mc"], range(len(data.CHAT_QUESTIONS)))}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    return on_eval


def edit_setup(args):
    """Edit test, step 1: the shared table and A's and B's readers, trained together on D0."""
    rng = random.Random(0)
    torch.manual_seed(0)
    memory = SharedMemory().to(DEVICE)
    train_wiki, _, d0, _, banned_words, banned_rows = fact_world(rng, memory.n_rows)
    probe = d0[:N_D0_PROBE]  # stage 0's probe facts and candidates
    cands = []
    for _, v in probe:
        cs = rng.sample([c for c in data.CITIES if c != v], 9) + [v]
        rng.shuffle(cs)
        cands.append(cs)
    lms = {k: MemLM(name, memory, device=DEVICE) for k, name in MODELS.items()}
    log = []
    fit_together(lms, memory, d0, train_wiki, args, rng, torch.Generator().manual_seed(0), True,
                 edit_curve(lms, probe, cands, log, time.time()), kl=SETUP_KL)
    torch.save({"table": memory.table.detach().cpu(), "readers": {k: lm.reader_state() for k, lm in lms.items()},
                "d0": d0, "probe": probe, "cands": cands, "banned_words": sorted(banned_words),
                "banned_rows": sorted(banned_rows), "models": MODELS, "args": vars(args), "log": log}, OUT / "setup.pt")


def edit_onboard(args):
    """Step 2: C joins later. The table stays as step 1 left it; only C's new readers learn, on step 1's texts."""
    st = torch.load(OUT / "setup.pt", weights_only=False)
    memory = SharedMemory().to(DEVICE)
    memory.table.data.copy_(st["table"])
    torch.manual_seed(1)
    lms = {k: MemLM(name, memory, device=DEVICE) for k, name in ADDED.items()}
    log = []
    fit_together(lms, memory, st["d0"], data.wiki_split(CORPUS)[0], args, random.Random(1),
                 torch.Generator().manual_seed(1), False, edit_curve(lms, st["probe"], st["cands"], log, time.time()),
                 kl=SETUP_KL)
    assert torch.equal(memory.table.detach().cpu(), st["table"])  # C adapted to the table, not the reverse
    torch.save({"readers": {k: lm.reader_state() for k, lm in lms.items()}, "models": ADDED, "args": vars(args),
                "log": log}, OUT / "onboard.pt")


def load_edit():
    """The table as step 1 left it, and all three models with their frozen readers."""
    st = torch.load(OUT / "setup.pt", weights_only=False)
    readers = {**st["readers"], **torch.load(OUT / "onboard.pt", weights_only=False)["readers"]}
    memory = SharedMemory().to(DEVICE)
    memory.table.data.copy_(st["table"])
    lms = {k: MemLM(name, memory, device=DEVICE, offload=OFFLOAD) for k, name in {**MODELS, **ADDED}.items()}
    for k, lm in lms.items():
        lm.load_reader_state(readers[k])
    return st, memory, lms, memory.table.detach().clone()


def edit_episodes(args, rounds=12, swap_every=4):
    """A, B and C each write a short newsletter paragraph about each of their ten new facts per seed,
    with memory off: what the "paragraphs" arm writes. A slot whose paragraph keeps failing gets a
    fresh company name every `swap_every` rounds."""
    st = torch.load(OUT / "setup.pt", weights_only=False)
    memory = SharedMemory(init_std=0)  # the table is never read during generation
    lms = {k: MemLM(name, memory, device=DEVICE, offload=OFFLOAD) for k, name in {**MODELS, **ADDED}.items()}
    banned_words, banned_rows = set(st["banned_words"]), set(map(tuple, st["banned_rows"]))
    chat_text = [lm.chat(u.format(e="x"), "x") for lm in lms.values() for u in data.CHAT_USER + data.CHAT_QUESTIONS]
    banned_words |= data.vocabulary(chat_text)  # chat markup and questions are words too
    banned_rows |= text_rows(chat_text, memory.n_rows)
    for path in sorted(OUT.glob("edit_episodes_*.json")):  # names stay unique across every episode
        for key in lms:
            for s in json.load(open(path))[key]:
                banned_words.add(s["entity"].lower())
                banned_rows |= set(enumerate(rows_of(s["entity"].lower(), memory.n_rows)))
    seeds = [s for s in (args.seeds or EDIT_PILOT_SEEDS + EDIT_SEEDS) if not (OUT / f"edit_episodes_{s}.json").exists()]
    rngs = {seed: random.Random(seed) for seed in seeds}
    plan = {}
    for seed in seeds:  # names and cities are fixed per seed before any generation
        for key in lms:
            names = data.make_names(EDIT_FACTS, rngs[seed], banned_words, banned_rows, memory.n_rows)
            plan[seed, key] = [{"seed": seed, "entity": e, "city": c, "text": None, "attempts": 0, "rejected": [],
                                "_done": False} for e, c in zip(names, rngs[seed].sample(data.CITIES, EDIT_FACTS))]

    def take(s, reply, r):
        if data.accept(reply, s["entity"], s["city"]):
            s["text"] = reply
            return True
        if len(s["rejected"]) < 2:
            s["rejected"].append(reply)
        if (r + 1) % swap_every == 0:  # this name keeps failing: try another
            s["entity"] = data.make_names(1, rngs[s["seed"]], banned_words, banned_rows, memory.n_rows)[0]
        return False

    for key, lm in lms.items():
        torch.manual_seed(0)
        print(f"{key}: paragraphs", flush=True)
        generate_rounds(lm, [s for seed in seeds for s in plan[seed, key]],
                        lambda s: data.EPISODE_PROMPT.format(e=s["entity"], v=s["city"]), take, rounds)
    for seed in seeds:
        missing = [(k, s["city"]) for k in lms for s in plan[seed, k] if s["text"] is None]
        if missing:
            print(f"seed {seed}: no acceptable paragraph for {missing}; not saved", flush=True)
            continue
        for k in lms:
            for s in plan[seed, k]:
                del s["_done"], s["seed"]
        json.dump({"seed": seed, **{k: plan[seed, k] for k in lms}}, open(OUT / f"edit_episodes_{seed}.json", "w"),
                  indent=1)
        print(seed, {k: sum(s["attempts"] for s in plan[seed, k]) for k in lms}, flush=True)


def edit_facts(ep, w, n=None):
    """(facts, candidates) for writer w's facts; the candidates are the episode's own cities."""
    facts = [(s["entity"], s["city"]) for s in ep[w]][:n]
    return facts, [[c for _, c in facts]] * len(facts)


def edit_texts(lm, slots, arm):
    """What a writer writes: each fact's 6 phrasings and 3 chat statements (in the writer's own chat
    format), or the writer's own paragraphs."""
    if arm == "paragraphs":
        return [s["text"] for s in slots]
    return ([t.format(e=s["entity"], v=s["city"]) for s in slots for t in data.TRAIN_TEMPLATES]
            + [lm.chat(u.format(e=s["entity"]), data.TRAIN_TEMPLATES[k].format(e=s["entity"], v=s["city"]))
               for s in slots for k, u in enumerate(data.CHAT_USER)])


@functools.lru_cache(maxsize=1)
def general_text():
    """The general-text chunks writes replay: the corpus's training split."""
    return data.wiki_split(CORPUS)[0]


def edit_write(lm, slots, arm, cfg, seed):
    """One writer's write (run 3): Adam on the table with the writer's loss on its texts, through its
    frozen readers, with step 1's n-gram dropout, replaying general text so the write cannot move what
    the model predicts there. No restriction to the rows of the company names."""
    texts = edit_texts(lm, slots, arm)
    return write(lm, texts, cfg["steps"], cfg["lr"], EDIT_WRITE_DROPOUT, seed=seed,
                 replay=(general_text(), REPLAY_K, REPLAY_KL))


def edit_pilot(args):
    """Write settings per arm, chosen on the pilot seeds only; the confirmatory seeds are never touched.
    Each writer writes first, keeping the table at every checkpoint; then each model reads them all."""
    st, memory, lms, m0 = load_edit()
    n = args.quiz_facts or None
    probe, cands = st["probe"][:PILOT_D0][:n], st["cands"][:PILOT_D0][:n]
    lrs, checkpoints = ((EDIT_PILOT_LRS[1],), (2, 4)) if args.smoke else (EDIT_PILOT_LRS, EDIT_PILOT_STEPS)
    d0_m0 = {k: acc(quiz(lm, probe, cands, templates=HELD)) for k, lm in lms.items()}
    rows = []
    for seed in args.seeds or EDIT_PILOT_SEEDS:
        ep = json.load(open(OUT / f"edit_episodes_{seed}.json"))
        for w, lm in lms.items():
            facts, fc = edit_facts(ep, w, n)
            tables, d0 = {}, {}  # (arm, lr, steps) -> the table then, and the writer's D0 accuracy
            for arm in EDIT_ARMS:
                for lr in lrs:
                    memory.table.data.copy_(m0)
                    for step in edit_write(lm, ep[w], arm, {"lr": lr, "steps": max(checkpoints)}, seed):
                        if step in checkpoints:
                            tables[arm, lr, step] = memory.table.detach().to("cpu", copy=True)
                            d0[arm, lr, step] = acc(quiz(lm, probe, cands, templates=HELD))
            at_m0, after = {}, {key: {} for key in tables}
            for r, x in lms.items():  # every model's held-out accuracy on w's facts, before and after
                memory.table.data.copy_(m0)
                at_m0[r] = acc(quiz(x, facts, fc, templates=HELD))
                for key, table in tables.items():
                    memory.table.data.copy_(table)
                    after[key][r] = acc(quiz(x, facts, fc, templates=HELD))
            for (arm, lr, step), now in after.items():
                rows.append({"seed": seed, "writer": w, "arm": arm, "lr": lr, "steps": step, "m0": at_m0, "after": now,
                             "others_gain": round(mean(now[r] - at_m0[r] for r in lms if r != w), 4),
                             "d0_m0": d0_m0[w], "d0": d0[arm, lr, step]})
                print(json.dumps(rows[-1]), flush=True)
    memory.table.data.copy_(m0)
    # PLAN.md rule, per arm: maximise the other two models' held-out gain on the writer's facts minus the
    # writer's drop on 50 D0 facts, averaged over seeds and writers; ties go to fewer steps, then the smaller lr.
    objective = {arm: {} for arm in EDIT_ARMS}
    for r in rows:
        objective[r["arm"]].setdefault((r["lr"], r["steps"]), []).append(r["others_gain"] - (r["d0_m0"] - r["d0"]))
    chosen = {}
    for arm, configs in objective.items():
        best = max(configs, key=lambda c: (round(mean(configs[c]), 6), -c[1], -c[0]))
        chosen[arm] = {"lr": best[0], "steps": best[1], "objective": round(mean(configs[best]), 4)}
    # Scale test: the sentences lr, with the fewest steps reaching 90% of its best objective (each step of
    # 200 writes costs about 10 GPU-minutes).
    lr, best = chosen["sentences"]["lr"], chosen["sentences"]["objective"]
    steps = min(s for (l, s), v in objective["sentences"].items() if l == lr and (best <= 0 or mean(v) >= 0.9 * best))
    locked = {"models": {**MODELS, **ADDED}, "write": chosen, "scale": {"lr": lr, "steps": steps}}
    json.dump({"rows": rows, "chosen": chosen}, open(OUT / "edit_pilot.json", "w"), indent=1)
    json.dump(locked, open(OUT / "locked_config.json", "w"), indent=1)
    print("chosen", json.dumps(chosen), "scale", json.dumps(locked["scale"]), flush=True)


def check_lock(smoke=False):
    """The locked settings, once the plan, code, analyses and config match what LOCK.json hashed."""
    if smoke:
        return json.load(open(OUT / "locked_config.json"))
    lock = json.loads((OUT / "LOCK.json").read_text())
    for name, digest in lock["sha256"].items():
        path = OUT / name if name == "locked_config.json" else ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            sys.exit(f"{name} differs from {OUT / 'LOCK.json'}")
    if lock["locked_config"]["models"] != {**MODELS, **ADDED}:
        sys.exit(f"models differ from {OUT / 'LOCK.json'}")
    return lock["locked_config"]


def resume(path, key):
    """An output file opened for appending, keeping only the records of finished units (`key` values that
    have a "done" record); returns (file, finished keys)."""
    lines = path.read_text().splitlines() if path.exists() else []
    done = {json.loads(l)[key] for l in lines if '"kind": "done"' in l}
    out = open(path, "w")
    out.writelines(l + "\n" for l in lines if json.loads(l).get(key) in done)
    return out, done


def edit(args):
    """The edit test (PLAN.md): per seed, A, B and C each write ten new facts into the shared table, in
    the seed's order; after every write each model is quizzed on every writer's facts. All writes run
    first, keeping the table after each; then each model reads every state in turn."""
    cfg = check_lock(args.smoke)["write"]
    st, memory, lms, m0 = load_edit()
    n = args.quiz_facts or None
    probe, cands = st["probe"][:n], st["cands"][:n]
    eval_wiki = data.wiki_split(CORPUS)[1][:100]
    out, done = resume(OUT / "edit.jsonl", "seed")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    def read_state(r, lm, seed, arm, state, sets):
        for w, (facts, fc) in sets.items():
            for x in battery(lm, facts, fc):
                emit(seed=seed, arm=arm, state=state, reader=r, set=f"F_{w}", **x)
        for x in quiz(lm, probe, cands, templates=HELD):
            emit(kind="mc", seed=seed, arm=arm, state=state, reader=r, set="D0", **x)
        emit(kind="wiki_loss", seed=seed, arm=arm, state=state, reader=r, loss=round(lm.mean_loss(eval_wiki), 4))

    for seed in args.seeds or EDIT_SEEDS:
        if seed in done:
            continue
        t0 = time.time()
        ep = json.load(open(OUT / f"edit_episodes_{seed}.json"))
        sets = {w: edit_facts(ep, w, n) for w in lms}
        order = WRITE_ORDERS[seed % len(WRITE_ORDERS)]
        tables = {}  # (arm, pos) -> the table after that write
        for arm in EDIT_ARMS:
            memory.table.data.copy_(m0)
            for pos, w in enumerate(order, 1):
                before = memory.table.detach().clone()
                for _ in edit_write(lms[w], ep[w], arm, cfg[arm], seed):
                    pass
                delta = memory.table.detach() - before
                emit(kind="write", seed=seed, arm=arm, pos=pos, writer=w,
                     rows_moved=int((delta.abs().sum(-1) > 0).sum()), delta_norm=round(delta.norm().item(), 4))
                tables[arm, pos] = memory.table.detach().to("cpu", copy=True)
        for r, lm in lms.items():
            for w, (facts, fc) in sets.items():  # memory off: the same for both arms and every table
                for state, oracle in (("NOMEM", False), ("ORACLE", True)):
                    for x in battery(lm, facts, fc, memory=False, oracle=oracle):
                        emit(seed=seed, arm=None, state=state, reader=r, set=f"F_{w}", **x)
            memory.table.data.copy_(m0)
            read_state(r, lm, seed, None, "M0", sets)
            for (arm, pos), table in tables.items():
                memory.table.data.copy_(table)
                read_state(r, lm, seed, arm, f"M{pos}", sets)
        emit(kind="order", seed=seed, order=order, sec=round(time.time() - t0))
        emit(kind="done", seed=seed)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    memory.table.data.copy_(m0)
    out.close()


def fresh_facts(st, n_writes, rng, n_rows, lms):
    """n_writes groups of EDIT_FACTS new facts: names whose one-word rows nothing else uses (stage 0,
    general text, chat markup, the edit test's names) and EDIT_FACTS distinct cities per group."""
    banned_words, banned_rows = set(st["banned_words"]), set(map(tuple, st["banned_rows"]))
    chat_text = [lm.chat(u.format(e="x"), "x") for lm in lms.values() for u in data.CHAT_USER + data.CHAT_QUESTIONS]
    banned_words |= data.vocabulary(chat_text)
    banned_rows |= text_rows(chat_text, n_rows)
    for path in sorted(OUT.glob("edit_episodes_*.json")):
        for key in lms:
            for s in json.load(open(path))[key]:
                banned_words.add(s["entity"].lower())
                banned_rows |= set(enumerate(rows_of(s["entity"].lower(), n_rows)))
    return [[{"entity": e, "city": c} for e, c in zip(data.make_names(EDIT_FACTS, rng, banned_words, banned_rows, n_rows),
                                                     rng.sample(data.CITIES, EDIT_FACTS))] for _ in range(n_writes)]


def context_answers(lm, facts, questions, n, rng, batch):
    """Greedy chat replies, memory off, to each question about facts[i] with n of the facts stated first
    (facts[i] and n - 1 others, shuffled), scored by the first city named."""
    stop = end_of_turn(lm)
    prompts = []
    for i in questions:
        shown = [facts[i]] + rng.sample(facts[:i] + facts[i + 1:], n - 1)
        rng.shuffle(shown)
        prompts.append("\n".join(data.TRAIN_TEMPLATES[0].format(e=e, v=v) for e, v in shown) + "\n\n"
                       + data.CHAT_QUESTIONS[0].format(e=facts[i][0]))
    answers = []
    for k in range(0, len(prompts), batch):
        answers += lm.generate(prompts[k:k + batch], max_new_tokens=20, greedy=True)
    out = []
    for i, a in zip(questions, answers):
        a = a.split(stop)[0] if stop else a
        m = CITY.search(a)
        out.append((i, a, int(bool(m) and m.group(1) == facts[i][1])))
    return out


def scale(args):
    """Phase 2 (PLAN.md): SCALE_WRITES writes of 10 new facts each into the step 2 table, the models taking
    turns, until the text written is longer than any model's context window. All writes run first,
    keeping some tables; then each model reads them, and answers with the facts in its prompt instead."""
    cfg = check_lock(args.smoke)["scale"]
    st, memory, lms, m0 = load_edit()
    n_writes = args.scale_writes or SCALE_WRITES
    keep = sorted({k for k in SCALE_FOLLOW if k <= n_writes} | {n_writes})
    writes = fresh_facts(st, n_writes, random.Random(1000), memory.n_rows, lms)
    writers = list(lms)
    facts = [(s["entity"], s["city"]) for slots in writes for s in slots]
    group = [j for j in range(1, n_writes + 1) for _ in range(EDIT_FACTS)]  # the write each fact belongs to
    cands = [[s["city"] for s in writes[j - 1]] for j in group]
    json.dump(writes, open(OUT / "scale_facts.json", "w"))
    probe, d0_cands = st["probe"][:args.quiz_facts or None], st["cands"][:args.quiz_facts or None]
    eval_wiki = data.wiki_split(CORPUS)[1][:100]
    out = open(OUT / "scale.jsonl", "w")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    tables, t0 = {}, time.time()
    for k, slots in enumerate(writes, 1):
        w = writers[(k - 1) % len(writers)]
        before = memory.table.detach().clone()
        for _ in edit_write(lms[w], slots, "sentences", cfg, seed=k):
            pass
        delta = memory.table.detach() - before
        tokens = sum(len(lms[w].tok(t, add_special_tokens=False)["input_ids"]) for t in edit_texts(lms[w], slots, "sentences"))
        emit(kind="write", k=k, writer=w, rows_moved=int((delta.abs().sum(-1) > 0).sum()),
             delta_norm=round(delta.norm().item(), 4), tokens=tokens)
        if k in keep:
            tables[k] = memory.table.detach().to("cpu", copy=True)
        if k % 10 == 0 or k == n_writes:
            print(f"write {k}/{n_writes} by {w} at {time.time() - t0:.0f}s", flush=True)
    torch.save(tables[n_writes], OUT / "scale_table.pt")
    for k in SCALE_CURVE:  # for the benchmarks' harm curve
        if k in tables:
            torch.save(tables[k], OUT / f"scale_table_{k}.pt")

    def read(r, lm, k, idx):
        """Held-out forced choice and a chat reply for the facts at positions idx, with the table now."""
        fs, cs = [facts[i] for i in idx], [cands[i] for i in idx]
        write_of = {facts[i][0]: group[i] for i in idx}
        for x in quiz(lm, fs, cs, templates=HELD):
            emit(kind="mc", k=k, reader=r, j=write_of[x["e"]], **x)
        prompts = [lm.chat(data.CHAT_QUESTIONS[0].format(e=e)) for e, _ in fs]
        stop = end_of_turn(lm)
        for i, a in zip(idx, free_answers(lm, prompts, n_tokens=20)):
            a = a.split(stop)[0] if stop else a
            m = CITY.search(a)
            emit(kind="chat_free", k=k, reader=r, j=group[i], e=facts[i][0], answer=a,
                 correct=int(bool(m) and m.group(1) == facts[i][1]))

    everyone = list(range(len(facts)))
    sample = sorted(random.Random(2000).sample(everyone, min(5 if args.smoke else SCALE_SAMPLE, len(facts))))
    for r, lm in lms.items():
        for k, table in [(0, m0), *tables.items()]:
            memory.table.data.copy_(table)
            read(r, lm, k, everyone if k in (0, n_writes) else
                 [i for i in everyone if group[i] in SCALE_FOLLOW and group[i] <= k])
            if k in (0, n_writes):
                for x in quiz(lm, probe, d0_cands, templates=HELD):
                    emit(kind="d0", k=k, reader=r, **x)
                emit(kind="wiki_loss", k=k, reader=r, loss=round(lm.mean_loss(eval_wiki), 4))
        for n in (1, *SCALE_CONTEXT):  # memory off, n facts stated in the prompt (n = 1: the oracle)
            n = min(n, len(facts))
            batch = max(1, min(64, 16000 // (12 * n + 40)))  # about 16K prompt tokens per batch
            for i, a, c in context_answers(lm, facts, sample, n, random.Random(3000 + n), batch):
                emit(kind="context", reader=r, n=n, j=group[i], e=facts[i][0], answer=a, correct=c)
        print(f"reader {r} done at {time.time() - t0:.0f}s", flush=True)
    memory.table.data.copy_(m0)
    out.close()


def memory_lm(lm, memory_on):
    """An lm-evaluation-harness model whose log-likelihoods come from a MemLM, reading the shared table
    or not. Requests are scored sorted by length, BENCH_BATCH at a time (the same values as one at a
    time, within 4e-5)."""
    from lm_eval.api.model import LM

    class MemoryLM(LM):
        @torch.no_grad()
        def loglikelihood(self, requests, disable_tqdm=False):
            pairs = [r.args for r in requests]
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))  # less padding
            out = [None] * len(pairs)
            for k in range(0, len(order), BENCH_BATCH):
                idx = order[k:k + BENCH_BATCH]
                lp = lm.span_logprobs([pairs[i][0] + pairs[i][1] for i in idx], [len(pairs[i][0]) for i in idx],
                                      memory_on, strict=False)
                for i, v in zip(idx, lp.tolist()):
                    out[i] = (v, False)
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError

    return MemoryLM()


def curve_questions(smoke=False):
    """(task manager, {task: question indices}): lm-evaluation-harness questions none of which the main
    benchmark set scores. MMLU test questions 20-24 of each subject (main set: 0-19), HellaSwag
    validation 1000-1299 (0-999), and ARC-Challenge validation, WinoGrande and PIQA train 0-299 (main
    set: all ARC test, WinoGrande and PIQA validation questions); screen_tasks/ defines the last three.
    1,484 questions. The screen (screen.py) used the same set."""
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(include_path=str(ROOT / "screen_tasks"))
    subjects = []

    def walk(group):
        for v in group.values():
            walk(v) if isinstance(v, dict) else subjects.append(v.config.task)

    walk(get_task_dict(["mmlu"], tm))
    n = 2 if smoke else None
    samples = {s: list(range(20, 20 + (n or 5))) for s in (subjects[:2] if smoke else subjects)}
    samples.update({"hellaswag": list(range(1000, 1000 + (n or 300))), "arc_challenge_screen": list(range(n or 299)),
                    "winogrande_screen": list(range(n or 300)), "piqa_screen": list(range(n or 300))})
    return tm, samples


def bench(args):
    """General ability (PLAN.md): lm-evaluation-harness tasks scored by each model with memory off, with
    the step 2 table, and with the table after the scale test. Then the harm curve (run 3): the
    curve_questions() set with memory off, the step 2 table, and the scale test's table after each of
    SCALE_CURVE and the last write. One record per question."""
    import lm_eval
    check_lock(args.smoke)
    st, memory, lms, m0 = load_edit()
    scaled = torch.load(OUT / "scale_table.pt")
    states = {"off": None, "M0": m0, "scaled": scaled}
    curve = {"off": None, "M0": m0, **{f"after {k}": torch.load(OUT / f"scale_table_{k}.pt") for k in SCALE_CURVE
                                       if (OUT / f"scale_table_{k}.pt").exists()}, "after all": scaled}
    tm, samples = curve_questions(args.smoke)
    out, done = resume(OUT / "bench.jsonl", "unit")

    def run_unit(unit, r, lm, state, table, question_set, **kw):
        if unit in done:
            return
        t0 = time.time()
        if table is not None:
            memory.table.data.copy_(table)
        res = lm_eval.simple_evaluate(model=memory_lm(lm, table is not None), num_fewshot=0, log_samples=True,
                                      bootstrap_iters=0, **kw)
        for sub, xs in res["samples"].items():
            for x in xs:
                out.write(json.dumps({"kind": "q", "unit": unit, "set": question_set, "reader": r, "state": state,
                                      "task": sub, "doc": x["doc_id"], "acc": x.get("acc"),
                                      "acc_norm": x.get("acc_norm")}) + "\n")
        out.write(json.dumps({"kind": "done", "unit": unit}) + "\n")
        out.flush()
        print(f"{unit}: {sum(map(len, res['samples'].values()))} questions in {time.time() - t0:.0f}s", flush=True)

    for r, lm in lms.items():
        for state, table in states.items():
            for task, limit in BENCH_TASKS.items():
                run_unit(f"{r}|{state}|{task}", r, lm, state, table, "main", tasks=[task],
                         limit=args.bench_limit or limit)
        for state, table in curve.items():
            run_unit(f"{r}|curve|{state}", r, lm, state, table, "curve", tasks=list(samples), samples=samples,
                     task_manager=tm)
    memory.table.data.copy_(m0)
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["stage0", "episodes", "pilot", "confirm", "transfer",
                                   "edit-setup", "edit-onboard", "edit-episodes", "edit-pilot", "edit", "scale",
                                   "bench"])
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--fact-batch", type=int, default=48)
    p.add_argument("--wiki-batch", type=int, default=8)
    p.add_argument("--table-lr", type=float, default=1e-2)
    p.add_argument("--reader-lr", type=float, default=1e-3)
    p.add_argument("--order-dropout", type=float, default=0.5)
    p.add_argument("--window", type=int, default=0, help="recent words the window reader attends over; 0 = off")
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--quiz-facts", type=int, default=0, help="transfer, edit: quiz only this many facts per set (smoke)")
    p.add_argument("--smoke", action="store_true", help="edit-pilot: a tiny grid; edit, scale, bench: no lock (smoke runs only)")
    p.add_argument("--scale-writes", type=int, default=0, help="scale: this many writes (smoke)")
    p.add_argument("--bench-limit", type=int, default=0, help="bench: this many questions per task (smoke)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"stage0": stage0, "episodes": episodes, "pilot": pilot, "confirm": confirm, "transfer": transfer,
     "edit-setup": edit_setup, "edit-onboard": edit_onboard, "edit-episodes": edit_episodes,
     "edit-pilot": edit_pilot, "edit": edit, "scale": scale, "bench": bench}[args.cmd](args)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
