"""Pre-registered confirmation of the paragraph screen (PLAN.md, "Paragraph confirmation: three holds").

The screen (screen_para.py, exploratory) built the table and readers from paragraphs the models wrote. Holding
the memory to change nothing on STEM text as well as general text nearly removed its benchmark cost, while
paragraph writes passed on about half as well. Here three holds, each trained afresh from three setup seeds on
the same paragraphs:
  plain   KL(memory off || memory on) on 8 general-text chunks per step (the screen's "para")
  stem    the same plus 8 chunks of STEM text (the screen's "para_stem")
  double  the same plus 8 more general-text chunks: stem's amount of hold without the STEM text
In each, A, B and C write ten new facts each, in turn, as the four paragraphs per fact that the writer drafts
with its memory on, 80 steps per write (the screen's P4-80). Two changes from the screen, agreed with the
operator: paragraphs may name the company and its city in either order, and the quiz adds reverse questions
("Which company is headquartered in Prague?"), for evaluation only.

usage (SMEM_OUT is the output directory; SMEM_A/B/C swap in small models for a dry run):
  python confirm_para.py config                          locked_config.json, for lock.py
  python confirm_para.py gen-setup                       A and B write the setup paragraphs (memory off)
  python confirm_para.py facts                           the new facts each model may write, per write seed
  python confirm_para.py setup --arm plain --seed 1      the table with A's and B's readers, then C's
  python confirm_para.py write-gen --arm plain --seed 1  each writer's paragraphs about its facts, memory on
  python confirm_para.py chains --arm plain --seed 1     the writes, and every read
  python confirm_para.py bench --arm plain --seed 1      the main benchmark set with the table (--arm off: memory off)
  python confirm_para.py analyze                         the pre-registered analysis
Every command but config and analyze refuses to run on files that differ from LOCK.json (--smoke skips that).
"""
import argparse
import gc
import json
import math
import random
import time
from collections import Counter, defaultdict
from statistics import mean, stdev

import torch

import data
import run
import screen_para as sp
from run import (ADDED, BENCH_TASKS, CORPUS, DEVICE, EDIT_FACTS, EDIT_WRITE_DROPOUT, HELD, MODELS, N_D0_PROBE,
                 OFFLOAD, OUT, REPLAY_K, REPLAY_KL, WRITE_ORDERS)
from smem import MemLM, SharedMemory, text_rows, words

ARMS = {"plain": ("general",), "stem": ("general", "stem"), "double": ("general", "general")}
SETUP_SEEDS = (1, 2, 3)  # the screen's setup was seed 0
WRITE_SEEDS = (400, 401)  # the screen wrote seeds 300-301
PARAGRAPHS, WRITE_STEPS = 4, 80  # the screen's P4-80
RULES = " Do not mention any other city, and do not use the words \"based\", \"located\" or \"found\"."
REVERSE = ("Which company is headquartered in {v}? The company is called",
           "The company headquartered in {v} is called",
           "Q: Which company has its headquarters in {v}?\nA:")
TRANSFER_BAR = 0.10  # the edit test's bar for the paragraph arm
HARM_BAR = -2.0  # benchmark points, as runs 2 and 3
SETUP_ARGS = ("steps", "onboard_steps", "eval_every", "fact_batch", "wiki_batch", "table_lr", "reader_lr",
              "order_dropout", "micro", "micro_tokens")


def accept(text, e, v):
    """A paragraph counts if it names the company and its city, in either order, no other city, and none of
    the held-out probes' distinctive words, in 12-120 words (data.accept without its order rule)."""
    ws = [w for w, _, _ in words(text)]
    e, v = e.lower(), v.lower()
    return (e in ws and v in ws and 12 <= len(ws) <= 120 and not data.PROBE_CUES & set(ws)
            and not any(w in data.CITY_WORDS and w != v for w in ws))


def city_first(text, e, v):
    ws = [w for w, _, _ in words(text)]
    return ws.index(v.lower()) < ws.index(e.lower())


def prompt(s):
    return sp.STYLES[s["style"]].format(e=s["entity"], v=s["city"]) + RULES


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


def holds(arm):
    """The text an arm's hold covers, as (chunks, how many per step), one KL term per pool."""
    pools = {"general": run.general_text()}
    if "stem" in ARMS[arm]:
        pools["stem"] = sp.stem_text()["train"]
    return [(pools[p], REPLAY_K) for p in ARMS[arm]]


def locked(args):
    """The locked config, once the files match LOCK.json; outside a smoke run, setup's settings come from it."""
    cfg = run.check_lock(args.smoke)
    if not args.smoke:
        for k in SETUP_ARGS:
            setattr(args, k, cfg["setup"][k])
    return cfg


def config(args):
    cfg = {"models": {**MODELS, **ADDED}, "arms": ARMS, "setup_seeds": SETUP_SEEDS, "write_seeds": WRITE_SEEDS,
           "setup": {k: getattr(args, k) for k in SETUP_ARGS},
           "write": {"paragraphs": PARAGRAPHS, "steps": WRITE_STEPS, "lr": sp.WRITE_LR,
                     "order_dropout": EDIT_WRITE_DROPOUT, "replay_k": REPLAY_K, "replay_kl": REPLAY_KL,
                     "rounds": args.rounds, "fallback_off": args.fallback_off, "candidates": sp.CANDIDATES},
           "styles": sp.STYLES, "rules": RULES, "reverse": REVERSE, "bench_tasks": BENCH_TASKS,
           "bars": {"transfer": TRANSFER_BAR, "harm": HARM_BAR}}
    json.dump(cfg, open(OUT / "locked_config.json", "w"), indent=1)
    print(json.dumps(cfg, indent=1))


# ---- the texts ---------------------------------------------------------------------------------------------

def gen_setup(args):
    """A and B write the setup paragraphs, memory off, as the screen's gen-setup but in either order: for
    setup fact i, A writes the styles s with (s + i) even and B the others."""
    locked(args)
    _, _, d0, _, _, _ = run.fact_world(random.Random(0), sp.N_ROWS)  # the screen's and run 3's 500 setup facts
    d0 = d0[:args.facts or None]
    memory = SharedMemory(init_std=0)  # never read: no table exists yet
    lms = {k: MemLM(name, memory, device=DEVICE, offload=OFFLOAD) for k, name in MODELS.items()}
    slots = [{"i": i, "style": s, "writer": "AB"[(s + i) % 2], "entity": e, "city": v, "text": None,
              "attempts": 0, "_done": False} for i, (e, v) in enumerate(d0) for s in range(len(sp.STYLES))]
    for key, lm in lms.items():
        torch.manual_seed(0)
        print(f"{key}: setup paragraphs", flush=True)
        run.generate_rounds(lm, [s for s in slots if s["writer"] == key], prompt, take_reply(args.smoke), args.rounds,
                            max_new_tokens=args.max_new_tokens)
    texts = [[{"style": s["style"], "writer": s["writer"], "text": s["text"]}
              for s in slots if s["i"] == i and s["text"] is not None] for i in range(len(d0))]
    done = [s for s in slots if s["text"] is not None]
    stats = {"attempts": {k: sum(s["attempts"] for s in slots if s["writer"] == k) for k in lms},
             "missing": len(slots) - len(done), "facts_without_text": sum(not t for t in texts),
             "city_first": sum(city_first(s["text"], s["entity"], s["city"]) for s in done if not args.smoke)}
    json.dump({"d0": d0, "texts": texts, "stats": stats}, open(OUT / "setup_texts.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


def facts(args):
    """sp.CANDIDATES new facts per writer and write seed, as the screen's: made-up names whose one-word rows
    nothing else uses, each writer's cities distinct. The same facts serve every arm and setup seed."""
    locked(args)
    st = json.load(open(OUT / "setup_texts.json"))
    _, _, _, _, banned_words, banned_rows = run.fact_world(random.Random(0), sp.N_ROWS)
    paras = [x["text"] for xs in st["texts"] for x in xs]
    markup = sp.chat_markup()
    banned_words |= data.vocabulary(paras + markup)
    banned_rows |= text_rows(paras + markup, sp.N_ROWS)
    plan = {}
    for seed in WRITE_SEEDS:
        rng = random.Random(seed)
        plan[seed] = {}
        for w in (*MODELS, *ADDED):
            names = sp.clean_names(sp.CANDIDATES, rng, banned_words, banned_rows)
            plan[seed][w] = [{"entity": e, "city": c} for e, c in zip(names, rng.sample(data.CITIES, sp.CANDIDATES))]
    json.dump(plan, open(OUT / "write_facts.json", "w"), indent=1)
    print({seed: {w: len(fs) for w, fs in p.items()} for seed, p in plan.items()}, flush=True)


# ---- setup -------------------------------------------------------------------------------------------------

def setup(args):
    """The screen's setup with the arm's hold, from setup seed --seed: the table and A's and B's readers
    together, then C's readers alone on the table as A and B left it. The seed sets the initial table and
    readers, the batches, the held chunks and the n-gram dropout; the setup facts, their paragraphs, the probe
    and its candidates are the same for every arm and seed."""
    locked(args)
    arm, seed = args.arm, args.seed
    st = json.load(open(OUT / "setup_texts.json"))
    world = random.Random(0)
    _, _, d0, _, banned_words, banned_rows = run.fact_world(world, sp.N_ROWS)
    d0 = d0[:len(st["d0"])]
    assert [list(x) for x in d0] == st["d0"]
    probe = d0[:N_D0_PROBE]
    cands = []
    for _, v in probe:  # edit_setup's probe candidates, as in the screen
        cs = world.sample([c for c in data.CITIES if c != v], 9) + [v]
        world.shuffle(cs)
        cands.append(cs)
    paras = [[x["text"] for x in xs] for xs in st["texts"]]
    plain = [p for ps in paras for p in ps]
    chat = [(u.format(e=e), p) for (e, _), ps in zip(d0, paras) for p in ps for u in data.CHAT_USER]
    banned_words |= data.vocabulary(plain)
    banned_rows |= text_rows(plain, sp.N_ROWS)
    held = holds(arm)
    print(f"{arm} {seed}: {len(plain)} setup paragraphs for {sum(map(bool, paras))} of {len(d0)} facts; holds "
          f"{[len(pool) for pool, _ in held]} chunks", flush=True)

    t0 = time.time()
    torch.manual_seed(seed)
    memory = SharedMemory().to(DEVICE)
    lms = {k: MemLM(m, memory, device=DEVICE) for k, m in MODELS.items()}
    log = []
    sp.fit_items(lms, memory, plain, chat, held, args, random.Random(seed), torch.Generator().manual_seed(seed), True,
                 run.edit_curve(lms, probe, cands, log, t0))
    torch.save({"table": memory.table.detach().cpu(), "readers": {k: lm.reader_state() for k, lm in lms.items()},
                "d0": d0, "probe": probe, "cands": cands, "banned_words": sorted(banned_words),
                "banned_rows": sorted(banned_rows), "models": MODELS, "arm": arm, "seed": seed, "args": vars(args),
                "log": log}, OUT / f"{arm}_{seed}_setup.pt")
    print(f"A and B set up in {time.time() - t0:.0f}s", flush=True)
    for lm in lms.values():  # A and B make room for C
        lm.model.to("cpu")
    lms = MemLM.resident = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    table = memory.table.detach().cpu().clone()
    torch.manual_seed(1000 + seed)
    lms = {k: MemLM(m, memory, device=DEVICE) for k, m in ADDED.items()}
    log = []
    onboard = argparse.Namespace(**{**vars(args), "steps": args.onboard_steps})
    sp.fit_items(lms, memory, plain, chat, held, onboard, random.Random(1000 + seed),
                 torch.Generator().manual_seed(1000 + seed), False, run.edit_curve(lms, probe, cands, log, time.time()))
    assert torch.equal(memory.table.detach().cpu(), table)  # C adapted to the table, not the reverse
    torch.save({"readers": {k: lm.reader_state() for k, lm in lms.items()}, "models": ADDED, "arm": arm, "seed": seed,
                "args": vars(args), "log": log}, OUT / f"{arm}_{seed}_onboard.pt")
    print(f"C onboarded; setup done in {time.time() - t0:.0f}s", flush=True)


def load_arm(arm, seed):
    """An arm's table and all three models with their frozen readers (as sp.load_setup)."""
    st = torch.load(OUT / f"{arm}_{seed}_setup.pt", weights_only=False)
    readers = {**st["readers"], **torch.load(OUT / f"{arm}_{seed}_onboard.pt", weights_only=False)["readers"]}
    memory = SharedMemory().to(DEVICE)
    memory.table.data.copy_(st["table"])
    lms = {k: MemLM(n, memory, device=DEVICE, offload=OFFLOAD) for k, n in {**MODELS, **ADDED}.items()}
    for k, lm in lms.items():
        lm.load_reader_state(readers[k])
    return st, memory, lms


# ---- the writes ----------------------------------------------------------------------------------------------

def write_gen(args):
    """Each writer drafts its four paragraphs about each of its candidate facts with the memory on (the arm's
    setup table and its own readers); the first EDIT_FACTS facts with all four accepted are kept. Drafts still
    missing after the locked rounds get up to the locked memory-off rounds, and each records which way it was
    written."""
    cfg = locked(args)
    rounds, fallback = (args.rounds, args.fallback_off) if args.smoke else (cfg["write"]["rounds"],
                                                                            cfg["write"]["fallback_off"])
    st, memory, lms = load_arm(args.arm, args.seed)
    plan = json.load(open(OUT / "write_facts.json"))
    kept, stats = {seed: {} for seed in plan}, {}
    for key, lm in lms.items():
        slots = [{"seed": seed, "k": k, "style": s, **f, "text": None, "memory": None, "attempts": 0, "_done": False}
                 for seed, p in plan.items() for k, f in enumerate(p[key]) for s in range(len(sp.STYLES))]
        take = take_reply(args.smoke)
        t0, first = time.time(), None
        for r, on in [(r, True) for r in range(rounds)] + [(r, False) for r in range(fallback)]:
            pending = [s for s in slots if not s["_done"]]
            if not pending:
                break
            for s, reply in zip(pending, sp.generate_on(lm, [prompt(s) for s in pending], seed=r if on else 1000 + r,
                                                        max_new_tokens=args.max_new_tokens, memory=on)):
                s["attempts"] += 1
                s["_done"] = take(s, reply, r)
                s["memory"] = ("on" if on else "off") if s["_done"] else None
            first = len(slots) - sum(not s["_done"] for s in slots) if first is None else first
            print(f"  {key} round {r}{'' if on else ' (memory off)'}: {len(pending)} pending, "
                  f"{time.time() - t0:.0f}s", flush=True)
        for seed, p in plan.items():
            full = []
            for k, f in enumerate(p[key]):
                mine = [s for s in slots if s["seed"] == seed and s["k"] == k]
                if all(s["text"] for s in mine) and len(full) < EDIT_FACTS:
                    full.append({**f, "texts": [s["text"] for s in mine], "memory": [s["memory"] for s in mine]})
            kept[seed][key] = full
        done = [s for s in slots if s["_done"]]
        stats[key] = {"attempts": sum(s["attempts"] for s in slots), "missing": len(slots) - len(done),
                      "first_round": first, "slots": len(slots), "memory_off": sum(s["memory"] == "off" for s in slots),
                      "city_first": sum(city_first(s["text"], s["entity"], s["city"]) for s in done if not args.smoke),
                      "kept": {seed: len(kept[seed][key]) for seed in plan}}
        print(key, json.dumps(stats[key]), flush=True)
    json.dump({"writes": kept, "stats": stats}, open(OUT / f"writes_{args.arm}_{args.seed}.json", "w"), indent=1)


def reverse_cands(sets, wseed):
    """For each fact: its name and 9 other new names of the write seed whose cities differ, shuffled."""
    pool = [(e, v) for fs in sets.values() for e, v in fs]
    out = {}
    for w, fs in sets.items():
        out[w] = []
        for e, v in fs:
            rng = random.Random(f"{wseed}|{e}")
            others = sorted(x for x, c in pool if c != v and x != e)
            cs = rng.sample(others, min(9, len(others))) + [e]
            rng.shuffle(cs)
            out[w].append(cs)
    return out


def reverse_quiz(lm, facts, cands):
    """Forced choice of the company's name after each REVERSE question about its city."""
    out = []
    for t, template in enumerate(REVERSE):
        lp = lm.choice_logprobs([template.format(v=v) for _, v in facts], cands)
        for i, ((e, _), cs) in enumerate(zip(facts, cands)):
            out.append({"kind": "reverse", "t": t, "e": e, "correct": int(lp[i].argmax() == cs.index(e))})
    return out


def chains(args):
    """Per write seed: A, B and C write in the seed's order, each write on top of the last, keeping the table
    after each; then each model reads every state: forward questions (run.battery) and reverse questions on
    every writer's facts, and before the writes and after the last also greedy chat answers, the setup probe and
    the loss on held-out general and STEM text (and once with memory off)."""
    locked(args)
    arm, seed = args.arm, args.seed
    writes = json.load(open(OUT / f"writes_{arm}_{seed}.json"))["writes"]
    short = {f"{s}|{w}": len(fs) for s, ws in writes.items() for w, fs in ws.items() if len(fs) < EDIT_FACTS}
    if short:
        raise SystemExit(f"writes_{arm}_{seed}.json is short of facts: {short}")
    st, memory, lms = load_arm(arm, seed)
    held = holds(arm)
    m0 = memory.table.detach().clone()
    n = args.quiz_facts or None
    probe, pcands = st["probe"][:n], st["cands"][:n]
    texts_eval = {"wiki": data.wiki_split(CORPUS)[1][:100], "stem": sp.stem_text()["eval"][:100]}
    out, done = run.resume(OUT / "confirm_para.jsonl", "unit")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    for wseed in WRITE_SEEDS:
        unit = f"{arm}|{seed}|{wseed}"
        if unit in done:
            continue
        t0 = time.time()
        slots = writes[str(wseed)]
        sets = {w: [(s["entity"], s["city"]) for s in slots[w]][:n] for w in lms}
        cands = {w: [[c for _, c in fs]] * len(fs) for w, fs in sets.items()}  # the writer's ten cities
        rcands = reverse_cands(sets, wseed)
        order = WRITE_ORDERS[wseed % len(WRITE_ORDERS)]
        memory.table.data.copy_(m0)
        tables = {}
        for pos, w in enumerate(order, 1):
            texts = [t for s in slots[w] for t in s["texts"][:PARAGRAPHS]]
            before, t1 = memory.table.detach().clone(), time.time()
            for _ in sp.write_held(lms[w], texts, args.write_steps or WRITE_STEPS, sp.WRITE_LR, wseed, held):
                pass
            delta = memory.table.detach() - before
            emit(kind="write", unit=unit, pos=pos, writer=w, texts=len(texts),
                 rows_moved=int((delta.abs().sum(-1) > 0).sum()), delta_norm=round(delta.norm().item(), 4),
                 sec=round(time.time() - t1))
            tables[pos] = memory.table.detach().to("cpu", copy=True)
        print(f"{unit}: written at {time.time() - t0:.0f}s", flush=True)

        def read(r, lm, pos, full):
            for w, fs in sets.items():
                for x in run.battery(lm, fs, cands[w], free_tokens=args.free_tokens if full else 0):
                    emit(unit=unit, pos=pos, reader=r, set=w, **x)
                for x in reverse_quiz(lm, fs, rcands[w]):
                    emit(unit=unit, pos=pos, reader=r, set=w, **x)
            if full:
                for x in run.quiz(lm, probe, pcands, templates=HELD):
                    emit(kind="d0", unit=unit, pos=pos, reader=r, **x)
                for name, texts in texts_eval.items():
                    emit(kind="loss", unit=unit, pos=pos, reader=r, text=name, loss=round(lm.mean_loss(texts), 4))

        for r, lm in lms.items():
            for name, texts in texts_eval.items():  # memory off: the same for every state
                emit(kind="loss", unit=unit, pos=-1, reader=r, text=name, loss=round(lm.mean_loss(texts, memory=False), 4))
            memory.table.data.copy_(m0)
            read(r, lm, 0, True)
            for pos, table in tables.items():
                memory.table.data.copy_(table)
                read(r, lm, pos, pos == len(order))
            print(f"{unit}: {r} read every state at {time.time() - t0:.0f}s", flush=True)
        emit(kind="order", unit=unit, order=order, sec=round(time.time() - t0))
        emit(kind="done", unit=unit)
        print(f"{unit} done in {time.time() - t0:.0f}s", flush=True)
    memory.table.data.copy_(m0)
    out.close()


# ---- benchmarks ----------------------------------------------------------------------------------------------

def bench(args):
    """The main benchmark set (run.BENCH_TASKS, as runs 2 and 3), each model with the arm's setup table, or with
    memory off (--arm off, once per machine: each machine runs one setup seed). One record per question."""
    import lm_eval
    locked(args)
    arm, seed = args.arm, args.seed
    if arm == "off":
        memory = SharedMemory(init_std=0).to(DEVICE)  # never read
        lms = {k: MemLM(n, memory, device=DEVICE, offload=OFFLOAD) for k, n in {**MODELS, **ADDED}.items()}
    else:
        _, memory, lms = load_arm(arm, seed)
    out, done = run.resume(OUT / "confirm_para_bench.jsonl", "unit")
    for r, lm in lms.items():
        for task, limit in BENCH_TASKS.items():
            unit = f"{seed}|{arm}|{r}|{task}"
            if unit in done:
                continue
            t0 = time.time()
            res = lm_eval.simple_evaluate(model=run.memory_lm(lm, arm != "off"), tasks=[task], num_fewshot=0,
                                          limit=2 if args.smoke else limit, log_samples=True, bootstrap_iters=0)
            for sub, xs in res["samples"].items():
                for x in xs:
                    out.write(json.dumps({"kind": "q", "unit": unit, "seed": seed, "arm": arm, "reader": r,
                                          "task": sub, "doc": x["doc_id"], "acc": x.get("acc"),
                                          "acc_norm": x.get("acc_norm")}) + "\n")
            out.write(json.dumps({"kind": "done", "unit": unit}) + "\n")
            out.flush()
            print(f"{unit}: {sum(map(len, res['samples'].values()))} questions in {time.time() - t0:.0f}s", flush=True)
    out.close()


# ---- analysis ------------------------------------------------------------------------------------------------

def t_test(xs, n_tests=1):
    """Two-sided one-sample t-test of mean 0, with a Bonferroni-adjusted t interval (analyze_edit.py's)."""
    from scipy import stats
    n, m = len(xs), mean(xs)
    if n < 2:
        return {"n": n, "mean": round(m, 4)}
    se, alpha = stdev(xs) / math.sqrt(n), 0.05 / n_tests
    if se > 0:
        t = m / se
        p, h = 2 * float(stats.t.sf(abs(t), n - 1)), float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
    else:
        t, p, h = (math.copysign(math.inf, m) if m else 0.0), (0.0 if m else 1.0), 0.0
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "ci_level": 1 - alpha,
            "t": round(t, 3), "p": p, "p_bonf": min(1.0, p * n_tests)}


def harm(scores, arm, r, seeds, n_boot=10_000):
    """Per model and arm: the mean over the five tasks of (accuracy with the setup table - memory off), each
    question's change averaged over the setup seeds; 95% bootstrap interval resampling questions within tasks
    (runs 2 and 3's rule, over seeds). n_boot=0: the mean alone."""
    import numpy as np
    diffs = {}
    for task in sp.METRIC:
        per_q = defaultdict(list)
        for seed in seeds:
            x, y = scores.get((seed, arm, r, task), {}), scores.get((seed, "off", r, task), {})
            for q in set(x) & set(y):
                per_q[q].append(x[q] - y[q])
        diffs[task] = [mean(v) for _, v in sorted(per_q.items()) if len(v) == len(seeds)]
    diffs = {t: np.asarray(d, dtype=float) for t, d in diffs.items() if d}
    if not diffs:
        return None
    m = float(np.mean([d.mean() for d in diffs.values()])) * 100
    out = {"diff": round(m, 2), "seeds": list(seeds), "by_task": {t: round(float(d.mean()) * 100, 2) for t, d in diffs.items()}}
    if n_boot:
        rng, boot = np.random.default_rng(0), np.zeros(n_boot)
        for d in diffs.values():
            for i in range(0, n_boot, 1000):  # 1,000 resamples at a time
                boot[i:i + 1000] += d[rng.integers(0, len(d), size=(min(1000, n_boot - i), len(d)))].mean(1)
        boot = np.sort(boot / len(diffs))
        out["ci"] = [round(float(boot[int(0.025 * n_boot)]) * 100, 2), round(float(boot[int(0.975 * n_boot) - 1]) * 100, 2)]
        out["harm"] = m < HARM_BAR and out["ci"][1] < 0
    return out


def analyze(args):
    """The pre-registered analysis (PLAN.md, "Paragraph confirmation: three holds").
    Primary 1, harm: harm() per arm and model; harm if the mean is below -2 points and the interval excludes 0.
    Primary 2, transfer: per arm, one value per episode (setup seed x write seed x writer): the two non-writers'
    held-out forced-choice accuracy on the writer's facts right after its write minus before any write, averaged
    over the two; t-test with Bonferroni over the three arms; shown if the adjusted p < 0.05 and the mean is at
    least 10 points.
    Secondary: the arms paired per episode (stem - double: STEM text at equal hold; double - plain: more hold;
    stem - plain), 95% t intervals; per arm the writer's own gain, chat forced choice, greedy chat answers after
    the last write, reverse questions, the first write's facts after the last, the setup probe, loss on held-out
    general and STEM text with the table minus memory off, harm per seed and per task, setup recall at the end
    of setup, first-round acceptance of the drafts and paragraphs with the city first."""
    cfg = json.load(open(OUT / "locked_config.json"))
    seeds_all = tuple(cfg["setup_seeds"])
    summary = {"harm": {}, "transfer": {}, "paired": {}, "secondary": {}, "harm_by_seed": {}, "setup_recall": {},
               "drafts": {}}
    path = OUT / "confirm_para_bench.jsonl"
    if path.exists():
        scores = defaultdict(dict)  # (seed, arm, reader, task) -> {(subtask, doc): correct}
        for rec in map(json.loads, open(path)):
            if rec["kind"] == "q":
                task = "mmlu" if rec["task"].startswith("mmlu") else rec["task"]
                scores[rec["seed"], rec["arm"], rec["reader"], task][rec["task"], rec["doc"]] = rec[sp.METRIC[task]]
        for arm in ARMS:
            for r in (*MODELS, *ADDED):
                seeds = tuple(s for s in seeds_all if (s, arm, r, "mmlu") in scores and (s, "off", r, "mmlu") in scores)
                if seeds:
                    summary["harm"][f"{arm}|{r}"] = harm(scores, arm, r, seeds)
                    summary["harm_by_seed"][f"{arm}|{r}"] = {s: harm(scores, arm, r, (s,), n_boot=0)["diff"] for s in seeds}

    logs = json.load(open(OUT / "setup_logs.json")) if (OUT / "setup_logs.json").exists() else {}
    for arm in ARMS:
        for seed in seeds_all:
            key, files = f"{arm}|{seed}", [OUT / f"{arm}_{seed}_{x}.pt" for x in ("setup", "onboard")]
            if all(f.exists() for f in files):  # else the logs saved from them
                logs[key] = [torch.load(f, weights_only=False, map_location="cpu")["log"] for f in files]
            if key in logs:
                a, c = (log[-1] for log in logs[key])
                summary["setup_recall"][key] = {k: {"held_out": v["held_out"], "chat": v["chat"]}
                                                for k, v in {**a, **c}.items() if isinstance(v, dict)}
            wpath = OUT / f"writes_{arm}_{seed}.json"
            if wpath.exists():
                summary["drafts"][key] = json.load(open(wpath))["stats"]

    path = OUT / "confirm_para.jsonl"
    recs = [json.loads(line) for line in open(path)] if path.exists() else []
    orders = {r["unit"]: r["order"] for r in recs if r["kind"] == "order"}
    cells = defaultdict(list)  # (unit, pos, reader, set, kind) -> correct; held-out templates only for "mc"
    d0, loss = defaultdict(list), {}
    for r in recs:
        if r["kind"] in ("mc", "chat_mc", "chat_free", "reverse") and (r["kind"] != "mc" or r["t"] in HELD):
            cells[r["unit"], r["pos"], r["reader"], r["set"], r["kind"]].append(r["correct"])
        elif r["kind"] == "d0":
            d0[r["unit"], r["pos"], r["reader"]].append(r["correct"])
        elif r["kind"] == "loss":
            loss[r["unit"], r["pos"], r["reader"], r["text"]] = r["loss"]

    def acc(unit, pos, reader, w, kind="mc"):
        return mean(cells[unit, pos, reader, w, kind])

    eps = defaultdict(dict)  # arm -> {(seed, wseed, writer): others}
    for arm in ARMS:
        res = defaultdict(list)
        for unit, order in sorted(orders.items()):
            a, seed, wseed = unit.split("|")
            if a != arm:
                continue
            last = len(order)
            for pos, w in enumerate(order, 1):
                others = [r for r in order if r != w]
                eps[arm][seed, wseed, w] = mean(acc(unit, pos, r, w) - acc(unit, 0, r, w) for r in others)
                res["own"].append(acc(unit, pos, w, w) - acc(unit, 0, w, w))
                res["chat"].append(mean(acc(unit, pos, r, w, "chat_mc") - acc(unit, 0, r, w, "chat_mc") for r in others))
                res["free"].append(mean(acc(unit, last, r, w, "chat_free") - acc(unit, 0, r, w, "chat_free") for r in others))
                res["reverse"].append(mean(acc(unit, pos, r, w, "reverse") - acc(unit, 0, r, w, "reverse") for r in others))
                res["reverse_own"].append(acc(unit, pos, w, w, "reverse") - acc(unit, 0, w, w, "reverse"))
            first = order[0]
            res["kept"].append(mean(acc(unit, last, r, first) - acc(unit, 1, r, first) for r in order if r != first))
            for r in order:
                res["d0"].append(mean(d0[unit, last, r]) - mean(d0[unit, 0, r]))
                for text in ("wiki", "stem"):
                    off = loss[unit, -1, r, text]
                    res[f"{text}_m0"].append(loss[unit, 0, r, text] - off)
                    res[f"{text}_end"].append(loss[unit, last, r, text] - off)
            res["rows_moved"] += [x["rows_moved"] for x in recs if x["kind"] == "write" and x["unit"] == unit]
        if eps[arm]:
            test = t_test(list(eps[arm].values()), n_tests=len(ARMS))
            test["shown"] = test.get("p_bonf", 1.0) < 0.05 and test["mean"] >= TRANSFER_BAR
            summary["transfer"][arm] = test
            summary["secondary"][arm] = {k: round(mean(v), 4) for k, v in res.items()}
    for a, b in (("stem", "double"), ("double", "plain"), ("stem", "plain")):
        keys = sorted(set(eps[a]) & set(eps[b]))
        if len(keys) > 1:
            summary["paired"][f"{a}-{b}"] = t_test([eps[a][k] - eps[b][k] for k in keys])
    json.dump(summary, open(OUT / "confirm_para_summary.json", "w"), indent=1)
    json.dump(logs, open(OUT / "setup_logs.json", "w"), indent=1)

    print("harm (benchmark points, table attached - memory off; harm if below -2 and the interval excludes 0):")
    for key, v in summary["harm"].items():
        print(f"  {key:10s} {v['diff']:>6} {v['ci']} harm: {v['harm']}  by seed {summary['harm_by_seed'][key]}")
    print("transfer (non-writers' held-out forced choice right after the write; shown if p_bonf < 0.05 and >= 0.10):")
    for arm, v in summary["transfer"].items():
        print(f"  {arm:10s} {v['mean']:>7} {v.get('ci')} n={v['n']} p_bonf={v.get('p_bonf', float('nan')):.2g} "
              f"shown: {v['shown']}")
    for key, v in summary["paired"].items():
        print(f"  {key:10s} {v['mean']:>7} {v.get('ci')} n={v['n']}")
    for arm, v in summary["secondary"].items():
        print(f"  {arm:10s} {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("config", "gen-setup", "facts", "setup", "write-gen", "chains", "bench",
                                       "analyze"))
    p.add_argument("--arm", choices=(*ARMS, "off"), default="plain")
    p.add_argument("--seed", type=int, default=SETUP_SEEDS[0])
    p.add_argument("--smoke", action="store_true", help="dry run: no lock check, keep every reply, 2 questions per task")
    p.add_argument("--facts", type=int, default=0, help="gen-setup: setup facts to write about (dry run)")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--fallback-off", type=int, default=12, help="write-gen: memory-off rounds for what is still missing")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--onboard-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--fact-batch", type=int, default=48)
    p.add_argument("--wiki-batch", type=int, default=8, help="chunks per held pool per setup step")
    p.add_argument("--table-lr", type=float, default=1e-2)
    p.add_argument("--reader-lr", type=float, default=1e-3)
    p.add_argument("--order-dropout", type=float, default=0.5)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--micro-tokens", type=int, default=1024, help="padded tokens per setup micro-batch")
    p.add_argument("--write-steps", type=int, default=0, help="steps per write (dry run)")
    p.add_argument("--quiz-facts", type=int, default=0, help="quiz only this many facts per set (dry run)")
    p.add_argument("--free-tokens", type=int, default=20)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"config": config, "gen-setup": gen_setup, "facts": facts, "setup": setup, "write-gen": write_gen,
     "chains": chains, "bench": bench, "analyze": analyze}[args.command](args)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
