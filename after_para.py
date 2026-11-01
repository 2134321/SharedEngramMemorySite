"""Follow-up to the paragraph confirmation (PLAN.md, "Paragraph confirmation, follow-up: general ability after the
writes, and a blob past every context window").

The confirmation (confirm_para.py) benchmarked each hold's memory as setup left it, before the writes. Here its
plain and STEM tables after the writes are rebuilt from the saved setup checkpoints by replaying the same writes
(the same paragraphs, order, settings and seeds), checked against the confirmation's answers, and benchmarked.
Then, exploratory, each table after the writes takes a blob of model-written paragraphs about 2,200 more
companies, more text than any model's context window, in one pass, and is quizzed and benchmarked again.

usage (SMEM_OUT is the output directory; SMEM_A/B/C swap in small models for a dry run):
  python after_para.py config --checkpoints DIR ...     locked_config.json, with the checkpoints' sha256, for lock.py
  python after_para.py blob-gen                         the blob's paragraphs, drafted with memory off (machine 1)
  python after_para.py replay --seed 1                  each hold's tables after both write sequences, and reads
  python after_para.py blob --seed 1                    the blob written on each table after the writes, and reads
  python after_para.py bench --seed 1 --states off      the main set: off, setup, writes or blob (comma-separated)
  python after_para.py analyze                          the analysis
Every command but config and analyze refuses to run on files that differ from LOCK.json (--smoke skips that).
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

import confirm_para as cp
import data
import run
import screen_para as sp
from run import ADDED, BENCH_TASKS, CORPUS, DEVICE, EDIT_WRITE_DROPOUT, HELD, MODELS, OFFLOAD, OUT, REPLAY_KL, WRITE_ORDERS
from smem import WRITE_TOKENS, MemLM, SharedMemory, address_rows, drop_orders, order_drop_mask, text_rows

HOLDS = ("plain", "stem")  # the operator dropped the double hold
CONFIRM = Path(os.environ.get("SMEM_CONFIRM", run.ROOT / "results_confirm_para"))  # the confirmation's results
BLOB_COMPANIES, BLOB_BATCH, BLOB_SAMPLE, BLOB_SEED = 2200, 40, 100, 5000
WINDOW = 262144  # the largest context window of the three models (Qwen3.5-4B and Nemotron Nano)


def locked(args):
    return run.check_lock(args.smoke)


def config(args):
    ckpts = {}
    for d in args.checkpoints:
        for f in sorted(Path(d).glob("*_*_*.pt")):  # {hold}_{seed}_setup.pt, {hold}_{seed}_onboard.pt
            hold, seed, _ = f.stem.split("_")
            if hold in HOLDS and int(seed) in cp.SETUP_SEEDS:
                ckpts[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    cfg = {"models": {**MODELS, **ADDED}, "holds": {h: cp.ARMS[h] for h in HOLDS}, "setup_seeds": cp.SETUP_SEEDS,
           "write_seeds": cp.WRITE_SEEDS, "checkpoints": ckpts,
           "write": json.load(open(CONFIRM / "locked_config.json"))["write"],
           "blob": {"companies": BLOB_COMPANIES, "batch": BLOB_BATCH, "sample": BLOB_SAMPLE, "seed": BLOB_SEED,
                    "rounds": args.rounds, "min_tokens": WINDOW},
           "bench_tasks": BENCH_TASKS, "bars": {"harm": cp.HARM_BAR}}
    json.dump(cfg, open(OUT / "locked_config.json", "w"), indent=1)
    print(json.dumps(cfg, indent=1))


class Tables:
    """The three models on one shared table, with either hold's frozen readers from this setup seed's
    checkpoints, and any table loaded in (as cp.load_arm, without loading the models again per hold)."""

    def __init__(self, seed):
        self.seed, self.hold = seed, None
        self.memory = SharedMemory().to(DEVICE)
        self.lms = {k: MemLM(n, self.memory, device=DEVICE, offload=OFFLOAD) for k, n in {**MODELS, **ADDED}.items()}

    def use(self, hold, table=None):
        if hold != self.hold:
            self.st = torch.load(OUT / f"{hold}_{self.seed}_setup.pt", weights_only=False)
            onboard = torch.load(OUT / f"{hold}_{self.seed}_onboard.pt", weights_only=False)
            readers = {**self.st["readers"], **onboard["readers"]}
            for k, lm in self.lms.items():
                lm.load_reader_state(readers[k])
            self.hold = hold
        self.memory.table.data.copy_(self.st["table"] if table is None else table)


def check_checkpoints(cfg, seed):
    """The uploaded checkpoints of this setup seed are the ones the lock names."""
    for name, digest in cfg["checkpoints"].items():
        if int(name.split("_")[1]) == seed and hashlib.sha256((OUT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} differs from the locked config")


def writer(path):
    """An emit(**record) appending to `path` (resuming as run.resume), and the units already done."""
    out, done = run.resume(path, "unit")

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()
    return emit, done, out


def sequence(hold, seed, wseed, n=None):
    """A confirmation write sequence: its order, each writer's paragraphs, facts and candidate cities."""
    slots = json.load(open(CONFIRM / f"writes_{hold}_{seed}.json"))["writes"][str(wseed)]
    sets = {w: [(s["entity"], s["city"]) for s in slots[w]][:n] for w in "ABC"}
    cands = {w: [[c for _, c in fs]] * len(fs) for w, fs in sets.items()}  # the writer's ten cities
    texts = {w: [t for s in slots[w] for t in s["texts"][:cp.PARAGRAPHS]] for w in "ABC"}
    return WRITE_ORDERS[wseed % len(WRITE_ORDERS)], texts, sets, cands


def reads(lm, emit, r, state, sets, cands, probe, pcands, texts_eval, sample, free_tokens, **unit):
    """One model's reads with the table now: the written facts (run.battery: forced choice, chat forced
    choice, greedy chat answers), the blob sample, the setup probe and the loss on held-out text."""
    for w, fs in sets.items():
        for x in run.battery(lm, fs, cands[w], free_tokens=free_tokens):
            emit(**unit, state=state, reader=r, set=w, **x)
    if sample:
        for x in run.quiz(lm, [(s["entity"], s["city"]) for s in sample], [s["cands"] for s in sample], templates=HELD):
            emit(kind="blob_mc", **unit, state=state, reader=r, **x)
    for x in run.quiz(lm, probe, pcands, templates=HELD):
        emit(kind="d0", **unit, state=state, reader=r, **x)
    for name, texts in texts_eval.items():
        emit(kind="loss", **unit, state=state, reader=r, text=name, loss=round(lm.mean_loss(texts), 4))


def eval_texts(smoke=False):
    n = 4 if smoke else 100
    return {"wiki": data.wiki_split(CORPUS)[1][:n], "stem": sp.stem_text()["eval"][:n]}


# ---- the tables after the writes -------------------------------------------------------------------------------

def replay(args):
    """Per hold and write sequence: the confirmation's three writes again on the setup table (cp.chains' loop:
    the same paragraphs, order, settings and seed), then every model reads the setup table and the table after
    the writes; that table is saved for the benchmarks and the blob."""
    cfg = locked(args)
    seed, n = args.seed, args.quiz_facts or None
    if not args.smoke:
        check_checkpoints(cfg, seed)
    tb = Tables(seed)
    texts_eval = eval_texts(args.smoke)
    emit, done, out = writer(OUT / "after_para.jsonl")
    for hold in HOLDS:
        tb.use(hold)
        held = cp.holds(hold)
        m0 = tb.memory.table.detach().clone()
        probe, pcands = tb.st["probe"][:n], tb.st["cands"][:n]
        for wseed in cp.WRITE_SEEDS:
            unit = f"replay|{hold}|{seed}|{wseed}"
            if unit in done:
                continue
            t0 = time.time()
            order, texts, sets, cands = sequence(hold, seed, wseed, n)
            tb.use(hold)
            for pos, w in enumerate(order, 1):
                before, t1 = tb.memory.table.detach().clone(), time.time()
                for _ in sp.write_held(tb.lms[w], texts[w], args.write_steps or cp.WRITE_STEPS, sp.WRITE_LR, wseed, held):
                    pass
                delta = tb.memory.table.detach() - before
                emit(kind="write", unit=unit, pos=pos, writer=w, texts=len(texts[w]),
                     rows_moved=int((delta.abs().sum(-1) > 0).sum()), delta_norm=round(delta.norm().item(), 4),
                     sec=round(time.time() - t1))
            after = tb.memory.table.detach().to("cpu", copy=True)
            torch.save(after, OUT / f"writes_{hold}_{seed}_{wseed}.pt")
            for r, lm in tb.lms.items():
                for name, ts in texts_eval.items():
                    emit(kind="loss", unit=unit, state="off", reader=r, text=name, loss=round(lm.mean_loss(ts, memory=False), 4))
                for state, table in (("setup", m0), ("writes", after)):
                    tb.memory.table.data.copy_(table)
                    reads(lm, emit, r, state, sets, cands, probe, pcands, texts_eval, None, args.free_tokens, unit=unit)
            emit(kind="order", unit=unit, order=order, sec=round(time.time() - t0))
            emit(kind="done", unit=unit)
            print(f"{unit} done in {time.time() - t0:.0f}s", flush=True)
    out.close()


# ---- the blob ----------------------------------------------------------------------------------------------------

def blob_gen(args):
    """Paragraphs about BLOB_COMPANIES more made-up companies, one per style, drafted with memory off by A, B and
    C in turn (cp.prompt and cp.accept, as the confirmation's setup paragraphs), with more companies added
    until every model's tokenizer counts more than WINDOW tokens; and a fixed sample of fully drafted companies
    to quiz, each with 9 other cities."""
    locked(args)
    st = json.load(open(CONFIRM / "setup_texts.json"))
    _, _, _, _, banned_words, banned_rows = run.fact_world(random.Random(0), sp.N_ROWS)
    paras = [x["text"] for xs in st["texts"] for x in xs]
    written = [f["entity"] for p in json.load(open(CONFIRM / "write_facts.json")).values() for fs in p.values() for f in fs]
    markup = sp.chat_markup()
    banned_words |= data.vocabulary(paras + markup + written)
    banned_rows |= text_rows(paras + markup + written, sp.N_ROWS)  # no blob name shares a row with a written one
    memory = SharedMemory(init_std=0)  # never read: memory off
    lms = {k: MemLM(n, memory, device=DEVICE, offload=OFFLOAD) for k, n in {**MODELS, **ADDED}.items()}
    rng = random.Random(BLOB_SEED)
    names, cities, slots, tokens = [], [], [], {k: 0 for k in lms}
    want = args.blob_companies or BLOB_COMPANIES
    while len(names) < want or (not args.smoke and min(tokens.values()) <= WINDOW):
        more = sp.clean_names(want - len(names) if len(names) < want else want // 10, rng, banned_words, banned_rows)
        new = [{"i": len(names) + j, "style": s, "drafter": "ABC"[(len(names) + j + s) % 3], "entity": e,
                "city": v, "text": None, "attempts": 0, "_done": False}
               for j, (e, v) in enumerate(zip(more, [rng.choice(data.CITIES) for _ in more])) for s in range(len(sp.STYLES))]
        names += more
        cities += [s["city"] for s in new if s["style"] == 0]
        for key, lm in lms.items():
            torch.manual_seed(BLOB_SEED + len(slots))
            print(f"{key}: blob paragraphs", flush=True)
            run.generate_rounds(lm, [s for s in new if s["drafter"] == key], cp.prompt, cp.take_reply(args.smoke),
                                args.rounds, max_new_tokens=args.max_new_tokens)
        slots += new
        for k, lm in lms.items():
            tokens[k] += sum(len(lm.tok(s["text"], add_special_tokens=False)["input_ids"]) for s in new if s["text"])
        print(f"{len(names)} companies, tokens {tokens}", flush=True)
    k = len(sp.STYLES)
    full = [i for i in range(len(names)) if all(s["text"] for s in slots[k * i:k * i + k])]
    pick = random.Random(BLOB_SEED + 1)
    sample = []
    for i in sorted(pick.sample(full, min(args.blob_sample or BLOB_SAMPLE, len(full)))):
        cs = pick.sample([c for c in data.CITIES if c != cities[i]], 9) + [cities[i]]
        pick.shuffle(cs)
        sample.append({"entity": names[i], "city": cities[i], "cands": cs})
    done = [s for s in slots if s["text"]]
    stats = {"companies": len(names), "slots": len(slots), "paragraphs": len(done), "fully_drafted": len(full),
             "attempts": {k: sum(s["attempts"] for s in slots if s["drafter"] == k) for k in lms}, "tokens": tokens,
             "words": sum(len(s["text"].split()) for s in done), "past_every_window": min(tokens.values()) > WINDOW}
    json.dump({"paragraphs": [{k: s[k] for k in ("i", "style", "drafter", "entity", "city", "text")} for s in done],
               "sample": sample, "stats": stats}, open(OUT / "blob.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


def write_blob(lms, held, paragraphs, seed, batch=BLOB_BATCH):
    """Each model writes its share of the blob as one write: Adam on the table (sp.WRITE_LR), one pass through
    its paragraphs in shuffled batches, one step per batch, n-gram dropout, the hold replayed at every step, and
    only rows its paragraphs address can move (sp.write_held, with a new batch at each step)."""
    stats = {}
    for w, lm in lms.items():
        texts = [p["text"] for p in paragraphs if p["drafter"] == w]
        random.Random(f"{seed}|{w}").shuffle(texts)
        table = lm.memory.table
        opt = torch.optim.Adam([table], lr=sp.WRITE_LR)
        rows = address_rows(texts, lm.memory.n_rows).to(table.device)
        gen, pick = torch.Generator().manual_seed(seed), random.Random(seed)
        before, t0, steps = table.detach().clone(), time.time(), 0
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            opt.zero_grad(set_to_none=True)
            lm.backward_mean(drop_orders(lm.encode(chunk), order_drop_mask(len(chunk), EDIT_WRITE_DROPOUT, gen)),
                             max_tokens=WRITE_TOKENS)
            for pool, k in held:
                lm.kl_backward(lm.encode(pick.sample(pool, k), max_tokens=64), REPLAY_KL, max_tokens=WRITE_TOKENS)
            table.grad.mul_(rows.unsqueeze(-1))  # Adam leaves rows with no gradient where they are
            opt.step()
            steps += 1
        delta = table.detach() - before
        stats[w] = {"paragraphs": len(texts), "steps": steps, "rows_moved": int((delta.abs().sum(-1) > 0).sum()),
                    "delta_norm": round(delta.norm().item(), 4), "sec": round(time.time() - t0)}
        print(f"  {w} wrote its {len(texts)} blob paragraphs in {steps} steps, {time.time() - t0:.0f}s", flush=True)
    return stats


def blob(args):
    """Per hold and write sequence: the blob sample read with the table after the writes; the blob written on
    that table (write_blob); then every model reads it all again with the table after the blob, which is saved
    for the benchmarks."""
    cfg = locked(args)
    seed, n = args.seed, args.quiz_facts or None
    b = json.load(open(OUT / "blob.json"))
    if not args.smoke and not b["stats"]["past_every_window"]:
        raise SystemExit("blob.json does not pass every context window")
    tb = Tables(seed)
    texts_eval = eval_texts(args.smoke)
    emit, done, out = writer(OUT / "after_para.jsonl")
    for hold in HOLDS:
        tb.use(hold)
        held = cp.holds(hold)
        probe, pcands = tb.st["probe"][:n], tb.st["cands"][:n]
        for wseed in cp.WRITE_SEEDS:
            unit = f"blob|{hold}|{seed}|{wseed}"
            if unit in done:
                continue
            t0 = time.time()
            _, _, sets, cands = sequence(hold, seed, wseed, n)
            after = torch.load(OUT / f"writes_{hold}_{seed}_{wseed}.pt")
            tb.use(hold, after)
            for r, lm in tb.lms.items():
                for x in run.quiz(lm, [(s["entity"], s["city"]) for s in b["sample"]], [s["cands"] for s in b["sample"]],
                                  templates=HELD):
                    emit(kind="blob_mc", unit=unit, state="writes", reader=r, **x)
            emit(kind="blob_write", unit=unit, **write_blob(tb.lms, held, b["paragraphs"], BLOB_SEED))
            torch.save(tb.memory.table.detach().to("cpu", copy=True), OUT / f"blob_{hold}_{seed}_{wseed}.pt")
            for r, lm in tb.lms.items():
                reads(lm, emit, r, "blob", sets, cands, probe, pcands, texts_eval, b["sample"], args.free_tokens, unit=unit)
            emit(kind="done", unit=unit)
            print(f"{unit} done in {time.time() - t0:.0f}s", flush=True)
    out.close()


# ---- benchmarks --------------------------------------------------------------------------------------------------

def bench(args):
    """The main benchmark set (run.BENCH_TASKS), each model with memory off, either hold's setup table, its
    tables after the writes, or after the blob, on this machine. One record per question."""
    import lm_eval
    cfg = locked(args)
    seed = args.seed
    if not args.smoke and args.states != "off":  # memory off runs before the checkpoints arrive
        check_checkpoints(cfg, seed)
    states = []
    for s in args.states.split(","):
        states += ([("off", None, None)] if s == "off" else [(f"setup:{h}", h, None) for h in HOLDS] if s == "setup" else
                   [(f"{s}:{h}:{w}", h, OUT / f"{s}_{h}_{seed}_{w}.pt") for h in HOLDS for w in cp.WRITE_SEEDS])
    tb = Tables(seed)
    emit, done, out = writer(OUT / "after_para_bench.jsonl")
    for state, hold, path in states:
        if hold:
            tb.use(hold, torch.load(path) if path else None)
        for r, lm in tb.lms.items():
            for task, limit in BENCH_TASKS.items():
                unit = f"{seed}|{state}|{r}|{task}"
                if unit in done:
                    continue
                t0 = time.time()
                res = lm_eval.simple_evaluate(model=run.memory_lm(lm, hold is not None), tasks=[task], num_fewshot=0,
                                              limit=2 if args.smoke else limit, log_samples=True, bootstrap_iters=0)
                for sub, xs in res["samples"].items():
                    for x in xs:
                        emit(kind="q", unit=unit, seed=seed, state=state, reader=r, task=sub, doc=x["doc_id"],
                             acc=x.get("acc"), acc_norm=x.get("acc_norm"))
                emit(kind="done", unit=unit)
                print(f"{unit}: {sum(map(len, res['samples'].values()))} questions in {time.time() - t0:.0f}s", flush=True)
    out.close()


# ---- analysis ----------------------------------------------------------------------------------------------------

def change(scores, pairs, r, n_boot=10_000):
    """Per model: the mean over the five tasks of the per-question change a - b, each question's change averaged
    over the (seed, a, b) pairs; 95% bootstrap interval resampling questions within tasks (cp.harm's rule)."""
    import numpy as np
    diffs = {}
    for task in sp.METRIC:
        per_q = defaultdict(list)
        for seed, a, b in pairs:
            x, y = scores.get((seed, a, r, task), {}), scores.get((seed, b, r, task), {})
            for q in set(x) & set(y):
                per_q[q].append(x[q] - y[q])
        diffs[task] = [mean(v) for _, v in sorted(per_q.items()) if len(v) == len(pairs)]
    diffs = {t: np.asarray(d, dtype=float) for t, d in diffs.items() if d}
    if len(diffs) < len(sp.METRIC):
        return None
    m = float(np.mean([d.mean() for d in diffs.values()])) * 100
    rng, boot = np.random.default_rng(0), np.zeros(n_boot)
    for d in diffs.values():
        for i in range(0, n_boot, 1000):
            boot[i:i + 1000] += d[rng.integers(0, len(d), size=(min(1000, n_boot - i), len(d)))].mean(1)
    boot = np.sort(boot / len(diffs))
    ci = [round(float(boot[int(0.025 * n_boot)]) * 100, 2), round(float(boot[int(0.975 * n_boot) - 1]) * 100, 2)]
    return {"diff": round(m, 2), "ci": ci, "harm": m < cp.HARM_BAR and ci[1] < 0, "pairs": len(pairs),
            "by_task": {t: round(float(d.mean()) * 100, 2) for t, d in diffs.items()}}


def t95(xs):
    m = mean(xs)
    if len(xs) < 2:
        return {"n": len(xs), "mean": round(m, 4)}
    from scipy import stats
    h = float(stats.t.ppf(0.975, len(xs) - 1)) * stdev(xs) / math.sqrt(len(xs))
    return {"n": len(xs), "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def analyze(args):
    """Part 1 (pre-registered in the follow-up): harm with the tables after the writes, per hold and model, and
    the writes' own cost and the setup tables again. The replay check against the confirmation. Part 2
    (exploratory): the written facts, the blob sample, the setup probe, the loss and the benchmarks after the blob."""
    cfg = json.load(open(OUT / "locked_config.json"))
    seeds, wseeds = cfg["setup_seeds"], cfg["write_seeds"]
    summary = {"harm": {}, "replay": {}, "facts": {}, "blob": {}, "loss": {}, "d0": {}}
    path = OUT / "after_para_bench.jsonl"
    scores = defaultdict(dict)  # (seed, state, reader, task) -> {(subtask, doc): correct}
    if path.exists():
        for rec in map(json.loads, open(path)):
            if rec["kind"] == "q":
                task = "mmlu" if rec["task"].startswith("mmlu") else rec["task"]
                scores[rec["seed"], rec["state"], rec["reader"], task][rec["task"], rec["doc"]] = rec[sp.METRIC[task]]
    for h in HOLDS:
        for r in "ABC":
            ss = [s for s in seeds if (s, "off", r, "mmlu") in scores]
            comps = {"after_writes": [(s, f"writes:{h}:{w}", "off") for s in ss for w in wseeds],
                     "setup_again": [(s, f"setup:{h}", "off") for s in ss],
                     "added_by_writes": [(s, f"writes:{h}:{w}", f"setup:{h}") for s in ss for w in wseeds],
                     "after_blob": [(s, f"blob:{h}:{w}", "off") for s in ss for w in wseeds],
                     "added_by_blob": [(s, f"blob:{h}:{w}", f"writes:{h}:{w}") for s in ss for w in wseeds]}
            for name, pairs in comps.items():
                if pairs and all((s, a, r, "mmlu") in scores for s, a, _ in pairs):
                    summary["harm"].setdefault(name, {})[f"{h}|{r}"] = change(scores, pairs, r, n_boot=args.n_boot)

    recs = [json.loads(line) for line in open(OUT / "after_para.jsonl")] if (OUT / "after_para.jsonl").exists() else []
    cells = defaultdict(list)  # (hold, seed, wseed, state, reader, set, kind) -> correct (held-out templates for mc)
    for x in recs:
        if x["kind"] in ("mc", "chat_mc", "chat_free") and (x["kind"] != "mc" or x["t"] in HELD):
            _, h, s, w = x["unit"].split("|")
            cells[h, int(s), int(w), x["state"], x["reader"], x["set"], x["kind"]].append(x["correct"])

    # the replay against the confirmation: held-out forced choice before the writes (pos 0) and after them (pos 3)
    mine = {(*x["unit"].split("|")[1:], {"setup": 0, "writes": 3}[x["state"]], x["reader"], x["set"], x["t"], x["e"]): x["correct"]
            for x in recs if x["kind"] == "mc" and x["unit"].startswith("replay|") and x["t"] in HELD}
    theirs = {}
    if mine and (CONFIRM / "confirm_para.jsonl").exists():
        for line in open(CONFIRM / "confirm_para.jsonl"):
            if '"kind": "mc"' in line:
                x = json.loads(line)
                if x["pos"] in (0, 3) and x["t"] in HELD:
                    key = (*x["unit"].split("|"), x["pos"], x["reader"], x["set"], x["t"], x["e"])
                    if key in mine:
                        theirs[key] = x["correct"]
    for h in HOLDS:
        for pos in (0, 3):
            keys = [k for k in theirs if k[0] == h and k[3] == pos]
            if keys:
                summary["replay"][f"{h}|{'setup' if pos == 0 else 'writes'}"] = {
                    "answers": len(keys), "same": sum(mine[k] == theirs[k] for k in keys)}
    old = {(x["unit"], x["pos"]): x for x in map(json.loads, open(CONFIRM / "confirm_para.jsonl")) if x["kind"] == "write"} \
        if mine else {}
    rows = [(x["rows_moved"], old[x["unit"].split("|", 1)[1], x["pos"]]["rows_moved"]) for x in recs
            if x["kind"] == "write" and (x["unit"].split("|", 1)[1], x["pos"]) in old]
    if rows:
        summary["replay"]["writes_with_the_same_rows_moved"] = f"{sum(a == b for a, b in rows)} of {len(rows)}"

    off = {(x["unit"].split("|", 1)[1], x["reader"], x["text"]): x["loss"] for x in recs
           if x["kind"] == "loss" and x["state"] == "off"}

    def acc(h, s, w, state, r, setname, kind="mc"):
        xs = cells[h, s, w, state, r, setname, kind]
        return mean(xs) if xs else None

    for h in HOLDS:
        for state in ("writes", "blob"):
            for kind in ("mc", "chat_free"):
                others, own = [], []
                for s in seeds:
                    for w in wseeds:
                        if acc(h, s, w, state, "A", "A", kind) is None:
                            continue
                        others.append(mean(mean(acc(h, s, w, state, r, ws, kind) - acc(h, s, w, "setup", r, ws, kind)
                                                for r in "ABC" if r != ws) for ws in "ABC"))
                        own.append(mean(acc(h, s, w, state, ws, ws, kind) - acc(h, s, w, "setup", ws, ws, kind)
                                        for ws in "ABC"))
                if others:  # one value per write sequence: the gain after all three writes, or after the blob
                    summary["facts"][f"{h}|{state}|{kind}"] = {"others": t95(others), "own": t95(own)}
        before, after = [], []
        for x in recs:
            if x["kind"] == "blob_mc" and x["unit"].split("|")[1] == h:
                (before if x["state"] == "writes" else after).append(x["correct"])
        if before and after:
            summary["blob"][h] = {"sample_before": round(mean(before), 4), "sample_after": round(mean(after), 4)}
        for state in ("setup", "writes", "blob"):
            d0 = [x["correct"] for x in recs if x["kind"] == "d0" and x["unit"].split("|")[1] == h and x["state"] == state]
            if d0:
                summary["d0"][f"{h}|{state}"] = round(mean(d0), 4)
            for text in ("wiki", "stem"):  # loss with the table minus memory off, the same model and sequence
                diff = [x["loss"] - off[x["unit"].split("|", 1)[1], x["reader"], text] for x in recs
                        if x["kind"] == "loss" and x["state"] == state and x["text"] == text and x["unit"].split("|")[1] == h
                        and (x["unit"].split("|", 1)[1], x["reader"], text) in off]
                if diff:
                    summary["loss"][f"{h}|{state}|{text}"] = round(mean(diff), 4)
    if (OUT / "blob.json").exists():
        summary["blob"]["stats"] = json.load(open(OUT / "blob.json"))["stats"]
    writes = [x for x in recs if x["kind"] == "blob_write"]
    if writes:
        summary["blob"]["writes"] = {w: {k: round(mean(x[w][k] for x in writes), 1) for k in ("steps", "rows_moved", "sec")}
                                     for w in "ABC"}
    json.dump(summary, open(OUT / "after_para_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("config", "blob-gen", "replay", "blob", "bench", "analyze"))
    p.add_argument("--seed", type=int, default=cp.SETUP_SEEDS[0])
    p.add_argument("--states", default="off", help="bench: off, setup, writes and/or blob, comma-separated")
    p.add_argument("--checkpoints", nargs="+", default=[], help="config: folders holding the setup checkpoints")
    p.add_argument("--smoke", action="store_true", help="dry run: no lock check, keep every reply, 2 questions per task")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--blob-companies", type=int, default=0, help="blob-gen: companies (dry run)")
    p.add_argument("--blob-sample", type=int, default=0, help="blob-gen: blob facts to quiz (dry run)")
    p.add_argument("--write-steps", type=int, default=0, help="replay: steps per write (dry run)")
    p.add_argument("--quiz-facts", type=int, default=0, help="quiz only this many facts per set (dry run)")
    p.add_argument("--free-tokens", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"config": config, "blob-gen": blob_gen, "replay": replay, "blob": blob, "bench": bench,
     "analyze": analyze}[args.command](args)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
