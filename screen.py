"""Exploratory screen (not pre-registered): a setup recipe that leaves the models' general ability alone
while facts still transfer, with unrestricted writes. Models A and B only. Each variant builds its own
table and readers as run.py edit-setup does, then writes a chain of new fact groups with no row
restriction. The winner is pre-registered and confirmed separately. See LOG.md (Oct 27, evening).

  V0  run 2's recipe: setup also trains the memory to predict general text; the table starts random
      (std 0.02) and the readers' gates half open
  V1  setup holds the memory to changing nothing on general text instead (KL between memory off and on,
      weight 1), and every write step replays general text at the same weight
  V2  V1, with the table starting at zero and the gates nearly closed (sigmoid(-4) = 0.018)
  V3  V2 with KL weight 10

Measured per variant: benchmark accuracy on questions run.py's benchmarks do not use, with memory off
(once per model), with the new table (M0) and after the writes; held-out recall of the setup facts;
every write's facts for both models after every write; general-text loss with memory on and off.
"""
import argparse
import gc
import json
import random
import time

import lm_eval
import torch

import data
import run
from run import CORPUS, DEVICE, EDIT_WRITE_DROPOUT, HELD, MODELS, N_D0_PROBE, OFFLOAD, OUT
from smem import MemLM, SharedMemory, write

VARIANTS = {"V0": {"kl": None, "init_std": 0.02, "gate_bias": 0.0},
            "V1": {"kl": 1.0, "init_std": 0.02, "gate_bias": 0.0},
            "V2": {"kl": 1.0, "init_std": 0.0, "gate_bias": -4.0},
            "V3": {"kl": 10.0, "init_std": 0.0, "gate_bias": -4.0}}
WRITES = 6  # groups of EDIT_FACTS new facts, written in turn by A and B, each on top of the last
WRITE_CFG = {"lr": 0.1, "steps": 80}  # run 2's locked setting for sentence writes
REPLAY_K = 8  # general-text chunks replayed per write step (KL variants)
SAMPLE_SEED = 200  # the written facts; the edit test used seeds 0-5 and 100-101


def main(args):
    out_path = OUT / "screen.jsonl"
    lines = out_path.read_text().splitlines() if out_path.exists() else []
    done = {json.loads(l)["unit"] for l in lines if '"kind": "done"' in l}
    out = open(out_path, "w")  # keep only the records of finished units
    out.writelines(l + "\n" for l in lines if json.loads(l).get("unit") in done)

    def emit(**rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    tm, samples = run.curve_questions(args.smoke)
    n_q = args.quiz_facts or None
    train_wiki, eval_wiki = data.wiki_split(CORPUS)
    eval_wiki = eval_wiki[:100]

    def bench(unit, state, r, lm, memory_on):
        t0 = time.time()
        res = lm_eval.simple_evaluate(model=run.memory_lm(lm, memory_on), tasks=list(samples), num_fewshot=0,
                                      samples=samples, log_samples=True, bootstrap_iters=0, task_manager=tm)
        for task, xs in res["samples"].items():
            for x in xs:
                emit(kind="q", unit=unit, state=state, reader=r, task=task, doc=x["doc_id"], acc=x.get("acc"),
                     acc_norm=x.get("acc_norm"))
        n = sum(map(len, res["samples"].values()))
        print(f"{unit}|{state}|{r}: {n} questions in {time.time() - t0:.0f}s", flush=True)

    for name, v in VARIANTS.items():
        if args.variants and name not in args.variants or name in done:
            continue
        t0 = time.time()
        rng = random.Random(0)  # the same setup facts and probe for every variant
        torch.manual_seed(0)
        memory = SharedMemory(init_std=v["init_std"]).to(DEVICE)
        _, _, d0, _, banned_words, banned_rows = run.fact_world(rng, memory.n_rows)
        probe = d0[:N_D0_PROBE]
        cands = []
        for _, city in probe:
            cs = rng.sample([c for c in data.CITIES if c != city], 9) + [city]
            rng.shuffle(cs)
            cands.append(cs)
        lms = {k: MemLM(m, memory, device=DEVICE, offload=OFFLOAD, gate_bias=v["gate_bias"]) for k, m in MODELS.items()}
        if "off" not in done:  # memory off: the same for every variant
            for r, lm in lms.items():
                bench("off", "off", r, lm, False)
                emit(kind="wiki_loss", unit="off", reader=r, loss=round(lm.mean_loss(eval_wiki, memory=False), 4))
            emit(kind="done", unit="off")
            done.add("off")
        log = []
        run.fit_together(lms, memory, d0, train_wiki, args, rng, torch.Generator().manual_seed(0), True,
                         run.edit_curve(lms, probe, cands, log, time.time()), kl=v["kl"])
        for row in log:
            emit(kind="setup_curve", unit=name, **row)
        for lm in lms.values():
            for p in lm.reader_parameters():
                p.requires_grad_(False)
        torch.save({"table": memory.table.detach().cpu(), "readers": {k: lm.reader_state() for k, lm in lms.items()},
                    "variant": v, "args": vars(args)}, OUT / f"screen_{name}.pt")
        print(f"{name} setup done at {time.time() - t0:.0f}s", flush=True)

        st = {"banned_words": sorted(banned_words), "banned_rows": sorted(banned_rows)}
        groups = run.fresh_facts(st, WRITES, random.Random(SAMPLE_SEED), memory.n_rows, lms)
        facts = [[(s["entity"], s["city"]) for s in g][:n_q] for g in groups]
        fcands = [[[c for _, c in f]] * len(f) for f in facts]

        def read(state):
            """Both models on every group's facts (held-out forced choice); the full battery with chat on
            the group just written (on every group before any write); the setup probe and general-text
            loss at the start and the end."""
            for r, lm in lms.items():
                for j, (f, fc) in enumerate(zip(facts, fcands), 1):
                    for x in run.quiz(lm, f, fc, templates=HELD):
                        emit(kind="mc", unit=name, state=state, reader=r, group=j, **x)
                for j in (range(1, WRITES + 1) if state == 0 else (state,)):
                    for x in run.battery(lm, facts[j - 1], fcands[j - 1]):
                        emit(unit=name, state=state, reader=r, group=j, battery=True, **x)
                if state in (0, WRITES):
                    for x in run.quiz(lm, probe, cands, templates=HELD):
                        emit(kind="d0", unit=name, state=state, reader=r, **x)
                    emit(kind="wiki_loss", unit=name, state=state, reader=r, loss=round(lm.mean_loss(eval_wiki), 4))

        read(0)
        for r, lm in lms.items():
            bench(name, "M0", r, lm, True)
        replay = (train_wiki, REPLAY_K, v["kl"]) if v["kl"] else None
        writers = list(lms)
        for k, g in enumerate(groups, 1):
            w = writers[(k - 1) % len(writers)]
            lm = lms[w]
            texts = run.edit_texts(lm, g[:n_q], "sentences")
            before = memory.table.detach().clone()
            t1 = time.time()
            for _ in write(lm, texts, args.write_steps or WRITE_CFG["steps"], WRITE_CFG["lr"], EDIT_WRITE_DROPOUT,
                           seed=SAMPLE_SEED + k, replay=replay):
                pass
            delta = memory.table.detach() - before
            emit(kind="write", unit=name, state=k, writer=w, rows_moved=int((delta.abs().sum(-1) > 0).sum()),
                 delta_norm=round(delta.norm().item(), 4), sec=round(time.time() - t1))
            read(k)
            print(f"{name} write {k} by {w}: {time.time() - t1:.0f}s with its reads", flush=True)
        for r, lm in lms.items():
            bench(name, "end", r, lm, True)
        torch.save(memory.table.detach().cpu(), OUT / f"screen_{name}_end.pt")
        emit(kind="done", unit=name)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
        for lm in lms.values():  # the next variant loads its own models and table
            lm.model.to("cpu")
        MemLM.resident = lms = memory = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--fact-batch", type=int, default=48)
    p.add_argument("--wiki-batch", type=int, default=8)
    p.add_argument("--table-lr", type=float, default=1e-2)
    p.add_argument("--reader-lr", type=float, default=1e-3)
    p.add_argument("--order-dropout", type=float, default=0.5)
    p.add_argument("--micro", type=int, default=16)
    p.add_argument("--quiz-facts", type=int, default=0, help="quiz only this many facts per group (smoke)")
    p.add_argument("--write-steps", type=int, default=0, help="steps per write (smoke)")
    p.add_argument("--smoke", action="store_true", help="two questions per benchmark task, two MMLU subjects")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    main(args)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
