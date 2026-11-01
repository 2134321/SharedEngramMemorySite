"""Summary of screen.jsonl (screen.py, exploratory). Prints a table per variant, writes screen_summary.json.

harm: benchmark accuracy with memory on minus off on the same questions, per task and averaged over the
five tasks, with 95% bootstrap intervals (questions resampled within each task); at M0 (the new table)
and at the end (after the writes). transfer: a model's held-out forced choice on a group's facts right
after the group was written minus before any write, for the model that did not write it and for the
writer; the same for the first city named in a chat reply. unwritten: groups not yet written, against
before any write. retained: each group at the end, against right after its write.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
METRIC = {"mmlu": "acc", "arc_challenge_screen": "acc_norm", "hellaswag": "acc_norm", "winogrande_screen": "acc",
          "piqa_screen": "acc_norm"}
READERS = ("A", "B")

recs = [json.loads(line) for line in open(OUT / "screen.jsonl")]
variants = [r["unit"] for r in recs if r["kind"] == "done" and r["unit"] != "off"]
bench = defaultdict(dict)  # (unit, state, reader, task) -> {question: score}
for r in recs:
    if r["kind"] == "q":
        task = "mmlu" if r["task"].startswith("mmlu") else r["task"]
        bench[r["unit"], r["state"], r["reader"], task][r["task"], r["doc"]] = r[METRIC[task]]


def harm(unit, state, readers=READERS, n_boot=10_000):
    """Memory on minus off, in points, per task and averaged over tasks (and over `readers`)."""
    diffs = {}
    for task in METRIC:
        on = [bench[unit, state, r, task] for r in readers]
        off = [bench["off", "off", r, task] for r in readers]
        diffs[task] = np.array([[a[q] - b[q] for q in sorted(set(a) & set(b))] for a, b in zip(on, off)]).mean(0)
    rng = np.random.default_rng(0)
    boot = np.mean([d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) for d in diffs.values()], 0)
    out = {t: round(float(d.mean()) * 100, 2) for t, d in diffs.items()}
    out["mean"] = round(float(np.mean([d.mean() for d in diffs.values()])) * 100, 2)
    out["ci"] = [round(float(np.percentile(boot, q)) * 100, 2) for q in (2.5, 97.5)]
    return out


def accuracy(unit, kind, state, reader, group=None, battery=None):
    xs = [r["correct"] for r in recs if r["kind"] == kind and r.get("unit") == unit and r.get("state") == state
          and r["reader"] == reader and (group is None or r.get("group") == group)
          and (battery is None or bool(r.get("battery")) == battery)]
    return mean(xs) if xs else None


summary = {}
for v in variants:
    writes = {r["state"]: r for r in recs if r["kind"] == "write" and r["unit"] == v}
    last = max(writes)
    s = {"harm_M0": harm(v, "M0"), "harm_end": harm(v, "end"),
         "harm_M0_by_reader": {r: harm(v, "M0", (r,))["mean"] for r in READERS},
         "harm_end_by_reader": {r: harm(v, "end", (r,))["mean"] for r in READERS}}
    s["setup_recall"] = {r: accuracy(v, "d0", 0, r) for r in READERS}
    s["setup_recall_end"] = {r: accuracy(v, "d0", last, r) for r in READERS}
    gains = defaultdict(list)
    for k, w in ((k, x["writer"]) for k, x in writes.items()):
        for r in READERS:
            who = "writer" if r == w else "other"
            gains[who].append(accuracy(v, "mc", k, r, k, False) - accuracy(v, "mc", 0, r, k, False))
            gains[who + "_chat"].append(accuracy(v, "chat_free", k, r, k, True) - accuracy(v, "chat_free", 0, r, k, True))
            for j in range(k + 1, last + 1):
                gains["unwritten"].append(accuracy(v, "mc", k, r, j, False) - accuracy(v, "mc", 0, r, j, False))
            if k < last:
                gains["retained"].append(accuracy(v, "mc", last, r, k, False) - accuracy(v, "mc", k, r, k, False))
    s.update({key: round(mean(xs) * 100, 1) for key, xs in gains.items()})
    s["wiki_loss"] = {r: {"off": next(x["loss"] for x in recs if x["kind"] == "wiki_loss" and x["unit"] == "off"
                                      and x["reader"] == r),
                          "M0": next(x["loss"] for x in recs if x["kind"] == "wiki_loss" and x["unit"] == v
                                     and x["state"] == 0 and x["reader"] == r),
                          "end": next(x["loss"] for x in recs if x["kind"] == "wiki_loss" and x["unit"] == v
                                      and x["state"] == last and x["reader"] == r)} for r in READERS}
    s["rows_moved_per_write"] = round(mean(x["rows_moved"] for x in writes.values()))
    s["sec_per_write"] = round(mean(x["sec"] for x in writes.values()))
    summary[v] = s

json.dump(summary, open(OUT / "screen_summary.json", "w"), indent=1)
cols = [("harm at M0", lambda s: f'{s["harm_M0"]["mean"]:+.1f} ({s["harm_M0"]["ci"][0]:+.1f}, {s["harm_M0"]["ci"][1]:+.1f})'),
        ("harm after writes", lambda s: f'{s["harm_end"]["mean"]:+.1f} ({s["harm_end"]["ci"][0]:+.1f}, {s["harm_end"]["ci"][1]:+.1f})'),
        ("setup recall A/B", lambda s: "/".join(f"{x:.0%}" for x in s["setup_recall"].values())),
        ("other model gain", lambda s: f'{s["other"]:+.1f}'), ("writer gain", lambda s: f'{s["writer"]:+.1f}'),
        ("other, chat", lambda s: f'{s["other_chat"]:+.1f}'), ("unwritten", lambda s: f'{s["unwritten"]:+.1f}'),
        ("retained", lambda s: f'{s.get("retained", float("nan")):+.1f}')]
print("variant  " + " | ".join(name for name, _ in cols))
for v, s in summary.items():
    print(f"{v:8s} " + " | ".join(f(s) for _, f in cols))
print(json.dumps(summary, indent=1))
