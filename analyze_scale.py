"""Pre-registered analysis of scale.jsonl (PLAN.md, "Phase 2"). Prints the tests, writes scale_summary.json."""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
READERS = ("A", "B", "C")
OLDEST = 20  # primary: the facts of the first 20 writes
FOLLOW = (1, 3, 10, 30, 100)
WINDOW = {"Qwen/Qwen3-0.6B": 40960, "Qwen/Qwen3.5-4B": 262144, "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16": 65536,
          "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": 262144}

recs = [json.loads(line) for line in open(OUT / "scale.jsonl")]
models = json.load(open(OUT / "locked_config.json"))["models"]
writes = sorted((r for r in recs if r["kind"] == "write"), key=lambda r: r["k"])
last = writes[-1]["k"]
writer = {r["k"]: r["writer"] for r in writes}
cells, write_of = defaultdict(list), {}
for r in recs:
    if r["kind"] in ("mc", "chat_free"):
        cells[r["kind"], r["k"], r["reader"], r["e"]].append(r["correct"])
        write_of[r["e"]] = r["j"]


def acc(kind, k, reader, e):
    """One fact's accuracy for a reader with the table after write k (0 = before any)."""
    return mean(cells[kind, k, reader, e])


def boot(values, n_boot=10_000):
    """Mean with a 95% bootstrap interval over the given units."""
    rng = random.Random(0)
    means = sorted(mean(values[rng.randrange(len(values))] for _ in values) for _ in range(n_boot))
    return {"n": len(values), "mean": round(mean(values), 4),
            "ci": [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot) - 1], 4)]}


def others_gain(facts, kind="mc", k=None):
    """Per fact: the two non-writers' accuracy after write k (default: the last) minus at M0, averaged."""
    k = last if k is None else k
    return [mean(acc(kind, k, r, e) - acc(kind, 0, r, e) for r in READERS if r != writer[write_of[e]]) for e in facts]


everyone = sorted(write_of, key=lambda e: write_of[e])
oldest = [e for e in everyone if write_of[e] <= OLDEST]
summary = {"primary": boot(others_gain(oldest)), "secondary": {}}
summary["primary"]["shown"] = summary["primary"]["mean"] >= 0.10 and summary["primary"]["ci"][0] > 0
summary["decision"] = ("the shared memory keeps facts past every context window" if summary["primary"]["shown"]
                       else "not shown")
sec = summary["secondary"]
sec["all_facts"] = {kind: boot(others_gain(everyone, kind)) for kind in ("mc", "chat_free")}
sec["oldest_chat_free"] = boot(others_gain(oldest, "chat_free"))
sec["writer_x_reader_at_end"] = {
    kind: {f"{w} writes, {r} reads": {
        "m0": round(mean(acc(kind, 0, r, e) for e in everyone if writer[write_of[e]] == w), 4),
        "end": round(mean(acc(kind, last, r, e) for e in everyone if writer[write_of[e]] == w), 4)}
        for w in READERS for r in READERS} for kind in ("mc", "chat_free")}
kept = sorted({k for (_, k, _, _) in cells} - {0, last})
sec["followed_writes"] = {  # non-writers' held-out accuracy on write j's facts after write k, as writes pile up
    f"write {j}": {f"after write {k}": round(mean(
        mean(acc("mc", k, r, e) for r in READERS if r != writer[j]) for e in everyone if write_of[e] == j), 4)
        for k in [*kept, last] if k >= j and cells.get(("mc", k, READERS[0], next(e for e in everyone if write_of[e] == j)))}
    for j in FOLLOW if j <= last}
context = defaultdict(dict)
for r in recs:
    if r["kind"] == "context":
        context[r["reader"], r["n"]][r["e"]] = r["correct"]
sec["in_context_vs_memory"] = {}
for reader in READERS:
    sample = sorted({e for (rd, n), d in context.items() if rd == reader for e in d})
    row = {f"{n} facts in the prompt (memory off)": round(mean(context[reader, n].values()), 4)
           for (rd, n) in sorted(context) if rd == reader}
    for who, keep in (("others' facts", lambda e: writer[write_of[e]] != reader),
                      ("own facts", lambda e: writer[write_of[e]] == reader)):
        es = [e for e in sample if keep(e)]
        if es:
            row[f"memory after all writes, {who} (n={len(es)})"] = round(mean(acc("chat_free", last, reader, e) for e in es), 4)
            row[f"memory before any write, {who}"] = round(mean(acc("chat_free", 0, reader, e) for e in es), 4)
    sec["in_context_vs_memory"][reader] = row
d0 = defaultdict(list)
for r in recs:
    if r["kind"] == "d0":
        d0[r["reader"], r["k"]].append(r["correct"])
sec["d0_held_out"] = {r: {"m0": round(mean(d0[r, 0]), 4), "end": round(mean(d0[r, last]), 4)} for r in READERS}
wiki = {(r["reader"], r["k"]): r["loss"] for r in recs if r["kind"] == "wiki_loss"}
sec["wiki_loss"] = {r: {"m0": wiki[r, 0], "end": wiki[r, last]} for r in READERS}
tokens, crossed = 0, {}
for r in writes:
    tokens += r["tokens"]
    for k, name in models.items():
        if name in WINDOW and k not in crossed and tokens > WINDOW[name]:
            crossed[k] = r["k"]
sec["written"] = {"writes": last, "facts": len(everyone), "tokens": tokens,
                  "window": {k: WINDOW.get(name) for k, name in models.items()},
                  "write_where_text_passes_window": crossed,
                  "rows_moved_per_write": round(mean(r["rows_moved"] for r in writes), 1)}
json.dump(summary, open(OUT / "scale_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
