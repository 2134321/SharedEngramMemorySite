"""Pre-registered analysis of bench.jsonl (PLAN.md, "General ability"; run 3 adds the harm curve). Prints the
tests, writes bench_summary.json."""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
METRIC = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "winogrande": "acc", "piqa": "acc_norm"}
READERS = ("A", "B", "C")

scores = defaultdict(dict)  # (question set, reader, state, task) -> {question: correct}
for line in open(OUT / "bench.jsonl"):
    r = json.loads(line)
    if r["kind"] == "q":  # the curve set's ARC, WinoGrande and PIQA tasks carry a _screen suffix
        task = "mmlu" if r["task"].startswith("mmlu") else r["task"].removesuffix("_screen")
        scores[r.get("set", "main"), r["reader"], r["state"], task][r["task"], r["doc"]] = r[METRIC[task]]


def paired(reader, state, n_boot=10_000, qset="main"):
    """Per task and averaged over tasks: accuracy with `state`'s table minus with memory off, on the same
    questions; 95% bootstrap intervals resampling questions within each task."""
    diffs = {}
    for task in METRIC:
        on, off = scores[qset, reader, state, task], scores[qset, reader, "off", task]
        diffs[task] = [on[q] - off[q] for q in sorted(set(on) & set(off))]
    rng = random.Random(0)
    boots = {task: [] for task in METRIC}
    overall = []
    for _ in range(n_boot):
        means = {t: mean(d[rng.randrange(len(d))] for _ in d) for t, d in diffs.items()}
        for t, m in means.items():
            boots[t].append(m)
        overall.append(mean(means.values()))

    def interval(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * n_boot)], 4), round(xs[int(0.975 * n_boot) - 1], 4)]

    out = {t: {"n": len(d), "diff": round(mean(d), 4), "ci": interval(boots[t])} for t, d in diffs.items()}
    out["mean_over_tasks"] = {"diff": round(mean(mean(d) for d in diffs.values()), 4), "ci": interval(overall)}
    return out


summary = {"accuracy": {}, "primary": {}, "secondary": {}}
for reader in READERS:
    summary["accuracy"][reader] = {state: {t: round(mean(scores["main", reader, state, t].values()), 4) for t in METRIC}
                                   for state in ("off", "M0", "scaled")}
    test = paired(reader, "M0")
    m = test["mean_over_tasks"]
    m["harm"] = m["diff"] < -0.02 and m["ci"][1] < 0
    summary["primary"][reader] = test
    summary["secondary"][reader] = {"after_scale_test": paired(reader, "scaled")}
    curve = sorted({s for (q, r, s, _) in scores if q == "curve" and r == reader} - {"off"},
                   key=lambda s: (s != "M0", s == "after all", int(s.split()[-1]) if s.split()[-1].isdigit() else 0))
    if curve:  # run 3: the harm curve over the scale test, on the curve question set
        summary["secondary"][reader]["harm_curve"] = {s: paired(reader, s, qset="curve")["mean_over_tasks"] for s in curve}
summary["decision"] = {r: ("costs general ability" if t["mean_over_tasks"]["harm"] else "no harm shown")
                       for r, t in summary["primary"].items()}
json.dump(summary, open(OUT / "bench_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
