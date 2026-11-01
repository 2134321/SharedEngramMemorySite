"""Pre-registered analysis of transfer.jsonl (PLAN.md, "Transfer test"). Prints the tests, writes transfer_summary.json."""
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
HELD, TRAINED, QA = (1, 2, 3), (0,), (2,)
DIRECTIONS = (("A", "B"), ("B", "A"))  # (writer, reader)

recs = [json.loads(line) for line in open(OUT / "transfer.jsonl")]
cells = defaultdict(lambda: defaultdict(list))
for r in recs:
    if r["kind"] in ("choice", "free"):
        cells[(r["kind"], r["reader"], r["table"], r["fit"], r["set"])][r["e"]].append((r["t"], r["correct"]))


def per_fact(kind, reader, table, fit, set_, templates=HELD):
    """Accuracy per fact over the given templates."""
    return {e: mean(c for t, c in v if t in templates)
            for e, v in cells[(kind, reader, table, fit, set_)].items() if any(t in templates for t, _ in v)}


def paired(a, b, n_tests=len(DIRECTIONS), n_boot=10_000):
    """Mean of a - b over shared facts, with a Bonferroni-adjusted paired bootstrap interval."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(0)
    boots = sorted(mean(d[rng.randrange(len(d))] for _ in d) for _ in range(n_boot))
    alpha = 0.05 / n_tests
    lo, hi = boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"n": len(d), "diff": round(mean(d), 4), "ci": [round(lo, 4), round(hi, 4)], "ci_level": 1 - alpha}


summary = {"primary": {}, "secondary": {}, "levels": {}}
for w, r in DIRECTIONS:
    test = paired(per_fact("choice", r, w, "fresh", "T"), per_fact("choice", r, "random", "fresh", "T"))
    test["transfer_shown"] = test["ci"][0] > 0 and test["diff"] >= 0.10
    summary["primary"][f"{w}_writes_{r}_reads"] = test
    summary["secondary"][f"{w}_writes_{r}_reads"] = {
        name: paired(per_fact(kind, r, w, "fresh", "T", tpl), per_fact(kind, r, "random", "fresh", "T", tpl))
        for name, kind, tpl in (("free_text_held_out", "free", HELD), ("qa_template", "choice", QA),
                                ("trained_phrasing", "choice", TRAINED))}
summary["decision"] = ("transfer shown: go on to edits" if any(t["transfer_shown"] for t in summary["primary"].values())
                       else "no transfer: replace the shared n-gram table before edits")

conditions = sorted({k[1:4] for k in cells})
for reader, table, fit in conditions:
    row = {}
    for s in ("F", "T"):
        for name, kind, tpl in (("held_out", "choice", HELD), ("qa_template", "choice", QA),
                                ("trained_phrasing", "choice", TRAINED), ("free_text", "free", HELD)):
            vals = per_fact(kind, reader, table, fit, s, tpl)
            if vals:
                row[f"{s}_{name}"] = round(mean(vals.values()), 4)
    summary["levels"][f"reader {reader} | table {table} | {fit}"] = row
summary["wiki_loss"] = {f"reader {r['reader']} | table {r['table']} | {r['fit']}": r["loss"]
                        for r in recs if r["kind"] == "wiki_loss"}

json.dump(summary, open(OUT / "transfer_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
