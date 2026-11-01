"""Post-hoc check on the paragraph confirmation (REVIEW_PARA.md). Not pre-registered; the locked analysis
(confirm_para.py analyze) is unchanged. Writes results_confirm_para/posthoc.json.

The analysis unit. The pre-registered transfer test treats each hold's 18 writes (setup seed x write seed x
writer) as independent, but the three writes of a sequence (one setup seed and write seed) land on one table,
each on top of the last, as run 3's review found for its edit test (REVIEW_RUN3.md, correction 4). Here each
sequence's three writes are averaged first: 6 values per hold, with a t interval at the pre-registered level
(Bonferroni over the three holds). The per-write values are computed as in confirm_para.py analyze, and their
mean and interval reproduce its summary. The holds' paired comparisons (pre-registered per write, 95% t) are
repeated with the 6 sequences as the unit, and with the 3 setup seeds (each the mean of its two sequences),
the only fully independent training runs.

usage: python posthoc_para.py   (standard library and scipy only)
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results_confirm_para"
HELD = (1, 2, 3)  # run.HELD: the held-out probe templates
HOLDS = ("plain", "stem", "double")
LEVEL = 1 - 0.05 / len(HOLDS)


def interval(xs, level=LEVEL):
    m = mean(xs)
    h = float(stats.t.ppf(1 - (1 - level) / 2, len(xs) - 1)) * stdev(xs) / math.sqrt(len(xs))
    return {"n": len(xs), "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "ci_level": round(level, 4)}


cells, orders = defaultdict(list), {}
for line in open(RUN / "confirm_para.jsonl"):
    if '"kind": "mc"' in line:
        r = json.loads(line)
        if r["t"] in HELD:
            cells[r["unit"], r["pos"], r["reader"], r["set"]].append(r["correct"])
    elif '"kind": "order"' in line:
        r = json.loads(line)
        orders[r["unit"]] = r["order"]


def acc(unit, pos, reader, w):
    return mean(cells[unit, pos, reader, w])


out = {"level": round(LEVEL, 4)}
for arm in HOLDS:
    writes, sequences = [], {}
    for unit, order in sorted(orders.items()):
        if unit.split("|")[0] != arm:
            continue
        gains = [mean(acc(unit, pos, r, w) - acc(unit, 0, r, w) for r in order if r != w)
                 for pos, w in enumerate(order, 1)]
        writes += gains
        sequences[unit] = round(mean(gains), 4)
    out[arm] = {"writes": interval(writes), "sequences": interval(list(sequences.values())), "by_sequence": sequences}
    w, s = out[arm]["writes"], out[arm]["sequences"]
    print(f"{arm:7s} 18 writes: {w['mean'] * 100:+.1f} ({w['ci'][0] * 100:.1f} to {w['ci'][1] * 100:.1f})   "
          f"6 sequences: {s['mean'] * 100:+.1f} ({s['ci'][0] * 100:.1f} to {s['ci'][1] * 100:.1f})")

out["paired"] = {}
for a, b in (("stem", "double"), ("double", "plain"), ("stem", "plain")):
    seqs = sorted(u.split("|", 1)[1] for u in out[a]["by_sequence"])  # "seed|write seed"
    diff = {q: out[a]["by_sequence"][f"{a}|{q}"] - out[b]["by_sequence"][f"{b}|{q}"] for q in seqs}
    seeds = sorted({q.split("|")[0] for q in seqs})
    by_seed = [mean(d for q, d in diff.items() if q.split("|")[0] == s) for s in seeds]
    out["paired"][f"{a}-{b}"] = {"sequences": interval(list(diff.values()), 0.95), "seeds": interval(by_seed, 0.95)}
    p = out["paired"][f"{a}-{b}"]
    print(f"{a} - {b}: 6 sequences {p['sequences']['mean'] * 100:+.1f} ({p['sequences']['ci'][0] * 100:.1f} to "
          f"{p['sequences']['ci'][1] * 100:.1f}), 3 seeds ({p['seeds']['ci'][0] * 100:.1f} to {p['seeds']['ci'][1] * 100:.1f})")
json.dump(out, open(RUN / "posthoc.json", "w"), indent=1)
