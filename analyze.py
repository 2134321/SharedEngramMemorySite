"""Pre-registered analysis of confirm.jsonl (see PLAN.md), per write arm. Prints the tests, writes summary.json."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
HELD, TRAINED = (1, 2, 3), (0,)
OTHER = {"A": "B", "B": "A"}

recs = [json.loads(line) for line in open(OUT / "confirm.jsonl")]
orders = {r["seed"]: r["order"] for r in recs if r.get("kind") == "order"}
arms = sorted({r["arm"] for r in recs if r.get("arm")})
cells, losses = defaultdict(list), {}
for r in recs:
    key = (r.get("seed"), r.get("arm"), r.get("state"), r.get("reader"), r.get("set"))
    if "t" in r:
        cells[key].append(r)
    elif "loss" in r:
        losses[key] = r["loss"]


def val(seed, arm, state, reader, set_, templates=HELD, field="correct"):
    """States before the first write (NOMEM, M0, ORACLE) are shared by every arm."""
    arm = arm if state in ("M1", "M2") else None
    return mean(r[field] for r in cells[(seed, arm, state, reader, set_)] if r["t"] in templates)


def test(xs, n_tests=1):
    """Two-sided one-sample t-test against zero, 95% CI, and Bonferroni-adjusted p over `n_tests`."""
    n, m = len(xs), mean(xs)
    if n < 2:
        return {"n": n, "mean": round(m, 4)}
    se = stdev(xs) / math.sqrt(n)
    t = m / se if se else float("inf")
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(t), n - 1)
    return {"n": n, "mean": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)], "t": round(t, 2),
            "p": float(f"{p:.3g}"), "p_bonferroni": float(f"{min(1.0, p * n_tests):.3g}"),
            "n_positive": sum(x > 0 for x in xs)}


def episodes_for(arm):
    """One row per episode = (seed, writer); gains are measured right after that writer's write."""
    rows = []
    for seed, order in sorted(orders.items()):
        for pos, w in enumerate(order, 1):
            e = {"seed": seed, "writer": w}
            for role, r in (("writer", w), ("nonwriter", OTHER[w])):
                for tag, tpl in (("held", HELD), ("trained", TRAINED)):
                    for field in ("correct", "fc"):
                        m0 = val(seed, arm, "M0", r, f"F_{w}", tpl, field)
                        key = f"{role}_{tag}_{field}"
                        e[key + "_m0"] = m0
                        e[key + "_nomem"] = val(seed, arm, "NOMEM", r, f"F_{w}", tpl, field)
                        e[key + "_gain"] = val(seed, arm, f"M{pos}", r, f"F_{w}", tpl, field) - m0
                        e[key + "_oracle_gain"] = val(seed, arm, "ORACLE", r, f"F_{w}", tpl, field) - m0
                e[f"{role}_text_loss_change"] = (losses[(seed, arm, f"M{pos}", r, f"text_{w}")]
                                                 - losses[(seed, None, "M0", r, f"text_{w}")])
            rows.append(e)
    return rows


summary, all_episodes = {"n_seeds": len(orders), "arms": arms}, {}
for arm in arms:
    eps = all_episodes[arm] = episodes_for(arm)

    def col(key, writer=None):
        return [e[key] for e in eps if writer in (None, e["writer"])]

    s = summary[arm] = {"primary_nonwriter_heldout_gain": test(col("nonwriter_held_correct_gain"), len(arms))}
    for label, w in (("A_writes_B_reads", "A"), ("B_writes_A_reads", "B")):
        s[label] = {k: test(col(k, w)) for k in (
            "nonwriter_held_correct_gain", "nonwriter_held_correct_oracle_gain", "nonwriter_held_fc_gain",
            "writer_held_correct_gain", "writer_held_correct_oracle_gain",
            "nonwriter_trained_correct_gain", "writer_trained_correct_gain",
            "nonwriter_text_loss_change", "writer_text_loss_change")}
        oracle = mean(col("nonwriter_held_correct_oracle_gain", w))
        s[label]["nonwriter_fraction_of_ceiling"] = round(mean(col("nonwriter_held_correct_gain", w)) / oracle, 4) if oracle else None
        s[label]["levels"] = {k: round(mean(col(k, w)), 4) for k in (
            "nonwriter_held_correct_nomem", "nonwriter_held_correct_m0", "writer_held_correct_m0")}
    # Controls: the second writer's facts after only the first write (not written yet), and the
    # first writer's facts after the second write (interference).
    s["unwritten_facts_at_M1"] = {r: test([val(sd, arm, "M1", r, f"F_{o[1]}") - val(sd, arm, "M0", r, f"F_{o[1]}")
                                           for sd, o in orders.items()]) for r in "AB"}
    s["first_writer_facts_after_second_write"] = {
        r: test([val(sd, arm, "M2", r, f"F_{o[0]}") - val(sd, arm, "M1", r, f"F_{o[0]}") for sd, o in orders.items()])
        for r in "AB"}
    s["D0_change"] = {f"{r}_{st}": test([val(sd, arm, st, r, "D0") - val(None, arm, "M0", r, "D0") for sd in orders])
                      for r in "AB" for st in ("M1", "M2")}
    s["wiki_loss_change"] = {f"{r}_{st}": test([losses[(sd, arm, st, r, "wiki")] - losses[(None, None, "M0", r, "wiki")]
                                                for sd in orders]) for r in "AB" for st in ("M1", "M2")}

summary["D0_levels"] = {f"{r}_{st}": round(val(None, None, st, r, "D0"), 4) for r in "AB" for st in ("NOMEM", "M0")}
grads = [r for r in recs if r.get("kind") == "grad_cos"]
summary["grad_cos_at_M0"] = {k: test([g[k] for g in grads]) for k in ("all", "entity_rows")}

json.dump({"summary": summary, "episodes": all_episodes}, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
