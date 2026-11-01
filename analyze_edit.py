"""Pre-registered analysis of edit.jsonl (PLAN.md, "Edit test"). Prints the tests, writes edit_summary.json."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

OUT = Path(os.environ.get("SMEM_OUT", Path(__file__).resolve().parent / "results"))
HELD, TRAINED, CHAT = (1, 2, 3), (0,), (0, 1, 2)
ARMS = ("sentences", "paragraphs")
READERS = ("A", "B", "C")
BEFORE = ("NOMEM", "ORACLE", "M0")  # states before any write, shared by both arms

recs = [json.loads(line) for line in open(OUT / "edit.jsonl")]
orders = {r["seed"]: r["order"] for r in recs if r["kind"] == "order"}
cells = defaultdict(list)
for r in recs:
    if r["kind"] in ("mc", "chat_mc", "chat_free"):
        cells[r["seed"], r["arm"], r["state"], r["reader"], r["set"], r["kind"]].append((r["t"], r["correct"]))


def accuracy(seed, arm, state, reader, set_, kind="mc", templates=HELD):
    """Mean correct over the set's facts x templates."""
    return mean(c for t, c in cells[seed, None if state in BEFORE else arm, state, reader, set_, kind] if t in templates)


def gain(seed, arm, w, r, state, kind="mc", templates=HELD, base="M0"):
    """Reader r's accuracy on writer w's facts at `state`, minus at `base`."""
    return accuracy(seed, arm, state, r, f"F_{w}", kind, templates) - accuracy(seed, arm, base, r, f"F_{w}", kind, templates)


def episodes(arm, kind="mc", templates=HELD, who="others", when="own"):
    """One value per episode (seed, writer W): the gain on W's facts, averaged over the readers.
    who: "others" = every model but W, "writer" = W. when: "own" = right after W's write, "end" = after all writes."""
    vals = []
    for seed, order in sorted(orders.items()):
        for pos, w in enumerate(order, 1):
            state = f"M{pos}" if when == "own" else f"M{len(order)}"
            readers = [w] if who == "writer" else [r for r in READERS if r != w]
            vals.append(mean(gain(seed, arm, w, r, state, kind, templates) for r in readers))
    return vals


def t_test(xs, n_tests=1):
    """Two-sided one-sample t-test of mean 0, with a Bonferroni-adjusted t interval."""
    n, m = len(xs), mean(xs)
    if n < 2:
        return {"n": n, "mean": round(m, 4)}
    se, alpha = stdev(xs) / math.sqrt(n), 0.05 / n_tests
    if se > 0:
        t = m / se
        p, h = 2 * stats.t.sf(abs(t), n - 1), stats.t.ppf(1 - alpha / 2, n - 1) * se
    else:  # every episode gave the same value
        t, p, h = (math.copysign(math.inf, m) if m else 0.0), (0.0 if m else 1.0), 0.0
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "ci_level": 1 - alpha,
            "t": round(t, 3), "p": float(p), "p_bonf": float(min(1.0, p * n_tests))}


summary = {"primary": {}, "secondary": {}, "levels": {}, "writes": {}, "examples": []}
oracle = [mean(gain(seed, None, w, r, "ORACLE") for r in READERS if r != w)
          for seed, order in sorted(orders.items()) for w in order]
for arm in ARMS:
    test = t_test(episodes(arm), n_tests=len(ARMS))
    test["edit_transfer_shown"] = test.get("p_bonf", 1.0) < 0.05 and test["mean"] >= 0.10
    test["oracle_gain"] = round(mean(oracle), 4)
    test["fraction_of_oracle"] = round(test["mean"] / mean(oracle), 4) if mean(oracle) else None
    summary["primary"][arm] = test

    unwritten, retained = [], []
    for seed, order in sorted(orders.items()):
        for pos in range(1, len(order)):
            for w in order[pos:]:  # facts whose writer has not written yet
                unwritten.append(mean(gain(seed, arm, w, r, f"M{pos}") for r in READERS))
            w = order[pos - 1]  # an earlier write, after all the later ones
            retained.append(mean(gain(seed, arm, w, r, f"M{len(order)}", base=f"M{pos}") for r in READERS if r != w))
    summary["secondary"][arm] = {
        "writer_gain": t_test(episodes(arm, who="writer")),
        "chat_choice": t_test(episodes(arm, "chat_mc", CHAT)),
        "chat_free_text": t_test(episodes(arm, "chat_free", CHAT)),
        "trained_phrasing": t_test(episodes(arm, "mc", TRAINED)),
        "after_all_writes": t_test(episodes(arm, when="end")),
        "unwritten_facts": t_test(unwritten),
        "earlier_writes_after_later_ones": t_test(retained),
        # Writer x reader, all 9 cells (the diagonal is the writer itself): accuracy on the writer's facts
        # at M0 and right after its write, and the gain over the 6 seeds.
        "writer_x_reader": {name: {f"{w} writes, {r} reads": {
            "m0": round(mean(accuracy(seed, arm, "M0", r, f"F_{w}", kind, tpl) for seed in orders), 4),
            "after": round(mean(accuracy(seed, arm, f"M{order.index(w) + 1}", r, f"F_{w}", kind, tpl)
                                for seed, order in orders.items()), 4),
            "gain": t_test([gain(seed, arm, w, r, f"M{order.index(w) + 1}", kind, tpl) for seed, order in sorted(orders.items())])}
            for w in READERS for r in READERS}
            for name, kind, tpl in (("held_out", "mc", HELD), ("chat_choice", "chat_mc", CHAT),
                                    ("chat_free_text", "chat_free", CHAT))},
        "d0_held_out_after_all_writes": {r: t_test([accuracy(seed, arm, f"M{len(o)}", r, "D0") - accuracy(seed, arm, "M0", r, "D0")
                                                    for seed, o in sorted(orders.items())]) for r in READERS},
        "wiki_loss_after_all_writes": {r: round(mean(x["loss"] for x in recs if x["kind"] == "wiki_loss" and x["arm"] == arm
                                                     and x["reader"] == r and x["state"] == "M3")
                                                - mean(x["loss"] for x in recs if x["kind"] == "wiki_loss"
                                                       and x["reader"] == r and x["state"] == "M0"), 4) for r in READERS},
    }
    summary["writes"][arm] = {w: {k: round(mean(x[k] for x in recs if x["kind"] == "write" and x["arm"] == arm
                                                and x["writer"] == w), 2) for k in ("rows_moved", "delta_norm")}
                              for w in READERS}

for r in READERS:  # accuracy levels on other models' facts (and, after a write, on the reader's own)
    row = {}
    for name, kind, tpl in (("held_out", "mc", HELD), ("trained", "mc", TRAINED), ("chat_choice", "chat_mc", CHAT),
                            ("chat_free_text", "chat_free", CHAT)):
        others = [(seed, pos, w) for seed, order in sorted(orders.items()) for pos, w in enumerate(order, 1) if w != r]
        for state in BEFORE:
            row[f"{name} | {state}"] = round(mean(accuracy(s, None, state, r, f"F_{w}", kind, tpl) for s, _, w in others), 4)
        for arm in ARMS:
            row[f"{name} | {arm} | others' facts after their write"] = round(
                mean(accuracy(s, arm, f"M{p}", r, f"F_{w}", kind, tpl) for s, p, w in others), 4)
            row[f"{name} | {arm} | own facts after own write"] = round(
                mean(accuracy(s, arm, f"M{o.index(r) + 1}", r, f"F_{r}", kind, tpl) for s, o in sorted(orders.items())), 4)
    row["d0_held_out | M0"] = round(mean(accuracy(s, None, "M0", r, "D0") for s in orders), 4)  # onboarding, for C
    summary["levels"][r] = row

seed0, order0 = min(orders.items())
for w in order0:  # a few chat answers: other models asked about w's first two facts, before and after w's write
    for r in READERS:
        if r == w:
            continue
        for e in [x["e"] for x in recs if x["kind"] == "chat_free" and x["seed"] == seed0 and x["state"] == "M0"
                  and x["reader"] == r and x["set"] == f"F_{w}" and x["t"] == 0][:2]:
            ans = {x["state"]: x["answer"] for x in recs if x["kind"] == "chat_free" and x["seed"] == seed0
                   and x["arm"] in (None, ARMS[0]) and x["reader"] == r and x["e"] == e and x["t"] == 0
                   and x["state"] in ("M0", f"M{order0.index(w) + 1}")}
            summary["examples"].append({"writer": w, "reader": r, "company": e, "before": ans.get("M0"),
                                        f"after ({ARMS[0]})": ans.get(f"M{order0.index(w) + 1}")})

summary["decision"] = {arm: ("edits transfer" if t["edit_transfer_shown"] else "not shown")
                       for arm, t in summary["primary"].items()}
json.dump(summary, open(OUT / "edit_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
