"""Post-hoc checks on run 3 that came out of the review (REVIEW_RUN3.md, "Corrections after review").
Not pre-registered; the locked analyses are unchanged. Writes results_run3/posthoc.json.

1. Context windows, counted per reader. The scale test logged each write's tokens with its writer's
   tokenizer, and analyze_scale.py compared the sum of those mixed counts with every model's window.
   Here every write's texts are counted with each reader's own tokenizer.
2. The edit test's analysis unit. The pre-registered test treats the 18 writes as independent, but the
   three writes of a seed share one evolving table; here the gains are averaged within each seed first.
3. Harm added by the scale test's writes: the harm-curve states against the table before any write
   (not against memory off), paired by question, with the same bootstrap as analyze_bench.py.
4. Memory-off agreement between the screen and run 3 on the harm-curve questions: only correctness is
   saved per question, so this is agreement of correct/incorrect, not of the answer chosen.

usage: python posthoc_run3.py   (part 1 downloads the three tokenizers from Hugging Face)
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results_run3"
METRIC = {"mmlu": "acc", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "winogrande": "acc", "piqa": "acc_norm"}
READERS = ("A", "B", "C")
out = {}


def windows():
    """Part 1: the write after which the text written so far passes each reader's window, in its tokens."""
    from transformers import AutoConfig, AutoTokenizer

    import run
    from smem import MemLM, document_start

    class Writer:  # what MemLM.chat needs: the tokenizer and the document-start token
        def __init__(self, name):
            self.tok = AutoTokenizer.from_pretrained(name)
            self.sep = document_start(self.tok, AutoConfig.from_pretrained(name).model_type)

        chat = MemLM.chat

    models = json.load(open(RUN / "locked_config.json"))["models"]
    lms = {k: Writer(name) for k, name in models.items()}
    writes = json.load(open(RUN / "scale_facts.json"))
    logged = {r["k"]: r for r in map(json.loads, open(RUN / "scale.jsonl")) if r["kind"] == "write"}
    limits = scale_windows()
    window = {k: limits[name] for k, name in models.items()}
    totals, crossed, own, per_write = defaultdict(int), {}, 0, []
    for k, slots in enumerate(writes, 1):
        w = logged[k]["writer"]
        texts = run.edit_texts(lms[w], slots, "sentences")
        counts = {r: sum(len(lms[r].tok(t, add_special_tokens=False)["input_ids"]) for t in texts) for r in READERS}
        assert counts[w] == logged[k]["tokens"], (k, counts[w], logged[k]["tokens"])  # the logged count, recomputed
        own += counts[w]
        per_write.append(counts)
        for r in READERS:
            totals[r] += counts[r]
            if r not in crossed and totals[r] > window[r]:
                crossed[r] = k
    after_20 = {r: sum(c[r] for c in per_write[20:]) for r in READERS}  # text written after the primary's facts
    return {"logged_total_writer_tokens": own, "total_tokens_per_reader": dict(totals), "window": window,
            "write_where_text_passes_window": crossed, "tokens_after_write_20": after_20,
            "first_20_writes_past_window": {r: after_20[r] > window[r] for r in READERS},
            "tokens_per_write": per_write}


def scale_windows():
    """analyze_scale.py's WINDOW table, read without running that script (it rewrites its summary)."""
    import ast
    for node in ast.parse((ROOT / "analyze_scale.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "WINDOW" for t in node.targets):
            return ast.literal_eval(node.value)


def t_interval(xs, n_tests):
    n, m = len(xs), mean(xs)
    se, alpha = stdev(xs) / math.sqrt(n), 0.05 / n_tests
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)], "ci_level": 1 - alpha,
            "p_bonf": float(min(1.0, 2 * stats.t.sf(abs(m / se), n - 1) * n_tests))}


def seeds():
    """Part 2: the pre-registered primary with the six seeds as the unit instead of the 18 writes."""
    recs = [json.loads(line) for line in open(RUN / "edit.jsonl")]
    orders = {r["seed"]: r["order"] for r in recs if r["kind"] == "order"}
    cells = defaultdict(list)
    for r in recs:
        if r["kind"] == "mc":
            cells[r["seed"], r["arm"], r["state"], r["reader"], r["set"]].append((r["t"], r["correct"]))

    def acc(seed, arm, state, reader, set_):
        return mean(c for t, c in cells[seed, None if state == "M0" else arm, state, reader, set_] if t in (1, 2, 3))

    res = {}
    for arm in ("sentences", "paragraphs"):
        per_seed, per_write = [], []
        for seed, order in sorted(orders.items()):
            gains = [mean(acc(seed, arm, f"M{pos}", r, f"F_{w}") - acc(seed, arm, "M0", r, f"F_{w}")
                          for r in READERS if r != w) for pos, w in enumerate(order, 1)]
            per_write += gains
            per_seed.append(mean(gains))
        res[arm] = {"writes_as_unit (pre-registered)": t_interval(per_write, 2), "seeds_as_unit": t_interval(per_seed, 2)}
    return res


def curve_scores(path, key):
    scores = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["kind"] != "q":
            continue
        task = "mmlu" if r["task"].startswith("mmlu") else r["task"].removesuffix("_screen")
        scores[key(r), r["reader"], task][r["task"], r["doc"]] = r[METRIC[task]]
    return scores


def paired(scores, a, b, reader, n_boot=10_000):
    """Mean over tasks of the per-question change from state b to state a, 95% bootstrap interval resampling
    questions within each task (analyze_bench.py's procedure)."""
    diffs = {}
    for task in METRIC:
        x, y = scores[a, reader, task], scores[b, reader, task]
        diffs[task] = [x[q] - y[q] for q in sorted(set(x) & set(y))]
    rng = random.Random(0)
    overall = sorted(mean(mean(d[rng.randrange(len(d))] for _ in d) for d in diffs.values()) for _ in range(n_boot))
    return {"diff": round(mean(mean(d) for d in diffs.values()), 4),
            "ci": [round(overall[int(0.025 * n_boot)], 4), round(overall[int(0.975 * n_boot) - 1], 4)]}


def added_harm():
    """Part 3: each harm-curve state against the table before any write."""
    scores = curve_scores(RUN / "bench.jsonl", lambda r: r["state"] if r.get("set") == "curve" else None)
    return {r: {s: paired(scores, s, "M0", r) for s in ("after 10", "after 30", "after 100", "after all")} for r in READERS}


def agreement():
    """Part 4: memory-off correctness on the same questions, screen (H100) against run 3 (A100)."""
    run3 = curve_scores(RUN / "bench.jsonl", lambda r: r["state"] if r.get("set") == "curve" else None)
    screen = curve_scores(ROOT / "results_screen" / "screen.jsonl", lambda r: r["state"])
    res = {}
    for r in ("A", "B"):
        pairs = [(screen["off", r, t][q], run3["off", r, t][q]) for t in METRIC for q in screen["off", r, t]
                 if q in run3["off", r, t]]
        res[r] = {"questions": len(pairs), "same_correctness": sum(x == y for x, y in pairs),
                  "fraction": round(mean(x == y for x, y in pairs), 4)}
    return res


if __name__ == "__main__":
    out["edit_unit_of_analysis"] = seeds()
    out["harm_added_by_scale_writes"] = added_harm()
    out["memory_off_correctness_agreement_screen_vs_run3"] = agreement()
    out["context_windows_per_reader"] = windows()
    json.dump(out, open(RUN / "posthoc.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
