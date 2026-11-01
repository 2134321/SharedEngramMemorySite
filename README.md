# Rosetta Memory

Three frozen language models with different tokenizers and architectures share one writable memory.
One model writes new facts into it, and the other two can then answer questions about those facts.

Results page: https://claude.ai/artifact/V6kXwNJTAoUaqkN2AipU71

| Model | Architecture |
|---|---|
| Qwen3.5-4B (`Qwen/Qwen3.5-4B`) | Gated DeltaNet layers mixed with attention |
| NVIDIA Nemotron 3.5 Bolt E4B (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`) | attention only |
| NVIDIA Nemotron 3 Nano 4B (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) | Mamba-2 layers mixed with attention |

## How it works

- **The memory** is 12 hashed tables, four each for one-, two- and three-word keys, with 65,536 rows
  of 64 numbers. Keys are lowercase whole words taken from the text itself, so every tokenizer lands
  on the same rows.
- **Reading:** at every token, each model looks up the rows for the last one to three words and adds
  them into its residual stream through two small gated readers, after block 2 and a third of the way up.
  The models themselves never change.
- **Setup** trains the table and the readers on facts about made-up companies, while a penalty holds
  the memory to changing nothing on general text (the "hold").
- **A write** trains only the table: 80 steps on the writing model's own text, through its frozen
  layers and readers. The other two models then read what it wrote.

The design, every test and every analysis rule were written down in a plan before the runs they
cover. That plan is not published here yet; the code's docstrings describe each stage.

## Results

Transfer is the gain, in points on multiple-choice questions about the written facts, for the two
models that did not write them. Harm is the change in accuracy on five benchmarks (MMLU,
ARC-Challenge, HellaSwag, WinoGrande, PIQA); a model counts as harmed when its change is below
-2 points and the confidence interval excludes 0.

| Run | Facts written as | Transfer | Harm: Qwen3.5 / Bolt / Nemotron |
|---|---|---|---|
| Run 3 (pre-registered) | fixed sentences | **+62.7** (97.5% CI 56.0 to 69.4) | -4.5 / -1.1 / -5.7 |
| Confirmation, plain hold (pre-registered) | the models' own paragraphs | **+26.7** (98.3% CI 21.0 to 32.3) | -1.06 / -0.14 / -1.99 |
| Confirmation, STEM hold (pre-registered) | the models' own paragraphs | **+20.0** (98.3% CI 15.3 to 24.7) | -0.30 / -0.17 / -0.57 |
| Follow-up: the confirmation's writes replayed from the same setup checkpoints, then benchmarked (pre-registered) | | | plain -1.23 / -0.35 / -2.00; STEM -0.38 / -0.13 / -0.50 |

The intervals in the table treat each of a run's 18 writes as independent, but the writes in one
sequence share a table, and the confirmation's six sequences come from three setup seeds.
`posthoc_run3.py` and `posthoc_para.py` (not pre-registered) redo them with each write sequence as
the unit: run 3 +62.7 (97.5% CI 56.5 to 68.9), plain +26.7 (98.3% CI 21.1 to 32.2), STEM +20.0
(98.3% CI 11.2 to 28.8).

The follow-up rebuilt the tables by replaying the confirmation's writes from its saved setup
checkpoints. Before the writes, the rebuilt tables gave the same right or wrong answer as the
original run on all 1,620 quiz answers per hold; after the writes, on 96.4% (plain) and 98.4% (STEM),
and every write changed the same number of rows.

Run 3's scale test: after 200 further writes, more text than any model's context window, the oldest
facts still transfer at +58.0 (95% CI 53.7 to 62.3).

## What is here

| | |
|---|---|
| `smem.py` | the memory, the readers and the frozen-model wrapper |
| `data.py` | made-up company names, the fact templates and the text corpus split |
| `run.py` | run 3: setup, onboarding of the third model, the write test, the scale test, benchmarks |
| `screen.py`, `analyze_screen.py` | the exploratory screen that chose run 3's hold |
| `screen_para.py` | the exploratory screen with the models' own paragraphs |
| `confirm_para.py` | the pre-registered paragraph confirmation (three holds) |
| `after_para.py` | the pre-registered follow-up: benchmarks after the writes (the script also holds an exploratory test whose results are not included here) |
| `analyze*.py`, `posthoc_*.py` | analyses; `posthoc_*` are not pre-registered |
| `lock.py`, `verify_locks.py` | the pre-registration lock, and a check of every lock |
| `screen_tasks/` | the benchmark question sets used by run 3's harm curve |
| `pod_setup.sh` | the GPU environment, with pinned versions, and the model downloads |
| `results_*/` | each run's inputs (setup texts, facts, the paragraphs written), every per-question record (`*.jsonl.xz`), summaries, and for pre-registered runs `LOCK.json` |

Not included yet: the general text that setup and the hold use (`corpus_2026.txt`, paragraphs from
English Wikipedia articles created in 2026). Republishing it needs a link to each source article, so
it will come with those. The check of the numbers below does not need it.

## Reproduce

### The numbers, from the records (CPU, about 15 minutes)

Needs Python 3 with torch (the CPU build is enough; no model is loaded), transformers, numpy and scipy,
or the environment `pod_setup.sh` builds, activated with `source .venv/bin/activate`.

```
for f in results_*/*.xz; do xz -dk "$f"; done   # unpack the per-question records

SMEM_OUT=results_screen       python analyze_screen.py
SMEM_OUT=results_run3         python analyze_edit.py
SMEM_OUT=results_run3         python analyze_scale.py
SMEM_OUT=results_run3         python analyze_bench.py
SMEM_OUT=results_run3         python posthoc_run3.py
SMEM_OUT=results_screen_para  python screen_para.py analyze
SMEM_OUT=results_confirm_para python confirm_para.py analyze
python posthoc_para.py
SMEM_OUT=results_after_para   python after_para.py analyze
```

Each command rewrites its summary in place (`*_summary.json`, `posthoc.json`). Every one comes out
byte for byte identical to the committed file; `git diff` stays empty.

### The pre-registrations

Each `LOCK.json` records the time a run's code and analysis were frozen, before the run produced any
data, with the sha256 of every frozen file. `python verify_locks.py` checks the files against it. The
plan each lock also covers is not published here yet, so its line reads "not included".

### The runs (GPU)

`bash pod_setup.sh` builds the environment in `.venv` (Python 3.12, torch 2.14.1 for CUDA 12.6,
transformers 5.20.2, lm-eval 0.4.13, flash-linear-attention 0.5.2) and downloads the three models.
Then run `source .venv/bin/activate`, so that `python` below runs in it. Each command reads its
output folder from `SMEM_OUT`; we set `SMEM_WRITE_TOKENS=1024`, and `SMEM_OFFLOAD=1` everywhere but
the first screen. The general text comes from `SMEM_CORPUS` (default `corpus_2026.txt`, not included
yet); with other general text the runs work the same way but will not match our numbers exactly. We
used one A100 40 GB per machine (an H100 for part of the first screen). Each script's docstring
lists its commands.

1. **Screen** (exploratory, about $16): `python screen.py`, which runs the four variants, then
   `python analyze_screen.py`. We ran V0 and V1 on one machine and V2 on another
   (`--variants V2 V3`; V3 was stopped).
2. **Run 3** (pre-registered, about $26): `python run.py edit-setup`, `edit-onboard`,
   `edit-episodes` (our episodes are committed, so this can be skipped), `edit-pilot`, then
   `python lock.py results_run3` before `python run.py edit`, `scale` and `bench`, each followed
   by its `analyze_*.py`.
3. **Paragraph screen** (exploratory, about $16): `python screen_para.py gen-setup` and `facts`, then
   `setup --control general` (the `para` setup) and `setup --control stem` (`para_stem`), both with
   `--onboard-steps 500`; then `write-gen`, `chains` and `bench` with `--setup para`, `para_stem` and
   `template` (run 3's setup checkpoints, copied in as `template_setup.pt` and `template_onboard.pt`);
   then `analyze`.
4. **Confirmation** (pre-registered, about $43, three machines, one per setup seed): `python
   confirm_para.py config` and `python lock.py results_confirm_para` first, then on machine 1 `gen-setup`
   and `facts`; on machine `s`, for each arm in `plain`, `stem`, `double`: `setup`, `write-gen`,
   `chains` and `bench` with `--arm <arm> --seed <s>`, plus `bench --arm off --seed <s>`; then `analyze`.
5. **Follow-up** (pre-registered, about $24, three machines): it replays the confirmation's writes
   from the confirmation's saved setup checkpoints, which are not in this repository (2.6 GB); rerun
   step 4's `setup` to make them. Then `python after_para.py config --checkpoints <dir>`, `python
   lock.py results_after_para`, and on machine `s`: `bench --seed <s> --states off`, `replay --seed
   <s>` and `bench --seed <s> --states setup,writes`; then `analyze`. (The published cost includes
   the exploratory test.)

Costs are at $1.99 per A100 hour.

## License

The code is under the Apache License 2.0 (`LICENSE`); `smem.py` adapts parts of XMemTransfer, also
Apache 2.0. See `NOTICE`.
