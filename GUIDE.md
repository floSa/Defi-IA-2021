# Development guide — Defi-IA-2021

Working guide for this project: how it's built, how to run it, the decisions
taken, and the milestone plan. Kept up to date as we go.

## Goal

Revisit the 2021 INSA Toulouse Défi IA (28-way job classification from
biographies) with a **2026 state-of-the-art** pipeline and **beat** what was
achievable in 2021 (a fine-tuned BERT back then). Two deliverables:

- **Accuracy track** — maximise Macro-F1 (leaderboard).
- **Fairness track** — minimise macro disparate impact at low Macro-F1 cost.

Full plan: [`reports/PLAN.md`](reports/PLAN.md). Results log:
[`reports/experiments.md`](reports/experiments.md).

## Environment

- Working laptop: WSL Ubuntu, **no GPU** (16 CPU cores, 7.4 GB RAM). Runs the
  classical pipeline (Steps A/B).
- GPU for transformers: **Kaggle Kernels API** (T4 16 GB) as the reproducible
  backbone; owner's **RTX 4060 Ti** (8 GB) available from tomorrow for
  quota-free iteration.
- Python env is a `virtualenv` at `.venv/` (plain `venv` is broken on this box;
  see the memory note). Always use `.venv/bin/python`.

## How to run

```bash
make data          # extract raw zip → data/raw/
make install       # .venv + core CPU deps
make test          # metrics pinned to organisers' reference (must stay green)
make eda           # dataset summary

# Classical model (Steps A/B), CPU:
.venv/bin/python scripts/train_classical.py                     # strong word+char SVM
.venv/bin/python scripts/train_classical.py --no-char --classifier logistic  # baseline
.venv/bin/python scripts/train_classical.py --scrub-gender      # fairness variant
.venv/bin/python scripts/train_classical.py --full --submit submissions/classical.csv
```

## Cadence & conventions

- **Autonomous milestones**: build one milestone, show results, adapt the code
  to what the numbers say, then move on. Owner reviews asynchronously.
- **Commit regularly** with clear messages; the owner pushes to their remote
  later. Repo name: `Defi-IA-2021`.
- Every modelling change logs a row in `reports/experiments.md` and keeps
  `make test` green.
- Config-driven experiments via `config/config.yaml`; no magic numbers in code.

## Milestones

- [x] M0 — Scaffolding, data, evaluation layer, tests (DI reproduces 3.898).
- [x] M1 — Hardened baseline (word LogReg): Macro-F1 0.7335.
- [x] M2 — Strong classical (word+char LinearSVC): Macro-F1 0.7641.
- [x] M3 — Strong word+char (hashed) LinearSVC full-data submission shipped
      (submissions/classical_wordchar_svm.csv, ~0.764). OOM fixed via HashingVectorizer.
- [~] M4 — Fairness track: scrub-gender done (DI 3.49); name masking (NER) coded.
- [ ] M5 — Transformer fine-tune (DeBERTa-v3 / ModernBERT) on GPU (target ≈ 0.81+).
- [ ] M6 — Ensemble (classical + transformer) + final submissions (both tracks).
- [ ] M7 — Writeup + reproducibility polish.

## Decision log

- **2026-07-13** — Metrics implemented to match the organisers' notebooks
  exactly (test pins DI = 3.898171170378378).
- **2026-07-13** — Classical model = word(1–2) + char_wb(2–5) TF-IDF +
  class-balanced LinearSVC. Char n-grams chosen for robustness to noisy text.
- **2026-07-13** — GPU route decided: **Kaggle Kernels API** primary (see PLAN
  §8). Colab-via-MCP ruled out (no connector, no headless run API).
- **2026-07-18** — **Every tuned number must be measured on data that did not
  tune it.** The per-class threshold gain was learned and reported on the same
  holdout, which overstated it by ×2.2 (+0.0082 claimed → +0.0032 real) and hid
  a +0.26 disparate-impact cost. `tune_thresholds.py` and `build_ensemble.py`
  now split validation into calib/eval, report from eval, and refit on
  everything only for the artifact that ships. Audit:
  `scripts/audit_threshold_tuning.py`, write-up in `reports/experiments.md`.
- **2026-07-18** — Checkpoint selection counts as tuning too. `--full` used a
  sample of the *training* rows as its eval set, so `load_best_model_at_end`
  picked the most overfit checkpoint. It now holds out 3 % that training never
  sees.
- **2026-07-18** — **Smoke test before every long run**, and it earns its keep:
  `--smoke` caught `np.save` appending `.npy` to the atomic-write temp file,
  which would have crashed the GPU run *after* training, at the moment it saved
  its logits. Pinned by `tests/test_atomic_save.py`.

## RESUME HERE (2026-07-18 evening — desktop, RTX 4060 Ti)

The Kaggle route is retired (GPU quota exhausted). Everything now runs locally.

**Environment is rebuilt and verified**: `uv venv` + `uv pip install -r
requirements.txt -e .`; data extracted from the committed
`defi-ia-insa-toulouse.zip` into `data/raw/`; torch 2.13+cu130 sees the 4060 Ti;
15 tests green (the disparate-impact metric still reproduces the organisers'
3.898171170378378 under pandas 3 / sklearn 1.9).

Note `transformers` installs as **5.x**, not the 4.x this code was written for.
The smoke test confirms training, per-epoch eval, checkpointing and early
stopping all still work on 5.x.

**The GPU is shared until ~09:30.** `scripts/gpu_queue.sh` waits for utilisation
to drop below 20 %, then runs the whole sequence unattended:

```bash
bash scripts/gpu_queue.sh          # waits, then: roberta-base repro ->
                                   # roberta-large -> deberta-v3 bf16 ->
                                   # threshold tuning on the winner
```

Every stage is skipped if its `metrics.json` exists and resumes from its newest
checkpoint, so re-running the script after any interruption is safe.

Stage 1 reproduces roberta-base and must land near **0.8035**. If it does not,
something in the stack changed and every later comparison is void — do not skip
this check.

Accuracy safety net already shipped: submissions/classical_wordchar_svm.csv
(~0.764 Macro-F1). Note `submissions/classical_tuned.csv` should be expected
around **0.767**, not the 0.7714 originally logged — see the audit in
`reports/experiments.md`.
