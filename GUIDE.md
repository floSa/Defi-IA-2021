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
- [~] M3 — First valid submission (word-only LinearSVC, full-data) shipped.
      Strong char model + tuning deferred to 64 GB desktop / Kaggle (local OOM).
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
