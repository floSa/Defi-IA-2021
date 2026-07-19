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

## STATE (2026-07-19 afternoon) — read this first

**Submitted to Kaggle: public score 0.82166.** 4th against the 2021 leaderboard
(1st 0.84247, 2nd 0.82932, 3rd 0.82733) — **0.0057 off the podium, 0.0208 off
first place**.

⚠️ **The holdout estimate was 0.8329 — optimistic by 0.0112**, twice the distance
to the podium. Do not trust an offline figure from this project without
discounting it. See [`reports/PRECONISATIONS.md`](reports/PRECONISATIONS.md) for
the analysis, the ranked list of what to try next, and what is already measured
as worthless.

Best model: roberta-large, holdout Macro-F1 0.8241, **converged** (its best epoch
is not its last). The only converged run on the project; every roberta-base run
was still improving when it stopped.

### Submissions, all validated against the template (54,300 rows, 28 classes)

| file | Macro-F1 | DI | track |
|---|---:|---:|---|
| **`final_accuracy_track.csv`** | **0.8329** | 5.14 | **accuracy — ship this** |
| `roberta_large_6ep_ensemble.csv` | 0.8288 | 4.43 | better DI, −0.4 pt |
| **`roberta_counterfactual_fairness.csv`** | **0.8018** | **3.41** | **fairness — ship this** |
| `classical_counterfactual_fairness.csv` | 0.7522 | 3.28 | lowest DI, but −5 pt |
| `classical_wordchar_svm.csv` | 0.7643 | 3.89 | classical safety net |

Reference points: 2021 public leaderboard top ≈0.81–0.82; this project started
the session at 0.8035 (Kaggle roberta-base) and 0.7643 (classical).

### What is still open

1. **Nothing has been submitted to Kaggle.** Every figure above is offline
   holdout. Drop a token in `~/.kaggle/kaggle.json` and the submissions are
   ready to go.
2. **Single seed for every transformer number.** The classical work is 3–5
   seeds; the GPU work is not. The 0.8329 also won a 4-way pipeline choice on
   its judging half, so it carries a little selection optimism.
3. **DeBERTa-v3 in bf16 never ran** — the ROADMAP's open question about whether
   bf16 stabilises it is still open.
4. **Counterfactual training on roberta-large** was never tried; on roberta-base
   it cost 0.0009 Macro-F1 for 0.70 DI, so on large it could plausibly give a
   fairness submission above 0.82.

### Traps already found and closed — do not re-learn these

- **The DI metric is gameable**: a job predicted for one gender only scores as
  *perfect parity*. Always report `count_single_gender_jobs` beside a DI figure.
- Anything tuned must be scored on rows that did not tune it. Three published
  numbers on this project were in-sample.
- **Never read a single epoch without the LR schedule.** A flat spot mid-schedule
  is not convergence — roberta-large looked plateaued at epoch 4 and then gained
  1.8 points at epoch 5. Use `scripts/check_convergence.py`.
- **After `kill -9` on a GPU job, confirm VRAM actually dropped before
  relaunching.** A stale 7 GB allocation cost a 2.9× slowdown and looked exactly
  like a model problem.
- `--smoke` before any long run; it caught a bug that would have crashed the GPU
  job *after* training.

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

**The GPU queue is already armed and waiting.** `scripts/gpu_queue.sh` was
launched on the evening of 2026-07-18; it polls the card every 2 minutes and
starts on its own after four consecutive readings below 20 % utilisation (one
reading is not enough — the neighbouring job dips to zero between epochs). It
then runs, unattended and in order:

1. `roberta_base_repro` — **must land near 0.8035.** If it does not, something in
   the stack changed and every later comparison is void. Do not skip this.
2. `roberta_large` — batch 16 + grad-accum 2, the biggest expected jump.
3. `deberta_v3_bf16` — the open question: bf16 (which the T4 lacked) is the
   leading suspect for stabilising the divergence that killed it on Kaggle.
4. Threshold tuning + ensemble on whichever backbone won.

Every stage is skipped if its `metrics.json` exists and resumes from its newest
checkpoint, so re-running the script after any interruption is safe. Progress:
`reports/gpu_queue.log`.

De-risked in advance, so none of these can waste GPU hours:
- checkpoint-resume **verified by SIGKILLing a real run** and restarting it;
- the post-GPU chain (`tune_thresholds`, `build_ensemble`) dry-run end to end on
  correctly-shaped artifacts (`scripts/dryrun_postprocessing.py`);
- `protobuf` installed — without it DeBERTa-v3's tokenizer dies before touching
  the GPU;
- weights for all three backbones pre-downloaded (3.1 GB cached).

### Expectations to hold the results against

| claim in the older docs | what the measurements say |
|---|---|
| threshold tuning = +0.8 pt, free | between −0.003 and +0.003, i.e. **≈ 0**, and it costs +0.26 DI |
| rare classes drag Macro-F1 down | **false** — the 7 rarest average 0.738 F1, the 7 weakest 0.659 |
| name masking helps fairness | **no measurable effect** (−0.015 DI, inside the noise) |
| `classical_tuned.csv` ≈ 0.7714 | expect ≈ **0.764** — that figure was in-sample |

Accuracy safety net already shipped: `submissions/classical_wordchar_svm.csv`
(~0.764 Macro-F1). For the fairness track, **counterfactual training** is the
lever to ship: DI 3.345 for −0.005 Macro-F1, ~2.6× more efficient than
threshold-based fairness.

⚠️ Before optimising disparate impact by any means, read the gaming vector in
`reports/experiments.md` — a job predicted for a single gender scores as perfect
parity, so a DI-directed optimiser can "win" by emptying a class of one gender.
Report `count_single_gender_jobs` alongside every DI figure.
