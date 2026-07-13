# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A modernised revisit of the 2021 INSA Toulouse Defi IA Kaggle competition:
assign one of **28 job categories** to an English biography. Scored on two axes
that trade off against each other:

- **Macro-F1** (leaderboard) — unweighted mean per-class F1, so the 28 classes
  count equally despite heavy imbalance (professor 32% -> rapper 0.4%).
- **Macro disparate impact** (fairness tie-break) — mean per-job max(M,F)/min(M,F);
  lower is fairer (1.0 = parity). Biographies leak gender via pronouns/names, so
  this is a real accuracy/fairness trade-off.

Two deliverables are pursued in parallel: an **accuracy track** and a
**fairness track**. See reports/PLAN.md (roadmap), GUIDE.md (milestones +
decision log), reports/experiments.md (results table).

## Environment (important, non-obvious)

- The project lives on **WSL** (`/home/florian/mes_projets/Defi-IA 2021`) but
  Claude Code runs on **Windows**. Run everything through
  `wsl.exe -e bash -lc "..."`. Prefer running script *files* over inline
  Python with nested quotes (the PowerShell/UNC bridge mangles nested quotes).
- `python -m venv` is **broken** here (no ensurepip). The project venv at
  `.venv/` was made with `virtualenv`. Always use `.venv/bin/python`.
- **No local GPU.** ~7.4 GB RAM. Transformers run on Kaggle T4 (see below).
- Repeated OOM can crash the whole WSL VM; symptoms are tool commands returning
  exit 1 with empty output and an invalidated cwd. Recover with
  `wsl.exe --shutdown` then re-issue (first command after may fail once).

## Commands

```bash
# All run inside WSL, from the project root, using the virtualenv.
.venv/bin/python -m pytest -q                 # full test suite
.venv/bin/python -m pytest tests/test_metrics.py::test_disparate_impact_matches_reference_notebook  # single test
.venv/bin/ruff check src scripts kaggle tests # lint

.venv/bin/python scripts/explore_data.py      # dataset summary (EDA)

# Classical model (CPU). Default = word + hashed-char TF-IDF + LinearSVC.
.venv/bin/python scripts/train_classical.py                       # holdout eval (Macro-F1 + DI)
.venv/bin/python scripts/train_classical.py --no-char --classifier logistic   # baseline floor
.venv/bin/python scripts/train_classical.py --scrub-gender        # fairness variant
.venv/bin/python scripts/train_classical.py --full --submit submissions/x.csv # fit all data -> submission
```

The Makefile mirrors these (`make test/lint/eda`) but assumes a working
`python -m venv`; on this box call `.venv/bin/python` directly.

## Transformer on Kaggle (Step C) — how it actually works

There is no local GPU, so the transformer is fine-tuned on a **Kaggle T4 via the
API**, driven entirely from this repo. Credentials live at `~/.kaggle/kaggle.json`.

```bash
.venv/bin/kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4   # push + run
.venv/bin/kaggle kernels status flosal/defi-ia-2021-transformer        # poll
bash scripts/kaggle_monitor.sh flosal/defi-ia-2021-transformer         # poll until done, auto-pull outputs
```

**Hard-won constraints baked into kaggle/train_transformer_kernel.py — do not
regress these:**

- **Force `--accelerator NvidiaTeslaT4`.** Kaggle otherwise assigns a P100
  (sm_60), which the current Kaggle PyTorch build no longer supports
  (`no kernel image is available`).
- **Do not `pip install -U torch/transformers`** in the kernel; a newer torch
  wheel drops P100 support and can break the runtime. Only add lightweight deps
  (sentencepiece, protobuf for the DeBERTa-v3 tokenizer).
- **Train in fp32** (`fp16=False`). fp16 AMP raised
  `Attempting to unscale FP16 gradients` on this stack.
- The weighted loss casts logits to float32 (`logits.float()`) before
  cross_entropy so class weights and logits share a dtype.
- **No aggressive early stopping.** Eval per epoch, not every 500 steps: with
  Macro-F1 near 0 during warmup, step-wise early stopping killed training at
  ~0.13 epoch and the model collapsed. Let it run the full epochs.
- The kernel auto-discovers the data dir by globbing
  `/kaggle/input/**/train.json` (the competition mounts under its slug).

Outputs pulled to `models/kaggle_out/`: `submission.csv`, `test_logits.npy`
(for ensembling), `valid_metrics.json`.

## Architecture

`src/defi_ia/` is an installed package (`pip install -e .`); scripts import it.

- **paths.py** — single source of truth for all file locations, overridable via
  `DEFI_IA_ROOT` (so the same code runs locally or on Kaggle).
- **data/load.py** — loaders indexed by `Id`; joins labels + human-readable
  `job` names. **evaluation/metrics.py** — the two competition metrics; the
  disparate-impact implementation mirrors the organisers' reference notebook
  **exactly** and is pinned by a test to their published value 3.898171170378378.
  Treat this module as fixed ground truth.
- **models/tfidf_linear.py** — the classical model. Key design point: the char
  channel uses a **HashingVectorizer** (fixed `n_features=2**20`), not a
  vocabulary TfidfVectorizer, because building a char (2-5) vocabulary on the
  full 217k set OOM-crashes the 7.4 GB box. Score is unchanged (~0.764) but
  memory is bounded. Use LinearSVC, not SGD (untuned SGD scored 0.52).
- **models/transformer.py** — HF Trainer wrapper (class-weighted loss,
  DeBERTa-v3 / ModernBERT). torch/transformers are imported lazily inside
  functions so the module imports on the CPU-only box (e.g. for lint). The
  standalone `kaggle/train_transformer_kernel.py` mirrors this logic but is
  self-contained (no package import) for reproducibility on Kaggle.
- **preprocessing/text.py** — `scrub_gender` (pronoun/honorific neutralisation)
  and `mask_person_names` (spaCy NER, no-op if spaCy absent) are the two
  fairness levers; kept separate/optional so the accuracy/fairness trade-off can
  be measured rather than baked in.
- **evaluation/submission.py** — builds and validates the Kaggle CSV
  (sorted by Id, integer Category, checked against the template).

Every modelling change should log a row in reports/experiments.md and keep the
metrics tests green.
