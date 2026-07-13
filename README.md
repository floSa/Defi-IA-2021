# Défi IA 2021 — Job classification from biographies (modernised)

Assign one of **28 job categories** to an English-language biography, taken
from CommonCrawl. Originally the [INSA Toulouse Défi IA 2021 Kaggle
competition](https://www.kaggle.com/c/defi-ia-insa-toulouse); this repository
revisits it with a modern, reproducible, fairness-aware pipeline.

The task is scored on **two axes**:

| Axis | Metric | Direction | Who it ranks |
|------|--------|-----------|--------------|
| Accuracy | **Macro-F1** (unweighted mean per-class F1) | higher | Kaggle leaderboard |
| Fairness | **Macro disparate impact** (mean per-job `max(M,F)/min(M,F)`) | lower → 1.0 | top-10 tie-break |

Macro-F1 means the 28 classes count equally despite heavy imbalance
(`professor` 32 % → `rapper` 0.4 %). The fairness axis penalises models that
learn to correlate a job with a gender — and the biographies leak gender
through pronouns and names, so this is a real trade-off, not a free lunch.

## Dataset at a glance

| | rows | columns |
|---|------|---------|
| `train.json` | 217,197 | `Id`, `description`, `gender` |
| `test.json`  | 54,300  | `Id`, `description`, `gender` |
| `train_label.csv` | 217,197 | `Id`, `Category` (int 0–27) |

- Descriptions are short biographies: median **62 words**, p95 **123**.
- Gender: **M 117,953 / F 99,244**.
- Ground-truth labels already carry a macro disparate impact of **3.90**.

## Repository layout

```
.
├── config/config.yaml          # single source of experiment knobs
├── data/
│   ├── raw/                    # extracted competition files (git-ignored)
│   ├── interim/ processed/     # derived artifacts
├── notebooks/
│   ├── 00_baseline_tfidf_logreg.ipynb      # organisers' baseline
│   └── 01_fairness_metric_reference.ipynb  # organisers' fairness metric
├── scripts/explore_data.py     # `make eda` dataset summary
├── src/defi_ia/
│   ├── paths.py                # environment-agnostic path config
│   ├── data/                   # tidy loaders (indexed by Id)
│   ├── preprocessing/          # cleaning + gender scrubbing
│   ├── features/               # (planned) vectorizers / embeddings
│   ├── models/                 # (planned) tf-idf, transformer, ensemble
│   ├── fairness/               # (planned) bias mitigation + Pareto analysis
│   └── evaluation/             # metrics + submission builder
├── tests/                      # metrics pinned to organisers' reference values
├── reports/                    # brief + generated figures
├── submissions/                # Kaggle-ready CSVs
├── Makefile                    # orchestration entry points
├── requirements.txt            # core CPU stack
└── requirements-dl.txt         # transformer extras (GPU)
```

## Quickstart

```bash
make data        # extract the raw zip into data/raw/
make install     # create .venv and install the core (CPU) stack
make test        # metrics reproduce the organisers' reference exactly
make eda         # print the dataset summary
```

Transformer fine-tuning needs a GPU (`make install-dl` on a Kaggle/Colab/cloud
machine); the classical pipeline runs comfortably on CPU.

## Status

Scaffolding and the fixed evaluation layer are in place. The modelling roadmap
(baseline → transformer → fairness → ensemble) is described in
[`reports/PLAN.md`](reports/PLAN.md) and implemented after review.
