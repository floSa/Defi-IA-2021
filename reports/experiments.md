# Experiment log

Offline metrics on a stratified 15 % hold-out (seed 42), unless noted.
Macro-F1 higher is better; disparate impact (DI) lower → 1.0 is better.
Ground-truth labels' own DI = 3.898.

| # | Date | Model | Preproc | Macro-F1 | DI | Notes |
|---|------|-------|---------|---------:|---:|-------|
| A | 2026-07-13 | word TF-IDF (1–2) + LogReg, balanced | lower | 0.7335 | 4.10 | hardened baseline, floor |
| B | 2026-07-13 | word(1–2)+char_wb(2–5) TF-IDF + LinearSVC, balanced | lower | **0.7641** | 3.85 | strong classical, ~9 min, 6.5 GB RAM peak |
| F1 | 2026-07-13 | = B + **gender scrubbing** | lower + scrub | 0.7601 | **3.49** | fairness track: −0.36 DI for −0.4 F1 pt |
| — | 2026-07-13 | word(1–2) + SGD(hinge), balanced | lower | 0.5230 | 9.06 | SGD untuned = poor; abandoned, use LinearSVC |
| B' | 2026-07-13 | word(1–2)+**hashed** char_wb(2–5) + LinearSVC | lower | 0.7639 | 3.86 | HashingVectorizer = same score as B, bounded RAM → full-data submission now possible locally |
| C | 2026-07-13 | **roberta-base** fine-tuned (fp32, LR 1e-5, class-weighted, 2 ep) | strip+lower none | **0.8035** | 4.15 | Kaggle T4; loss 6.6->1.0 clean, still rising at ep2; beats classical +4pts, near 2021 top |

## ⚠️ Local compute constraint (7.4 GB WSL, no GPU)

Full-data fits with **char n-grams OOM-crash the WSL VM** (vocabulary
construction spikes >7.4 GB). Confirmed repeatedly on 2026-07-13. Consequences:

- Offline **holdout** numbers above are valid (holdout fits under the ceiling).
- **Submissions** generated locally are limited to the lighter **word-only**
  config. The strong word+char model (B, 0.7641) and the transformer are
  deferred to the **64 GB desktop** (tomorrow) or **Kaggle** (T4).
- **Fixed (2026-07-13):** the char channel now uses a `HashingVectorizer`
  (fixed `n_features=2**20`), so the full-data strong model fits locally
  without OOM. Score is unchanged (0.7639 vs 0.7641).

## Reading

- Step B lifts Macro-F1 by +3 pts over the baseline and, unlike the baseline,
  does **not** amplify the labels' bias (DI 3.85 < 3.898).
- Reference: 2021 public leaderboard top ≈ 0.81–0.82 Macro-F1. The gap to close
  is the transformer (Step C) + tuning + ensemble.

## Fairness track (Pareto so far)

| variant | Macro-F1 | DI |
|---|---:|---:|
| no mitigation (B) | 0.7641 | 3.85 |
| gender scrubbing | 0.7601 | 3.49 |
| + name masking (NER) | *todo* | *todo* |

## Next candidates

- Classical tuning: `min_df`, `C`, `sublinear_tf`, char range (3–5), TF-IDF
  `max_features`; add a Multinomial-NB feature (NB-SVM trick).
- Fairness track: quantify the `--scrub-gender` accuracy/DI trade-off.
- Transformer (Step C): DeBERTa-v3-base fine-tune on GPU.

## Ablation study (technique deltas)

Measured on a classical testbed (hashed word+char LinearSVC, 72k/18k stratified
subsample, zero GPU) while the transformer waits on GPU. Deltas transfer
qualitatively; augmentation is expected to help the transformer, not the linear
model.

| Technique | Macro-F1 | delta |
|---|---:|---:|
| baseline (argmax) | 0.7420 | — |
| + per-class threshold tuning | 0.7502 | **+0.0082** |
| + rare-class augmentation | 0.7392 | -0.0028 |
| + augmentation + thresholds | 0.7474 | +0.0054 |

**Findings:** (1) threshold tuning is a real, free Macro-F1 lever (+0.8pt) and
should transfer to the transformer. (2) EDA/counterfactual augmentation *hurts*
the bag-of-words model (paraphrase diversity is noise to TF-IDF) — it must be
tested on the contextual transformer, where it is expected to help the rare
classes, not on the linear model.
