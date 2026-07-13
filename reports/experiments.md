# Experiment log

Offline metrics on a stratified 15 % hold-out (seed 42), unless noted.
Macro-F1 higher is better; disparate impact (DI) lower → 1.0 is better.
Ground-truth labels' own DI = 3.898.

| # | Date | Model | Preproc | Macro-F1 | DI | Notes |
|---|------|-------|---------|---------:|---:|-------|
| A | 2026-07-13 | word TF-IDF (1–2) + LogReg, balanced | lower | 0.7335 | 4.10 | hardened baseline, floor |
| B | 2026-07-13 | word(1–2)+char_wb(2–5) TF-IDF + LinearSVC, balanced | lower | **0.7641** | 3.85 | strong classical, ~9 min, 6.5 GB RAM |
| F1 | 2026-07-13 | = B + **gender scrubbing** | lower + scrub | 0.7601 | **3.49** | fairness track: −0.36 DI for −0.4 F1 pt |

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
