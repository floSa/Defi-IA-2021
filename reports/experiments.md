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
| C-thr | 2026-07-14 | classical wordchar + **per-class threshold tuning** | lower | ~~0.7714~~ | — | ⚠️ in-sample, see audit below; expect ≈0.767 |
| B'' | 2026-07-18 | = B' re-run on the rebuilt desktop env | lower | **0.7643** | 3.89 | **reproduction check**: matches the logged 0.7641 to 0.0002 under pandas 3.0.3 / sklearn 1.9.0 / numpy 2.1.3 — the stack upgrade did not move the baseline |

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

## ⚠️ Finding: the Macro-F1 bottleneck is NOT the rare classes (2026-07-18)

The whole augmentation strategy rests on a premise stated in
`preprocessing/augment.py` and ROADMAP §3 — *"rare classes are the ones dragging
macro-F1 down"*. On the classical model, measured on the 32.6k holdout
(`reports/error_analysis.md`), that premise is **false**.

| | mean F1 | median support |
|---|---:|---:|
| the 7 **rarest** classes | **0.738** | ~130 |
| the 7 **weakest** classes | **0.659** | 225 |

Spearman correlation between class support and class F1 is only **+0.37**.
Rarity explains little of the spread — `class_weight="balanced"` is already doing
its job. `rapper` (117 examples) scores 0.802, better than `teacher` (1372
examples) at 0.587 and `architect` (876) at 0.673.

What actually costs Macro-F1 is **semantic confusion between neighbouring
professions**, dominated by the `professor` mega-class (32 % of the data)
bleeding into every academically-adjacent job and back:

| true → predicted | count |
|---|---:|
| professor → teacher | 194 |
| professor → psychologist | 180 |
| professor → physician | 165 |
| psychologist → professor | 150 |
| physician → professor | 136 |
| architect ↔ software_engineer | 95 / 80 |
| nurse → physician | 92 |
| physician → surgeon | 78 |

**Consequences:**

1. This is the likely explanation for the augmentation result (−0.0028):
   `augment_rare_classes` synthesises examples for classes *below a count
   threshold*, i.e. precisely the classes that already score best. It adds
   paraphrase noise where there was no deficit, and touches none of the
   confusions above.
2. Augmentation should be **retargeted at confusable pairs**, not at rare
   classes — or dropped in favour of work that separates `professor` from
   `teacher`/`psychologist`/`physician`.
3. It also predicts **where the transformer's +4 pts come from**: telling
   "professor" from "teacher" needs context, which is exactly what a contextual
   encoder does and bag-of-words cannot. Expect the transformer's gain to be
   concentrated in these confusable mid-frequency classes — worth verifying
   per-class once a GPU run lands, because if it is *not* concentrated there,
   the ensemble has more to gain than a single blend weight suggests.

## ⚠️ The disparate-impact metric is gameable — read before optimising it

Found while designing the λ-sweep threshold tuner (ROADMAP §3 item 16), and
pinned by `tests/test_di_edge_cases.py`.

`macro_disparate_impact` averages `max(M,F)/min(M,F)` per predicted job. For a
job predicted for a **single gender**, the other column is NaN, and pandas'
`max`/`min` skip NaN — so both return the same count and the ratio is **1.0,
i.e. scored as perfect parity**. Concretely:

| predictions | DI |
|---|---:|
| job a = 10 M / 1 F, job b = 5 M / 5 F | **5.50** |
| job a = 11 M / **0 F**, job b = 5 M / 5 F | **1.00** |

Removing every woman from a job *improves* the fairness score by 4.5 points.
(A job never predicted at all simply drops out of the mean — that case is
harmless.)

This is inherited from the organisers' reference notebook and **must not be
"fixed"**: the metric is pinned to their published 3.898171170378378 and is what
the competition scores. But it has two consequences:

1. **Any procedure that optimises DI directly will find this** — the proposed
   λ-sweep tuner, adversarial debiasing, or a manual post-processing rule. It
   would produce an excellent DI that is maximally unfair.
2. Every fairness number must therefore be reported with
   `count_single_gender_jobs`. **Current status: 0 of 28 predicted jobs are
   single-gender on the holdout**, so none of the results in this file are
   affected — the trap is latent, not sprung.

The docstring in `metrics.py` previously claimed such jobs "drop out of the
mean", which is not what the code does; corrected without touching the
computation.

## Fairness track — Pareto front, full data (2026-07-18)

All five variants fit on the same 184.6k train split and scored on the same
32.6k holdout, so the numbers are directly comparable. No variant was used to
select anything, so none carries selection bias.
(`scripts/fairness_pareto.py`, results in `reports/fairness_pareto.json`.)

| variant | Macro-F1 | ΔF1 | DI | ΔDI | F1 cost per DI point |
|---|---:|---:|---:|---:|---:|
| none | 0.7643 | — | 3.891 | — | — |
| name masking (NER) | 0.7634 | −0.0010 | 3.875 | −0.015 | 0.065 |
| **gender scrubbing** | 0.7609 | −0.0035 | **3.478** | −0.413 | **0.008** |
| scrubbing + masking | 0.7600 | −0.0043 | 3.526 | −0.364 | 0.012 |
| **counterfactual training** | 0.7591 | −0.0052 | **3.345** | **−0.546** | 0.010 |

**⚠️ Noise floor.** These are single-seed measurements. The threshold audit gave
a seed-to-seed sd of **0.062 DI** on this model, so any DI difference below
~0.06 is not distinguishable from noise. Read the table accordingly:

- **Counterfactual training and gender scrubbing are real, large effects**
  (−0.55 and −0.41 DI, both far outside the noise band). Counterfactual reaches
  the lowest DI; scrubbing has the slightly better exchange rate.
- **Name masking does essentially nothing** (−0.015 DI, well inside the noise).
  This closes a *todo* that had been open since the start of the project. The
  hypothesis was that first names leak gender as strongly as pronouns do; on
  this data, once pronouns are present, masking names buys no measurable
  fairness. Not worth the spaCy dependency and the extra 190 s per fit.
- **"scrubbing + masking" vs "scrubbing" is within noise** (ΔDI 0.048 < 0.06).
  The script flags the combination as dominated, but on one seed that verdict
  is not safe — it should be re-run across seeds before being acted on. What
  *is* safe: adding masking on top of scrubbing does not help.

**Shipping recommendation:** counterfactual training for the fairness-track
submission (best DI, and it *adds* signal instead of removing it, so the test
text stays untouched), with gender scrubbing as the cheaper alternative if the
0.5 pt of Macro-F1 matters more than the last 0.13 of DI. The accuracy track
keeps no mitigation.

Both mitigations are worth re-testing on the transformer: it can exploit context
that a bag-of-words cannot, so the accuracy cost may well be smaller there.

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

> ⚠️ **Finding (1) above is retracted — see the audit below.** The +0.8 pt was
> measured on the same rows that chose the thresholds. The real gain is +0.3 pt.

## ⚠️ Audit: the threshold-tuning gain was measured in-sample (2026-07-18)

`tune_thresholds.py:68-69` and `ablation_classical.py:87-88` learned the
per-class bias on the holdout and then reported Macro-F1 **on that same
holdout**. The search has 28 free parameters over a 33-point grid for up to 12
rounds — ample capacity to fit one split's noise. Every "+0.8 pt" figure in this
file and in ROADMAP §2 carried that optimism.

Re-measured honestly with a nested split (`scripts/audit_threshold_tuning.py`):
the classical model is fit on 152k rows, the bias is tuned on a 32.6k
**calibration** set, and the score is reported on a 32.6k **eval** set that
neither the model nor the tuner ever saw. Three seeds.

| | Macro-F1 delta | sd (3 seeds) |
|---|---:|---:|
| tuned on calib, judged on eval — **the real gain** | **+0.0032** | 0.0009 |
| tuned on eval, judged on eval — the old methodology | +0.0071 | 0.0014 |
| what the tuner believes it achieved, on its own calib rows | +0.0074 | — |

The old method **overstates the gain by ×2.2**. The tuner's self-report
(+0.0074) lands on the ROADMAP's published +0.0082, which corroborates that the
published figure is the self-reported one rather than a generalising estimate.

**The gain is real but small**: +0.0032 with a seed-to-seed sd of 0.0009 is
consistently positive, so this is not noise — it is simply a third of what was
claimed.

**It also costs fairness, which was never measured**: the old script computed no
disparate impact for the tuned predictions.

| | DI (lower = fairer) |
|---|---:|
| argmax | 3.87 |
| + tuned thresholds | 4.14 (**+0.262**, sd 0.062) |

The mechanism is now measured rather than assumed (`reports/error_analysis.md`):
**rare jobs are the gender-skewed ones.**

| | mean per-job DI ratio |
|---|---:|
| 10 rarest jobs | **5.36** |
| 10 most frequent jobs | **3.03** |

Spearman correlation between support and DI ratio is **−0.387**. The worst
offenders are `dietitian` (12.0), `nurse` (10.1), `rapper` (9.9), `model` (6.7),
`dj` (6.7) — mostly rare, all strongly stereotyped. Threshold tuning exists
precisely to push more predictions into rare classes, so it feeds the most
skewed jobs and drags the average of the ratios up. The +0.26 DI is not a
side-effect to be engineered away; it is what the technique does.

Since DI is the **tie-break for the top 10**, a +0.3 pt Macro-F1 gain bought
with +0.26 DI is a trade-off to decide deliberately, not a free lever. ROADMAP
§3 item 16 proposes the fix: run the same coordinate ascent against
`Macro-F1 − λ·DI` and sweep λ, which traces the whole front from one set of
saved logits at zero GPU cost.

**Consequences:**
- `submissions/classical_tuned.csv` should be expected to score ≈ **0.767**, not
  the 0.7714 logged in row C-thr.
- Threshold tuning stays worth applying to the transformer (Tier 1), but budget
  +0.3 pt, and re-check the DI cost there.
- `tune_thresholds.py` now splits the validation set in half — tunes on one,
  reports on the other — and refits on everything only for the deployed bias.
