# Modelling roadmap — Défi IA 2021

> Draft for review. Nothing below is implemented yet beyond the evaluation
> layer; we build after you validate the plan and the compute strategy.

## 1. Objective

Beat a strong classical baseline on **Macro-F1** while keeping **macro
disparate impact** competitive, with clean, reproducible, well-documented code
(the competition's third judged axis: reproducibility, readability, creativity).

Reference points from 2021: top public-leaderboard Macro-F1 landed around
**0.81–0.82**; the organisers' TF-IDF + logistic baseline sits well below that.

## 2. Constraints that shape the plan

- **No local GPU** (16 CPU cores, ~7.4 GB RAM). Classical models run locally;
  transformer fine-tuning needs an external GPU (Kaggle/Colab free tiers, or a
  rented cloud GPU). **This is the one decision that changes the roadmap** —
  see §7.
- **Rules**: no external datasets; pretrained models allowed but only
  fine-tuned on the provided data. Everything here respects that.
- **Imbalance**: 32 % → 0.4 %. Macro-F1 rewards getting the rare classes right,
  so class weighting / thresholding matter more than raw accuracy.

## 3. Evaluation protocol (already built)

- Single stratified train/validation split (85/15) for fast iteration, plus a
  5-fold stratified CV harness for the numbers we trust.
- Every experiment reports **both** Macro-F1 and disparate impact on the same
  validation fold, logged to `reports/`.
- Metrics are pinned by tests to the organisers' reference values.

## 4. Modelling ladder (each step is a checkpoint)

**Step A — Reproduce & harden the baseline.**
TF-IDF + Logistic Regression, but done right: `max_iter` fixed, class-weighted,
light cleaning. Establishes our floor and confirms the harness end-to-end.

**Step B — Strong classical model (CPU, local).**
- TF-IDF over **word (1–2)** *and* **char (2–5)** n-grams — char n-grams are
  robust to the noisy CommonCrawl text.
- Linear SVM / SGD with class weights; calibrated for probabilities.
- Hyper-parameter search on `min_df`, `max_features`, `C`.
- Expected: a large jump over the baseline, fully reproducible on CPU. This is
  our safety net and a competitive submission on its own.

**Step C — Transformer fine-tuning (needs GPU).**
- Fine-tune **DeBERTa-v3-base** (and compare **RoBERTa-base**) with a 28-way
  classification head, `max_length≈256` (covers p95=123 words comfortably).
- Class-weighted loss, mixed precision, early stopping on validation Macro-F1.
- Expected to be the single best model; this is where state-of-the-art shows.

**Step D — Ensemble.**
Blend calibrated probabilities from the classical model (B) and the
transformer(s) (C). Cheap, usually worth 0.5–1.5 Macro-F1 points, and diversity
between char-TF-IDF and a transformer tends to help.

## 5. Fairness track (parallel, measured throughout)

The lever: biographies leak gender via pronouns and names. Options, in
increasing order of effort:

1. **Gender scrubbing (pre-processing)** — neutralise pronouns/honorifics
   (already implemented) and mask first names via NER. Measure the
   accuracy/fairness trade-off.
2. **Post-processing** — adjust per-class decision thresholds to equalise
   gender rates where it costs little Macro-F1.
3. **Report the Pareto front** — pick the submission that best balances the two
   leaderboards rather than optimising one blindly.

We measure disparate impact at every step so fairness is never an afterthought.

## 6. Deliverables

- Reproducible training scripts + a single `predict → submission.csv` command.
- A short experiment log (`reports/`) with the Macro-F1 / fairness of each step.
- Optional: a couple of figures (class balance, Pareto front) for the writeup.

## 7. Two tracks (validated)

We ship **two distinct deliverables**, evaluated on the same harness:

- **Track ACCURACY ("smash the board")** — maximise Macro-F1 with no fairness
  compromise: word+char TF-IDF ensemble → DeBERTa-v3 fine-tune → probability
  blend. This is the leaderboard submission.
- **Track FAIRNESS** — a separate submission that minimises macro disparate
  impact at an acceptable Macro-F1 cost: gender scrubbing + first-name masking
  (NER) + per-class threshold post-processing, reported on the accuracy/fairness
  Pareto front.

Both reuse Steps A–D; the fairness track adds the mitigation layer on top.

## 8. Compute strategy (validated)

No GPU on the working laptop. Available: an **RTX 4060 Ti (8 GB VRAM), 64 GB
RAM, Ryzen 9600X** desktop, plus free remote GPU.

- **Steps A/B (classical)** — run **locally now**, CPU only. GPU-independent.
- **Step C/D (transformer)** — decided at that point, with this order of
  preference:
  1. **Kaggle Kernels API** (`kaggle kernels push/output`): scriptable,
     reproducible, T4 16 GB, ~30 h/week free, competition data already hosted.
     The reproducible backbone.
  2. **The RTX 4060 Ti** for quota-free iteration — driven via Claude Code on
     the desktop, or SSH, or a git sync. DeBERTa-v3-base fits in 8 GB at fp16.
  3. **Colab** only as a manual fallback (no clean automation / reproducibility).

Not viable: driving Colab through an MCP — no such connector exists and Colab
has no headless run API.
