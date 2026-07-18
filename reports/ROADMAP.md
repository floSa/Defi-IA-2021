# Roadmap & handoff — Defi-IA 2021

Definitive handoff for resuming on the **RTX 4060 Ti (16 GB VRAM, 64 GB RAM)**
desktop, with no Kaggle GPU quota limit. Read this first, then run locally.

- Competition: 28-way job classification from English biographies.
- Metrics: **Macro-F1** (leaderboard) + **macro disparate impact** (fairness, lower→1.0).
- Current best: **RoBERTa-base 0.8035 Macro-F1** (holdout); classical 0.764; 2021 top ≈ 0.81–0.82.

---

## 1. Where we are (results)

| Model | Macro-F1 (holdout) | Disparate impact | Submission file |
|---|---:|---:|---|
| word TF-IDF + LogReg (baseline) | 0.7335 | 4.10 | — |
| word+char (hashed) TF-IDF + LinearSVC | 0.764 | 3.86 | `submissions/classical_wordchar_svm.csv` |
| classical + gender scrubbing (fairness) | 0.760 | **3.49** | — |
| classical + per-class thresholds | 0.7714 | — | `submissions/classical_tuned.csv` |
| **RoBERTa-base fine-tuned** | **0.8035** | 4.15 | `submissions/roberta_holdout.csv` |
| classical + RoBERTa ensemble (blind α) | ? | — | `submissions/ensemble.csv` |

Full log: [`reports/experiments.md`](experiments.md). Milestones/decision log:
[`GUIDE.md`](../GUIDE.md).

---

## 2. What has been TESTED (and the verdict)

**Models**
- `microsoft/deberta-v3-base` — **ABANDONED**: NaN-diverges in fp32 on this stack
  (loss spikes at the peak LR, gradients go NaN). Even LR 1e-5 diverged near
  epoch 1. Do not use unless you find a stabilised recipe (see §4).
- `roberta-base` — **WORKS**, stable, clean loss curve (6.6→1.0, no NaN),
  0.8035, still rising at epoch 2. This is the current workhorse.
- `answerdotai/ModernBERT-base` — never got a clean run (env issues); untested on merit.

**Ablation (classical testbed, zero-GPU, 72k/18k subsample)**
| Technique | Macro-F1 delta |
|---|---:|
| per-class threshold tuning | ~~+0.0082~~ → **+0.0032** after audit (see below) |
| rare-class augmentation (EDA + gender-counterfactual) | −0.0028 (hurts the *linear* model) |

> ⚠️ **Corrected 2026-07-18.** The +0.0082 was tuned and evaluated on the same
> holdout. Nested re-measurement on the full 217k (3 seeds, fit 152k / calib 33k
> / eval 33k) gives a **real gain of +0.0032 ± 0.0009** — the old method
> overstates by ×2.2 — and the technique **worsens disparate impact by +0.26**,
> which was never measured. Full write-up in [`experiments.md`](experiments.md);
> reproduce with `scripts/audit_threshold_tuning.py`.

Verdict: threshold tuning is a **real but small** Macro-F1 lever (+0.3 pt, not
+0.8 pt) that **costs fairness**. Still worth applying to the transformer, but
budget the smaller gain and re-check DI — DI is the top-10 tie-break.
**Augmentation hurts bag-of-words** (paraphrase = noise to TF-IDF) but is
expected to **help the contextual transformer**, especially rare classes — that
is the #1 thing to verify on the 4060 Ti.

**Infra lessons (do not re-learn these):**
- Kaggle assigns a **P100** by default whose sm_60 the current PyTorch build no
  longer supports → force `--accelerator NvidiaTeslaT4`. (Moot locally.)
- Do **not** `pip install -U torch/transformers` in a Kaggle kernel (breaks P100).
- fp16 works for RoBERTa; the "unscale FP16 gradients" error was DeBERTa-specific.
- Weighted CE must cast `logits.float()` before `cross_entropy`.
- Eval per **epoch**, not per 500 steps: step-wise early stopping killed training
  at 0.13 epoch (Macro-F1 is ~0 during warmup and looked "not improving").
- **Kaggle GPU quota (30 h/week) is now exhausted** — that is why we move to the
  4060 Ti.

---

## 3. Techniques TO TRY (prioritised — the real engineering layer)

Measure each as a **delta vs the roberta-base baseline** (ablation table).
Rough impact/cost on a 16 GB card (roberta-base, fp16, batch 32, maxlen 192,
~15–25 min/run):

**Tier 1 — high ROI, cheap**
1. **Per-class threshold tuning** on the transformer's holdout logits
   (`scripts/tune_thresholds.py` — needs the kernel/local run to save
   `valid_logits.npy` + `valid_meta.csv`, already wired). Proven +0.8pt classical.
2. **Data augmentation** (`src/defi_ia/preprocessing/augment.py`): rare-class
   gender-counterfactual + EDA. Expected to lift rare-class F1 on the transformer.
3. **Ensemble tuning**: blend RoBERTa + classical probabilities, sweep α on the
   holdout (not the blind 0.65 in `ensemble.csv`). `scripts/build_ensemble.py`.
4. **Longer training / roberta-base 3–4 epochs full-data** (score still rose at ep2).

**Tier 2 — strong transformer craft**
5. **FGM adversarial training** (perturb `word_embeddings` by normalised grad,
   second forward/backward). Classic +0.3–0.5 Macro-F1 in Kaggle NLP.
6. **Layer-wise LR decay (LLRD)** — lower LR for lower layers.
7. **R-drop / multi-sample dropout** — regularisation.
8. **Pseudo-labeling**: predict the provided (unlabeled) test set, add high-
   confidence rows to training, retrain. Semi-supervised, allowed (test is provided).

**Tier 3 — bigger guns**
9. **roberta-large** (16 GB fits it, fp16, batch 8–16) — likely the single biggest jump.
10. **deberta-v3-large** IF stabilised (lower LR 5e-6, longer warmup, grad clip 0.5, bf16).
11. **Stacking**: k-fold OOF preds from {NB-SVM, char-TFIDF, RoBERTa} → meta-learner.
12. **SWA / EMA** weight averaging.

**Fairness track**
13. **Counterfactual training**: train on gender-swapped duplicates → job becomes
    gender-invariant → lowers disparate impact. Already have the augmenter, and
    `fairness_pareto.py` measures it.
14. **Adversarial debiasing** (gradient-reversal gender head) — advanced.
15. Report the **accuracy/fairness Pareto front**; pick the shipped submission.
16. **Fairness-objective threshold tuning** — new, suggested by the audit. The
    per-class bias search currently maximises Macro-F1 and, as a side effect,
    pushes DI from 3.87 to 4.14. The same coordinate ascent can optimise
    `Macro-F1 − λ·DI` instead: sweeping λ traces a whole accuracy/fairness front
    from one set of saved logits, at zero GPU cost and no retraining. That makes
    the *same* machinery serve both tracks — Macro-F1 at λ=0 for the leaderboard
    submission, a larger λ for the fairness submission. PLAN.md §5.2 already
    called for threshold post-processing as a fairness lever; this is the
    concrete form of it.

---

## 4. What to INVESTIGATE (open questions)

- Does augmentation actually help RoBERTa on rare classes? (Ablation #2 above.)
- Best α for the classical+transformer ensemble on holdout?
- Can DeBERTa-v3(-large) be stabilised (bf16 on the 4060 Ti, which supports it —
  T4 did not)? If so it likely beats roberta-large.
- How much does threshold tuning give on the transformer? Budget **+0.3 pt**, not
  +0.8 — and measure the DI cost, which was +0.26 on the classical model.
- Optimal max_length: p95 = 116 words (~150 tokens); is 192 enough or does 256 help?
- Class-weight scheme: full inverse-freq vs sqrt+clip (current) vs none — which best for Macro-F1?

---

## 4b. Measurement discipline (2026-07-18)

Three published numbers on this project turned out to be measured on the data
that produced them. The rule that follows: **anything fitted, tuned, or selected
must be scored on rows that played no part in fitting, tuning or selecting it.**
That covers more than it first appears — threshold biases, blend weights,
hyper-parameter choices, *and* the choice of which checkpoint to keep.

Tooling that now enforces it, all zero-GPU and all with `--smoke`:

| script | what it measures | discipline built in |
|---|---|---|
| `audit_threshold_tuning.py` | the real threshold gain | fit / calib / eval, 3 seeds, prints the optimism of the old method |
| `tune_thresholds.py` | deployed per-class bias | splits validation in half; refits on all rows only for the shipped artifact |
| `build_ensemble.py` | blend weight α | α swept on one half, judged on the other; refuses to run if the classical model was fit on those rows |
| `sweep_classical.py` | hyper-parameters | fit / select / report; **excludes configs that hit `max_iter`** rather than ranking unconverged fits |
| `fairness_pareto.py` | accuracy/fairness trade-off | no selection at all — every variant reported on the same unseen rows |
| `error_analysis.py` | per-class F1 and DI drivers | reads a model fit on the training split only |

Two habits that paid for themselves within the hour:

- **Reproduce the reference before comparing anything.** The classical baseline
  re-ran at 0.7643 vs 0.7641 logged, which is what licenses trusting every other
  number measured on this rebuilt stack.
- **Smoke test before every long run.** `--smoke` caught an `np.save` filename
  bug that would have crashed the GPU run *after* training, at the moment it
  wrote its logits.

---

## 5. Resume on the 4060 Ti — setup

The desktop has a real GPU, so run the transformer **locally** (no Kaggle):

```bash
# 0. Data: defi-ia-insa-toulouse.zip (44 MB) IS committed to the repo as of
#    2026-07-18, so a clone brings it. Extract it (there is no `unzip` on a bare
#    WSL install, hence python):
python -c "import zipfile; zipfile.ZipFile('defi-ia-insa-toulouse.zip').extractall('data/raw')"
#    Expect 5 files in data/raw/: train.json (95.9 MB), test.json (23.9 MB),
#    train_label.csv, categories_string.csv, template_submissions.csv.

# 1. Environment (uv, per the machine's convention — plain venv is broken there):
uv venv --python 3.12
uv pip install -r requirements.txt && uv pip install -e . --no-deps
uv pip install torch transformers datasets accelerate    # torch ships CUDA on linux
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
# Verified 2026-07-18: torch 2.13.0+cu130 sees the RTX 4060 Ti.
# NOTE: transformers resolves to 5.x, not the 4.x this code was written against.
# The smoke test confirms training, per-epoch eval, checkpointing and early
# stopping still work — run it before trusting a long job.

# Train RoBERTa locally (proven-stable config lives in scripts/train_transformer.py;
# it now defaults to roberta-base, fp16, no premature early stopping):
python scripts/train_transformer.py --model roberta-base --batch-size 32 \
    --max-length 192 --epochs 3            # holdout eval -> Macro-F1 + saves logits

# Then zero-GPU optimisation:
python scripts/tune_thresholds.py          # per-class thresholds
python scripts/build_ensemble.py           # classical + transformer blend
```

**Notes for the local run**
- `scripts/train_transformer.py` uses the `defi_ia` package and reads
  `data/raw/` via `paths.py` — no Kaggle paths. The self-contained Kaggle kernel
  (`kaggle/train_transformer_kernel.py`) is only for the Kaggle path.
- 16 GB VRAM: roberta-base batch 32 fp16 is comfortable; roberta-large use batch
  8–16; deberta-v3 try bf16 (`--bf16`, the 4060 Ti supports it — a real advantage
  over the T4 for stabilising DeBERTa).
- Keep the ablation discipline: one technique per run, log the delta in
  `reports/experiments.md`, commit + `git push origin main`.

**Config invariants that must not regress** (see [`CLAUDE.md`](../CLAUDE.md)):
weighted-loss `logits.float()`, no premature early stopping, eval per epoch,
roberta-base as the stable default.

---

## 6. Repo state

24 commits, pushed to https://github.com/floSa/Defi-IA-2021 (branch `main`).
Nothing sensitive tracked (no kaggle.json, no data, no zip). Tests green
(`.venv/bin/python -m pytest -q`). Everything is reproducible from the scripts.
