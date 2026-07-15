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
| classical + per-class thresholds | ~0.772 | — | `submissions/classical_tuned.csv` |
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
| per-class threshold tuning | **+0.0082** (real lever) |
| rare-class augmentation (EDA + gender-counterfactual) | −0.0028 (hurts the *linear* model) |

Verdict: **threshold tuning is a free, real Macro-F1 lever** — apply it to the
transformer. **Augmentation hurts bag-of-words** (paraphrase = noise to TF-IDF)
but is expected to **help the contextual transformer**, especially rare classes
— that is the #1 thing to verify on the 4060 Ti.

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
    gender-invariant → lowers disparate impact. Already have the augmenter.
14. **Adversarial debiasing** (gradient-reversal gender head) — advanced.
15. Report the **accuracy/fairness Pareto front**; pick the shipped submission.

---

## 4. What to INVESTIGATE (open questions)

- Does augmentation actually help RoBERTa on rare classes? (Ablation #2 above.)
- Best α for the classical+transformer ensemble on holdout?
- Can DeBERTa-v3(-large) be stabilised (bf16 on the 4060 Ti, which supports it —
  T4 did not)? If so it likely beats roberta-large.
- How much does threshold tuning give on the transformer vs the +0.8pt classical?
- Optimal max_length: p95 = 116 words (~150 tokens); is 192 enough or does 256 help?
- Class-weight scheme: full inverse-freq vs sqrt+clip (current) vs none — which best for Macro-F1?

---

## 5. Resume on the 4060 Ti — setup

The desktop has a real GPU, so run the transformer **locally** (no Kaggle):

```bash
# In the project on the desktop (WSL or native Linux), in the venv:
pip install -r requirements.txt -r requirements-dl.txt   # torch+cuda, transformers, datasets
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"

# Data: ensure data/raw/ has the 5 competition files (unzip defi-ia-insa-toulouse.zip).

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
