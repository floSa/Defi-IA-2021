"""Zero-GPU ablation of the 'clever' techniques on the classical model.

While the transformer waits on GPU, we prove the engineering techniques on the
classical TF-IDF+SVM testbed: each technique's Macro-F1 delta on a stratified
holdout. The deltas transfer qualitatively to the transformer.

Techniques measured:
  - baseline                         (argmax of hashed word+char LinearSVC)
  - + per-class threshold tuning     (macro-F1 lever for rare classes)
  - + rare-class data augmentation   (gender-counterfactual + EDA)
  - + both

Uses a subsample so two full fits stay fast and memory-safe on the 7.4 GB box;
relative deltas are what matter here.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import f1_score

from defi_ia.data.load import load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.augment import augment_rare_classes
from defi_ia.preprocessing.text import basic_clean

SUBSAMPLE = 90_000
SEED = 42


def _prep(df):
    df = df.copy()
    df["text"] = df["description"].map(lambda t: basic_clean(t, lower=True))
    return df


def _holdout_scores(train_df, valid_df):
    cfg = TfidfLinearConfig()  # hashed char -> bounded memory
    model = build_model(cfg)
    model.fit(train_df["text"], train_df["Category"])
    scores = model.decision_function(valid_df["text"])
    return scores


def _tune_bias(logp, y, n_classes, rounds=10):
    bias = np.zeros(n_classes)

    def mf1(b):
        return f1_score(y, (logp + b).argmax(1), average="macro")

    grid = np.linspace(-4, 4, 33)
    cur = mf1(bias)
    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            best_v, best_f = bias[c], mf1(bias)
            for v in grid:
                bias[c] = v
                f = mf1(bias)
                if f > best_f:
                    best_f, best_v = f, v
            bias[c] = best_v
            if best_f > cur + 1e-9:
                cur, improved = best_f, True
        if not improved:
            break
    return bias


def main() -> None:
    full = _prep(load_train(with_labels=True))
    full = full.sample(n=min(SUBSAMPLE, len(full)), random_state=SEED)
    tr, va = stratified_holdout(full, valid_size=0.2, seed=SEED)
    y = va["Category"].to_numpy()
    n_classes = 28

    print(f"ablation on {len(tr):,} train / {len(va):,} holdout\n")

    # 1) baseline
    scores = _holdout_scores(tr, va)
    logp = log_softmax(scores, axis=1)
    base = f1_score(y, scores.argmax(1), average="macro")

    # 2) + threshold tuning
    bias = _tune_bias(logp, y, n_classes)
    thr = f1_score(y, (logp + bias).argmax(1), average="macro")

    # 3) + augmentation (rare classes), fresh fit
    tr_aug = augment_rare_classes(tr, target=2000, seed=SEED)
    tr_aug["text"] = tr_aug["description"].map(lambda t: basic_clean(t, lower=True))
    scores_a = _holdout_scores(tr_aug, va)
    aug = f1_score(y, scores_a.argmax(1), average="macro")

    # 4) + both
    logp_a = log_softmax(scores_a, axis=1)
    bias_a = _tune_bias(logp_a, y, n_classes)
    both = f1_score(y, (logp_a + bias_a).argmax(1), average="macro")

    print("=== ABLATION (classical testbed, Macro-F1) ===")
    print(f"  baseline                 {base:.4f}")
    print(f"  + threshold tuning       {thr:.4f}   ({thr - base:+.4f})")
    print(f"  + augmentation           {aug:.4f}   ({aug - base:+.4f})")
    print(f"  + augmentation+threshold {both:.4f}   ({both - base:+.4f})")


if __name__ == "__main__":
    main()
