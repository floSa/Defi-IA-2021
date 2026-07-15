"""Ship a threshold-tuned classical submission — zero GPU.

Applies the proven per-class threshold lever (+0.8pt in the ablation) to the
classical model: tune the bias on a holdout, then apply it to the full model's
test scores. A tangible optimised submission while Kaggle GPU is quota-blocked.
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import f1_score

from defi_ia.data.load import load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.submission import make_submission
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.text import basic_clean

SEED = 42


def _tune_bias(logp, y, n_classes=28, rounds=10):
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
    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))

    # Tune bias on a holdout using a split-trained model.
    tr, va = stratified_holdout(train, valid_size=0.15, seed=SEED)
    split_model = build_model(TfidfLinearConfig())
    split_model.fit(tr["text"], tr["Category"])
    val_logp = log_softmax(split_model.decision_function(va["text"]), axis=1)
    y = va["Category"].to_numpy()
    base = f1_score(y, val_logp.argmax(1), average="macro")
    bias = _tune_bias(val_logp, y)
    tuned = f1_score(y, (val_logp + bias).argmax(1), average="macro")
    print(f"holdout Macro-F1: {base:.4f} -> {tuned:.4f}  ({tuned - base:+.4f})")

    # Apply the bias to the full-data model's test scores.
    full = joblib.load("models/classical_wordchar_svm.joblib")
    test = load_test()
    test["text"] = test["description"].map(lambda t: basic_clean(t, lower=True))
    test_logp = log_softmax(full.decision_function(test["text"]), axis=1)
    preds = (test_logp + bias).argmax(1)
    out = make_submission(test.index, preds, "submissions/classical_tuned.csv")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
