"""Blend classical + transformer probabilities into a final submission.

Both models predict over the same 28 Category indices, so we blend their
per-class probabilities and take the argmax.

- Transformer: softmax of the saved test logits (models/kaggle_out/test_logits.npy).
- Classical:   probabilities from a saved joblib model. LinearSVC has no
  predict_proba, so we softmax its decision_function as a surrogate.

The row order of test_logits.npy matches load_test() (both read test.json in the
same order), so we align by position and re-key to Id at the end.

Example:
    python scripts/build_ensemble.py --alpha 0.6 --out submissions/ensemble.csv
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
from scipy.special import softmax

from defi_ia.data.load import load_test
from defi_ia.evaluation.submission import make_submission
from defi_ia.preprocessing.text import basic_clean


def _classical_test_proba(model, texts):
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(texts)
    scores = model.decision_function(texts)
    return softmax(scores, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transformer-logits", default="models/kaggle_out/test_logits.npy")
    p.add_argument("--classical-model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--alpha", type=float, default=0.6, help="weight on the transformer (0..1)")
    p.add_argument("--out", default="submissions/ensemble.csv")
    args = p.parse_args()

    test = load_test()
    test["text"] = test["description"].map(lambda t: basic_clean(t, lower=True))

    probs_t = softmax(np.load(args.transformer_logits), axis=1)
    clf = joblib.load(args.classical_model)
    probs_c = _classical_test_proba(clf, test["text"].values)

    if probs_t.shape != probs_c.shape:
        raise ValueError(f"shape mismatch: {probs_t.shape} vs {probs_c.shape}")

    blend = args.alpha * probs_t + (1 - args.alpha) * probs_c
    preds = blend.argmax(axis=1)
    out = make_submission(test.index, preds, args.out)
    print(f"ensemble (alpha={args.alpha}) -> {out}  [{len(preds)} rows]")


if __name__ == "__main__":
    main()
