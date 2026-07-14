"""Per-class decision-bias tuning to maximise Macro-F1 (zero GPU).

Macro-F1 weights all 28 classes equally, so the plain argmax under-predicts
rare classes. We learn an additive per-class bias on the log-probabilities by
coordinate ascent on the HOLDOUT split, then apply the same bias to the test
logits. This is a classic, cheap Macro-F1 lever that argmax leaves on the table.

Inputs (from the Kaggle kernel's holdout run):
  models/kaggle_out/valid_logits.npy   (n_val, 28)
  models/kaggle_out/valid_meta.csv     (Id, Category, gender)
  models/kaggle_out/test_logits.npy    (n_test, 28)

Writes the learned bias and an optimised test submission.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import f1_score

from defi_ia.data.load import load_test
from defi_ia.evaluation.submission import make_submission


def tune_bias(logp: np.ndarray, y: np.ndarray, n_classes: int, rounds: int = 12) -> np.ndarray:
    """Coordinate-ascent on a per-class additive log-prob bias, maximising Macro-F1."""
    bias = np.zeros(n_classes)

    def mf1(b):
        return f1_score(y, (logp + b).argmax(1), average="macro")

    grid = np.linspace(-4.0, 4.0, 33)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kaggle-out", default="models/kaggle_out")
    p.add_argument("--out", default="submissions/roberta_tuned.csv")
    args = p.parse_args()

    val_logits = np.load(f"{args.kaggle_out}/valid_logits.npy")
    meta = pd.read_csv(f"{args.kaggle_out}/valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    n_classes = val_logits.shape[1]

    logp = log_softmax(val_logits, axis=1)
    base = f1_score(y, logp.argmax(1), average="macro")
    bias = tune_bias(logp, y, n_classes)
    tuned = f1_score(y, (logp + bias).argmax(1), average="macro")
    print(f"holdout Macro-F1: argmax {base:.4f} -> tuned {tuned:.4f}  (+{tuned - base:.4f})")

    np.save(f"{args.kaggle_out}/threshold_bias.npy", bias)

    test = load_test()
    test_logp = log_softmax(np.load(f"{args.kaggle_out}/test_logits.npy"), axis=1)
    preds = (test_logp + bias).argmax(1)
    out = make_submission(test.index, preds, args.out)
    print(f"wrote tuned submission -> {out}")


if __name__ == "__main__":
    main()
