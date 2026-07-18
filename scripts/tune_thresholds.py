"""Per-class decision-bias tuning to maximise Macro-F1 (zero GPU).

Macro-F1 weights all 28 classes equally, so a plain argmax under-predicts the
rare ones. We learn an additive per-class bias on the log-probabilities by
coordinate ascent, which pushes rare classes over the line.

METHOD — read this before trusting any number it prints
-------------------------------------------------------
The bias has 28 free parameters searched over a 33-point grid for up to 12
rounds. That is more than enough capacity to fit the noise of a single split,
so a score measured on the same rows that chose the bias is optimistic by
construction. The earlier version of this script did exactly that and reported
+0.8 pt; the honest, nested measurement (``scripts/audit_threshold_tuning.py``)
puts the real generalising gain at roughly +0.3 pt, and shows it *worsens*
disparate impact.

So this script splits the validation set in two:

    calib half -> the only rows the bias search may see
    eval  half -> reports the gain, having had no influence on it

and then, separately, refits the bias on the FULL validation set to produce the
bias actually applied to the test logits. Using all available data for the
deployed bias is right; the number you quote must still come from the eval half.

Inputs (written by scripts/train_transformer.py into models/<run>/):
  valid_logits.npy   (n_val, 28)
  valid_meta.csv     (Id, Category, gender)
  test_logits.npy    (n_test, 28)

Example
-------
    python scripts/tune_thresholds.py --run-dir models/roberta_holdout \
        --out submissions/roberta_tuned.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_test
from defi_ia.evaluation.metrics import macro_disparate_impact
from defi_ia.evaluation.submission import make_submission


def tune_bias(logp: np.ndarray, y: np.ndarray, n_classes: int, rounds: int = 12) -> np.ndarray:
    """Coordinate-ascent on a per-class additive log-prob bias, maximising Macro-F1."""
    bias = np.zeros(n_classes)
    grid = np.linspace(-4.0, 4.0, 33)

    def mf1(b: np.ndarray) -> float:
        return f1_score(y, (logp + b).argmax(1), average="macro")

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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="models/roberta_holdout",
                   help="dir holding valid_logits.npy / valid_meta.csv / test_logits.npy")
    p.add_argument("--out", default="submissions/roberta_tuned.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-submit", action="store_true",
                   help="measure the gain only; do not touch the test logits")
    args = p.parse_args()

    run = Path(args.run_dir)
    val_logits = np.load(run / "valid_logits.npy")
    meta = pd.read_csv(run / "valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    gender = meta["gender"].to_numpy()
    n_classes = val_logits.shape[1]
    names = load_categories()

    logp = log_softmax(val_logits, axis=1)

    # --- honest measurement, on rows the tuner never sees -------------------
    idx = np.arange(len(y))
    calib_idx, eval_idx = train_test_split(
        idx, test_size=0.5, random_state=args.seed, stratify=y
    )
    bias_calib = tune_bias(logp[calib_idx], y[calib_idx], n_classes)

    y_eval = y[eval_idx]
    base_pred = logp[eval_idx].argmax(1)
    tuned_pred = (logp[eval_idx] + bias_calib).argmax(1)
    base_f1 = f1_score(y_eval, base_pred, average="macro")
    tuned_f1 = f1_score(y_eval, tuned_pred, average="macro")
    base_di = macro_disparate_impact([names[c] for c in base_pred], gender[eval_idx])
    tuned_di = macro_disparate_impact([names[c] for c in tuned_pred], gender[eval_idx])

    # What the tuner believed it had achieved, on its own calibration rows —
    # printed only so the optimism of the old methodology stays visible.
    self_reported = f1_score(
        y[calib_idx], (logp[calib_idx] + bias_calib).argmax(1), average="macro"
    ) - f1_score(y[calib_idx], logp[calib_idx].argmax(1), average="macro")

    print("=== per-class threshold tuning (honest, nested) ===")
    print(f"  calib {len(calib_idx):,} rows / eval {len(eval_idx):,} rows")
    print(f"  Macro-F1  argmax {base_f1:.4f} -> tuned {tuned_f1:.4f}  ({tuned_f1 - base_f1:+.4f})")
    print(f"  disparate imp.   {base_di:.3f} -> {tuned_di:.3f}  ({tuned_di - base_di:+.3f}, "
          f"lower is fairer)")
    print(f"  [what tuning on the eval rows would have claimed: {self_reported:+.4f}]")
    if tuned_di > base_di:
        print("  ⚠ thresholding buys Macro-F1 at the cost of fairness — the DI tie-break")
        print("    is what separates the top 10, so check this against the fairness track.")

    summary = {
        "n_calib": len(calib_idx), "n_eval": len(eval_idx),
        "eval_argmax_f1": base_f1, "eval_tuned_f1": tuned_f1,
        "delta_f1_honest": tuned_f1 - base_f1,
        "delta_f1_if_measured_in_sample": self_reported,
        "eval_argmax_di": base_di, "eval_tuned_di": tuned_di,
    }
    (run / "threshold_report.json").write_text(json.dumps(summary, indent=2))

    if args.no_submit:
        return

    # --- deployed bias: refit on ALL validation rows ------------------------
    # More data for the bias itself is strictly better; only the *reported*
    # number has to come from rows the search never touched.
    bias_full = tune_bias(logp, y, n_classes)
    np.save(run / "threshold_bias.npy", bias_full)

    test_logits_path = run / "test_logits.npy"
    if not test_logits_path.exists():
        print(f"  no {test_logits_path}; skipping submission")
        return
    test = load_test()
    test_logp = log_softmax(np.load(test_logits_path), axis=1)
    preds = (test_logp + bias_full).argmax(1)
    print(f"  wrote tuned submission -> {make_submission(test.index, preds, args.out)}")
    print(f"  expected gain vs argmax: {tuned_f1 - base_f1:+.4f} (not {self_reported:+.4f})")


if __name__ == "__main__":
    main()
