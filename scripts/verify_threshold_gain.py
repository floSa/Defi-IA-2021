"""Is the transformer's +0.0191 threshold gain real, or one lucky split? (zero GPU)

The accuracy-track submission lives or dies on this number: 0.807 without the
per-class biases, 0.826 with. It was measured on a single calibration/judging
split of the validation set.

That is not good enough here, and this project has the receipts: on the
classical model the same procedure gave +0.0032 with 32.6k calibration rows and
−0.0030 with 16.3k. The gain is sensitive to how much data fits the 28
parameters, and the transformer figure used the smaller regime.

So: re-run the nested measurement over several independent splits of the same
saved logits. No retraining, no GPU — just a different partition each time.

    python scripts/verify_threshold_gain.py --run-dir models/roberta_large_6ep
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

from defi_ia.data.load import load_categories
from defi_ia.evaluation.metrics import macro_disparate_impact
from defi_ia.io_utils import atomic_save

N_CLASSES = 28


def tune_bias(logp: np.ndarray, y: np.ndarray, rounds: int = 12) -> np.ndarray:
    bias = np.zeros(N_CLASSES)
    grid = np.linspace(-4.0, 4.0, 33)

    def mf1(b):
        return f1_score(y, (logp + b).argmax(1), average="macro")

    cur = mf1(bias)
    for _ in range(rounds):
        improved = False
        for c in range(N_CLASSES):
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
    p.add_argument("--run-dir", default="models/roberta_large_6ep")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024, 13, 99])
    p.add_argument("--out", default="reports/threshold_gain_verification.json")
    args = p.parse_args()

    run = Path(args.run_dir)
    meta = pd.read_csv(run / "valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    gender = meta["gender"].to_numpy()
    names = load_categories()
    logp = log_softmax(np.load(run / "valid_logits.npy"), axis=1)

    rows = []
    for seed in args.seeds:
        cal, ev = train_test_split(np.arange(len(y)), test_size=0.5,
                                   random_state=seed, stratify=y)
        bias = tune_bias(logp[cal], y[cal])
        base_pred, tuned_pred = logp[ev].argmax(1), (logp[ev] + bias).argmax(1)
        base = f1_score(y[ev], base_pred, average="macro")
        tuned = f1_score(y[ev], tuned_pred, average="macro")
        di_b = macro_disparate_impact([names[c] for c in base_pred], gender[ev])
        di_t = macro_disparate_impact([names[c] for c in tuned_pred], gender[ev])
        rows.append({"seed": seed, "argmax_f1": base, "tuned_f1": tuned,
                     "delta": tuned - base, "argmax_di": di_b, "tuned_di": di_t,
                     "delta_di": di_t - di_b})
        print(f"  seed {seed:<5} argmax {base:.4f} -> tuned {tuned:.4f} "
              f"({tuned - base:+.4f})   DI {di_b:.3f} -> {di_t:.3f}")

    d = np.array([r["delta"] for r in rows])
    ddi = np.array([r["delta_di"] for r in rows])
    sd = d.std(ddof=1) if len(d) > 1 else 0.0

    print("\n" + "=" * 68)
    print(f"threshold gain over {len(rows)} splits: {d.mean():+.4f} ± {sd:.4f}")
    print(f"  range {d.min():+.4f} … {d.max():+.4f}")
    print(f"  DI cost {ddi.mean():+.3f} ± {ddi.std(ddof=1) if len(ddi) > 1 else 0:.3f}")
    verdict = ("HOLDS — every split agrees on the sign and the size"
               if d.min() > 0 and sd < abs(d.mean()) / 3
               else "FRAGILE — the gain varies more than it should across splits"
               if d.min() > 0 else
               "DOES NOT HOLD — at least one split gives a negative gain")
    print(f"  verdict: {verdict}")
    print(f"\n  the single-split figure that was reported was {rows[0]['delta']:+.4f}")

    atomic_save(Path(args.out), lambda q: q.write_text(json.dumps(
        {"run": str(run), "n_splits": len(rows), "mean_delta": float(d.mean()),
         "sd_delta": float(sd), "min_delta": float(d.min()), "max_delta": float(d.max()),
         "mean_delta_di": float(ddi.mean()), "verdict": verdict, "splits": rows},
        indent=2)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
