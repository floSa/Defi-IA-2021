"""Trace the accuracy/fairness front from per-class thresholds alone (zero GPU).

The existing threshold tuner maximises Macro-F1 and, as a side effect, pushes
disparate impact from 3.87 to 4.14 — rare jobs are the gender-skewed ones and
thresholding exists to predict more of them (see reports/experiments.md).

The same coordinate ascent can optimise ``Macro-F1 − λ·DI`` instead. Sweeping λ
traces a whole front from **one** set of saved scores: no retraining, no GPU,
minutes of CPU. λ=0 reproduces the accuracy-track tuning; larger λ buys fairness
back. That makes one piece of machinery serve both submissions.

⚠️ **The guard is the point of this script.** `macro_disparate_impact` scores a
job predicted for a single gender as ratio 1.0 — perfect parity — so an
optimiser pointed at DI can "win" by emptying a class of one gender
(tests/test_di_edge_cases.py). Any bias vector that creates a single-gender job
is therefore rejected outright rather than scored. Without that, this script
would produce a beautiful DI and a maximally unfair model.

Measurement discipline: the bias is fitted on a calibration half and every number
reported comes from the other half, which the search never saw.

    python scripts/threshold_fairness_front.py --smoke
    python scripts/threshold_fairness_front.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import count_single_gender_jobs, macro_disparate_impact
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean

N_CLASSES = 28
RESULTS = Path("reports/threshold_fairness_front.json")
LAMBDAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def di_and_single_gender(pred: np.ndarray, is_m: np.ndarray) -> tuple[float, int]:
    """Fast reimplementation of macro_disparate_impact, with the skew counter.

    Mirrors the organisers' semantics exactly, including the quirk: a job that
    is present but predicted for one gender only contributes a ratio of 1.0,
    and a job never predicted contributes nothing.
    """
    cm = np.bincount(pred[is_m], minlength=N_CLASSES).astype(float)
    cf = np.bincount(pred[~is_m], minlength=N_CLASSES).astype(float)
    present = (cm + cf) > 0
    both = (cm > 0) & (cf > 0)
    ratios = np.ones(N_CLASSES)
    hi, lo = np.maximum(cm, cf), np.minimum(cm, cf)
    ratios[both] = hi[both] / lo[both]
    return float(ratios[present].mean()), int((present & ~both).sum())


def tune(logp: np.ndarray, y: np.ndarray, is_m: np.ndarray, lam: float,
         rounds: int = 8) -> np.ndarray:
    """Coordinate ascent on Macro-F1 − λ·DI, refusing single-gender solutions."""
    bias = np.zeros(N_CLASSES)
    grid = np.linspace(-4.0, 4.0, 33)

    def objective(b: np.ndarray) -> float:
        pred = (logp + b).argmax(1)
        di, n_single = di_and_single_gender(pred, is_m)
        if n_single > 0:
            # Emptying a class of one gender is how this metric is gamed; make
            # such solutions strictly worse than anything legitimate.
            return -1e9
        return f1_score(y, pred, average="macro") - lam * di

    cur = objective(bias)
    for _ in range(rounds):
        improved = False
        for c in range(N_CLASSES):
            best_v, best_o = bias[c], cur
            for v in grid:
                bias[c] = v
                o = objective(bias)
                if o > best_o:
                    best_o, best_v = o, v
            bias[c] = best_v
            if best_o > cur + 1e-9:
                cur, improved = best_o, True
        if not improved:
            break
    return bias


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()

    lambdas = LAMBDAS
    if args.smoke:
        lambdas = [0.0, 0.02]
        if args.out == str(RESULTS):
            args.out = str(RESULTS.with_suffix(".smoke.json"))
        print("[smoke] 2 lambdas\n")

    names = load_categories()
    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    _, va = stratified_holdout(train, 0.15, args.seed)
    if args.smoke:
        va = va.sample(n=3_000, random_state=args.seed)

    model = joblib.load(args.model)
    logp = log_softmax(model.decision_function(va["text"]), axis=1)
    y = va["Category"].to_numpy()
    gender = va["gender"].to_numpy()
    is_m = gender == "M"

    # Sanity-check the fast DI against the pinned reference implementation.
    ref = macro_disparate_impact([names[c] for c in logp.argmax(1)], gender)
    fast, _ = di_and_single_gender(logp.argmax(1), is_m)
    assert abs(ref - fast) < 1e-9, f"fast DI {fast} disagrees with reference {ref}"
    print(f"holdout {len(y):,} rows; fast DI matches the reference ({ref:.4f})\n")

    idx = np.arange(len(y))
    calib, ev = train_test_split(idx, test_size=0.5, random_state=args.seed, stratify=y)

    rows = []
    for lam in lambdas:
        bias = tune(logp[calib], y[calib], is_m[calib], lam)
        pred = (logp[ev] + bias).argmax(1)
        f1 = f1_score(y[ev], pred, average="macro")
        di, n_single = di_and_single_gender(pred, is_m[ev])
        # Cross-check the winner against the reference implementation too.
        assert n_single == count_single_gender_jobs([names[c] for c in pred], gender[ev])
        rows.append({"lambda": lam, "macro_f1": f1, "disparate_impact": di,
                     "single_gender_jobs": n_single, "bias": bias.tolist()})
        print(f"  λ={lam:<6} Macro-F1 {f1:.4f}   DI {di:.3f}   "
              f"mono-gender jobs {n_single}")

    base_pred = logp[ev].argmax(1)
    base_f1 = f1_score(y[ev], base_pred, average="macro")
    base_di, base_single = di_and_single_gender(base_pred, is_m[ev])

    atomic_save(Path(args.out), lambda q: q.write_text(json.dumps(
        {"seed": args.seed, "n_eval": len(ev), "smoke": args.smoke,
         "argmax": {"macro_f1": base_f1, "disparate_impact": base_di,
                    "single_gender_jobs": base_single},
         "front": rows}, indent=2)))

    print("\n" + "=" * 72)
    print("THRESHOLD-ONLY ACCURACY/FAIRNESS FRONT (judged on unseen rows)")
    print("=" * 72)
    print(f"  argmax (no tuning)   Macro-F1 {base_f1:.4f}   DI {base_di:.3f}   "
          f"mono-gender jobs {base_single}")
    for r in rows:
        print(f"  λ={r['lambda']:<6}            Macro-F1 {r['macro_f1']:.4f} "
              f"({r['macro_f1'] - base_f1:+.4f})   DI {r['disparate_impact']:.3f} "
              f"({r['disparate_impact'] - base_di:+.3f})   "
              f"mono-gender jobs {r['single_gender_jobs']}")

    worst = max(r["single_gender_jobs"] for r in rows)
    if worst > base_single:
        print(f"\n⚠ tuned solutions reach {worst} single-gender job(s) against "
              f"{base_single} for plain argmax.")
        print("  The guard only constrains the CALIBRATION half — it cannot stop a bias")
        print("  from emptying a class on the eval half, where rare classes get few")
        print("  predictions. Their DI is inflated in the fair direction and must not be")
        print("  quoted. This is mostly a small-sample effect: with a full-size holdout")
        print("  every class has both genders. Re-check before trusting any row above.")
    else:
        print(f"\nno solution created a single-gender job beyond argmax's {base_single}: "
              "the DI figures are honest.")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
