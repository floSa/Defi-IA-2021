"""Compose both post-processing levers into the accuracy-track submission (zero GPU).

Measured separately on roberta-large, each lever is real:

    per-class thresholds   +0.0191 Macro-F1
    classical ensemble     +0.0139

They act on different things — the ensemble changes *which* scores you have, the
thresholds change *where you cut* them — so composing them should beat either
alone. This script does that and, crucially, measures whether it actually does.

Protocol, same as everywhere else in this project: the validation set is split in
half. **Both** the blend weight α and the 28 per-class biases are fitted on the
calibration half only; every number reported comes from the judging half, which
neither saw. The artifacts that ship are then refitted on all validation rows —
more data for the parameters is right, as long as the quoted figure comes from
data that did not choose them.

Fitting α and 28 biases on the same 16k rows is a lot of capacity, so the
composed gain is the one most at risk of not surviving the split. That is
precisely why it is measured rather than assumed.

    python scripts/final_submission.py --run-dir models/roberta_large_6ep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.evaluation.metrics import count_single_gender_jobs, macro_disparate_impact
from defi_ia.evaluation.submission import make_submission
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean

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


def sweep_alpha(p_t, p_c, y, grid) -> float:
    best_a, best_f = 0.0, -1.0
    for a in grid:
        f = f1_score(y, (a * p_t + (1 - a) * p_c).argmax(1), average="macro")
        if f > best_f:
            best_a, best_f = float(a), f
    return best_a


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="models/roberta_large_6ep")
    p.add_argument("--classical-model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--out", default="submissions/final_accuracy_track.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run = Path(args.run_dir)
    meta = pd.read_csv(run / "valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    gender = meta["gender"].to_numpy()
    names = load_categories()

    p_t = softmax(np.load(run / "valid_logits.npy"), axis=1)
    train = load_train(with_labels=True)
    texts = train.loc[meta.index, "description"].map(lambda t: basic_clean(t, lower=True))
    clf = joblib.load(args.classical_model)
    p_c = softmax(clf.decision_function(texts.values), axis=1)

    idx = np.arange(len(y))
    cal, ev = train_test_split(idx, test_size=0.5, random_state=args.seed, stratify=y)
    grid = np.linspace(0.0, 1.0, 41)

    def report(pred, label):
        f1 = f1_score(y[ev], pred, average="macro")
        di = macro_disparate_impact([names[c] for c in pred], gender[ev])
        n_single = count_single_gender_jobs([names[c] for c in pred], gender[ev])
        flag = f"  ⚠ {n_single} single-gender job(s)" if n_single else ""
        print(f"  {label:<34} Macro-F1 {f1:.4f}   DI {di:.3f}{flag}")
        return f1, di

    print(f"calibrate on {len(cal):,} rows, judge on {len(ev):,} rows\n")
    base_f1, base_di = report(p_t[ev].argmax(1), "transformer alone")

    # Lever 1 alone.
    b_t = tune_bias(log_softmax(np.log(p_t[cal] + 1e-12), axis=1), y[cal])
    thr_f1, thr_di = report((np.log(p_t[ev] + 1e-12) + b_t).argmax(1), "+ thresholds")

    # Lever 2 alone.
    alpha = sweep_alpha(p_t[cal], p_c[cal], y[cal], grid)
    blend_ev = alpha * p_t[ev] + (1 - alpha) * p_c[ev]
    ens_f1, ens_di = report(blend_ev.argmax(1), f"+ ensemble (a={alpha:.3f})")

    # Both. Thresholds are fitted on the BLENDED calibration scores, not reused
    # from the transformer-only fit — the blend changes the score distribution.
    blend_cal = alpha * p_t[cal] + (1 - alpha) * p_c[cal]
    b_both = tune_bias(np.log(blend_cal + 1e-12), y[cal])
    both_pred = (np.log(blend_ev + 1e-12) + b_both).argmax(1)
    both_f1, both_di = report(both_pred, "+ ensemble + thresholds")

    variants = {"transformer": base_f1, "thresholds": thr_f1,
                "ensemble": ens_f1, "ensemble+thresholds": both_f1}
    winner = max(variants, key=variants.get)
    print(f"\n  best on unseen rows: {winner} ({variants[winner]:.4f})")
    if both_f1 < max(thr_f1, ens_f1):
        print("  composing the two levers did NOT beat the better one alone —")
        print("  fitting α + 28 biases on the same 16k rows overfits them.")
    print("  (the winner's figure is mildly optimistic: it won a 4-way choice on")
    print("   these same rows. With a 4-option choice the inflation is small, but")
    print("   it is not zero.)")

    # Ship the variant that actually won, not whichever the script happens to
    # compute last — a file named "final" must not contain a runner-up.
    test = load_test()
    test["text"] = test["description"].map(lambda t: basic_clean(t, lower=True))
    p_t_test = softmax(np.load(run / "test_logits.npy"), axis=1)
    p_c_test = softmax(clf.decision_function(test["text"].values), axis=1)

    # Deployed parameters are refitted on ALL validation rows.
    alpha_full = sweep_alpha(p_t, p_c, y, grid) if "ensemble" in winner else 0.0
    if "ensemble" in winner:
        blend_full = alpha_full * p_t + (1 - alpha_full) * p_c
        scores_full, scores_test = blend_full, (alpha_full * p_t_test
                                                + (1 - alpha_full) * p_c_test)
    else:
        scores_full, scores_test = p_t, p_t_test

    if "thresholds" in winner:
        bias_full = tune_bias(np.log(scores_full + 1e-12), y)
        preds = (np.log(scores_test + 1e-12) + bias_full).argmax(1)
    else:
        preds = scores_test.argmax(1)

    print(f"\nshipping the '{winner}' pipeline")
    out = make_submission(test.index, preds, args.out)
    atomic_save(run / "final_submission_report.json", lambda q: q.write_text(json.dumps({
        "alpha_calib": alpha, "alpha_deployed": alpha_full,
        "eval_transformer_f1": base_f1, "eval_thresholds_f1": thr_f1,
        "eval_ensemble_f1": ens_f1, "eval_both_f1": both_f1,
        "eval_transformer_di": base_di, "eval_thresholds_di": thr_di,
        "eval_ensemble_di": ens_di, "eval_both_di": both_di,
        "winner": winner, "shipped": str(out), "expected_f1": variants[winner],
    }, indent=2)))
    print(f"wrote {out}  (expect ≈ {variants[winner]:.4f}, not an in-sample figure)")


if __name__ == "__main__":
    main()
