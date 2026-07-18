"""Blend classical + transformer probabilities, with the blend weight chosen honestly.

Both models score the same 28 Category indices, so we mix their per-class
probabilities and take the argmax. The only free parameter is alpha, the weight
on the transformer.

METHOD — the trap this script is built to avoid
-----------------------------------------------
Sweeping alpha on a set and then reporting the winning alpha's score on that
same set is the same in-sample mistake that inflated the threshold-tuning gain
by ×2.2 (see reports/experiments.md). Alpha is only one parameter, so the
optimism is smaller than for the 28-parameter bias — but "smaller" is not
"zero", and the sweep picks the maximum of a noisy curve by construction.

So the validation rows are split in half: alpha is swept on the calib half and
the gain is reported on the eval half. The deployed alpha is then refit on all
validation rows, which is the right thing to do for the model that ships.

**Both models must be blind to these validation rows.** The classical model has
to come from a run that excluded them — the same seed and valid_size as the
transformer's split. If it was fit with --full it has already seen them, its
validation probabilities are near-perfect, and every alpha this script picks
will be wrong. The check below refuses to guess about that.

Example
-------
    python scripts/build_ensemble.py --run-dir models/roberta_holdout \
        --classical-model models/classical_wordchar_svm.joblib \
        --out submissions/ensemble.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.evaluation.metrics import macro_disparate_impact
from defi_ia.evaluation.submission import make_submission
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean


def _classical_proba(model, texts) -> np.ndarray:
    if hasattr(model[-1], "predict_proba"):
        return model.predict_proba(texts)
    # LinearSVC has no predict_proba; softmax of the margins is the usual surrogate.
    return softmax(model.decision_function(texts), axis=1)


def _sweep_alpha(p_t: np.ndarray, p_c: np.ndarray, y: np.ndarray,
                 grid: np.ndarray) -> tuple[float, float]:
    """Return the (alpha, Macro-F1) that maximises Macro-F1 on the rows given."""
    best_a, best_f = 0.0, -1.0
    for a in grid:
        f = f1_score(y, (a * p_t + (1 - a) * p_c).argmax(1), average="macro")
        if f > best_f:
            best_a, best_f = float(a), f
    return best_a, best_f


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="models/roberta_holdout",
                   help="transformer run dir (valid_logits.npy, valid_meta.csv, test_logits.npy)")
    p.add_argument("--classical-model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--out", default="submissions/ensemble.csv")
    p.add_argument("--alpha", type=float, default=None,
                   help="skip the sweep and force this weight on the transformer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assume-classical-is-blind", action="store_true",
                   help="confirm the classical model was NOT fit on these validation rows")
    args = p.parse_args()

    run = Path(args.run_dir)
    meta = pd.read_csv(run / "valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    gender = meta["gender"].to_numpy()
    names = load_categories()

    p_t = softmax(np.load(run / "valid_logits.npy"), axis=1)

    # Re-score the transformer's exact validation rows with the classical model.
    train = load_train(with_labels=True)
    missing = meta.index.difference(train.index)
    if len(missing):
        raise ValueError(f"{len(missing)} validation Ids are absent from train.json")
    texts = train.loc[meta.index, "description"].map(lambda t: basic_clean(t, lower=True))

    clf = joblib.load(args.classical_model)
    p_c = _classical_proba(clf, texts.values)
    if p_t.shape != p_c.shape:
        raise ValueError(f"shape mismatch: transformer {p_t.shape} vs classical {p_c.shape}")

    # A classical model that trained on these rows scores absurdly well on them.
    # That is the signature of the --full model being passed in by mistake, which
    # would silently corrupt every alpha below.
    f1_c = f1_score(y, p_c.argmax(1), average="macro")
    f1_t = f1_score(y, p_t.argmax(1), average="macro")
    print(f"on {len(y):,} validation rows:  classical {f1_c:.4f}   transformer {f1_t:.4f}")
    if f1_c > 0.95 and not args.assume_classical_is_blind:
        raise SystemExit(
            f"\nclassical Macro-F1 is {f1_c:.4f} on these rows — it was almost certainly\n"
            "trained on them (a --full fit). Its probabilities here are memorised, not\n"
            "predicted, so any alpha swept against them is meaningless.\n"
            "Refit the classical model on the same split as the transformer, or pass\n"
            "--assume-classical-is-blind if you are certain this is a genuine holdout."
        )

    if args.alpha is not None:
        alpha_deployed, honest = args.alpha, None
        print(f"using forced alpha={alpha_deployed}")
    else:
        grid = np.linspace(0.0, 1.0, 41)
        idx = np.arange(len(y))
        calib_idx, eval_idx = train_test_split(
            idx, test_size=0.5, random_state=args.seed, stratify=y
        )
        alpha_calib, _ = _sweep_alpha(p_t[calib_idx], p_c[calib_idx], y[calib_idx], grid)

        # Honest read: the alpha chosen on calib, judged on rows it never saw.
        y_e = y[eval_idx]
        blend_e = alpha_calib * p_t[eval_idx] + (1 - alpha_calib) * p_c[eval_idx]
        honest = f1_score(y_e, blend_e.argmax(1), average="macro")
        t_only = f1_score(y_e, p_t[eval_idx].argmax(1), average="macro")
        # What sweeping and reporting on the same rows would have claimed.
        _, insample = _sweep_alpha(p_t[eval_idx], p_c[eval_idx], y_e, grid)

        di_t = macro_disparate_impact([names[c] for c in p_t[eval_idx].argmax(1)],
                                      gender[eval_idx])
        di_b = macro_disparate_impact([names[c] for c in blend_e.argmax(1)], gender[eval_idx])

        print("\n=== ensemble (alpha swept on calib, judged on eval) ===")
        print(f"  alpha chosen on calib     {alpha_calib:.3f}")
        print(f"  transformer alone (eval)  {t_only:.4f}")
        print(f"  blend             (eval)  {honest:.4f}  ({honest - t_only:+.4f})  <- REAL")
        print(f"  [sweeping on eval itself would claim {insample:.4f} "
              f"({insample - t_only:+.4f})]")
        print(f"  disparate impact  {di_t:.3f} -> {di_b:.3f}  ({di_b - di_t:+.3f})")

        # Deployed alpha: refit on every validation row now that the honest
        # estimate is banked.
        alpha_deployed, _ = _sweep_alpha(p_t, p_c, y, grid)
        print(f"  alpha refit on all validation rows: {alpha_deployed:.3f}")

        atomic_save(run / "ensemble_report.json", lambda q: q.write_text(json.dumps({
            "alpha_calib": alpha_calib, "alpha_deployed": alpha_deployed,
            "eval_transformer_f1": t_only, "eval_blend_f1": honest,
            "delta_honest": honest - t_only,
            "delta_if_measured_in_sample": insample - t_only,
            "eval_transformer_di": di_t, "eval_blend_di": di_b,
        }, indent=2)))

    test_logits = run / "test_logits.npy"
    if not test_logits.exists():
        print(f"\nno {test_logits}; not writing a submission")
        return
    test = load_test()
    test["text"] = test["description"].map(lambda t: basic_clean(t, lower=True))
    blend = (alpha_deployed * softmax(np.load(test_logits), axis=1)
             + (1 - alpha_deployed) * _classical_proba(clf, test["text"].values))
    out = make_submission(test.index, blend.argmax(axis=1), args.out)
    print(f"\nensemble (alpha={alpha_deployed:.3f}) -> {out}  [{len(test)} rows]")
    if honest is not None:
        print(f"expect roughly {honest:.4f} Macro-F1, not the in-sample figure")


if __name__ == "__main__":
    main()
