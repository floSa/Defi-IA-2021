"""Train and evaluate the classical TF-IDF + linear model (Steps A & B).

Examples
--------
Quick validation run (holdout, prints Macro-F1 + fairness)::

    python scripts/train_classical.py

Word-only baseline (Step A floor)::

    python scripts/train_classical.py --no-char --classifier logistic

Scrub gender and compare fairness::

    python scripts/train_classical.py --scrub-gender

Fit on all data and write a submission::

    python scripts/train_classical.py --full --submit submissions/classical.csv
"""

from __future__ import annotations

import argparse
import time

import joblib
import pandas as pd

from defi_ia import paths
from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.evaluation.submission import make_submission
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.augment import gender_counterfactual
from defi_ia.preprocessing.text import basic_clean, scrub_gender


def _prepare_text(series, scrub: bool):
    cleaned = series.map(lambda t: basic_clean(t, lower=True))
    if scrub:
        cleaned = cleaned.map(scrub_gender)
    return cleaned


def _add_counterfactuals(texts, labels):
    """Append the gender-swapped twin of every row, keeping its label.

    Seeing each biography under both genders makes the job label gender-invariant
    by construction. Unlike scrubbing, this *adds* signal rather than deleting
    it, so the test text needs no transformation at inference time.
    """
    swapped = texts.map(gender_counterfactual)
    return (pd.concat([texts, swapped], ignore_index=True),
            pd.concat([labels, labels], ignore_index=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier", default="linear_svm",
                   choices=["linear_svm", "logistic", "sgd"])
    p.add_argument("--no-char", action="store_true", help="word n-grams only")
    p.add_argument("--scrub-gender", action="store_true",
                   help="neutralise gender markers (fairness track)")
    p.add_argument("--counterfactual", action="store_true",
                   help="train on each bio AND its gender-swapped twin (fairness track). "
                        "Measured best DI on the 3-seed front: 3.281 vs 3.828 unmitigated, "
                        "for -0.007 Macro-F1. Leaves the test text untouched.")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true",
                   help="fit on all training data (for submission)")
    p.add_argument("--submit", metavar="PATH", default=None,
                   help="write a submission CSV to PATH (implies test inference)")
    p.add_argument("--save-model", metavar="PATH", default=None)
    args = p.parse_args()

    paths.ensure_dirs()
    cfg = TfidfLinearConfig(
        use_char=not args.no_char,
        classifier=args.classifier,
        seed=args.seed,
        calibrate=False,
    )

    print(f"Loading data … (char={cfg.use_char}, clf={cfg.classifier}, "
          f"scrub_gender={args.scrub_gender}, counterfactual={args.counterfactual})")
    train = load_train(with_labels=True)
    train["text"] = _prepare_text(train["description"], args.scrub_gender)

    if not args.full:
        tr, va = stratified_holdout(train, args.valid_size, args.seed)
        fit_text, fit_labels = tr["text"], tr["Category"]
        if args.counterfactual:
            # Only the TRAINING side is augmented — the holdout must stay the
            # untouched distribution the competition actually scores.
            fit_text, fit_labels = _add_counterfactuals(fit_text, fit_labels)
        model = build_model(cfg)
        t0 = time.time()
        model.fit(fit_text, fit_labels)
        fit_s = time.time() - t0

        pred = model.predict(va["text"])
        f1 = macro_f1(va["Category"], pred)
        # Fairness is measured on the *predicted* job names.
        names = load_categories()
        di = macro_disparate_impact([names[c] for c in pred], va["gender"])

        print("\n=== Validation ===")
        print(f"  fit time      : {fit_s:5.1f}s on {len(fit_text):,} docs")
        print(f"  Macro-F1      : {f1:.4f}")
        print(f"  disparate imp.: {di:.4f}  (labels' own = 3.898, target → 1.0)")
    else:
        fit_text, fit_labels = train["text"], train["Category"]
        if args.counterfactual:
            fit_text, fit_labels = _add_counterfactuals(fit_text, fit_labels)
        model = build_model(cfg)
        print(f"Fitting on all {len(fit_text):,} docs …")
        model.fit(fit_text, fit_labels)

    if args.save_model:
        joblib.dump(model, args.save_model)
        print(f"  saved model → {args.save_model}")

    if args.submit:
        if not args.full:
            print("  [warn] --submit without --full uses the holdout model.")
        test = load_test()
        test["text"] = _prepare_text(test["description"], args.scrub_gender)
        preds = model.predict(test["text"])
        out = make_submission(test.index, preds, args.submit)
        print(f"  wrote submission → {out}")


if __name__ == "__main__":
    main()
