"""Accuracy/fairness Pareto front for the classical model (zero GPU).

The competition scores two numbers that pull against each other: Macro-F1
(maximise, decides the leaderboard) and macro disparate impact (minimise toward
1.0, breaks ties in the top 10). Biographies leak gender through pronouns,
honorifics and first names, so anything that removes that signal costs some
accuracy. The question is not "which mitigation is best" — it is **what does
each one cost, and is the trade worth it**.

This script measures every mitigation on the same split and prints the front, so
the choice of shipped submission is made on numbers instead of intuition.

Variants
--------
none                 no mitigation — the reference point
scrub                pronouns, honorifics and gendered nouns neutralised
mask-names           first names replaced via spaCy NER (they leak gender as
                     strongly as pronouns); no-op and reported as skipped if
                     spaCy is absent
scrub+mask           both
counterfactual       every training bio is duplicated with its genders swapped,
                     so the job label becomes gender-invariant by construction.
                     Unlike the other three this adds signal rather than removing
                     it, and it leaves the test text untouched.

No model selection happens here — every variant is reported on the same held-out
rows, and none of them was used to choose anything. So these numbers are
directly comparable and none carries selection bias.

    python scripts/fairness_pareto.py --smoke
    python scripts/fairness_pareto.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from defi_ia.data.load import load_categories, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.io_utils import atomic_save
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.augment import gender_counterfactual
from defi_ia.preprocessing.text import basic_clean, mask_person_names, scrub_gender

RESULTS = Path("reports/fairness_pareto.json")
VARIANTS = ["none", "scrub", "mask-names", "scrub+mask", "counterfactual"]


def _spacy_available() -> bool:
    try:
        import spacy  # noqa: F401

        spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
        return True
    except Exception:
        return False


def _apply(texts: pd.Series, variant: str) -> pd.Series:
    """Text transform for a variant (counterfactual acts on rows, not text)."""
    out = texts
    if variant in ("mask-names", "scrub+mask"):
        out = pd.Series(mask_person_names(out.tolist()), index=out.index)
    if variant in ("scrub", "scrub+mask"):
        out = out.map(scrub_gender)
    return out


def run_variant(variant: str, tr: pd.DataFrame, va: pd.DataFrame, names, seed: int) -> dict:
    tr_text, va_text = _apply(tr["text"], variant), _apply(va["text"], variant)
    tr_labels = tr["Category"]

    if variant == "counterfactual":
        # Train on each bio AND its gender-swapped twin, so the model cannot key
        # the job on gender. The evaluation text stays untouched.
        swapped = tr_text.map(gender_counterfactual)
        tr_text = pd.concat([tr_text, swapped], ignore_index=True)
        tr_labels = pd.concat([tr_labels, tr_labels], ignore_index=True)

    model = build_model(TfidfLinearConfig(seed=seed))
    t0 = time.time()
    model.fit(tr_text, tr_labels)
    fit_s = time.time() - t0

    pred = model.predict(va_text)
    f1 = macro_f1(va["Category"], pred)
    di = macro_disparate_impact([names[c] for c in pred], va["gender"])
    return {"variant": variant, "n_train": len(tr_text), "macro_f1": f1,
            "disparate_impact": di, "fit_seconds": round(fit_s, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()

    variants, subsample = list(VARIANTS), args.subsample
    if args.smoke:
        variants, subsample = ["none", "scrub", "counterfactual"], 8_000
        # Never write smoke output to the real results path: cpu_queue.sh reads
        # that file's existence as "stage already done".
        if args.out == str(RESULTS):
            args.out = str(RESULTS.with_suffix(".smoke.json"))
        print("[smoke] 3 variants on 8k rows\n")

    if "mask-names" in variants and not _spacy_available():
        print("spaCy/en_core_web_sm unavailable -> skipping the name-masking variants.")
        print("  install with: uv pip install spacy && "
              ".venv/bin/python -m spacy download en_core_web_sm\n")
        variants = [v for v in variants if "mask" not in v]

    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    if subsample:
        train = train.sample(n=min(subsample, len(train)), random_state=42)
    names = load_categories()
    tr, va = stratified_holdout(train, args.valid_size, args.seed)
    print(f"train {len(tr):,} / holdout {len(va):,}\n")

    out_path = Path(args.out)
    payload = {"seed": args.seed, "n_rows": len(train), "smoke": args.smoke, "runs": []}
    if out_path.exists() and not args.smoke:
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("n_rows") == len(train) and prev.get("seed") == args.seed:
                payload = prev
                print(f"resuming: {len(payload['runs'])} variant(s) already done\n")
        except (json.JSONDecodeError, KeyError):
            pass

    done = {r["variant"] for r in payload["runs"]}
    for variant in variants:
        if variant in done:
            print(f"  {variant}: cached")
            continue
        res = run_variant(variant, tr, va, names, args.seed)
        payload["runs"].append(res)
        atomic_save(out_path, lambda q, d=payload: q.write_text(json.dumps(d, indent=2)))
        print(f"  {res['variant']:<16} Macro-F1 {res['macro_f1']:.4f}   "
              f"DI {res['disparate_impact']:.3f}   ({res['fit_seconds']:.0f}s)")

    runs = payload["runs"]
    base = next((r for r in runs if r["variant"] == "none"), None)

    print("\n" + "=" * 70)
    print("ACCURACY / FAIRNESS TRADE-OFF   (F1 higher better, DI lower better)")
    print("=" * 70)
    print(f"{'variant':<16} {'Macro-F1':>9} {'ΔF1':>8} {'DI':>7} {'ΔDI':>8}"
          f" {'F1 cost / DI point':>20}")
    for r in sorted(runs, key=lambda x: x["disparate_impact"]):
        if base:
            d_f1 = r["macro_f1"] - base["macro_f1"]
            d_di = r["disparate_impact"] - base["disparate_impact"]
            # How much accuracy each unit of fairness costs — the number that
            # actually decides whether a mitigation is worth shipping.
            rate = f"{-d_f1 / -d_di:.4f}" if d_di < -1e-9 else "—"
            print(f"{r['variant']:<16} {r['macro_f1']:9.4f} {d_f1:+8.4f} "
                  f"{r['disparate_impact']:7.3f} {d_di:+8.3f} {rate:>22}")
        else:
            print(f"{r['variant']:<16} {r['macro_f1']:9.4f} {'':>8} "
                  f"{r['disparate_impact']:7.3f}")

    # A variant that is worse on BOTH axes than another is never the right ship.
    dominated = [
        r["variant"] for r in runs
        if any(o["macro_f1"] >= r["macro_f1"] and o["disparate_impact"] <= r["disparate_impact"]
               and o["variant"] != r["variant"] for o in runs)
    ]
    if dominated:
        print(f"\ndominated (worse or equal on both axes, never worth shipping): "
              f"{', '.join(dominated)}")
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
