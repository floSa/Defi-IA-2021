"""Hyper-parameter sweep for the classical model, with convergence checking (zero GPU).

Two disciplines are baked in, because without them the output would be a
plausible-looking table of meaningless numbers.

**1. Convergence is verified, not assumed.**
``LinearSVC`` stops at ``max_iter`` whether or not it has converged, and sklearn
only emits a warning. A config that hit the ceiling has not finished learning,
so comparing it against one that converged measures the iteration budget, not
the hyper-parameter. Every run below records ``n_iter_`` and is marked
NOT CONVERGED if it reached the cap — such rows are excluded from the
recommendation rather than silently ranked.

**2. The config is chosen on different data from the one that scores it.**
Picking the best of N configs on a split and then reporting that config's score
on the same split is selection bias — the same in-sample error that inflated the
threshold-tuning gain by ×2.2 (reports/experiments.md). So:

    fit    (70 %)  trains every candidate
    select (15 %)  ranks them and picks the winner
    report (15 %)  scores the winner, having played no part in choosing it

The gap between the winner's select score and its report score is printed, since
that gap *is* the selection bias, measured rather than assumed away.

    python scripts/sweep_classical.py --smoke     # 3 configs on 8k rows
    python scripts/sweep_classical.py             # full sweep, resumable
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_train
from defi_ia.evaluation.metrics import macro_disparate_impact
from defi_ia.io_utils import atomic_save
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.text import basic_clean

RESULTS = Path("reports/classical_sweep.json")

# One change at a time from the current default, so each delta is attributable.
CANDIDATES: list[dict] = [
    {"name": "baseline (C=1, min_df=5, char 2-5, sublinear)"},
    {"name": "C=0.5", "C": 0.5},
    {"name": "C=2", "C": 2.0},
    {"name": "C=4", "C": 4.0},
    {"name": "min_df=2", "min_df": 2},
    {"name": "min_df=10", "min_df": 10},
    {"name": "char 3-5", "char_ngram_range": (3, 5)},
    {"name": "char 2-6", "char_ngram_range": (2, 6)},
    {"name": "word 1-3", "word_ngram_range": (1, 3)},
    {"name": "no sublinear_tf", "sublinear_tf": False},
    {"name": "char hash 2**21", "char_n_features": 2**21},
]


def three_way(df, seed: int, select_size=0.15, report_size=0.15):
    rest, report = train_test_split(
        df, test_size=report_size, random_state=seed, stratify=df["Category"]
    )
    fit, select = train_test_split(
        rest, test_size=select_size / (1 - report_size),
        random_state=seed, stratify=rest["Category"],
    )
    return fit.copy(), select.copy(), report.copy()


def run_one(spec: dict, fit_df, select_df, report_df, names) -> dict:
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    cfg = TfidfLinearConfig(**kwargs)
    model = build_model(cfg)

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(fit_df["text"], fit_df["Category"])
        hit_cap = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    fit_s = time.time() - t0

    clf = model[-1]
    n_iter = getattr(clf, "n_iter_", None)
    n_iter = int(np.max(n_iter)) if n_iter is not None else None
    max_iter = getattr(clf, "max_iter", None)
    converged = not hit_cap and (n_iter is None or max_iter is None or n_iter < max_iter)

    sel_pred = model.predict(select_df["text"])
    rep_pred = model.predict(report_df["text"])
    return {
        "name": spec["name"],
        "select_f1": f1_score(select_df["Category"], sel_pred, average="macro"),
        "report_f1": f1_score(report_df["Category"], rep_pred, average="macro"),
        "report_di": macro_disparate_impact([names[c] for c in rep_pred], report_df["gender"]),
        "fit_seconds": round(fit_s, 1),
        "n_iter": n_iter, "max_iter": max_iter, "converged": bool(converged),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default=str(RESULTS))
    args = p.parse_args()

    candidates, subsample = CANDIDATES, args.subsample
    if args.smoke:
        candidates, subsample = CANDIDATES[:3], 8_000
        print("[smoke] 3 configs on 8k rows — checks the path, not the science\n")

    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    if subsample:
        train = train.sample(n=min(subsample, len(train)), random_state=42)
    names = load_categories()

    fit_df, select_df, report_df = three_way(train, args.seed)
    print(f"fit {len(fit_df):,} / select {len(select_df):,} / report {len(report_df):,}\n")

    out_path = Path(args.out)
    payload = {"seed": args.seed, "n_rows": len(train), "smoke": args.smoke, "runs": []}
    if out_path.exists() and not args.smoke:
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("n_rows") == len(train) and prev.get("seed") == args.seed:
                payload = prev
                print(f"resuming: {len(payload['runs'])} config(s) already done\n")
        except (json.JSONDecodeError, KeyError):
            pass

    done = {r["name"] for r in payload["runs"]}
    for spec in candidates:
        if spec["name"] in done:
            print(f"  {spec['name']}: cached")
            continue
        res = run_one(spec, fit_df, select_df, report_df, names)
        payload["runs"].append(res)
        atomic_save(out_path, lambda q, d=payload: q.write_text(json.dumps(d, indent=2)))
        flag = "" if res["converged"] else "   ⚠ NOT CONVERGED"
        print(f"  {res['name']:<44} select {res['select_f1']:.4f}  "
              f"report {res['report_f1']:.4f}  ({res['fit_seconds']:.0f}s){flag}")

    runs = payload["runs"]
    ok = [r for r in runs if r["converged"]]
    stalled = [r for r in runs if not r["converged"]]

    print("\n" + "=" * 78)
    if stalled:
        print(f"{len(stalled)} config(s) hit max_iter and are excluded from the ranking:")
        for r in stalled:
            print(f"  - {r['name']} (n_iter {r['n_iter']}/{r['max_iter']}) — "
                  "raise max_iter before drawing any conclusion from it")
    if not ok:
        print("no config converged; nothing can be concluded")
        return

    baseline = next((r for r in ok if r["name"].startswith("baseline")), None)
    winner = max(ok, key=lambda r: r["select_f1"])
    print(f"\nchosen on the select split : {winner['name']}")
    print(f"  select Macro-F1 {winner['select_f1']:.4f}")
    print(f"  report Macro-F1 {winner['report_f1']:.4f}   <- the honest score")
    print(f"  selection bias  {winner['select_f1'] - winner['report_f1']:+.4f} "
          "(what picking the max of a noisy set buys you for free)")
    print(f"  disparate imp.  {winner['report_di']:.3f}")
    if baseline and winner["name"] != baseline["name"]:
        gain = winner["report_f1"] - baseline["report_f1"]
        print(f"\n  vs baseline on the report split: {gain:+.4f}")
        if gain <= 0:
            print("  the winner does NOT beat the baseline once judged on unseen rows —")
            print("  its select-split lead was selection noise. Keep the baseline.")
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
