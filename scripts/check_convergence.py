"""Refuse to compare transformer runs that had not finished learning (zero GPU).

The failure this exists to prevent: two models trained for the same number of
epochs are *not* comparable if either was still improving when the budget ran
out. A bigger model that needs more epochs then looks worse than a smaller one,
and the conclusion is backwards — the measurement recorded the epoch budget, not
the model.

The test is simple and mechanical: if a run's **best epoch is its last epoch**,
it was still improving and the run is truncated. Its score is a lower bound, not
a result, and it must not be ranked against a run that plateaued.

Reads the `log_history` that `train_transformer.py` stores in each run's
`metrics.json`, so it costs nothing and needs no GPU.

    python scripts/check_convergence.py
    python scripts/check_convergence.py --strict   # exit 1 if anything is truncated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyse(metrics_path: Path) -> dict | None:
    try:
        d = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if d.get("smoke"):
        return None

    evals = [(h.get("epoch"), h["eval_macro_f1"])
             for h in d.get("log_history", []) if "eval_macro_f1" in h]
    if not evals:
        return {"run": metrics_path.parent.name, "macro_f1": d.get("macro_f1"),
                "epochs": None, "verdict": "no eval history recorded"}

    evals.sort(key=lambda e: e[0] or 0)
    scores = [s for _, s in evals]
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    truncated = best_i == len(scores) - 1 and len(scores) > 1
    # How much the last epoch still added — a large jump means a lot was left.
    last_gain = scores[-1] - scores[-2] if len(scores) > 1 else None

    return {
        "run": metrics_path.parent.name,
        "model": d.get("model"),
        "macro_f1": d.get("macro_f1"),
        "n_epochs": len(scores),
        "best_epoch": best_i + 1,
        "trajectory": [round(s, 4) for s in scores],
        "last_epoch_gain": last_gain,
        "truncated": truncated,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any run was still improving when it stopped")
    args = p.parse_args()

    runs = [r for r in (analyse(m) for m in sorted(Path(args.models_dir).glob("*/metrics.json")))
            if r]
    if not runs:
        print("no completed (non-smoke) runs found")
        return

    print("=" * 78)
    print("CONVERGENCE CHECK — a run whose best epoch is its last was still learning")
    print("=" * 78)
    truncated = []
    for r in runs:
        if r.get("epochs") is None and r.get("verdict"):
            print(f"\n{r['run']}: {r['verdict']}")
            continue
        flag = "  ⚠ TRUNCATED" if r["truncated"] else ""
        print(f"\n{r['run']}  ({r['model']}){flag}")
        print(f"  Macro-F1 per epoch : {r['trajectory']}")
        print(f"  best epoch         : {r['best_epoch']}/{r['n_epochs']}")
        if r["last_epoch_gain"] is not None:
            print(f"  gain on last epoch : {r['last_epoch_gain']:+.4f}")
        if r["truncated"]:
            truncated.append(r)

    print("\n" + "-" * 78)
    if truncated:
        print(f"{len(truncated)} run(s) stopped while still improving:")
        for r in truncated:
            print(f"  - {r['run']} (still +{r['last_epoch_gain']:.4f} on its final epoch)")
        print("\nTheir scores are LOWER BOUNDS, not results. Re-run them with more epochs")
        print("before ranking them against anything — comparing at a fixed epoch budget")
        print("measures the budget, not the model, and a larger model needing more")
        print("epochs will look worse than a smaller one that has plateaued.")
    else:
        print("every run plateaued before its budget ran out — the scores are comparable.")

    ranked = sorted((r for r in runs if r.get("macro_f1")),
                    key=lambda r: r["macro_f1"], reverse=True)
    if ranked:
        print("\nranking (⚠ = not safely comparable):")
        for r in ranked:
            print(f"  {r['macro_f1']:.4f}  {r['run']}{'  ⚠' if r.get('truncated') else ''}")

    if args.strict and truncated:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
