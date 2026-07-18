"""Aggregate the fairness Pareto front across seeds (zero GPU).

A single-seed front cannot separate a real mitigation effect from split noise.
Running the same five variants under several seeds gives each effect a mean and
a standard deviation, so the write-up can say which differences are findings and
which are not — instead of borrowing a noise estimate from another experiment.

Reads every `reports/fairness_pareto*.json` and reports, per variant, the change
against the no-mitigation baseline **computed within each seed** (paired), which
removes the seed's own difficulty from the comparison.

    python scripts/aggregate_fairness_seeds.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from defi_ia.io_utils import atomic_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="reports/fairness_pareto*.json")
    p.add_argument("--out", default="reports/fairness_pareto_multiseed.md")
    args = p.parse_args()

    files = [f for f in sorted(Path().glob(args.glob)) if "smoke" not in f.name]
    per_seed: dict[int, dict[str, dict]] = {}
    for f in files:
        d = json.loads(f.read_text())
        if d.get("smoke"):
            continue
        per_seed[d["seed"]] = {r["variant"]: r for r in d["runs"]}

    if not per_seed:
        print(f"no results matched {args.glob}")
        return

    seeds = sorted(per_seed)
    variants = [v for v in ["none", "mask-names", "scrub", "scrub+mask", "counterfactual"]
                if all(v in per_seed[s] for s in seeds)]
    print(f"{len(seeds)} seeds: {seeds}")

    def sd(a: np.ndarray) -> float:
        """Sample sd, or 0 when a single seed leaves nothing to spread."""
        return float(a.std(ddof=1)) if len(seeds) > 1 else 0.0

    rows = []
    for v in variants:
        # Paired deltas: each seed compared against its OWN baseline, so the
        # seed's intrinsic difficulty cancels instead of inflating the spread.
        d_f1 = np.array([per_seed[s][v]["macro_f1"] - per_seed[s]["none"]["macro_f1"]
                         for s in seeds])
        d_di = np.array([per_seed[s][v]["disparate_impact"]
                         - per_seed[s]["none"]["disparate_impact"] for s in seeds])
        f1 = np.array([per_seed[s][v]["macro_f1"] for s in seeds])
        di = np.array([per_seed[s][v]["disparate_impact"] for s in seeds])
        rows.append({"variant": v,
                     "f1": f1.mean(), "f1_sd": sd(f1),
                     "di": di.mean(), "di_sd": sd(di),
                     "d_f1": d_f1.mean(), "d_f1_sd": sd(d_f1),
                     "d_di": d_di.mean(), "d_di_sd": sd(d_di),
                     "d_di_all": d_di.tolist()})

    lines = [
        "# Fairness Pareto front — averaged over seeds",
        "",
        f"{len(seeds)} seeds ({', '.join(map(str, seeds))}), full 217k data, identical "
        "protocol per seed. Deltas are **paired**: each variant is compared against the "
        "no-mitigation baseline *of its own seed*, so the seed's intrinsic difficulty "
        "cancels out.",
        "",
        "| variant | Macro-F1 | ΔF1 (paired) | DI | ΔDI (paired) | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["variant"] == "none":
            verdict = "reference"
        elif abs(r["d_di"]) < 2 * max(r["d_di_sd"], 1e-9):
            verdict = "**no measurable effect**"
        else:
            verdict = "real effect"
        lines.append(
            f"| {r['variant']} | {r['f1']:.4f} ± {r['f1_sd']:.4f} | "
            f"{r['d_f1']:+.4f} ± {r['d_f1_sd']:.4f} | {r['di']:.3f} ± {r['di_sd']:.3f} | "
            f"{r['d_di']:+.3f} ± {r['d_di_sd']:.3f} | {verdict} |"
        )

    lines += ["", "## Reading", ""]
    for r in rows:
        if r["variant"] == "none":
            continue
        signal = abs(r["d_di"]) / max(r["d_di_sd"], 1e-9)
        rate = (-r["d_f1"] / -r["d_di"]) if r["d_di"] < -1e-9 else None
        line = (f"- **{r['variant']}** — ΔDI {r['d_di']:+.3f} (sd {r['d_di_sd']:.3f}, "
                f"{signal:.1f}× its own spread)")
        if rate is not None:
            line += f", costing {rate:.4f} Macro-F1 per DI point"
        line += f". Per-seed ΔDI: {[round(x, 3) for x in r['d_di_all']]}."
        lines.append(line)
    lines.append("")

    atomic_save(Path(args.out), lambda q: q.write_text("\n".join(lines)))
    print("\n".join(lines[4:len(rows) + 7]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
