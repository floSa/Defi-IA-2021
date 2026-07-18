"""Per-class error and fairness breakdown of a saved classical model (zero GPU).

Macro-F1 averages 28 per-class F1 scores with equal weight, so the score is
driven by whichever classes are worst — not by the bulk of the data. Likewise
the macro disparate impact averages 28 per-job gender ratios, so a handful of
jobs can dominate it. Both metrics hide *where* the problem is; this script
opens them up.

Outputs `reports/error_analysis.md`:

* per-class precision / recall / F1 / support, worst first — the Macro-F1
  bottleneck, i.e. what augmentation or threshold work should target;
* per-class predicted gender ratio — which jobs actually drive the DI;
* the confusion pairs that cost the most, so the failures can be read as
  "these two jobs look alike" rather than as an undifferentiated error rate.

Uses the model saved by ``train_classical.py --save-model``, which is fit on the
training split only, so the holdout below is genuinely unseen.

    python scripts/error_analysis.py --model models/classical_wordchar_svm.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from defi_ia.data.load import load_categories, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/error_analysis.md")
    args = p.parse_args()

    names = load_categories()
    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    _, va = stratified_holdout(train, args.valid_size, args.seed)

    model = joblib.load(args.model)
    pred = model.predict(va["text"])
    y = va["Category"].to_numpy()
    gender = va["gender"].to_numpy()

    overall_f1 = macro_f1(y, pred)
    overall_di = macro_disparate_impact([names[c] for c in pred], gender)

    labels = sorted(names)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )

    # Gender split of the PREDICTIONS, which is what the competition's disparate
    # impact is computed on (not the split of the true labels).
    rows = []
    for i, c in enumerate(labels):
        mask = pred == c
        n_m = int((gender[mask] == "M").sum())
        n_f = int((gender[mask] == "F").sum())
        di = max(n_m, n_f) / min(n_m, n_f) if min(n_m, n_f) > 0 else float("nan")
        rows.append({
            "class": c, "job": names[c], "support": int(support[i]),
            "precision": prec[i], "recall": rec[i], "f1": f1[i],
            "pred_M": n_m, "pred_F": n_f, "pred_DI": di,
        })
    df = pd.DataFrame(rows)

    cm = confusion_matrix(y, pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = [
        (names[labels[i]], names[labels[j]], int(cm[i, j]))
        for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:15], cm.shape),
                        strict=True)
    ]

    by_f1 = df.sort_values("f1")
    lines = [
        "# Per-class error & fairness breakdown (classical model)",
        "",
        f"Holdout: {len(va):,} rows (stratified, seed {args.seed}, "
        f"valid_size {args.valid_size}). Model: `{args.model}`.",
        "",
        f"- **Macro-F1 {overall_f1:.4f}** — the unweighted mean of the 28 F1 values below.",
        f"- **Disparate impact {overall_di:.3f}** — the mean of the per-job "
        "max(M,F)/min(M,F) ratios (lower is fairer; the labels' own value is 3.898).",
        "",
        "## Macro-F1 bottleneck — worst classes first",
        "",
        "Macro-F1 weights all classes equally, so the bottom of this table is worth "
        "as much as the top. `support` is the number of true holdout examples.",
        "",
        "| job | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in by_f1.iterrows():
        lines.append(f"| {r['job']} | {r['support']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} |")

    gap = by_f1.head(7)["f1"].mean()
    lines += [
        "",
        f"The 7 weakest classes average **{gap:.3f}** F1 against an overall "
        f"{overall_f1:.4f}. Lifting those is worth far more per unit of effort than "
        "improving the classes that already score well — one point on a weak class "
        "moves Macro-F1 as much as one point on `professor`.",
        "",
        "## What drives the disparate impact",
        "",
        "Computed on the model's **predictions**, as the competition does. A job "
        "predicted for only one gender has an undefined ratio and drops out of the "
        "mean, matching the organisers' reference implementation.",
        "",
        "| job | predicted M | predicted F | ratio |",
        "|---|---:|---:|---:|",
    ]
    for _, r in df.sort_values("pred_DI", ascending=False).iterrows():
        ratio = "—" if np.isnan(r["pred_DI"]) else f"{r['pred_DI']:.2f}"
        lines.append(f"| {r['job']} | {r['pred_M']} | {r['pred_F']} | {ratio} |")

    lines += [
        "",
        "## Most expensive confusions",
        "",
        "| true job | predicted as | count |",
        "|---|---|---:|",
    ]
    lines += [f"| {a} | {b} | {n} |" for a, b, n in pairs if n > 0]
    lines.append("")

    out = Path(args.out)
    atomic_save(out, lambda q: q.write_text("\n".join(lines)))
    print(f"Macro-F1 {overall_f1:.4f} | DI {overall_di:.3f}")
    print(f"weakest classes: {', '.join(by_f1.head(5)['job'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
