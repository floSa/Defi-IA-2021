"""Where does the transformer actually beat the classical model? (zero GPU)

The error analysis produced a falsifiable prediction: the classical model's
Macro-F1 is lost to **semantic confusion between neighbouring professions**
(professor ↔ teacher / psychologist / physician, architect ↔ software_engineer),
not to the rare classes, which already score well. Telling a professor from a
teacher needs context, which is exactly what a contextual encoder has and a
bag-of-words does not.

So the transformer's ~+4 pt should be **concentrated in those confusable
mid-frequency classes**. This script checks that, per class, on the shared
holdout — and it matters either way:

* prediction holds → the two models fail on *different* classes, and the
  ensemble has real diversity to exploit;
* prediction fails → the transformer is uniformly better, the models are
  redundant, and a blend will buy far less than the headline gap suggests.

Both models must have been fit on the same split; the transformer run dir and
`train_classical.py` (no `--full`) both default to
`stratified_holdout(0.15, seed 42)`.

    python scripts/compare_per_class.py --run-dir models/roberta_base_repro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from defi_ia.data.load import load_categories, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="transformer run dir")
    p.add_argument("--classical-model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports/per_class_comparison.md")
    args = p.parse_args()

    run = Path(args.run_dir)
    meta = pd.read_csv(run / "valid_meta.csv", index_col="Id")
    y = meta["Category"].to_numpy()
    t_pred = np.load(run / "valid_logits.npy").argmax(1)

    train = load_train(with_labels=True)
    texts = train.loc[meta.index, "description"].map(lambda t: basic_clean(t, lower=True))
    c_pred = joblib.load(args.classical_model).predict(texts)

    names = load_categories()
    labels = sorted(names)
    c_f1 = f1_score(y, c_pred, labels=labels, average=None, zero_division=0)
    t_f1 = f1_score(y, t_pred, labels=labels, average=None, zero_division=0)
    support = np.array([(y == c).sum() for c in labels])

    df = pd.DataFrame({
        "job": [names[c] for c in labels], "support": support,
        "classical_f1": c_f1, "transformer_f1": t_f1, "delta": t_f1 - c_f1,
    }).sort_values("delta", ascending=False)

    # The classes the error analysis singled out as the bag-of-words bottleneck.
    PREDICTED_WEAK = {"pastor", "teacher", "interior_designer", "architect",
                      "paralegal", "software_engineer", "chiropractor"}
    weak = df[df["job"].isin(PREDICTED_WEAK)]
    rest = df[~df["job"].isin(PREDICTED_WEAK)]

    _, holdout = stratified_holdout(train, 0.15, args.seed)
    lines = [
        "# Per-class: transformer vs classical",
        "",
        f"Shared holdout of {len(y):,} rows. Transformer run: `{run.name}`.",
        "",
        f"- classical Macro-F1   **{c_f1.mean():.4f}**",
        f"- transformer Macro-F1 **{t_f1.mean():.4f}**  ({t_f1.mean() - c_f1.mean():+.4f})",
        "",
        "## Testing the prediction from the error analysis",
        "",
        "The claim was that the transformer's gain concentrates in the classes the "
        "bag-of-words confuses with their semantic neighbours, not in the rare ones.",
        "",
        f"- mean gain on the 7 predicted-weak classes : **{weak['delta'].mean():+.4f}**",
        f"- mean gain on the other 21 classes         : **{rest['delta'].mean():+.4f}**",
        "",
    ]
    ratio = weak["delta"].mean() / rest["delta"].mean() if rest["delta"].mean() else float("nan")
    if weak["delta"].mean() > rest["delta"].mean():
        lines += [
            f"**Prediction holds** (×{ratio:.1f}). The two models fail on different "
            "classes, so the ensemble has genuine diversity to exploit — blend it.",
        ]
    else:
        lines += [
            "**Prediction does not hold.** The transformer is not specifically better "
            "on the confusable classes, so the two models are more redundant than the "
            "headline gap suggests and the ensemble will buy less than expected.",
        ]

    lines += ["", "## Per class, biggest gain first", "",
              "| job | support | classical | transformer | Δ |", "|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        mark = " ⭐" if r["job"] in PREDICTED_WEAK else ""
        lines.append(f"| {r['job']}{mark} | {r['support']} | {r['classical_f1']:.3f} | "
                     f"{r['transformer_f1']:.3f} | {r['delta']:+.3f} |")
    lines += ["", "⭐ = flagged as a bottleneck by `reports/error_analysis.md`.",
              f"\nHoldout built with seed {args.seed} ({len(holdout):,} rows).", ""]

    atomic_save(Path(args.out), lambda q: q.write_text("\n".join(lines)))
    print(f"classical {c_f1.mean():.4f} | transformer {t_f1.mean():.4f} "
          f"({t_f1.mean() - c_f1.mean():+.4f})")
    print(f"gain on predicted-weak classes {weak['delta'].mean():+.4f} vs "
          f"{rest['delta'].mean():+.4f} elsewhere")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
