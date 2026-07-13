"""Print a concise dataset summary (class balance, gender, text length, fairness).

Run with ``make eda`` or ``python scripts/explore_data.py`` once the core
environment is installed and the raw data extracted.
"""

from __future__ import annotations

from defi_ia.data.load import load_test, load_train
from defi_ia.evaluation.metrics import macro_disparate_impact


def main() -> None:
    train = load_train(with_labels=True)
    test = load_test()

    print(f"train: {len(train):,} rows | test: {len(test):,} rows")
    print(f"classes: {train['job'].nunique()}")

    words = train["description"].str.split().str.len()
    print(
        "description length (words): "
        f"median={words.median():.0f} mean={words.mean():.1f} "
        f"p95={words.quantile(0.95):.0f} max={words.max()}"
    )

    print("\ngender balance (train):")
    print(train["gender"].value_counts().to_string())

    print("\nclass distribution (train):")
    dist = train["job"].value_counts()
    for job, n in dist.items():
        print(f"  {job:20s} {n:6d} ({100 * n / len(train):4.1f}%)")

    di = macro_disparate_impact(train["job"], train["gender"])
    print(f"\nmacro disparate impact of the ground-truth labels: {di:.4f}")
    print("(a model scores 1.0 at perfect parity; the labels themselves are biased)")


if __name__ == "__main__":
    main()
