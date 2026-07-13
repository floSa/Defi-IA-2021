"""Reproducible train/validation splitting.

A single stratified hold-out for fast iteration; stratification keeps the rare
classes (down to 0.4 % of the data) present in both folds, which matters
because Macro-F1 weights every class equally.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_holdout(
    df: pd.DataFrame,
    valid_size: float = 0.15,
    seed: int = 42,
    label_col: str = "Category",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, valid), stratified on ``label_col``.

    Returns copies so downstream mutation never leaks across folds.
    """
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=df[label_col],
    )
    return train_df.copy(), valid_df.copy()
