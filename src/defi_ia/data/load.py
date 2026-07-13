"""Load the raw competition files into tidy, indexed pandas objects.

The raw layout (see ``reports/challenge_brief.txt``):

* ``train.json`` — columns ``Id``, ``description``, ``gender`` (217,197 rows)
* ``test.json``  — columns ``Id``, ``description``, ``gender``  (54,300 rows)
* ``train_label.csv`` — ``Id``, ``Category`` (integer label)
* ``categories_string.csv`` — mapping ``job name`` ↔ integer id

All loaders index by ``Id`` so that descriptions, labels and genders align.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from defi_ia import paths


@lru_cache(maxsize=1)
def load_categories() -> dict[int, str]:
    """Return the ``{integer_id: job_name}`` mapping used for scoring."""
    df = pd.read_csv(paths.CATEGORIES)
    # File has columns "0" (job name) and "1" (integer id).
    return dict(zip(df["1"].astype(int), df["0"].astype(str)))


def load_train(with_labels: bool = True) -> pd.DataFrame:
    """Load the training set indexed by ``Id``.

    Parameters
    ----------
    with_labels:
        When ``True`` (default) join the integer ``Category`` label and its
        human-readable ``job`` name onto each row.

    Returns
    -------
    DataFrame with columns ``description``, ``gender`` and — when requested —
    ``Category`` (int) and ``job`` (str).
    """
    df = pd.read_json(paths.TRAIN_JSON).set_index("Id")
    df["description"] = df["description"].str.strip()

    if with_labels:
        labels = pd.read_csv(paths.TRAIN_LABEL, index_col="Id")["Category"]
        df["Category"] = labels
        df["job"] = df["Category"].map(load_categories())

    return df


def load_test() -> pd.DataFrame:
    """Load the test set indexed by ``Id`` (columns ``description``, ``gender``)."""
    df = pd.read_json(paths.TEST_JSON).set_index("Id")
    df["description"] = df["description"].str.strip()
    return df
