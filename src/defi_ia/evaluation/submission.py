"""Build and validate a Kaggle submission file.

A valid submission (see the brief) must:

1. contain exactly two columns, ``Id`` and ``Category``;
2. be ordered by ``Id``;
3. use the *integer* job id in ``Category``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from defi_ia import paths


def make_submission(
    ids: pd.Index | pd.Series,
    categories: pd.Series | list[int],
    path: str | Path,
) -> Path:
    """Write a competition-ready submission and return its path.

    Raises if the shape or dtype would be rejected by Kaggle.
    """
    sub = pd.DataFrame({"Id": list(ids), "Category": list(categories)})
    sub = sub.sort_values("Id").reset_index(drop=True)

    if sub["Category"].isna().any():
        raise ValueError("Submission contains missing Category values.")
    if not pd.api.types.is_integer_dtype(sub["Category"]):
        # Coerce float-like integers, fail on genuine non-integers.
        as_int = sub["Category"].astype("int64")
        if not (as_int == sub["Category"]).all():
            raise ValueError("Category column must contain integer job ids.")
        sub["Category"] = as_int

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def check_against_template(submission_path: str | Path) -> None:
    """Assert a submission covers exactly the ids in the official template."""
    template = pd.read_csv(paths.SUBMISSION_TEMPLATE)
    sub = pd.read_csv(submission_path)
    if set(sub["Id"]) != set(template["Id"]):
        raise ValueError("Submission ids do not match the template ids.")
    if len(sub) != len(template):
        raise ValueError(f"Expected {len(template)} rows, got {len(sub)}.")
