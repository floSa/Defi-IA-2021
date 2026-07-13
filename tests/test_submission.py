"""Tests for the Kaggle submission builder."""

import pandas as pd
import pytest

from defi_ia.evaluation.submission import make_submission


def test_make_submission_sorts_by_id_and_writes(tmp_path):
    out = make_submission(
        ids=[2, 0, 1],
        categories=[5, 3, 4],
        path=tmp_path / "sub.csv",
    )
    df = pd.read_csv(out)
    assert list(df.columns) == ["Id", "Category"]
    assert list(df["Id"]) == [0, 1, 2]           # sorted
    assert list(df["Category"]) == [3, 4, 5]     # follows its Id
    assert pd.api.types.is_integer_dtype(df["Category"])


def test_make_submission_coerces_float_integers(tmp_path):
    out = make_submission([0, 1], [3.0, 4.0], tmp_path / "s.csv")
    assert pd.api.types.is_integer_dtype(pd.read_csv(out)["Category"])


def test_make_submission_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        make_submission([0, 1], [3, None], tmp_path / "s.csv")
