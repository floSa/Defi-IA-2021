"""Tests pinning our metrics to the organisers' reference values."""

import pytest

from defi_ia.evaluation.metrics import evaluate, macro_disparate_impact, macro_f1


def test_macro_f1_perfect():
    y = [0, 1, 2, 1, 0]
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_disparate_impact_parity_is_one():
    # Every job split 50/50 by gender → disparate impact exactly 1.0.
    jobs = ["nurse", "nurse", "surgeon", "surgeon"]
    genders = ["M", "F", "M", "F"]
    assert macro_disparate_impact(jobs, genders) == pytest.approx(1.0)


def test_disparate_impact_matches_reference_notebook():
    """Reproduce the exact value from the organisers' fairness notebook.

    Their notebook reports ``macro_disparate_impact`` over the *training
    labels* as 3.898171170378378. We recompute it here from the raw files so
    any drift in our implementation is caught immediately.
    """
    pytest.importorskip("pandas")
    from defi_ia import paths

    if not paths.TRAIN_JSON.exists():
        pytest.skip("raw data not present")

    from defi_ia.data.load import load_train

    df = load_train(with_labels=True)
    value = macro_disparate_impact(df["job"], df["gender"])
    assert value == pytest.approx(3.898171170378378, rel=1e-9)


def test_evaluate_returns_both_metrics():
    out = evaluate([0, 1], [0, 1], ["M", "F"])
    assert set(out) == {"macro_f1", "disparate_impact"}
