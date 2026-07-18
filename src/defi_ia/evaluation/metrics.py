"""Competition metrics.

Two numbers matter in this challenge:

1. **Macro-F1** — the public/private Kaggle leaderboard metric. Unweighted
   mean of the per-class F1 score, so every one of the 28 jobs counts equally
   regardless of frequency (rare jobs like ``rapper`` matter as much as
   ``professor``).

2. **Macro disparate impact** — the fairness tie-breaker for the top 10.
   For each *predicted* job, take ``max(M, F) / min(M, F)`` over the gender
   counts of the people assigned to it, then average across jobs. The lower
   the better; 1.0 is perfect demographic parity.

The disparate-impact implementation mirrors, line for line, the organisers'
reference notebook (``notebooks/01_fairness_metric_reference.ipynb``) so our
offline score matches the official one.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import f1_score


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Unweighted mean per-class F1 — the leaderboard metric."""
    return float(f1_score(y_true, y_pred, average="macro"))


def macro_disparate_impact(jobs: Sequence, genders: Sequence) -> float:
    """Average per-job gender disparate impact for a set of predictions.

    Parameters
    ----------
    jobs:
        Predicted job for each person (labels or names — only grouping matters).
    genders:
        Matching gender for each person, using ``"M"`` / ``"F"``.

    Returns
    -------
    Mean over jobs of ``max(M, F) / min(M, F)``. Lower is fairer; 1.0 is parity.
    """
    people = pd.DataFrame({"job": list(jobs), "gender": list(genders)})
    counts = people.groupby(["job", "gender"]).size().unstack("gender")
    # ⚠️ Two edge cases, both inherited deliberately from the organisers' notebook
    # (see test_metrics.py, which pins them):
    #
    #   * a job that is NEVER predicted has no row at all, so it drops out of
    #     the mean — harmless;
    #   * a job predicted for a SINGLE gender has NaN in the other column, and
    #     pandas' max/min skip NaN, so both return the same count and the ratio
    #     is **1.0 — scored as perfect parity**, not as maximal unfairness.
    #
    # The second one is exploitable: driving a class to a single gender *lowers*
    # this metric. Any procedure that optimises DI directly will find that, so
    # report ``count_single_gender_jobs`` alongside DI whenever you do.
    di = counts[["M", "F"]].max(axis="columns") / counts[["M", "F"]].min(axis="columns")
    return float(di.mean())


def count_single_gender_jobs(jobs: Sequence, genders: Sequence) -> int:
    """Number of predicted jobs assigned to exactly one gender.

    A companion diagnostic for :func:`macro_disparate_impact`, which scores such
    jobs as perfectly fair (ratio 1.0). A fairness result is only trustworthy if
    this count did not grow: a "better" DI obtained by emptying a class of one
    gender is the metric being gamed, not fairness being improved.
    """
    people = pd.DataFrame({"job": list(jobs), "gender": list(genders)})
    counts = people.groupby(["job", "gender"]).size().unstack("gender")
    for col in ("M", "F"):
        if col not in counts:
            counts[col] = float("nan")
    return int(counts[["M", "F"]].isna().any(axis="columns").sum())


def evaluate(y_true: Sequence, y_pred: Sequence, genders: Sequence) -> dict[str, float]:
    """Return both competition metrics in one call.

    ``disparate_impact`` is computed on the *predicted* jobs, matching how the
    organisers score a submission.
    """
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "disparate_impact": macro_disparate_impact(y_pred, genders),
    }
