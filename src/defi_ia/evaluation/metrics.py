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
    # A job predicted for only one gender has an undefined ratio in the
    # reference notebook (division by NaN/0). Keep the same behaviour: such
    # jobs drop out of the mean rather than being clipped.
    di = counts[["M", "F"]].max(axis="columns") / counts[["M", "F"]].min(axis="columns")
    return float(di.mean())


def evaluate(y_true: Sequence, y_pred: Sequence, genders: Sequence) -> dict[str, float]:
    """Return both competition metrics in one call.

    ``disparate_impact`` is computed on the *predicted* jobs, matching how the
    organisers score a submission.
    """
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "disparate_impact": macro_disparate_impact(y_pred, genders),
    }
