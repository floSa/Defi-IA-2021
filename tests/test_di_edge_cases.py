"""Pin the two edge cases of the disparate-impact metric.

These are not bugs to fix — `macro_disparate_impact` mirrors the organisers'
reference notebook and its value is pinned to their published 3.898171170378378.
But one of the two behaviours is exploitable, so it is pinned here explicitly:
if someone "improves" the implementation later, these tests make the change
visible instead of silently altering what the competition score means.
"""

from __future__ import annotations

from defi_ia.evaluation.metrics import count_single_gender_jobs, macro_disparate_impact


def test_job_predicted_for_a_single_gender_scores_as_perfect_parity():
    """The exploitable case: emptying a class of one gender LOWERS the metric."""
    both = macro_disparate_impact(
        ["a"] * 11 + ["b"] * 10,
        ["M"] * 10 + ["F"] * 1 + ["M"] * 5 + ["F"] * 5,
    )
    single = macro_disparate_impact(
        ["a"] * 11 + ["b"] * 10,
        ["M"] * 11 + ["M"] * 5 + ["F"] * 5,
    )
    # a = 10M/1F -> ratio 10, b = 5M/5F -> ratio 1  =>  mean 5.5
    assert both == 5.5
    # a = 11M/0F -> ratio 1 (NaN is skipped), b -> 1  =>  mean 1.0
    assert single == 1.0
    assert single < both, (
        "removing every woman from a job improves this metric — that is the "
        "gaming vector count_single_gender_jobs exists to expose"
    )


def test_job_never_predicted_drops_out_of_the_mean():
    """The harmless case: an unpredicted job simply has no row."""
    di = macro_disparate_impact(["a"] * 11, ["M"] * 10 + ["F"] * 1)
    assert di == 10.0  # only job "a" contributes; "b", "c", ... are absent


def test_count_single_gender_jobs_flags_the_exploitable_case():
    assert count_single_gender_jobs(["a"] * 11, ["M"] * 10 + ["F"] * 1) == 0
    assert count_single_gender_jobs(["a"] * 11, ["M"] * 11) == 1
    assert count_single_gender_jobs(
        ["a"] * 5 + ["b"] * 5, ["M"] * 5 + ["F"] * 5
    ) == 2  # both jobs are single-gender here
