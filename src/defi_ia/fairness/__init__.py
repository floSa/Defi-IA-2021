"""Fairness interventions to lower the macro disparate impact.

Populated once the modelling plan is validated. Intended contents:

* ``mitigation.py`` — pre-, in- and post-processing strategies
  (gender scrubbing, threshold adjustment, calibrated re-ranking) plus the
  accuracy/fairness Pareto analysis that decides which to ship.

The measurement of disparate impact itself lives in
:mod:`defi_ia.evaluation.metrics` so training code depends only on metrics.
"""
