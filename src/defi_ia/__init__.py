"""Défi IA 2021 — Job classification from biographies.

A modernised, reproducible take on the INSA Toulouse Défi IA 2021 NLP
competition: assign one of 28 job categories to an English-language
biography, optimising Macro-F1 while keeping gender bias (macro disparate
impact) low.

Public API is intentionally small; import submodules explicitly, e.g.::

    from defi_ia.data import load
    from defi_ia.evaluation import metrics
"""

__version__ = "0.1.0"
