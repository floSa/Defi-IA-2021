"""Centralised, environment-agnostic path configuration.

Every module resolves file locations through this single source of truth so
the code runs unchanged on a laptop, in WSL, or on a Kaggle/Colab kernel.

Override the project root with the ``DEFI_IA_ROOT`` environment variable
(useful on Kaggle where inputs live under ``/kaggle/input``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ``paths.py`` lives at ``<root>/src/defi_ia/paths.py`` → root is 3 levels up.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(os.environ.get("DEFI_IA_ROOT", _DEFAULT_ROOT))

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"

# Raw competition files (as extracted from the Kaggle zip).
TRAIN_JSON = DATA_RAW / "train.json"
TEST_JSON = DATA_RAW / "test.json"
TRAIN_LABEL = DATA_RAW / "train_label.csv"
CATEGORIES = DATA_RAW / "categories_string.csv"
SUBMISSION_TEMPLATE = DATA_RAW / "template_submissions.csv"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist yet."""
    for path in (DATA_INTERIM, DATA_PROCESSED, MODELS, REPORTS, SUBMISSIONS):
        path.mkdir(parents=True, exist_ok=True)
