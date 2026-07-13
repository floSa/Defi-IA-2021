"""TF-IDF + linear classifier — the strong classical model (Steps A & B).

Design:

* **Features**: a union of *word* (1–2 grams) and *character* (``char_wb``,
  2–5 grams) TF-IDF. Character n-grams are robust to the noisy, misspelled
  CommonCrawl text and capture morphology (``-ologist``, ``-er``) that maps
  cleanly onto job names.
* **Classifier**: a linear model with ``class_weight="balanced"`` so the 0.4 %
  classes are not drowned out — decisive for Macro-F1. ``linear_svm`` is the
  fastest strong option; ``logistic`` / ``sgd`` are available when calibrated
  probabilities are needed (e.g. for the ensemble in Step D).

The whole thing is a scikit-learn ``Pipeline`` so it serialises with ``joblib``
and exposes a uniform ``fit`` / ``predict`` / ``predict_proba`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass
class TfidfLinearConfig:
    """Hyper-parameters for the classical model (mirrors config.yaml)."""

    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (2, 5)
    use_char: bool = True
    min_df: int = 3
    max_features_word: int | None = 200_000
    max_features_char: int | None = 300_000
    sublinear_tf: bool = True
    classifier: str = "linear_svm"  # linear_svm | logistic | sgd
    C: float = 1.0
    class_weight: str | None = "balanced"
    calibrate: bool = False  # wrap classifier for predict_proba (needed for blends)
    seed: int = 42
    extra: dict = field(default_factory=dict)


def _build_vectorizer(cfg: TfidfLinearConfig) -> FeatureUnion | TfidfVectorizer:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngram_range,
        min_df=cfg.min_df,
        max_features=cfg.max_features_word,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents="unicode",
    )
    if not cfg.use_char:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram_range,
        min_df=cfg.min_df,
        max_features=cfg.max_features_char,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents="unicode",
    )
    return FeatureUnion([("word", word), ("char", char)])


def _build_classifier(cfg: TfidfLinearConfig):
    if cfg.classifier == "linear_svm":
        clf = LinearSVC(C=cfg.C, class_weight=cfg.class_weight, random_state=cfg.seed)
    elif cfg.classifier == "logistic":
        clf = LogisticRegression(
            C=cfg.C,
            class_weight=cfg.class_weight,
            max_iter=1000,
            random_state=cfg.seed,
        )
    elif cfg.classifier == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1.0 / (cfg.C * 1000),
            class_weight=cfg.class_weight,
            random_state=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {cfg.classifier!r}")

    # LinearSVC has no predict_proba; calibrate when probabilities are required.
    if cfg.calibrate and cfg.classifier == "linear_svm":
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    return clf


def build_model(cfg: TfidfLinearConfig | None = None) -> Pipeline:
    """Assemble the full ``Pipeline`` (vectorizer → linear classifier)."""
    cfg = cfg or TfidfLinearConfig()
    return Pipeline(
        [
            ("tfidf", _build_vectorizer(cfg)),
            ("clf", _build_classifier(cfg)),
        ]
    )


def predict_scores(model: Pipeline, texts) -> np.ndarray:
    """Return class scores as a ``(n, 28)`` array.

    Uses ``predict_proba`` when available, else ``decision_function`` — so the
    output can feed an ensemble regardless of the underlying classifier.
    """
    clf = model[-1]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(texts)
    return model.decision_function(texts)
