"""Text cleaning utilities.

Two independent concerns live here:

* :func:`basic_clean` — light normalisation shared by every model.
* :func:`scrub_gender` — replace explicit gender markers (pronouns, honorifics
  and gendered nouns) with neutral placeholders. This is the single most
  direct lever on the fairness metric: if the classifier cannot read "she" vs
  "he", it cannot easily correlate a job with a gender. It is deliberately a
  *separate, optional* step so we can measure the accuracy/fairness trade-off
  rather than baking it in.

Note: scrubbing pronouns does **not** neutralise first names, which also carry
gender signal. Name handling is left to the modelling plan (e.g. NER masking).
"""

from __future__ import annotations

import re

# Pairs and singletons of explicitly gendered tokens → neutral replacement.
# Kept intentionally small and auditable; extend from data inspection.
GENDERED_WORDS: dict[str, str] = {
    # subject / object / possessive pronouns
    "he": "they", "she": "they",
    "him": "them", "her": "them",
    "his": "their", "hers": "theirs",
    "himself": "themselves", "herself": "themselves",
    # honorifics
    "mr": "mx", "mrs": "mx", "ms": "mx", "miss": "mx",
    # common gendered nouns
    "man": "person", "woman": "person",
    "men": "people", "women": "people",
    "male": "person", "female": "person",
    "gentleman": "person", "lady": "person",
    "father": "parent", "mother": "parent",
    "son": "child", "daughter": "child",
    "husband": "spouse", "wife": "spouse",
    "boy": "child", "girl": "child",
}

_GENDER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in GENDERED_WORDS) + r")\b",
    flags=re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")


def basic_clean(text: str, lower: bool = True) -> str:
    """Trim, collapse whitespace and (optionally) lowercase.

    Transformer tokenizers prefer the original casing, so ``lower`` is exposed
    as a flag rather than always applied.
    """
    text = _WS_RE.sub(" ", text).strip()
    return text.lower() if lower else text


def scrub_gender(text: str) -> str:
    """Replace explicit gendered tokens with neutral equivalents.

    Case-insensitive on the match, but the replacement is always lower-case;
    pair with ``basic_clean(lower=True)`` for classical models.
    """
    return _GENDER_RE.sub(lambda m: GENDERED_WORDS[m.group(0).lower()], text)


# --- Person-name masking (fairness track) -----------------------------------
# First names leak gender just as strongly as pronouns. We mask PERSON spans
# with a neutral placeholder using spaCy NER when available, so the classifier
# can't key on "Brent" vs "Sara". spaCy is an *optional* dependency; without it
# the function is a no-op and logs once, so the core pipeline never breaks.

_NAME_PLACEHOLDER = "person"
_SPACY_NLP = None
_SPACY_TRIED = False


def _get_spacy():
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy

        # Small English model; NER only for speed.
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
    except Exception:  # pragma: no cover - environment dependent
        import warnings

        warnings.warn(
            "spaCy/en_core_web_sm not available; mask_person_names is a no-op. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm",
            RuntimeWarning,
            stacklevel=2,
        )
        _SPACY_NLP = None
    return _SPACY_NLP


def mask_person_names(texts, batch_size: int = 256, n_process: int = 1):
    """Replace PERSON entities with a neutral placeholder.

    Accepts an iterable of strings and returns a list (batched through spaCy's
    ``nlp.pipe`` for throughput). No-op if spaCy is unavailable.
    """
    nlp = _get_spacy()
    texts = list(texts)
    if nlp is None:
        return texts

    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        if not doc.ents:
            out.append(doc.text)
            continue
        pieces, last = [], 0
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                pieces.append(doc.text[last:ent.start_char])
                pieces.append(_NAME_PLACEHOLDER)
                last = ent.end_char
        pieces.append(doc.text[last:])
        out.append("".join(pieces))
    return out
