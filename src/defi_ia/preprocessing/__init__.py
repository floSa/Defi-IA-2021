"""Text preprocessing, including gender-signal scrubbing for fairness."""

from defi_ia.preprocessing.text import (
    GENDERED_WORDS,
    basic_clean,
    scrub_gender,
)

__all__ = ["basic_clean", "scrub_gender", "GENDERED_WORDS"]
