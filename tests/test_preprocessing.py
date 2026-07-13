"""Tests for text cleaning and gender scrubbing."""

from defi_ia.preprocessing.text import basic_clean, scrub_gender


def test_basic_clean_collapses_whitespace_and_lowers():
    assert basic_clean("  Hello   World  ") == "hello world"


def test_basic_clean_can_preserve_case():
    assert basic_clean("Hello World", lower=False) == "Hello World"


def test_scrub_gender_replaces_pronouns_case_insensitively():
    out = scrub_gender("She is a nurse. He is a surgeon.")
    assert "she" not in out.lower()
    assert "he " not in out.lower()
    assert "they" in out.lower()


def test_scrub_gender_only_touches_whole_words():
    # "theri" / substrings must not be altered; "the" should survive.
    assert scrub_gender("the theatre") == "the theatre"
