"""Text data augmentation — the "clever engineering" layer, no external data.

Three augmenters, composable, all operating only on the provided biographies:

1. `gender_counterfactual` — swap gendered tokens (he<->she, his<->her, mr<->mrs,
   man<->woman, ...). This is doubly useful: it *paraphrases* (augmentation) and
   it teaches the model that the job is independent of gender (directly lowers
   the disparate-impact metric). One clean idea that serves both tracks.

2. `eda` — a dependency-free subset of Easy Data Augmentation: random word
   swap and random word deletion. Cheap paraphrases that regularise the encoder.

3. `augment_rare_classes` — the macro-F1 lever: rare classes (rapper 0.4%,
   dj 0.4%, ...) are the ones dragging macro-F1 down, so we synthesise extra
   examples *for those classes only*, up to a target count, instead of blindly
   augmenting the whole set (which would just amplify `professor`).

Deterministic given a seed (no global RNG) so runs stay reproducible.
"""

from __future__ import annotations

import random
import re

import pandas as pd

# Bidirectional gender swaps (extends the one-way scrub map in text.py).
_SWAP = {
    "he": "she", "she": "he",
    "him": "her", "his": "her", "her": "his",
    "himself": "herself", "herself": "himself",
    "mr": "mrs", "mrs": "mr", "ms": "mr",
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "male": "female", "female": "male",
    "father": "mother", "mother": "father",
    "son": "daughter", "daughter": "son",
    "husband": "wife", "wife": "husband",
    "boy": "girl", "girl": "boy",
    "gentleman": "lady", "lady": "gentleman",
    "sir": "madam", "madam": "sir",
}
_SWAP_RE = re.compile(r"\b(" + "|".join(map(re.escape, _SWAP)) + r")\b", re.IGNORECASE)

# "her" is ambiguous (object vs possessive); the mapping above picks one
# direction consistently, which is fine for augmentation.


def _match_case(src: str, repl: str) -> str:
    if src.istitle():
        return repl.title()
    if src.isupper():
        return repl.upper()
    return repl


def gender_counterfactual(text: str) -> str:
    """Return the gender-swapped counterpart of `text`."""
    return _SWAP_RE.sub(lambda m: _match_case(m.group(0), _SWAP[m.group(0).lower()]), text)


def eda(text: str, rng: random.Random, p_swap: float = 0.1, p_del: float = 0.1) -> str:
    """Random adjacent-word swaps and random word deletions (dependency-free)."""
    words = text.split()
    if len(words) < 4:
        return text
    # random deletions
    words = [w for w in words if rng.random() > p_del] or words
    # random adjacent swaps
    n_swaps = max(1, int(len(words) * p_swap))
    for _ in range(n_swaps):
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_rare_classes(
    df: pd.DataFrame,
    target: int = 3000,
    text_col: str = "description",
    label_col: str = "Category",
    seed: int = 42,
    use_counterfactual: bool = True,
    use_eda: bool = True,
) -> pd.DataFrame:
    """Synthesise extra rows for under-represented classes up to `target` each.

    Returns the original df with augmented rows appended (new negative Ids so the
    index stays unique). Only classes below `target` are augmented.
    """
    rng = random.Random(seed)
    counts = df[label_col].value_counts()
    new_rows = []
    next_id = -1

    for label, n in counts.items():
        if n >= target:
            continue
        pool = df[df[label_col] == label]
        needed = target - n
        pool_records = pool.to_dict("records")
        for i in range(needed):
            base = pool_records[i % len(pool_records)].copy()
            txt = base[text_col]
            # Alternate the augmentation strategy for variety.
            strat = i % 3
            if strat == 0 and use_counterfactual:
                txt = gender_counterfactual(txt)
            elif strat == 1 and use_eda:
                txt = eda(txt, rng)
            elif use_counterfactual and use_eda:
                txt = eda(gender_counterfactual(txt), rng)
            base[text_col] = txt
            new_rows.append((next_id, base))
            next_id -= 1

    if not new_rows:
        return df.copy()

    aug = pd.DataFrame([r for _, r in new_rows], index=[i for i, _ in new_rows])
    aug.index.name = df.index.name
    return pd.concat([df, aug])
