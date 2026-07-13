"""Feature extraction.

Populated once the modelling plan is validated. Intended contents:

* ``vectorizers.py`` — configurable TF-IDF (word + char n-grams) factory.
* ``embeddings.py``  — sentence-embedding extractors for classical heads.

Deep-learning models tokenise text internally and bypass this package.
"""
