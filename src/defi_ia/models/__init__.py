"""Model implementations.

Populated once the modelling plan is validated. Intended contents:

* ``tfidf_linear.py`` — TF-IDF + linear classifier (SVM / logistic) baseline
  and strong classical reference.
* ``transformer.py``  — fine-tuned encoder (DeBERTa-v3 / RoBERTa) head.
* ``ensemble.py``     — probability blending across models.

Every model exposes the same minimal interface: ``fit(df)`` /
``predict_proba(df)`` returning a ``(n_samples, 28)`` array aligned with the
integer job ids, so the training and submission orchestration stays uniform.
"""
