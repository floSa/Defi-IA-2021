"""Fine-tune a transformer and evaluate both competition metrics (Step C).

GPU required. Install deps with ``pip install -r requirements-dl.txt``.

Examples
--------
Holdout run on the RTX 4060 Ti (8 GB → smaller batch + grad accum)::

    python scripts/train_transformer.py --model answerdotai/ModernBERT-base \
        --batch-size 8 --grad-accum 2

DeBERTa-v3 comparison::

    python scripts/train_transformer.py --model microsoft/deberta-v3-base

Fit and write a submission + saved logits for the ensemble::

    python scripts/train_transformer.py --full \
        --submit submissions/transformer.csv --save-logits models/transformer_test_logits.npy
"""

from __future__ import annotations

import argparse

import numpy as np

from defi_ia import paths
from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.evaluation.submission import make_submission
from defi_ia.models.transformer import TransformerConfig, fine_tune, predict_logits
from defi_ia.preprocessing.text import basic_clean, scrub_gender


def _prepare(series, scrub: bool):
    # Transformers keep original casing; only normalise whitespace.
    out = series.map(lambda t: basic_clean(t, lower=False))
    return out.map(scrub_gender) if scrub else out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--scrub-gender", action="store_true")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="train on all data")
    p.add_argument("--submit", metavar="PATH", default=None)
    p.add_argument("--save-logits", metavar="PATH", default=None,
                   help="save test-set logits (npy) for the ensemble")
    args = p.parse_args()

    paths.ensure_dirs()
    cfg = TransformerConfig(
        model_name=args.model, max_length=args.max_length, batch_size=args.batch_size,
        grad_accum=args.grad_accum, learning_rate=args.lr, epochs=args.epochs, seed=args.seed,
    )

    train = load_train(with_labels=True)
    train["text"] = _prepare(train["description"], args.scrub_gender)

    if args.full:
        tr, va = train, train.sample(frac=0.02, random_state=args.seed)  # tiny eval for early stop
    else:
        tr, va = stratified_holdout(train, args.valid_size, args.seed)

    trainer, tokenizer = fine_tune(cfg, tr, va)

    if not args.full:
        logits = predict_logits(trainer, tokenizer, va["text"], cfg.max_length)
        pred = np.argmax(logits, axis=-1)
        names = load_categories()
        f1 = macro_f1(va["Category"], pred)
        di = macro_disparate_impact([names[c] for c in pred], va["gender"])
        print("\n=== Validation ===")
        print(f"  model      : {cfg.model_name}")
        print(f"  Macro-F1   : {f1:.4f}")
        print(f"  disparate  : {di:.4f}")

    if args.submit or args.save_logits:
        test = load_test()
        test["text"] = _prepare(test["description"], args.scrub_gender)
        test_logits = predict_logits(trainer, tokenizer, test["text"], cfg.max_length)
        if args.save_logits:
            np.save(args.save_logits, test_logits)
            print(f"  saved logits → {args.save_logits}")
        if args.submit:
            preds = np.argmax(test_logits, axis=-1)
            out = make_submission(test.index, preds, args.submit)
            print(f"  wrote submission → {out}")


if __name__ == "__main__":
    main()
