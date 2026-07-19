"""Fine-tune a transformer and evaluate both competition metrics (Step C).

Run artifacts land in ``models/<run-name>/`` so nothing collides between runs:

    valid_logits.npy   holdout logits  -> feeds scripts/tune_thresholds.py
    valid_meta.csv     Id, Category, gender for the same rows
    test_logits.npy    test logits     -> feeds scripts/build_ensemble.py
    metrics.json       Macro-F1 + disparate impact + the run's config
    submission.csv     when --submit is given

Examples
--------
Cheap end-to-end check on CPU — exercises every step, proves nothing about
accuracy, and is the thing to run before any long GPU job::

    python scripts/train_transformer.py --smoke

Holdout run on the RTX 4060 Ti (proven-stable roberta-base config)::

    python scripts/train_transformer.py --run-name roberta_holdout \
        --batch-size 32 --max-length 192 --epochs 3

Resume a run that was killed::

    python scripts/train_transformer.py --run-name roberta_holdout --resume

Full-data fit + submission (keeps a real held-out slice for checkpoint choice)::

    python scripts/train_transformer.py --run-name roberta_full --full \
        --submit submissions/roberta_full.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from defi_ia import paths
from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.evaluation.submission import make_submission
from defi_ia.io_utils import atomic_save
from defi_ia.models.transformer import TransformerConfig, fine_tune, predict_logits
from defi_ia.preprocessing.augment import gender_counterfactual
from defi_ia.preprocessing.text import basic_clean, scrub_gender


def _prepare(series, scrub: bool):
    # Transformers keep original casing; only normalise whitespace.
    out = series.map(lambda t: basic_clean(t, lower=False))
    return out.map(scrub_gender) if scrub else out


_atomic_save = atomic_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # roberta-base is the proven-stable backbone; DeBERTa-v3 NaN-diverged on the
    # previous stack (reports/ROADMAP.md §2). Keep the default in sync with the
    # config dataclass so a bare invocation runs the config we trust.
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--run-name", default=None,
                   help="artifact dir under models/ (default: derived from --model)")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bf16", action="store_true",
                   help="bf16 instead of fp16 (Ada supports it; try this if a model diverges)")
    p.add_argument("--scrub-gender", action="store_true")
    p.add_argument("--counterfactual", action="store_true",
                   help="train on each bio AND its gender-swapped twin (fairness track). "
                        "Best DI lever measured on the classical model (3.28 vs 3.83); "
                        "untested on a transformer, which is the point of running it.")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full", action="store_true", help="train on all data (minus a small holdout)")
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    p.add_argument("--submit", metavar="PATH", default=None)
    p.add_argument("--cpu", action="store_true", help="force CPU (smoke tests only)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run through every step: train, checkpoint, eval, "
                        "logits, submission. Catches path/IO bugs before a long GPU job.")
    args = p.parse_args()

    paths.ensure_dirs()

    run_name = args.run_name or args.model.split("/")[-1].replace(".", "_")
    if args.smoke:
        run_name = f"smoke_{run_name}"
    out_dir = Path("models") / run_name

    epochs, max_length = args.epochs, args.max_length
    batch_size, use_cpu = args.batch_size, args.cpu
    if args.smoke:
        # Small enough to finish on CPU in a few minutes; 2 epochs so the
        # per-epoch eval/checkpoint/early-stop path is genuinely exercised.
        epochs, max_length, batch_size, use_cpu = 2, 48, 32, True
        print("[smoke] tiny CPU run — verifies the pipeline, not the science\n")

    cfg = TransformerConfig(
        model_name=args.model, max_length=max_length, batch_size=batch_size,
        grad_accum=args.grad_accum, learning_rate=args.lr, epochs=epochs, seed=args.seed,
        fp16=not args.bf16, bf16=args.bf16, use_cpu=use_cpu,
        output_dir=str(out_dir),
    )

    train = load_train(with_labels=True)
    train["text"] = _prepare(train["description"], args.scrub_gender)
    if args.smoke:
        train = train.sample(n=3_000, random_state=args.seed)

    if args.full:
        # A 3 % slice is EXCLUDED from training and used only to pick the best
        # checkpoint. The previous version sampled the eval set from the training
        # rows, so checkpoint selection scored the model on data it had already
        # seen and systematically favoured the most overfit checkpoint.
        tr, va = stratified_holdout(train, 0.03, args.seed)
    else:
        tr, va = stratified_holdout(train, args.valid_size, args.seed)

    if args.counterfactual:
        # Append the gender-swapped twin of every TRAINING row, label unchanged,
        # so the job becomes gender-invariant by construction. The validation
        # side is deliberately left alone: it must stay the distribution the
        # competition actually scores, or the DI would be flattering and false.
        swapped = tr.copy()
        swapped["text"] = swapped["text"].map(gender_counterfactual)
        tr = pd.concat([tr, swapped])
        print(f"  counterfactual: training set doubled to {len(tr):,}")

    print(f"run '{run_name}': train {len(tr):,} / valid {len(va):,} -> {out_dir}")
    trainer, tokenizer = fine_tune(cfg, tr, va, resume=args.resume)

    # Holdout logits are saved ALWAYS (not just on --full): scripts/tune_thresholds.py
    # needs valid_logits.npy + valid_meta.csv, and re-running a GPU fit just to
    # recover them would be pure waste.
    val_logits = predict_logits(trainer, tokenizer, va["text"], cfg.max_length)
    _atomic_save(out_dir / "valid_logits.npy", lambda p: np.save(p, val_logits))
    meta = pd.DataFrame(
        {"Category": va["Category"].to_numpy(), "gender": va["gender"].to_numpy()},
        index=va.index,
    )
    _atomic_save(out_dir / "valid_meta.csv", lambda p: meta.to_csv(p, index_label="Id"))

    pred = np.argmax(val_logits, axis=-1)
    names = load_categories()
    f1 = macro_f1(va["Category"], pred)
    di = macro_disparate_impact([names[c] for c in pred], va["gender"])
    print("\n=== Validation ===")
    print(f"  model      : {cfg.model_name}")
    print(f"  Macro-F1   : {f1:.4f}")
    print(f"  disparate  : {di:.4f}  (labels' own = 3.898, target -> 1.0)")

    metrics = {
        "run_name": run_name, "model": cfg.model_name, "macro_f1": f1, "disparate_impact": di,
        "n_train": len(tr), "n_valid": len(va), "full": args.full, "smoke": args.smoke,
        "epochs": cfg.epochs, "max_length": cfg.max_length, "batch_size": cfg.batch_size,
        "lr": cfg.learning_rate, "bf16": cfg.bf16, "fp16": cfg.fp16,
        "scrub_gender": args.scrub_gender, "seed": args.seed,
        "log_history": trainer.state.log_history,
    }
    _atomic_save(out_dir / "metrics.json",
                 lambda p: Path(p).write_text(json.dumps(metrics, indent=2, default=float)))

    test = load_test()
    if args.smoke:
        test = test.head(200)
    test["text"] = _prepare(test["description"], args.scrub_gender)
    test_logits = predict_logits(trainer, tokenizer, test["text"], cfg.max_length)
    _atomic_save(out_dir / "test_logits.npy", lambda p: np.save(p, test_logits))
    print(f"  saved logits + metrics -> {out_dir}/")

    if args.submit:
        # A smoke run only scores 200 rows, so its CSV would fail template
        # validation; write it inside the run dir instead of submissions/.
        target = out_dir / "submission_smoke.csv" if args.smoke else Path(args.submit)
        preds = np.argmax(test_logits, axis=-1)
        if args.smoke:
            pd.DataFrame({"Id": test.index, "Category": preds}).to_csv(target, index=False)
            print(f"  [smoke] wrote {target} (200 rows, not a valid submission)")
        else:
            print(f"  wrote submission -> {make_submission(test.index, preds, target)}")


if __name__ == "__main__":
    main()
