"""Produce a run's artifacts from a saved checkpoint, without training (GPU, minutes).

A fine-tune that runs out of wall-clock leaves checkpoints but no
`valid_logits.npy` / `test_logits.npy` / `metrics.json` — so every downstream
step (threshold tuning, ensemble, per-class comparison, submission) is blocked
even though a perfectly usable model is sitting on disk.

This loads the checkpoint of your choice, runs inference only, and writes the
exact same artifacts `train_transformer.py` would have written at the end. It
takes minutes instead of hours, so an interrupted run still ships.

The written `metrics.json` carries `from_checkpoint` and the checkpoint's epoch,
so nothing downstream can mistake a partially-trained model for a finished one.

    python scripts/artifacts_from_checkpoint.py --run-dir models/roberta_large_6ep
    python scripts/artifacts_from_checkpoint.py --run-dir models/x --checkpoint models/x/checkpoint-11540
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from defi_ia.data.load import load_categories, load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.evaluation.metrics import macro_disparate_impact, macro_f1
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean, scrub_gender


def _best_checkpoint(run: Path) -> tuple[Path, float, float | None]:
    """Pick the checkpoint with the highest recorded eval Macro-F1.

    Falls back to the newest one when no eval history is available. Returns
    (path, epoch, best_eval_macro_f1).
    """
    cands = []
    for c in run.glob("checkpoint-*"):
        state = c / "trainer_state.json"
        if not state.is_file():
            continue
        d = json.loads(state.read_text())
        evals = [h["eval_macro_f1"] for h in d.get("log_history", []) if "eval_macro_f1" in h]
        cands.append((max(evals) if evals else -1.0, d.get("epoch", 0.0), c))
    if not cands:
        raise SystemExit(f"no checkpoint with a trainer_state.json under {run}")
    score, epoch, path = max(cands)
    return path, epoch, (score if score >= 0 else None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="default: best by eval Macro-F1")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scrub-gender", action="store_true")
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run = Path(args.run_dir)
    if args.checkpoint:
        ckpt, epoch, best = Path(args.checkpoint), None, None
    else:
        ckpt, epoch, best = _best_checkpoint(run)
    print(f"checkpoint: {ckpt}" + (f"  (epoch {epoch:.1f}, eval Macro-F1 {best:.4f})"
                                   if best is not None else ""))

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def logits_for(texts) -> np.ndarray:
        ds = Dataset.from_dict({"text": list(texts)}).map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
            batched=True, remove_columns=["text"])
        out = []
        with torch.no_grad():
            for i in range(0, len(ds), args.batch_size):
                batch = ds[i:i + args.batch_size]
                enc = tok.pad({"input_ids": batch["input_ids"],
                               "attention_mask": batch["attention_mask"]},
                              return_tensors="pt").to(device)
                out.append(model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def prep(s):
        out = s.map(lambda t: basic_clean(t, lower=False))
        return out.map(scrub_gender) if args.scrub_gender else out

    train = load_train(with_labels=True)
    train["text"] = prep(train["description"])
    _, va = stratified_holdout(train, args.valid_size, args.seed)

    print(f"scoring {len(va):,} validation rows…")
    val_logits = logits_for(va["text"])
    atomic_save(run / "valid_logits.npy", lambda q: np.save(q, val_logits))
    meta = pd.DataFrame({"Category": va["Category"].to_numpy(),
                         "gender": va["gender"].to_numpy()}, index=va.index)
    atomic_save(run / "valid_meta.csv", lambda q: meta.to_csv(q, index_label="Id"))

    pred = val_logits.argmax(1)
    names = load_categories()
    f1 = macro_f1(va["Category"], pred)
    di = macro_disparate_impact([names[c] for c in pred], va["gender"])
    print(f"  Macro-F1 {f1:.4f} | disparate impact {di:.3f}")

    test = load_test()
    test["text"] = prep(test["description"])
    print(f"scoring {len(test):,} test rows…")
    atomic_save(run / "test_logits.npy", lambda q: np.save(q, logits_for(test["text"])))

    # Carry the provenance so no downstream step can mistake this for a run that
    # trained to completion.
    state = json.loads((ckpt / "trainer_state.json").read_text())
    metrics = {
        "run_name": run.name, "model": str(ckpt), "macro_f1": f1, "disparate_impact": di,
        "n_valid": len(va), "from_checkpoint": str(ckpt),
        "checkpoint_epoch": state.get("epoch"), "smoke": False,
        "max_length": args.max_length, "scrub_gender": args.scrub_gender, "seed": args.seed,
        "log_history": state.get("log_history", []),
        "note": "artifacts rebuilt from a checkpoint by artifacts_from_checkpoint.py; "
                "training did NOT run to its planned epoch budget",
    }
    atomic_save(run / "metrics.json",
                lambda q: q.write_text(json.dumps(metrics, indent=2, default=float)))
    print(f"wrote artifacts -> {run}/")


if __name__ == "__main__":
    main()
