"""Dry-run the post-GPU chain on correctly-shaped real data (zero GPU).

`tune_thresholds.py` and `build_ensemble.py` only ever run *after* a multi-hour
fine-tune, which is the worst possible moment to discover a shape mismatch or a
missing file. This script manufactures a run directory with exactly the artifacts
`train_transformer.py` writes — same shapes, same Ids, same dtypes — using the
classical model's scores in place of transformer logits, then leaves the real
post-processing scripts to run against it.

It proves the plumbing, not the science: blending the classical model with
itself says nothing about ensembling. What it does say is that tomorrow's
`tune_thresholds` and `build_ensemble` will not die on a missing column.

    python scripts/dryrun_postprocessing.py
    python scripts/tune_thresholds.py --run-dir models/_dryrun --no-submit
    python scripts/build_ensemble.py  --run-dir models/_dryrun --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from defi_ia.data.load import load_test, load_train
from defi_ia.data.split import stratified_holdout
from defi_ia.io_utils import atomic_save
from defi_ia.preprocessing.text import basic_clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/classical_wordchar_svm.joblib")
    p.add_argument("--out-dir", default="models/_dryrun")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    model = joblib.load(args.model)

    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    # Same split the transformer uses without --full, so the Ids line up exactly
    # the way they will tomorrow.
    _, va = stratified_holdout(train, 0.15, args.seed)

    valid_scores = model.decision_function(va["text"]).astype(np.float32)
    atomic_save(out / "valid_logits.npy", lambda q: np.save(q, valid_scores))
    meta = pd.DataFrame(
        {"Category": va["Category"].to_numpy(), "gender": va["gender"].to_numpy()},
        index=va.index,
    )
    atomic_save(out / "valid_meta.csv", lambda q: meta.to_csv(q, index_label="Id"))

    test = load_test()
    test["text"] = test["description"].map(lambda t: basic_clean(t, lower=True))
    test_scores = model.decision_function(test["text"]).astype(np.float32)
    atomic_save(out / "test_logits.npy", lambda q: np.save(q, test_scores))

    print(f"wrote a transformer-shaped run dir -> {out}/")
    print(f"  valid_logits.npy {valid_scores.shape}   valid_meta.csv {meta.shape}")
    print(f"  test_logits.npy  {test_scores.shape}")
    print("\nnow exercise the real post-processing:")
    print(f"  .venv/bin/python scripts/tune_thresholds.py --run-dir {out} --no-submit")
    print(f"  .venv/bin/python scripts/build_ensemble.py  --run-dir {out} "
          "--out /tmp/dryrun_ensemble.csv")


if __name__ == "__main__":
    main()
