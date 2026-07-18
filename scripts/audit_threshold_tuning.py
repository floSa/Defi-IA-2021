"""Honest, nested-split audit of the per-class threshold-tuning gain.

WHY THIS EXISTS
---------------
``scripts/tune_thresholds.py`` and ``scripts/ablation_classical.py`` both learn
the per-class bias on the holdout and then report the tuned Macro-F1 **on that
same holdout** (tune_thresholds.py:68-69, ablation_classical.py:87-88). The
coordinate ascent has 28 free parameters searched over a 33-point grid for up to
12 rounds — a lot of fitting capacity aimed at one split. A score measured on the
data that selected the parameters is optimistic by construction, so the reported
"+0.8 pt free lever" cannot be taken at face value.

This script measures the gain that actually **generalises**, using a three-way
stratified split:

    fit   (70 %)  -> trains the classical model
    calib (15 %)  -> the ONLY data the bias search may look at
    eval  (15 %)  -> never seen by the model or the tuner; the honest judge

and reports side by side:

    argmax                 baseline on eval
    tuned (calib -> eval)  the honest, generalising gain
    tuned (eval  -> eval)  the in-sample number, reproducing the old methodology

The gap between the last two is the optimism the previous figures carried.

Several seeds are run so the delta can be read against its own noise: a +0.008
"gain" means nothing if the seed-to-seed spread is +/- 0.01.

Usage
-----
    python scripts/audit_threshold_tuning.py --smoke      # 2 min, full code path
    python scripts/audit_threshold_tuning.py              # real run, 3 seeds
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from defi_ia.data.load import load_categories, load_train
from defi_ia.evaluation.metrics import macro_disparate_impact
from defi_ia.models.tfidf_linear import TfidfLinearConfig, build_model
from defi_ia.preprocessing.text import basic_clean

N_CLASSES = 28
RESULTS_PATH = Path("reports/threshold_audit.json")


def tune_bias(logp: np.ndarray, y: np.ndarray, n_classes: int = N_CLASSES,
              rounds: int = 12) -> np.ndarray:
    """Per-class additive bias by coordinate ascent on Macro-F1.

    Deliberately identical in semantics to ``tune_thresholds.tune_bias`` — the
    point is to audit *that* procedure, not a different one.
    """
    bias = np.zeros(n_classes)
    grid = np.linspace(-4.0, 4.0, 33)

    def mf1(b: np.ndarray) -> float:
        return f1_score(y, (logp + b).argmax(1), average="macro")

    cur = mf1(bias)
    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            best_v, best_f = bias[c], mf1(bias)
            for v in grid:
                bias[c] = v
                f = mf1(bias)
                if f > best_f:
                    best_f, best_v = f, v
            bias[c] = best_v
            if best_f > cur + 1e-9:
                cur, improved = best_f, True
        if not improved:
            break
    return bias


def three_way_split(df: pd.DataFrame, seed: int, calib_size: float = 0.15,
                    eval_size: float = 0.15):
    """Stratified fit / calib / eval split.

    Stratification matters: the rarest class is 0.4 % of the data and Macro-F1
    weights it like the 32 % one, so a class missing from a fold would make the
    metric jump for reasons unrelated to the technique under test.
    """
    rest, eval_df = train_test_split(
        df, test_size=eval_size, random_state=seed, stratify=df["Category"]
    )
    # calib_size is expressed as a fraction of the ORIGINAL frame.
    calib_frac = calib_size / (1.0 - eval_size)
    fit_df, calib_df = train_test_split(
        rest, test_size=calib_frac, random_state=seed, stratify=rest["Category"]
    )
    return fit_df.copy(), calib_df.copy(), eval_df.copy()


def _di(pred: np.ndarray, gender: pd.Series, names: dict[int, str]) -> float:
    return macro_disparate_impact([names[c] for c in pred], gender)


def run_seed(train: pd.DataFrame, seed: int, names: dict[int, str]) -> dict:
    """One full fit + honest-vs-in-sample comparison for a single seed."""
    fit_df, calib_df, eval_df = three_way_split(train, seed)
    print(f"  split: fit {len(fit_df):,} / calib {len(calib_df):,} / eval {len(eval_df):,}")

    model = build_model(TfidfLinearConfig(seed=seed))
    t0 = time.time()
    model.fit(fit_df["text"], fit_df["Category"])
    fit_s = time.time() - t0
    print(f"  fitted in {fit_s:.0f}s")

    # LinearSVC margins; log_softmax makes the additive bias act on a
    # comparable scale, exactly as the original scripts do.
    logp_calib = log_softmax(model.decision_function(calib_df["text"]), axis=1)
    logp_eval = log_softmax(model.decision_function(eval_df["text"]), axis=1)
    y_calib = calib_df["Category"].to_numpy()
    y_eval = eval_df["Category"].to_numpy()

    base = f1_score(y_eval, logp_eval.argmax(1), average="macro")

    # HONEST: the tuner only ever sees calib.
    t0 = time.time()
    bias_honest = tune_bias(logp_calib, y_calib)
    tune_s = time.time() - t0
    honest = f1_score(y_eval, (logp_eval + bias_honest).argmax(1), average="macro")
    # What the tuner believed it had achieved, on its own calibration data.
    calib_self = f1_score(y_calib, (logp_calib + bias_honest).argmax(1), average="macro")
    calib_base = f1_score(y_calib, logp_calib.argmax(1), average="macro")

    # IN-SAMPLE: reproduce the old methodology (tune on eval, score on eval).
    bias_insample = tune_bias(logp_eval, y_eval)
    insample = f1_score(y_eval, (logp_eval + bias_insample).argmax(1), average="macro")

    res = {
        "seed": seed,
        "n_fit": len(fit_df), "n_calib": len(calib_df), "n_eval": len(eval_df),
        "fit_seconds": round(fit_s, 1), "tune_seconds": round(tune_s, 1),
        "eval_argmax_f1": base,
        "eval_tuned_honest_f1": honest,
        "eval_tuned_insample_f1": insample,
        "calib_argmax_f1": calib_base,
        "calib_tuned_f1": calib_self,
        "delta_honest": honest - base,
        "delta_insample": insample - base,
        "delta_calib_selfreported": calib_self - calib_base,
        "eval_argmax_di": _di(logp_eval.argmax(1), eval_df["gender"], names),
        "eval_tuned_honest_di": _di((logp_eval + bias_honest).argmax(1),
                                    eval_df["gender"], names),
        "bias_honest": bias_honest.tolist(),
    }
    print(f"  argmax {base:.4f} | honest {honest:.4f} ({honest - base:+.4f}) "
          f"| in-sample {insample:.4f} ({insample - base:+.4f})")
    return res


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write via a temp file + rename so a kill mid-write cannot corrupt results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    p.add_argument("--subsample", type=int, default=None,
                   help="rows to subsample (default: all 217k)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run: 8k rows, 1 seed, exercises every step")
    p.add_argument("--out", default=str(RESULTS_PATH))
    args = p.parse_args()

    seeds = args.seeds
    subsample = args.subsample
    if args.smoke:
        seeds, subsample = [42], 8_000
        print("[smoke] 8k rows, 1 seed — verifying the whole path, not the science\n")

    train = load_train(with_labels=True)
    train["text"] = train["description"].map(lambda t: basic_clean(t, lower=True))
    if subsample:
        train = train.sample(n=min(subsample, len(train)), random_state=42)
    names = load_categories()
    print(f"loaded {len(train):,} rows\n")

    out_path = Path(args.out)
    # Resume: keep any seed already computed in a previous (possibly killed) run.
    payload: dict = {"smoke": args.smoke, "n_rows": len(train), "runs": []}
    if out_path.exists() and not args.smoke:
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("n_rows") == len(train) and not prev.get("smoke"):
                payload = prev
                print(f"resuming: {len(payload['runs'])} seed(s) already done")
        except (json.JSONDecodeError, KeyError):
            pass

    done = {r["seed"] for r in payload["runs"]}
    for seed in seeds:
        if seed in done:
            print(f"seed {seed}: cached, skipping")
            continue
        print(f"seed {seed}:")
        payload["runs"].append(run_seed(train, seed, names))
        # Checkpoint after every seed — a 30 min run must never lose finished work.
        _atomic_write_json(out_path, payload)
        print(f"  checkpointed -> {out_path}\n")

    runs = payload["runs"]
    dh = np.array([r["delta_honest"] for r in runs])
    di_ = np.array([r["delta_insample"] for r in runs])
    base = np.array([r["eval_argmax_f1"] for r in runs])

    print("=" * 72)
    print("PER-CLASS THRESHOLD TUNING — honest vs in-sample (Macro-F1 on eval)")
    print("=" * 72)
    print(f"  baseline argmax          {base.mean():.4f}  (sd {base.std():.4f})")
    print(f"  gain, tuned on calib     {dh.mean():+.4f}  (sd {dh.std():.4f})  <- REAL")
    print(f"  gain, tuned on eval      {di_.mean():+.4f}  (sd {di_.std():.4f})  <- optimistic")
    print(f"  optimism of old method   {(di_ - dh).mean():+.4f}")
    print(f"\n  results -> {out_path}")


if __name__ == "__main__":
    main()
