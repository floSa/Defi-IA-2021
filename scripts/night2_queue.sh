#!/usr/bin/env bash
# Second unattended GPU queue — closes the two gaps left by the first night.
#
# Gap 1: the fairness-track submission is stuck at classical accuracy (0.752
#        Macro-F1, DI 3.28) because counterfactual training was never tried on a
#        transformer. A counterfactual roberta should dominate it on BOTH axes.
# Gap 2: roberta-large stopped at epoch 3 of 6, still gaining.
#
# Fairness comes first: it is a brand-new deliverable, where finishing
# roberta-large only sharpens a number that is already the project's best.
#
#   bash scripts/night2_queue.sh
#
# Every stage resumes from its newest checkpoint and is skipped once its
# metrics.json exists, so re-running after any interruption is safe.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=reports/night2_queue.log
mkdir -p reports models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

stage() {
  local name=$1; shift
  if [ -f "models/$name/metrics.json" ]; then
    log "SKIP $name (already done)"
    return 0
  fi
  log "START $name"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    local f1 di
    f1=$($PY -c "import json;d=json.load(open('models/$name/metrics.json'));print(f\"{d['macro_f1']:.4f}\")" 2>/dev/null || echo "?")
    di=$($PY -c "import json;d=json.load(open('models/$name/metrics.json'));print(f\"{d['disparate_impact']:.3f}\")" 2>/dev/null || echo "?")
    log "DONE  $name — Macro-F1 $f1, DI $di ($(( (SECONDS-t0)/60 )) min)"
  else
    log "FAIL  $name — see $LOG; continuing"
  fi
}

log "=== night2 queue start ==="

# 1. Counterfactual roberta-base. 3 epochs over the doubled training set is the
#    same gradient budget as 6 epochs over the original, which is what the
#    non-counterfactual roberta-base got — so the two are comparable.
stage roberta_base_counterfactual $PY scripts/train_transformer.py \
  --model roberta-base --run-name roberta_base_counterfactual --counterfactual \
  --batch-size 32 --max-length 192 --epochs 3 --resume

# 2. Fairness-track submission from it, if it trained.
if [ -f models/roberta_base_counterfactual/test_logits.npy ]; then
  $PY - <<'EOF' >>"$LOG" 2>&1 && log "DONE fairness submission" || log "FAIL fairness submission"
import numpy as np
from defi_ia.data.load import load_test
from defi_ia.evaluation.submission import make_submission
logits = np.load("models/roberta_base_counterfactual/test_logits.npy")
out = make_submission(load_test().index, logits.argmax(1),
                      "submissions/roberta_counterfactual_fairness.csv")
print(f"wrote {out}")
EOF
fi

# 3. Finish roberta-large from where it stopped (epoch 3 of 6).
stage roberta_large_6ep_done $PY scripts/train_transformer.py \
  --model roberta-large --run-name roberta_large_6ep \
  --batch-size 16 --grad-accum 2 --max-length 192 --epochs 6 --lr 1e-5 --resume

log "--- convergence check ---"
$PY scripts/check_convergence.py 2>&1 | tee -a "$LOG" | tail -25

log "--- rebuilding the accuracy submission from the best run ---"
$PY scripts/final_submission.py --run-dir models/roberta_large_6ep \
    --out submissions/final_accuracy_track.csv >>"$LOG" 2>&1 \
    && log "DONE accuracy submission" || log "FAIL accuracy submission"

log "=== night2 queue done ==="
ls -la submissions/*.csv | tee -a "$LOG"
