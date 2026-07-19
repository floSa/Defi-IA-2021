#!/usr/bin/env bash
# Stop training at a deadline and ship whatever the best checkpoint gives.
#
# roberta-large sustains ~1.30 s/step after a resume, so its remaining epochs do
# not fit before the GPU has to be handed back. Rather than let it run into a
# wall and produce nothing, this waits for the next epoch checkpoint (or the
# deadline, whichever comes first), stops training, rebuilds the artifacts from
# the best checkpoint, and runs the whole post-processing chain.
#
#   bash scripts/finish_by_deadline.sh [RUN_DIR] [DEADLINE_HHMM]
#
# Defaults: models/roberta_large_6ep, 08:00.

set -uo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${1:-models/roberta_large_6ep}
DEADLINE=${2:-0800}
PY=.venv/bin/python
LOG=reports/gpu_queue.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# The step at which the next epoch boundary lands; a checkpoint appears there.
TARGET_CKPT=${TARGET_CKPT:-17310}

log "=== watchdog: stop at checkpoint-$TARGET_CKPT or $DEADLINE, then ship ==="

while true; do
  if [ -d "$RUN_DIR/checkpoint-$TARGET_CKPT" ]; then
    log "checkpoint-$TARGET_CKPT written — stopping training"
    break
  fi
  if [ "$(date +%H%M)" -ge "$DEADLINE" ] && [ "$(date +%H)" -lt 12 ]; then
    log "deadline $DEADLINE reached — stopping training at whatever is saved"
    break
  fi
  if ! pgrep -f "train_transformer.py --model roberta-large" >/dev/null; then
    log "training process is gone — proceeding with what is on disk"
    break
  fi
  sleep 60
done

# Let the checkpoint finish being written before pulling the rug.
sleep 20
pkill -f "bash scripts/gpu_queue.sh" 2>/dev/null
pkill -9 -f "train_transformer.py --model roberta-large" 2>/dev/null
sleep 10
log "training stopped; VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# 1. Rebuild the artifacts a completed run would have written.
log "--- rebuilding artifacts from the best checkpoint ---"
if $PY scripts/artifacts_from_checkpoint.py --run-dir "$RUN_DIR" --batch-size 16 >>"$LOG" 2>&1; then
  log "DONE artifacts for $RUN_DIR"
else
  log "FAIL artifacts for $RUN_DIR — post-processing will use roberta_base_6ep only"
fi

# 2. Convergence check across every run, so the ranking states plainly which
#    scores are lower bounds. roberta-large will be flagged: it got 3 epochs
#    against roberta-base's 6.
log "--- convergence check ---"
$PY scripts/check_convergence.py 2>&1 | tee -a "$LOG" | tail -20

# 3. Post-processing on the best run that actually has artifacts.
best=$($PY - <<'EOF'
import json, pathlib
best, name = -1, ""
for m in pathlib.Path("models").glob("*/metrics.json"):
    try:
        d = json.loads(m.read_text())
    except Exception:
        continue
    if d.get("smoke") or not (m.parent / "test_logits.npy").exists():
        continue
    if d["macro_f1"] > best:
        best, name = d["macro_f1"], m.parent.name
print(name)
EOF
)

if [ -z "$best" ]; then
  log "no run has usable artifacts — nothing to post-process"
  exit 1
fi
log "best usable run: $best"

$PY scripts/tune_thresholds.py --run-dir "models/$best" \
    --out "submissions/${best}_tuned.csv" >>"$LOG" 2>&1 \
    && log "DONE thresholds" || log "FAIL thresholds"

$PY scripts/build_ensemble.py --run-dir "models/$best" \
    --classical-model models/classical_wordchar_svm.joblib \
    --out "submissions/${best}_ensemble.csv" >>"$LOG" 2>&1 \
    && log "DONE ensemble" || log "FAIL ensemble"

$PY scripts/compare_per_class.py --run-dir "models/$best" \
    --classical-model models/classical_wordchar_svm.joblib >>"$LOG" 2>&1 \
    && log "DONE per-class comparison" || log "FAIL per-class comparison"

log "=== watchdog done — submissions in submissions/, reports in reports/ ==="
ls -la submissions/*.csv | tee -a "$LOG"
