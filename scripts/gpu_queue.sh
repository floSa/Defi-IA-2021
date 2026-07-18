#!/usr/bin/env bash
# Unattended GPU run queue — fires the whole Step-C sequence back to back.
#
# The GPU is shared until ~09:30; this script waits for it to be genuinely idle
# before starting, then runs every experiment in order without needing a
# decision in between. Each stage appends to reports/gpu_queue.log and its own
# run dir under models/, so a crash costs one stage, not the night.
#
#   bash scripts/gpu_queue.sh              # wait for a free GPU, then run
#   bash scripts/gpu_queue.sh --now        # skip the wait
#
# Every stage is resumable: re-running the script skips stages whose
# metrics.json already exists, and --resume picks a killed fit back up from its
# newest checkpoint.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=reports/gpu_queue.log
mkdir -p reports models

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# --- wait for the other job to release the GPU -------------------------------
# Utilisation, not free VRAM, is the binding constraint: 7 GB can be free while
# the card is at 100 % compute, and starting there halves both jobs' throughput.
wait_for_gpu() {
  # Require the card to stay quiet across several consecutive checks. A single
  # reading below the threshold is not enough: the neighbouring job dips to near
  # zero between epochs and while it saves checkpoints, and starting in one of
  # those gaps would put both jobs back in contention.
  # Two signals, both required:
  #   utilisation < 20 %  — the card is not computing;
  #   VRAM < 1500 MiB     — no training process is even resident. This is the
  #                         stronger of the two: a job merely pausing between
  #                         epochs still holds gigabytes, so low memory means it
  #                         actually exited rather than caught its breath.
  # Eight consecutive readings = 16 minutes of genuine silence. Starting too
  # early halves the throughput of BOTH jobs; starting late only costs idle time,
  # and there is plenty of it. The asymmetry justifies being conservative.
  local needed=8 interval=120 quiet=0 util mem
  log "waiting for GPU util <20% AND VRAM <1500MiB, ${needed} consecutive checks ${interval}s apart…"
  while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 20 ] && [ "$mem" -lt 1500 ]; then
      quiet=$((quiet + 1))
      log "  quiet reading ${quiet}/${needed} (util ${util}%, ${mem}MiB)"
      if [ "$quiet" -ge "$needed" ]; then
        log "GPU free — starting"
        return 0
      fi
    elif [ "$quiet" -gt 0 ]; then
      log "  busy again (util ${util:-?}%, ${mem:-?}MiB) — resetting the quiet counter"
      quiet=0
    fi
    sleep "$interval"
  done
}

# --- one stage ---------------------------------------------------------------
# Skips itself if already finished, so re-invoking the queue is always safe.
stage() {
  local name=$1; shift
  if [ -f "models/$name/metrics.json" ]; then
    log "SKIP $name (metrics.json already present)"
    return 0
  fi
  log "START $name"
  if "$@" >>"$LOG" 2>&1; then
    local f1
    f1=$($PY -c "import json;print(f\"{json.load(open('models/$name/metrics.json'))['macro_f1']:.4f}\")" 2>/dev/null || echo "?")
    log "DONE  $name — Macro-F1 $f1"
  else
    log "FAIL  $name (see $LOG) — continuing to the next stage"
  fi
}

[ "${1:-}" = "--now" ] || wait_for_gpu

log "=== GPU queue start ==="
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader | tee -a "$LOG"

# 1. Reproduce the known-good baseline first. If this does not land near 0.80,
#    something in the stack changed and every later number is suspect — that is
#    exactly the check that was missing when the 2020 project published a
#    retracted comparison.
stage roberta_base_repro $PY scripts/train_transformer.py \
  --model roberta-base --run-name roberta_base_repro \
  --batch-size 32 --max-length 192 --epochs 3 --resume

# 2. The single biggest expected jump (ROADMAP §3 tier 3). 16 GB fits it at
#    batch 16 in fp16; grad-accum keeps the effective batch at 32.
stage roberta_large $PY scripts/train_transformer.py \
  --model roberta-large --run-name roberta_large \
  --batch-size 16 --grad-accum 2 --max-length 192 --epochs 3 --lr 1e-5 --resume

# 3. DeBERTa-v3 in bf16 — the open question from the Kaggle era. The T4 had no
#    bf16, which is the leading suspect for its fp32/fp16 NaN divergence; Ada
#    does. Low LR + bf16 is the stabilisation recipe worth one shot.
stage deberta_v3_bf16 $PY scripts/train_transformer.py \
  --model microsoft/deberta-v3-base --run-name deberta_v3_bf16 \
  --batch-size 16 --grad-accum 2 --max-length 192 --epochs 3 --lr 5e-6 --bf16 --resume

# 4. Before ranking anything: did these runs actually finish learning? A model
#    whose best epoch is its last was still improving, so comparing it at a fixed
#    epoch budget measures the budget rather than the model. roberta-base was
#    already "still rising at epoch 2" on Kaggle, and roberta-large needs more
#    epochs than roberta-base, so this is a live risk here, not a formality.
log "--- convergence check ---"
$PY scripts/check_convergence.py 2>&1 | tee -a "$LOG" | tail -20

# 5. Zero-GPU post-processing on whichever backbone won.
best=$($PY - <<'EOF'
import json, pathlib
runs = []
for m in pathlib.Path("models").glob("*/metrics.json"):
    try:
        d = json.loads(m.read_text())
        if not d.get("smoke"):
            runs.append((d["macro_f1"], m.parent.name))
    except Exception:
        pass
print(max(runs)[1] if runs else "")
EOF
)
if [ -n "$best" ]; then
  log "best backbone: $best — running threshold tuning + ensemble on it"

  $PY scripts/tune_thresholds.py --run-dir "models/$best" \
      --out "submissions/${best}_tuned.csv" >>"$LOG" 2>&1 \
      && log "DONE  thresholds on $best" || log "FAIL  thresholds on $best"

  # The ensemble needs a classical model blind to the transformer's validation
  # rows. train_classical.py without --full and train_transformer.py without
  # --full both use stratified_holdout(0.15, seed 42), so the splits line up.
  # build_ensemble.py verifies this itself and refuses to run if they do not.
  if [ -f models/classical_wordchar_svm.joblib ]; then
    $PY scripts/build_ensemble.py --run-dir "models/$best" \
        --classical-model models/classical_wordchar_svm.joblib \
        --out "submissions/${best}_ensemble.csv" >>"$LOG" 2>&1 \
        && log "DONE  ensemble on $best" || log "FAIL  ensemble on $best (see $LOG)"
    # Tests the falsifiable prediction from reports/error_analysis.md: the
    # transformer's gain should concentrate in the classes the bag-of-words
    # confuses with their neighbours. If it does, the ensemble has real diversity
    # to exploit; if it does not, the models are redundant and the blend will
    # buy less than the headline gap suggests.
    $PY scripts/compare_per_class.py --run-dir "models/$best" \
        --classical-model models/classical_wordchar_svm.joblib >>"$LOG" 2>&1 \
        && log "DONE  per-class comparison" || log "FAIL  per-class comparison"
  else
    log "SKIP ensemble + per-class comparison (no classical model)"
  fi
else
  log "no completed run found — nothing to post-process"
fi

log "=== GPU queue done ==="
