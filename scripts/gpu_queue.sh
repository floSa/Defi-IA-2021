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

# --- wall-clock deadline -----------------------------------------------------
# No NEW training stage starts after this time, so the post-processing (threshold
# tuning, ensemble, per-class comparison) always gets to run and there are
# deliverables in the morning even if a fit overruns. A stage already running is
# left alone — killing it would waste everything it has done.
DEADLINE=${DEADLINE:-08:00}
past_deadline() {
  [ "$(date +%H%M)" -ge "$(echo "$DEADLINE" | tr -d ':')" ] && \
  [ "$(date +%H)" -lt 12 ]   # only meaningful in the morning half of the night
}

# --- one stage ---------------------------------------------------------------
# Skips itself if already finished, so re-invoking the queue is always safe.
stage() {
  local name=$1; shift
  if [ -f "models/$name/metrics.json" ]; then
    log "SKIP $name (metrics.json already present)"
    return 0
  fi
  if past_deadline; then
    log "SKIP $name — past the ${DEADLINE} deadline; going straight to post-processing"
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

# EPOCH BUDGET — 3 was not enough, measured, not assumed.
# roberta-base at 3 epochs went 0.7462 -> 0.7916 -> 0.7978 and was STILL rising
# (+0.0062 on its last epoch), so its 0.7978 is a lower bound and cannot be
# ranked against anything. 6 epochs with early stopping (patience 2) lets each
# model stop at ITS OWN plateau instead of at a shared arbitrary budget — which
# is the only way a bigger model that needs more epochs gets a fair hearing.
EPOCHS=${EPOCHS:-6}

# 1. Baseline at a budget that actually converges. Keeps the 3-epoch run
#    (roberta_base_repro) on disk as the evidence that 3 was too few.
stage roberta_base_6ep $PY scripts/train_transformer.py \
  --model roberta-base --run-name roberta_base_6ep \
  --batch-size 32 --max-length 192 --epochs "$EPOCHS" --resume

# 2. The single biggest expected jump (ROADMAP §3 tier 3). 16 GB fits it at
#    batch 16 in fp16; grad-accum keeps the effective batch at 32. This is the
#    priority run, which is why it comes before the research question below.
stage roberta_large_6ep $PY scripts/train_transformer.py \
  --model roberta-large --run-name roberta_large_6ep \
  --batch-size 16 --grad-accum 2 --max-length 192 --epochs "$EPOCHS" --lr 1e-5 --resume

# 3. DeBERTa-v3 in bf16 — the open question from the Kaggle era. The T4 had no
#    bf16, which is the leading suspect for its fp32/fp16 NaN divergence; Ada
#    does. This is a STABILITY PROBE, not a fair comparison: 4 epochs answers
#    "does it diverge?" and is deliberately the stage that gets cut if the
#    deadline arrives, because it is the least valuable of the three.
stage deberta_v3_bf16 $PY scripts/train_transformer.py \
  --model microsoft/deberta-v3-base --run-name deberta_v3_bf16 \
  --batch-size 16 --grad-accum 2 --max-length 192 --epochs 4 --lr 5e-6 --bf16 --resume

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

converged, truncated = [], []
for m in pathlib.Path("models").glob("*/metrics.json"):
    try:
        d = json.loads(m.read_text())
    except Exception:
        continue
    if d.get("smoke"):
        continue
    scores = [h["eval_macro_f1"] for h in sorted(
        d.get("log_history", []), key=lambda h: h.get("epoch") or 0)
        if "eval_macro_f1" in h]
    # Best epoch == last epoch means the run was still improving: its score is a
    # lower bound, so it must not win a ranking against a run that plateaued.
    still_rising = len(scores) > 1 and scores.index(max(scores)) == len(scores) - 1
    (truncated if still_rising else converged).append((d["macro_f1"], m.parent.name))

# Only fall back to truncated runs if nothing converged — better to post-process
# a lower-bound model than to produce nothing at all.
pool = converged or truncated
print(max(pool)[1] if pool else "")
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
