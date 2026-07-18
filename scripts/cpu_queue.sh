#!/usr/bin/env bash
# Unattended zero-GPU run queue — everything worth doing while the GPU is busy.
#
# Runs the CPU experiments back to back so no decision comes back mid-sequence.
# Each stage checkpoints its own results file and is skipped when that file is
# already complete, so re-running after an interruption resumes rather than
# restarts.
#
#   bash scripts/cpu_queue.sh
#
# Thread budget: the GPU job on the host shares these cores, so every stage is
# capped at 6 threads (the brief allows 8; 6 leaves headroom for its tokenizer).

set -uo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6

PY=.venv/bin/python
LOG=reports/cpu_queue.log
mkdir -p reports models

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

stage() {
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then
    log "SKIP $name ($marker present)"
    return 0
  fi
  log "START $name"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    log "DONE  $name (($((SECONDS - t0))s))"
  else
    log "FAIL  $name — see $LOG; continuing to the next stage"
  fi
}

log "=== CPU queue start ==="

# 1. Cheapest and most informative: where the Macro-F1 is actually lost, and
#    which jobs drive the disparate impact. Needs the model saved by the
#    baseline reproduction run.
if [ -f models/classical_wordchar_svm.joblib ]; then
  stage error-analysis reports/error_analysis.md \
    $PY scripts/error_analysis.py --model models/classical_wordchar_svm.joblib
else
  log "SKIP error-analysis (no saved classical model)"
fi

# 2. Accuracy/fairness Pareto front — 5 fits, feeds the fairness-track decision.
stage fairness-pareto reports/fairness_pareto.json \
  $PY scripts/fairness_pareto.py

# 3. Hyper-parameter sweep — 11 fits, the longest stage, so it goes last.
#    Its own JSON checkpoints per config, so a kill costs one config.
stage classical-sweep reports/classical_sweep.json \
  $PY scripts/sweep_classical.py

log "=== CPU queue done ==="
log "results: reports/error_analysis.md, reports/fairness_pareto.json, reports/classical_sweep.json"
