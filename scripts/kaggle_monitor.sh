#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
REF="${1:?usage: kaggle_monitor.sh <username/kernel-slug>}"
KG=.venv/bin/kaggle
for _ in $(seq 1 180); do
  S=$("$KG" kernels status "$REF" 2>&1 | tr -d '\r')
  echo "[$(date +%H:%M:%S)] $S"
  case "$S" in
    *RUNNING*|*QUEUED*) sleep 120 ;;
    *[Cc]omplete*|*COMPLETE*) echo DONE_OK; mkdir -p models/kaggle_out; "$KG" kernels output "$REF" -p models/kaggle_out/ --force 2>&1; exit 0 ;;
    *) echo DONE_OTHER; exit 2 ;;
  esac
done
echo TIMEOUT; exit 3
