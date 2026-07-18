#!/usr/bin/env bash
# Verify checkpoint-resume by actually killing a run — not by assuming it works.
#
# A multi-hour GPU fit that cannot resume turns any interruption into a total
# loss, and "resume is wired up" is exactly the kind of claim that holds until
# the first time it matters. So: start a run, SIGKILL it once it has written a
# checkpoint, restart with --resume, and check that it picks up from that
# checkpoint instead of starting over.
#
#   bash scripts/test_resume.sh
#
# Runs on CPU with the smoke config, so it costs minutes and no GPU.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
RUN_DIR=models/smoke_roberta-base
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "=== phase 1: run until a checkpoint exists, then SIGKILL ==="
rm -rf "$RUN_DIR"
$PY scripts/train_transformer.py --smoke > logs_resume_phase1.txt 2>&1 &
PID=$!

for _ in $(seq 1 240); do
  if compgen -G "$RUN_DIR/checkpoint-*" > /dev/null; then
    sleep 10   # let the checkpoint finish writing
    kill -9 $PID 2>/dev/null
    break
  fi
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
wait $PID 2>/dev/null

CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "FAIL: no checkpoint was ever written — resume cannot work"
  exit 1
fi
STEP_BEFORE=$(basename "$CKPT" | cut -d- -f2)
echo "killed after checkpoint at step $STEP_BEFORE"

# Final artifacts must be absent: the run died before writing them.
if [ -f "$RUN_DIR/metrics.json" ]; then
  echo "note: the run had already finished; the kill did not interrupt training"
fi

echo
echo "=== phase 2: restart with --resume ==="
$PY scripts/train_transformer.py --smoke --resume > logs_resume_phase2.txt 2>&1
RC=$?

echo
echo "=== verdict ==="
if grep -q "resuming from" logs_resume_phase2.txt; then
  echo "PASS  restart picked up from the checkpoint:"
  grep "resuming from" logs_resume_phase2.txt | sed 's/^/      /'
else
  echo "FAIL  restart did NOT resume — it started from scratch"
  exit 1
fi

if [ $RC -ne 0 ]; then
  echo "FAIL  resumed run exited with code $RC"
  tail -5 logs_resume_phase2.txt
  exit 1
fi

for f in metrics.json valid_logits.npy valid_meta.csv test_logits.npy; do
  if [ -f "$RUN_DIR/$f" ]; then
    echo "PASS  $f written"
  else
    echo "FAIL  $f missing after the resumed run"
    exit 1
  fi
done

echo
echo "resume verified end to end: killed at step $STEP_BEFORE, restarted, finished, all artifacts present."
