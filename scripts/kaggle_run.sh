#!/usr/bin/env bash
# One-shot Kaggle GPU run for the transformer (Step C).
#
# Usage:
#   1. Put your credentials once:
#        export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy
#      (or drop kaggle.json into ~/.kaggle/)
#   2. ./scripts/kaggle_run.sh [MODEL_NAME]
#
# Pushes kaggle/train_transformer_kernel.py to a private GPU kernel, waits for
# completion, then pulls submission.csv + test_logits.npy + valid_metrics.json
# into models/kaggle_out/.
set -euo pipefail

cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle
MODEL="${1:-answerdotai/ModernBERT-base}"

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME or place ~/.kaggle/kaggle.json}"
: "${KAGGLE_KEY:?set KAGGLE_KEY or place ~/.kaggle/kaggle.json}"

SLUG="defi-ia-2021-transformer"
REF="${KAGGLE_USERNAME}/${SLUG}"

# Stamp the username into the metadata and the chosen model into the kernel
# script's default (Kaggle kernels do not inherit local env vars).
sed -i "s#\"id\": \".*/${SLUG}\"#\"id\": \"${REF}\"#" kaggle/kernel-metadata.json
sed -i "s#\"MODEL_NAME\", \".*\"#\"MODEL_NAME\", \"${MODEL}\"#" kaggle/train_transformer_kernel.py

echo ">> pushing kernel ${REF} (model=${MODEL})"
$KAGGLE kernels push -p kaggle/ --accelerator NvidiaTeslaT4

echo ">> waiting for completion (polling every 60s)"
while true; do
  STATUS=$($KAGGLE kernels status "${REF}" 2>&1 | tr -d '\r')
  echo "   ${STATUS}"
  case "${STATUS}" in
    *complete*) break ;;
    *error*|*cancel*) echo "!! kernel failed"; exit 1 ;;
  esac
  sleep 60
done

mkdir -p models/kaggle_out
echo ">> pulling outputs"
$KAGGLE kernels output "${REF}" -p models/kaggle_out/
echo ">> done. See models/kaggle_out/{submission.csv,valid_metrics.json,test_logits.npy}"
cat models/kaggle_out/valid_metrics.json 2>/dev/null || true
