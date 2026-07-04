#!/usr/bin/env bash
# Classify attention by pattern, then copy maps into pattern_sorted/ (replaces classify_sample.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

SAMPLE_DIR="/home/ubuntu/work/attention_map/outputs/2k/niah_single_1/sample_001271_line0000"
OUT_DIR="/home/ubuntu/work/attention_map/outputs/2k/pattern_sorted"

echo "==> classify: ${SAMPLE_DIR}"
python -m attention_map.classify --sample_dir "${SAMPLE_DIR}"

echo "==> sort (clean old copies first): ${OUT_DIR}"
python -m attention_map.sort_by_pattern \
  --sample_dir "${SAMPLE_DIR}" \
  --out_dir "${OUT_DIR}" \
  --mode copy \
  --overwrite \
  --clean
