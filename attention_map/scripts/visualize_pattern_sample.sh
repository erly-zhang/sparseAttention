#!/usr/bin/env bash
# One full-resolution heatmap per pattern_sorted category (+ optional combined PNG).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

PATTERN_SORTED="/home/ubuntu/work/attention_map/outputs/2k/pattern_sorted"
SAMPLE_DIR="/home/ubuntu/work/attention_map/outputs/2k/niah_single_1/sample_001271_line0000"

python -m attention_map.visualize_pattern_picks \
  --pattern_sorted_dir "${PATTERN_SORTED}" \
  --classification_json "${SAMPLE_DIR}/pattern_classification.json" \
  --stream_file layer_05_head_00.npy \
  --vertical_file layer_05_head_12.npy \
  --block_file layer_05_head_10.npy \
  --downsample_max 0 \
  --cmap viridis
