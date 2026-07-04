#!/usr/bin/env bash
# K%% mass-coverage similarity: global N×N heatmap (all layers × all heads).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

DATASET_DIR="${DATASET_DIR:-/home/ubuntu/work/attention_map/outputs/2k/niah_single_1}"
K_PERCENT="${K_PERCENT:-95}"
MATRIX_KIND="${MATRIX_KIND:-directional}"
DPI="${DPI:-600}"

python -m attention_map.similarity \
  --dataset_dir "${DATASET_DIR}" \
  --k_percent "${K_PERCENT}" \
  --matrix_kind "${MATRIX_KIND}" \
  --dpi "${DPI}" \
  --cmap viridis
