#!/usr/bin/env bash
# Cross-sample directional similarity: input1 heads -> input2 heads.
# Memory: loads target sample once (~8GB); streams source heads (fits 16GB RAM).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE="/home/ubuntu/work/attention_map/outputs/2k/niah_single_1"
SAMPLE_SRC="${SAMPLE_SRC:-${BASE}/sample_001271_line0000}"
SAMPLE_TGT="${SAMPLE_TGT:-${BASE}/sample_000639_line0001}"
K_PERCENT="${K_PERCENT:-95}"
DPI="${DPI:-600}"

python -m attention_map.similarity \
  --sample_dir "${SAMPLE_SRC}" \
  --sample_dir_tgt "${SAMPLE_TGT}" \
  --k_percent "${K_PERCENT}" \
  --dpi "${DPI}" \
  --cmap viridis
