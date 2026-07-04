#!/usr/bin/env bash
# Record full attention maps [seq_len, seq_len] per (layer, head).
# 2k input from ruler_data/2k/niah_single_1 (~2048×2048 per head, ~4.8GB disk per sample).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Which row in validation.jsonl (0-based). 2k/niah_single_1 has 10 lines (0..9).
#   0 -> index 1271  (capable-radiosonde)   sample_001271_line0000  [already run]
#   1 -> index 639   (jittery-hospital)    sample_000639_line0001
#   2 -> index 4504  (roomy-devil)         sample_004504_line0002
START_LINE="${START_LINE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"

python -m attention_map.run \
  --model_path /home/ubuntu/work/model/Qwen2.5-3B \
  --data_root /home/ubuntu/work/ruler_data \
  --output_dir /home/ubuntu/work/attention_map/outputs \
  --split 2k \
  --task niah_single_1 \
  --max_samples "${MAX_SAMPLES}" \
  --start_line "${START_LINE}" \
  --query_slice all

# Other lengths: change --split (2k|4k|8k) and ensure ruler_data/<split>/<task>/validation.jsonl exists.
# Truncate for smoke: add --max_input_tokens 512
# Run multiple consecutive lines: e.g. START_LINE=1 MAX_SAMPLES=3 bash run.sh
