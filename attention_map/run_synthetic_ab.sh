#!/usr/bin/env bash
# Export full attention maps for the two synthetic A/B samples (~2k tokens).
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/work/model/Qwen2.5-3B}"
JSONL_PATH="${JSONL_PATH:-/home/ubuntu/work/datasets/synthetic_ab_pair/synthetic_ab_pair.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/work/attention_map/outputs_synthetic_ab}"

# Empty = no truncation (use full prompt from jsonl). Set e.g. 2048 to truncate.
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-}"
QUERY_SLICE="${QUERY_SLICE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
START_LINE="${START_LINE:-0}"

if [[ ! -f "${JSONL_PATH}" ]]; then
  echo "Dataset missing; generating ${JSONL_PATH} ..."
  python /home/ubuntu/work/datasets/synthetic_ab_pair/generate_synthetic_ab_pair.py \
    --model_path "${MODEL_PATH}"
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate attentionmap 2>/dev/null || true
fi

cmd=(
  python -m attention_map.run_longbench_v2
  --model_path "${MODEL_PATH}"
  --jsonl_path "${JSONL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --start_line "${START_LINE}"
  --query_slice "${QUERY_SLICE}"
)
if [[ -n "${MAX_INPUT_TOKENS}" ]]; then
  cmd+=(--max_input_tokens "${MAX_INPUT_TOKENS}")
fi
"${cmd[@]}"

echo "Done. Outputs under ${OUTPUT_DIR}"
