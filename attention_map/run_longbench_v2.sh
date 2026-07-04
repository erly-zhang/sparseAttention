#!/usr/bin/env bash
# Record LAST-token attention (the last row of each head's [seq_len, seq_len] map)
# for the rebuilt LongBench-v2 dataset (context already truncated to fit a 32k
# token window with the question kept intact).
#
# query_slice=last uses a memory-efficient TWO-STAGE extraction (see
# attention_map/model.py::extract_last_token_attention):
#   stage 1: prefill input_ids[:, :-1] with use_cache=True, output_attentions=False
#            (optionally chunked via CHUNK_SIZE) -> keep only past_key_values
#   stage 2: decode input_ids[:, -1:] with the cache, output_attentions=True
#            -> per-layer attention is [batch, num_heads, 1, seq_len] (the last row)
# The full [num_heads, seq_len, seq_len] map is never materialized.
#
# Memory note: with eager attention, set CHUNK_SIZE (e.g. 4096) so the prefill
# never builds a [num_heads, seq, seq] score matrix. This makes 32k feasible on
# the 46GB L40S. Without chunking, a single 32k prefill forward can still OOM.
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

START_LINE="${START_LINE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
QUERY_SLICE="${QUERY_SLICE:-last}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/work/model/Qwen2.5-7B}"
JSONL_PATH="${JSONL_PATH:-/home/ubuntu/work/datasets/longbench_v2/longbench_v2_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/work/attention_map/outputs_longbench_v2_7b_32k}"

DOMAIN="${DOMAIN:-}"
SUB_DOMAIN="${SUB_DOMAIN:-}"
DIFFICULTY="${DIFFICULTY:-}"
LENGTH="${LENGTH:-}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
QUERY_INDEX="${QUERY_INDEX:-}"

if [[ "${QUERY_SLICE}" == "index" && -z "${QUERY_INDEX}" ]]; then
  echo "QUERY_INDEX is required when QUERY_SLICE=index" >&2
  exit 1
fi

cmd=(
  python -m attention_map.run_longbench_v2
  --model_path "${MODEL_PATH}"
  --jsonl_path "${JSONL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --max_samples "${MAX_SAMPLES}"
  --start_line "${START_LINE}"
  --query_slice "${QUERY_SLICE}"
  --max_input_tokens "${MAX_INPUT_TOKENS}"
)

if [[ "${QUERY_SLICE}" == "last" && -n "${CHUNK_SIZE}" ]]; then
  cmd+=(--chunk_size "${CHUNK_SIZE}")
fi
if [[ -n "${DOMAIN}" ]]; then
  cmd+=(--domain "${DOMAIN}")
fi
if [[ -n "${SUB_DOMAIN}" ]]; then
  cmd+=(--sub_domain "${SUB_DOMAIN}")
fi
if [[ -n "${DIFFICULTY}" ]]; then
  cmd+=(--difficulty "${DIFFICULTY}")
fi
if [[ -n "${LENGTH}" ]]; then
  cmd+=(--length "${LENGTH}")
fi
if [[ "${QUERY_SLICE}" == "index" && -n "${QUERY_INDEX}" ]]; then
  cmd+=(--query_index "${QUERY_INDEX}")
fi

"${cmd[@]}"

# Defaults: 1 sample, QUERY_SLICE=last, MAX_INPUT_TOKENS=32768, CHUNK_SIZE=2048.
# Run several consecutive samples:
#   START_LINE=0 MAX_SAMPLES=3 bash run_longbench_v2.sh
# Run the whole dataset (503 samples; large + slow):
#   MAX_SAMPLES=503 bash run_longbench_v2.sh
# Filter by domain/sub-domain:
#   DOMAIN="Long In-context Learning" SUB_DOMAIN="New language translation" bash run_longbench_v2.sh
# Tune memory/speed via chunked prefill (smaller = less VRAM, slightly slower):
#   CHUNK_SIZE=2048 bash run_longbench_v2.sh
# Disable chunking (single prefill forward; may OOM at 32k):
#   CHUNK_SIZE= bash run_longbench_v2.sh
# Faster smoke test:
#   MAX_INPUT_TOKENS=512 bash run_longbench_v2.sh
# 128k variant (override defaults):
#   JSONL_PATH=/home/ubuntu/work/datasets/longbench_v2/longbench_v2_train_128k.jsonl \
#   OUTPUT_DIR=/home/ubuntu/work/attention_map/outputs_longbench_v2_128k \
#   MAX_INPUT_TOKENS=131072 CHUNK_SIZE=256 bash run_longbench_v2.sh
