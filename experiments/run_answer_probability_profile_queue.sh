#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-2}
wait_pid=${2:-49000}
work=/home/ubuntu/work
python=/home/ubuntu/miniconda3/envs/official_flex/bin/python
runner=$work/experiments/run_shareprefill_ae3_infinitebench.py
root=$work/experiments/outputs/infinitebench_answer_token_probability_profile_3samples
reference_root=$work/experiments/outputs/infinitebench_shareprefill_ae3_token_block_auto_retrieval/shareprefill_ae3_token_block_auto

mkdir -p "$root/logs"
echo "[$(date -Is)] waiting for RULER queue PID $wait_pid on GPU $gpu"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 60
done

while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
  echo "[$(date -Is)] GPU $gpu still has a compute process"
  sleep 60
done

run_profile() {
  local method=$1
  local task=$2
  shift 2
  echo "[$(date -Is)] start method=$method task=$task"
  CUDA_VISIBLE_DEVICES="$gpu" "$python" "$runner" \
    --model "$work/model/Qwen2.5-7B" \
    --method "$method" \
    --task "$task" \
    --limit 3 \
    --profile_one_token \
    --record_sparsity \
    --output_dir "$root" \
    --reference_metrics "$reference_root/$task/online_metrics.jsonl" \
    "$@" \
    >"$root/logs/${method}_${task}.log" 2>&1
  echo "[$(date -Is)] complete method=$method task=$task"
}

for method in \
  shareprefill_ae3_token_block_auto_fixed_mass_profile \
  shareprefill_ae3_token_block_auto_topp
do
  run_profile "$method" passkey \
    --watched_key_range 48:53 \
    --watched_key_range 58:63
  run_profile "$method" number_string \
    --watched_key_range 46:56 \
    --watched_key_range 61:71
done

echo "[$(date -Is)] all answer-probability profiles complete"
