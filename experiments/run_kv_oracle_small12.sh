#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/work
PYTHON=$ROOT/../miniconda3/envs/official_flex/bin/python
RUNNER=$ROOT/experiments/run_kv_oracle_infinitebench.py
MODEL=$ROOT/model/Llama-3.1-8B-Instruct
GROUP_CONFIG=$ROOT/experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json
REFERENCE=$ROOT/experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl
OUTPUT=$ROOT/experiments/outputs/infinitebench_llama31_kv_oracle_diagnostic_20260823
LOGS=$OUTPUT/logs
mkdir -p "$LOGS"

COMMON=(
  "$RUNNER"
  --model "$MODEL"
  --method shareprefill_ae3_token_block_auto_fixed_mass_profile
  --task kv_retrieval
  --max_length 131072
  --batch_size 1
  --limit 12
  --seed 42
  --chat
  --group_config "$GROUP_CONFIG"
  --calibration_scope benchmark_specific
  --flexprefill_root "$ROOT/FlexPrefill"
  --task_config_dir "$ROOT/experiments/data/kv_oracle_stratified12_20260823/task_configs"
  --reference_metrics "$REFERENCE"
  --record_sparsity
  --profile_kv_retrieval_ranges
)

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$ROOT/FlexPrefill:$ROOT" \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" "${COMMON[@]}" \
  --output_dir "$OUTPUT/stratified12_baseline_a3" \
  >"$LOGS/stratified12_baseline_a3.log" 2>&1 &
BASELINE_PID=$!

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$ROOT/FlexPrefill:$ROOT" \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" "${COMMON[@]}" \
  --output_dir "$OUTPUT/stratified12_oracle_a3" \
  --force_watched_key_range_blocks \
  >"$LOGS/stratified12_oracle_a3.log" 2>&1 &
ORACLE_PID=$!

echo "baseline_pid=$BASELINE_PID oracle_pid=$ORACLE_PID"
set +e
wait "$BASELINE_PID"
BASELINE_STATUS=$?
wait "$ORACLE_PID"
ORACLE_STATUS=$?
set -e
echo "baseline_status=$BASELINE_STATUS oracle_status=$ORACLE_STATUS"
test "$BASELINE_STATUS" -eq 0
test "$ORACLE_STATUS" -eq 0
