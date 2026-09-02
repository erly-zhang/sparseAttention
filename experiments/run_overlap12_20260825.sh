#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/work

ROOT=/home/ubuntu/work/experiments/outputs/infinitebench_llama31_finalmask_overlap12_20260825
LOG_ROOT="$ROOT/logs"
PYTHON=/home/ubuntu/miniconda3/envs/official_flex/bin/python
RUNNER=/home/ubuntu/work/experiments/run_dense_multimodel_infinitebench.py
MODEL=/home/ubuntu/work/model/Llama-3.1-8B-Instruct
GROUP_CONFIG=/home/ubuntu/work/experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json
TASK_CONFIG=/home/ubuntu/work/experiments/data/infinitebench_benchmark_specific_calibration/task_configs
REFERENCE=/home/ubuntu/work/experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/shareprefill_ae3_token_block_auto/kv_retrieval/online_metrics.jsonl

mkdir -p "$LOG_ROOT"

run_method() {
    local gpu="$1"
    local method="$2"
    local log="$LOG_ROOT/${method}.log"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH=/home/ubuntu/work/FlexPrefill:/home/ubuntu/work \
        "$PYTHON" "$RUNNER" \
        --model "$MODEL" \
        --method "$method" \
        --task kv_retrieval \
        --max_length 131072 \
        --batch_size 1 \
        --limit 12 \
        --seed 42 \
        --chat \
        --calibration_scope benchmark_specific \
        --group_config "$GROUP_CONFIG" \
        --task_config_dir "$TASK_CONFIG" \
        --output_dir "$ROOT" \
        --reference_metrics "$REFERENCE" \
        --record_sparsity \
        --dump_selector_details \
        --profile_one_token \
        >"$log" 2>&1
}

run_method 0 flexprefill &
pid_flex=$!
run_method 1 shareprefill_ae3_token_block_auto &
pid_original=$!
run_method 2 shareprefill_ae3_token_block_auto_target_protected &
pid_protected=$!

status=0
wait "$pid_flex" || status=1
wait "$pid_original" || status=1
wait "$pid_protected" || status=1
exit "$status"
