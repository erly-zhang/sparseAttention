#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 GPU_ID TASK [TASK ...]" >&2
    exit 2
fi

gpu_id="$1"
shift

work_root="${WORK_ROOT:-/home/ubuntu/work}"
output_root="${OUTPUT_ROOT:-$work_root/experiments/outputs/infinitebench_shareprefill_ae3_autoblock_topp_count125}"
runner="$work_root/experiments/run_shareprefill_ae3_infinitebench.py"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/official_flex/bin/python}"
reference_root="$work_root/experiments/outputs/infinitebench_benchmark_specific_comparison/shareprefill_ae3_full"
math_reference_root="$work_root/experiments/outputs/infinitebench_unified_math10_comparison/shareprefill_ae3_full"

mkdir -p "$output_root/logs"
cd "$work_root"

for task in "$@"; do
    task_reference_root="$reference_root"
    if [[ "$task" == "math_find" ]]; then
        task_reference_root="$math_reference_root"
    fi
    reference_metrics="$task_reference_root/$task/online_metrics.jsonl"
    if [[ ! -f "$reference_metrics" ]]; then
        echo "missing reference metrics: $reference_metrics" >&2
        exit 1
    fi

    echo "[$(date --iso-8601=seconds)] GPU $gpu_id starting $task"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$runner" \
        --method shareprefill_ae3_token_block_auto_topp \
        --task "$task" \
        --calibration_scope benchmark_specific \
        --output_dir "$output_root" \
        --reference_metrics "$reference_metrics" \
        --record_sparsity
    echo "[$(date --iso-8601=seconds)] GPU $gpu_id completed $task"
done
