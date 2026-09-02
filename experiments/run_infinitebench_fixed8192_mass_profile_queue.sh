#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 GPU_ID TASK [TASK ...]" >&2
    exit 2
fi

gpu_id="$1"
shift

work_root="${WORK_ROOT:-/home/ubuntu/work}"
output_root="${OUTPUT_ROOT:-$work_root/experiments/outputs/infinitebench_fixed8192_probability_mass_profile}"
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
    log_path="$output_root/logs/gpu${gpu_id}_${task}.log"

    echo "[$(date --iso-8601=seconds)] GPU $gpu_id starting $task" | tee -a "$log_path"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$runner" \
        --method shareprefill_ae3_token_block_auto_fixed_mass_profile \
        --task "$task" \
        --calibration_scope benchmark_specific \
        --output_dir "$output_root" \
        --reference_metrics "$reference_metrics" \
        --record_sparsity \
        --profile_one_token >>"$log_path" 2>&1
    echo "[$(date --iso-8601=seconds)] GPU $gpu_id completed $task" | tee -a "$log_path"
done
