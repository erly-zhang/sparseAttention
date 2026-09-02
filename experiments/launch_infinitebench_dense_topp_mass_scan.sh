#!/usr/bin/env bash
set -euo pipefail

work_root="/home/ubuntu/work"
scan_root="${work_root}/experiments/outputs/infinitebench_dense_topp_mass_layer_scan_20260818"
shard="${work_root}/experiments/run_infinitebench_dense_topp_mass_scan_shard.sh"

if [[ -e "${scan_root}" ]]; then
    echo "Refusing to overwrite existing scan root: ${scan_root}" >&2
    exit 1
fi
mkdir -p "${scan_root}/queue_logs"

launch() {
    local gpu_id="$1"
    shift
    nohup env SCAN_ROOT="${scan_root}" LIMIT=3 bash "${shard}" "${gpu_id}" "$@" \
        >"${scan_root}/queue_logs/gpu${gpu_id}.log" 2>&1 &
    echo "gpu${gpu_id} pid=$! starts=$*"
}

launch 0 0 19
launch 1 6 20
launch 2 10 21
launch 3 14 22
launch 4 15 24
launch 5 16
launch 6 17
launch 7 18
