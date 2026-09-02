#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 WAIT_PID GPU_ID TASK [TASK ...]" >&2
    exit 2
fi

wait_pid="$1"
gpu_id="$2"
shift 2

while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 60
done

exec /home/ubuntu/work/experiments/run_infinitebench_autoblock_topp_count_queue.sh \
    "$gpu_id" "$@"
