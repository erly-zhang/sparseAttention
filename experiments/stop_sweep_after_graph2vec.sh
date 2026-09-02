#!/usr/bin/env bash
# Watch graph2vec_top_p_0.8 completion and stop the sweep before svd_kmeans/bmm.
set -euo pipefail

SWEEP_PID="${1:?usage: stop_sweep_after_graph2vec.sh <sweep_bash_pid>}"
SWEEP_ROOT="${SWEEP_ROOT:-/home/ubuntu/work/experiments/outputs/comparison32k}"
LOG="${SWEEP_LOG:-/home/ubuntu/work/experiments/outputs/comparison32k_sweep.log}"
STOP_LOG="${SWEEP_ROOT}/sweep_stop_after_graph2vec.log"
TARGET_DIR="${SWEEP_ROOT}/graph2vec_top_p_0.8"
TARGET_SUMMARY="${TARGET_DIR}/eval_summary.json"
POLL_SEC="${POLL_SEC:-30}"

echo "$(date -Is) watcher started | sweep_pid=${SWEEP_PID} | waiting for ${TARGET_SUMMARY}" | tee -a "${STOP_LOG}"

while true; do
  if [[ -f "${TARGET_SUMMARY}" ]] && grep -q "Done. Outputs:.*graph2vec_top_p_0.8" "${LOG}"; then
    sleep 3
    if kill -0 "${SWEEP_PID}" 2>/dev/null; then
      kill "${SWEEP_PID}" || true
      echo "$(date -Is) stopped sweep pid=${SWEEP_PID} after graph2vec_top_p_0.8" | tee -a "${STOP_LOG}"
    else
      echo "$(date -Is) sweep pid=${SWEEP_PID} already exited" | tee -a "${STOP_LOG}"
    fi
    exit 0
  fi
  if ! kill -0 "${SWEEP_PID}" 2>/dev/null; then
    echo "$(date -Is) sweep pid=${SWEEP_PID} exited before graph2vec_top_p_0.8 finished" | tee -a "${STOP_LOG}"
    exit 1
  fi
  sleep "${POLL_SEC}"
done
