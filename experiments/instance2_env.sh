#!/usr/bin/env bash
# Runtime paths for GPU experiments on edu-ubuntu-instance2.
# Source this file before launching a job:
#   source /home/ubuntu/work/experiments/instance2_env.sh

if ! mountpoint -q /local; then
    echo "ERROR: /local is not mounted; refusing to use the root disk for caches." >&2
    return 1 2>/dev/null || exit 1
fi

export XDG_CACHE_HOME="/local/cache/xdg"
export HF_HOME="/local/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="/local/cache/torch"
export TRITON_CACHE_DIR="/local/cache/triton"
export TMPDIR="/local/tmp"

export SPARSEATTENTION_CHECKPOINT_DIR="/local/checkpoints"
export SPARSEATTENTION_RESULTS_DIR="/local/results"

mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TMPDIR}" \
    "${SPARSEATTENTION_CHECKPOINT_DIR}" \
    "${SPARSEATTENTION_RESULTS_DIR}"
