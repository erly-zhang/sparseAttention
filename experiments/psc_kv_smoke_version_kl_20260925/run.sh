#!/bin/bash
set -euo pipefail
cd /ocean/projects/cis260309p/jzhang114/sparseAttention
module purge
export LD_LIBRARY_PATH=/jet/packages/python/v3.12.13-gcc13/lib
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/repair_01/triton300:/ocean/projects/cis260309p/jzhang114/sparseAttention
export TRITON_CACHE_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_hybrid_smoke_20260925/vscode_l40s/gpu-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec timeout --signal=TERM --kill-after=10s 1500s experiments/psc_kv_layer_outputs_20260925/env/bin/python -u experiments/psc_kv_smoke_version_kl_20260925/run.py
