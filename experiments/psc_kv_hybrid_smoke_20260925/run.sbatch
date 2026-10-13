#!/bin/bash
#SBATCH --job-name=psc-hybrid-smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_hybrid_smoke_20260925/slurm-%j.out
set -euo pipefail
cd /ocean/projects/cis260309p/jzhang114/sparseAttention
source experiments/psc_kv_layer_outputs_20260925/environment.sh
export PYTHONPATH=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/repair_01/triton300:/ocean/projects/cis260309p/jzhang114/sparseAttention${PYTHONPATH:+:$PYTHONPATH}
export PSC_RUN_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_hybrid_smoke_20260925/run
export PSC_PIPELINE_SECONDS=1700
export PSC_APPROVED_TOTAL_SECONDS=7200
export PSC_PRIOR_ALLOCATION_SECONDS=4976
export TRITON_CACHE_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_hybrid_smoke_20260925/gpu-cache
exec timeout --signal=TERM --kill-after=10s 1740s experiments/psc_kv_layer_outputs_20260925/env/bin/python -u experiments/psc_kv_hybrid_smoke_20260925/pipeline.py
