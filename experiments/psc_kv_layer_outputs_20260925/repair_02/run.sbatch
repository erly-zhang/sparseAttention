#!/bin/bash
#SBATCH --job-name=psc-kv-layer5-r2
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=00:54:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/slurm-%j.out
set -euo pipefail
cd /ocean/projects/cis260309p/jzhang114/sparseAttention
source experiments/psc_kv_layer_outputs_20260925/environment.sh
export PYTHONPATH=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/repair_01/triton300:/ocean/projects/cis260309p/jzhang114/sparseAttention${PYTHONPATH:+:$PYTHONPATH}
export PSC_RUN_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/run_repair02
export PSC_PIPELINE_SECONDS=3160
export TRITON_CACHE_DIR=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kv_layer_outputs_20260925/repair_02/gpu-cache300
exec timeout --signal=TERM --kill-after=10s 3190s experiments/psc_kv_layer_outputs_20260925/env/bin/python -u experiments/psc_kv_layer_outputs_20260925/pipeline.py
