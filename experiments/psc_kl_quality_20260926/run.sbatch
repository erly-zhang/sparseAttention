#!/bin/bash
#SBATCH --job-name=psc-kl-quality
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kl_quality_20260926/slurm-%j.out
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
cd "$ROOT"
module purge
source experiments/psc_kv_layer_outputs_20260925/environment.sh
export OMP_NUM_THREADS=8
export PYTHONPATH="$ROOT/experiments/psc_kv_layer_outputs_20260925/repair_01/triton300:$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$ROOT/experiments/psc_kl_quality_20260926/triton-cache"
export QUALITY_SCRATCH="/local/psc-kl-quality-${SLURM_JOB_ID}"
mkdir "$QUALITY_SCRATCH"
exec timeout --signal=TERM --kill-after=15s 42900s \
  experiments/psc_kv_layer_outputs_20260925/env/bin/python -u \
  experiments/psc_kl_quality_20260926/quality.py run
