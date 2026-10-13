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
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kl_quality_retry01_20260927/slurm-%j.out
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
BASE="$ROOT/experiments/psc_kv_layer_outputs_20260925"
RUN="$ROOT/experiments/psc_kl_quality_retry01_20260927"
cd "$ROOT"
module purge
source "$BASE/environment.sh"
export LD_LIBRARY_PATH=/jet/packages/python/v3.12.13-gcc13/lib
export CC=/usr/bin/gcc
export COMPILER_PATH=/usr/bin
export OMP_NUM_THREADS=8
export PYTHONPATH="$BASE/repair_01/triton300:$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache"
export QUALITY_SCRATCH="/local/psc-kl-quality-${SLURM_JOB_ID}"
mkdir "$QUALITY_SCRATCH"
sha256sum "$RUN"/{quality.py,source_pins.json,retry.sbatch,inputs/manifest.json}
"$CC" --version
exec timeout --signal=TERM --kill-after=15s 42900s "$BASE/env/bin/python" -u "$RUN/quality.py" run
