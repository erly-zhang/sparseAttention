#!/bin/bash
#SBATCH --job-name=psc-routing-smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_timing_smoke_20260927/slurm-%j.out
set -euo pipefail
RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_timing_smoke_20260927
source /ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kl_quality_retry02_20260927/env.sh
cd "$ROOT"
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache"
hostname
nvidia-smi
sha256sum "$RUN/smoke.py" "$RUN/run.sbatch"
exec timeout --signal=TERM --kill-after=15s 1740s "$BASE/env/bin/python" -u "$RUN/smoke.py"
