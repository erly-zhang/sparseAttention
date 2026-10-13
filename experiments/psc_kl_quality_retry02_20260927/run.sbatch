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
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kl_quality_retry02_20260927/slurm-%j.out
set -euo pipefail
RUN=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/psc_kl_quality_retry02_20260927
source "$RUN/env.sh"
cd "$ROOT"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache"
export QUALITY_SCRATCH="/local/psc-kl-quality-${SLURM_JOB_ID}"
mkdir "$QUALITY_SCRATCH"
sha256sum "$RUN"/{quality.py,source_pins.json,run.sbatch,env.sh,inputs/manifest.json}
exec timeout --signal=TERM --kill-after=15s 42900s "$BASE/env/bin/python" -u "$RUN/quality.py" run
