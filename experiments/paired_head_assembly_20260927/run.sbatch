#!/bin/bash
#SBATCH --job-name=psc-paired-smoke
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/paired_head_assembly_20260927/slurm-%j.out
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
RUN="$ROOT/experiments/paired_head_assembly_20260927"
PRIOR="$ROOT/experiments/psc_kl_quality_retry02_20260927"
ENV_SHA=$(sha256sum "$PRIOR/env.sh")
[[ "${ENV_SHA%% *}" == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
cd "$ROOT"
export OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache"
export ROUTING_SOURCE="$ROOT/experiments/prefill_token_head_routing_20260924/routing.py"
hostname
nvidia-smi
"$BASE/env/bin/python" -u "$RUN/smoke.py" --check
"$BASE/env/bin/python" -u "$RUN/test_assembly.py"
exec timeout --signal=TERM --kill-after=15s 1680s "$BASE/env/bin/python" -u "$RUN/smoke.py"
