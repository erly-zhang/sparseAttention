#!/bin/bash
#SBATCH --job-name=psc-mapbench
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/shared_attention_map_bench_20260929/slurm-%A_%a.out
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments
RUN="$ROOT/shared_attention_map_bench_20260929"
PRIOR="$ROOT/psc_kl_quality_retry02_20260927"
[[ $(sha256sum "$PRIOR/env.sh" | cut -d' ' -f1) == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
cd "$RUN"
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache-$SLURM_ARRAY_TASK_ID"
hostname
nvidia-smi
exec "$BASE/env/bin/python" -u evaluate.py run
