#!/bin/bash
#SBATCH --job-name=psc-slotkv50
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260309p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments/head_slot_kv_ablation_20260928/slurm-%j.out
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
RUN="$ROOT/experiments/head_slot_kv_ablation_20260928"
PRIOR="$ROOT/experiments/psc_kl_quality_retry02_20260927"
[[ $(sha256sum "$PRIOR/env.sh" | cut -d' ' -f1) == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
cd "$RUN"
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$RUN/triton-cache"
hostname
nvidia-smi
"$BASE/env/bin/python" -u evaluate.py --check
exec "$BASE/env/bin/python" -u evaluate.py
