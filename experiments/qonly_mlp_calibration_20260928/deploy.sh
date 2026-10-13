#!/bin/bash
# CPU-only preparation; no submission and no GPU inference.
set -euo pipefail
STAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments
RUN="$ROOT/qonly_mlp_calibration_20260928"
PRIOR="$ROOT/psc_kl_quality_retry02_20260927"
cd "$STAGE"
sha256sum -c SHA256SUMS
[[ ! -e "$RUN" ]] || { printf 'Existing directory; do not redeploy\n'; exit 1; }
mkdir "$RUN"
cp ./*.py ./*.sh ./*.sbatch ./*.md SHA256SUMS "$RUN/"
cd "$RUN"
[[ $(sha256sum "$PRIOR/env.sh" | cut -d' ' -f1) == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
"$BASE/env/bin/python" -u validate_cpu.py > cpu-tests.log 2>&1
cat cpu-tests.log
"$BASE/env/bin/python" -u pilot.py prepare
"$BASE/env/bin/python" -u pilot.py check
sbatch --test-only run.sbatch > test-only.log 2>&1
cat test-only.log
printf 'VALIDATED ONLY. No job submitted.\n'
