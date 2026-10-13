#!/bin/bash
set -euo pipefail
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
PRIOR="$ROOT/experiments/psc_kl_quality_retry02_20260927"
RUN="$ROOT/experiments/psc_timing_smoke_20260927"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
hostname
squeue -u jzhang114
git -C "$ROOT" status --short --untracked-files=no
printf '%s  %s\n' 7c66db90d78a43ce19c876669e849414cf685389164c9819cb062cb91a1ffb61 "$PRIOR/quality.py" | sha256sum -c -
printf '%s  %s\n' c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 "$PRIOR/env.sh" | sha256sum -c -
# Fail before deployment on any existing run; never overwrite or submit twice.
mkdir "$RUN"
cp smoke.py run.sbatch deploy.sh "$RUN/"
source "$PRIOR/env.sh"
export PYTHONNOUSERSITE=1
"$BASE/env/bin/python" -m py_compile "$RUN/smoke.py"
"$BASE/env/bin/python" "$RUN/smoke.py" --check
bash -n "$RUN/run.sbatch"
sha256sum "$RUN/smoke.py" "$RUN/run.sbatch" "$RUN/deploy.sh"
sbatch --test-only "$RUN/run.sbatch" >"$RUN/test-only.log" 2>&1
printf 'TEST-ONLY PASSED (not a submitted job)\n'
job=$(sbatch --parsable "$RUN/run.sbatch")
printf '%s\n' "$job" >"$RUN/submitted-job.txt"
printf 'ACTUAL SUBMITTED JOB: %s\n' "$job"
squeue -j "$job"
scontrol show job "$job"
