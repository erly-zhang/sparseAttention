#!/bin/bash
set -euo pipefail
STAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
RUN="$ROOT/experiments/paired_head_assembly_20260927"
PRIOR="$ROOT/experiments/psc_kl_quality_retry02_20260927"
[[ $(hostname -s) != *login* ]]
cd "$STAGE"
sha256sum -c SHA256SUMS
[[ ! -e "$RUN" ]] || { printf 'Refusing existing run: %s\n' "$RUN"; exit 1; }
mkdir "$RUN"
cp assembly.py adapter.py smoke.py test_assembly.py run.sbatch README.md SHA256SUMS deploy.sh "$RUN/"
cd "$RUN"
sha256sum -c SHA256SUMS
ENV_SHA=$(sha256sum "$PRIOR/env.sh")
[[ "${ENV_SHA%% *}" == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
export OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1
export ROUTING_SOURCE="$ROOT/experiments/prefill_token_head_routing_20260924/routing.py"
"$BASE/env/bin/python" -u "$RUN/smoke.py" --check
"$BASE/env/bin/python" -u "$RUN/test_assembly.py"
sbatch --test-only "$RUN/run.sbatch" > "$RUN/test-only.log" 2>&1
cat "$RUN/test-only.log"
sbatch --parsable "$RUN/run.sbatch" > "$RUN/submitted-job.txt" 2> "$RUN/submit-stderr.log"
printf 'ACTUAL SUBMITTED JOB: '
cat "$RUN/submitted-job.txt"
