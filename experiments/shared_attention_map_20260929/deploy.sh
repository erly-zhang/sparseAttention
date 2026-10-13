#!/bin/bash
# CPU preparation only. submit.py is a separate, once-only action.
set -euo pipefail
STAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention/experiments
RUN="$ROOT/shared_attention_map_20260929"
cd "$STAGE"
sha256sum -c SHA256SUMS
[[ ! -e "$RUN" ]]
mkdir "$RUN"
cp attention.py adapter.py test_attention.py smoke.py validate_cpu.py submit.py run.sbatch deploy.sh README.md MONITOR.md SHA256SUMS "$RUN/"
cd "$RUN"
sha256sum -c SHA256SUMS
[[ $(sha256sum "$ROOT/psc_kl_quality_retry02_20260927/env.sh" | cut -d' ' -f1) == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$ROOT/psc_kl_quality_retry02_20260927/env.sh"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" "$BASE/env/bin/python" -u validate_cpu.py > cpu-tests.log 2>&1
cat cpu-tests.log
CUDA_VISIBLE_DEVICES="" "$BASE/env/bin/python" -u smoke.py prepare
CUDA_VISIBLE_DEVICES="" "$BASE/env/bin/python" -u smoke.py check
printf 'PREPARED ONLY. NO JOB SUBMITTED.\n'
