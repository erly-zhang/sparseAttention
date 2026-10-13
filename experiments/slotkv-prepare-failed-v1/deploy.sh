#!/bin/bash
# Prepare and validate only. Submission is a separate explicit step.
set -euo pipefail
STAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/ocean/projects/cis260309p/jzhang114/sparseAttention
RUN="$ROOT/experiments/head_slot_kv_ablation_20260928"
PRIOR="$ROOT/experiments/psc_kl_quality_retry02_20260927"
cd "$STAGE"
sha256sum -c SHA256SUMS
[[ ! -e "$RUN" ]] || { printf 'Existing run, do not redeploy: %s\n' "$RUN"; exit 1; }
mkdir "$RUN"
cp ./*.py ./*.sh ./*.sbatch ./*.md SHA256SUMS "$RUN/"
cd "$RUN"
sha256sum -c SHA256SUMS
[[ $(sha256sum "$PRIOR/env.sh" | cut -d' ' -f1) == c94cb1372ffeaff5ae4a32309e4a9d05364dd273c400ecfd0bed07822d8284d9 ]]
source "$PRIOR/env.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export ROUTING_SOURCE="$ROOT/experiments/prefill_token_head_routing_20260924/routing.py"
export METRICS_SOURCE="${ROOT%/*}/prefill-routing-deps/FlexPrefill/experiments/benchmark/infinitebench/metrics.py"
"$BASE/env/bin/python" -u test_assembly.py
"$BASE/env/bin/python" -u test_ablation.py
"$BASE/env/bin/python" -u evaluate.py --prepare
"$BASE/env/bin/python" -u evaluate.py --check
sbatch --test-only run.sbatch > test-only.log 2>&1
cat test-only.log
printf 'VALIDATED ONLY. No job submitted.\n'
