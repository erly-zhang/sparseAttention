"""Create a new immutable-source experiment; does not submit or overwrite results."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve()
old=E/'results/full_kv_random_reason1024_repaired_full500_20260928'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root.mkdir();(root/'source/scripts').mkdir(parents=True);(root/'logs').mkdir();(root/'random').mkdir()
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('bernoulli_math500.py','check_bernoulli_math500.py','prepare_bernoulli_math500.py'):shutil.copy2(E/'scripts'/name,root/'source/scripts'/name)
for name in ('inputs.jsonl','model_assets_sha256.json'):shutil.copy2(old/name,root/name)
cfg=json.loads((old/'config.json').read_text())
for k in ('smoke_source','repair_source','reasoning_budget'):cfg.pop(k,None)
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='qhead_shared_mask_bernoulli_recent_gpu',
    implementation='full_kv_bernoulli_r020_recent64_v1',random_keep_ratio=.2,
    random_selection='independent Bernoulli per eligible old position; no TopK, no count repair',
    seed_coordinates='private CUDA RNG: stable_hash(0x4245524E,42,sample_idx,decode_step,layer,original_rollout)',
    representative_policy='none; no Q/K scores; independent layers, shared positions across heads within each layer',
    protection='full prompt; W=min(64,ceil(.2*R)) latest reasoning protected; older p=(ceil(.2*R)-W)/(R-W)',
    count_rule='expected P+ceil(.2*R), actual count random; early W=min(64,ceil(.2*R))',
    budget_semantics='expected_count_not_fixed',mask_precompute_window=16,
    gather_backend='triton_fused_reusable_ragged',
    auto_full_submission=True,scope='MATH500: 3 full-length smoke questions, then all 500 fresh questions / 1000 rollouts',
    comparison_caveat='20% expected budget differs from historical 40% exact budget; Bernoulli also changes mask distribution. Do not attribute quality or speed differences solely to sampler. Compare synchronized total times including mask preparation; output lengths and hardware may differ.')
write(root/'config.json',cfg)
common='''#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=yl258
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2-00:00:00
'''
body='''set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN_ROOT="${1:?Frozen root required}"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/bernoulli_math500.py" '''
(root/'source/scripts/bernoulli_smoke.sbatch').write_text(common+'#SBATCH --job-name=bern20-smoke\n'+body+'smoke "$RUN_ROOT"\n')
(root/'source/scripts/bernoulli_array.sbatch').write_text(common+'#SBATCH --job-name=bern20-full\n#SBATCH --array=0-7%8\n'+body+'worker "$RUN_ROOT" "${SLURM_ARRAY_TASK_ID:?}"\n')
(root/'source/scripts/bernoulli_collect.sbatch').write_text(common.replace('--partition=gpu','--partition=general').replace('#SBATCH --gres=gpu:a100:1\n','').replace('--time=2-00:00:00','--time=04:00:00').replace('--mem=64G','--mem=16G')+'#SBATCH --job-name=bern20-grade\n'+body+'collect "$RUN_ROOT"\n')
(root/'README.md').write_text('''# Full KV Bernoulli 20% + recent64, MATH500

Qwen3-4B BF16, original chat preprocessing, 500 inputs, two sampled rollouts each,
seed42, temperature0.6/top_p0.95, max total length32768. Three full-length smoke
questions gate eight independent Slurm shards. No existing job is stopped.

All prompt tokens are selected. At reasoning length R, B=ceil(0.2 R), W=min(64,B).
Latest W are protected. Each of R-W older positions independently follows
Bernoulli p=(B-W)/(R-W), with the empty range handled directly. Thus expected
reasoning count is B; actual counts vary. Early histories may protect fewer
than64 as in the existing ratio-budgeted recent baseline. Never repair a draw
to an exact count. Never permanently delete KV.

Each layer, sample, original rollout and step has its own private seeded RNG.
All Q/KV heads within one layer share positions. Q/K content is unused. Generate
all-layer masks16 steps ahead, convert boolean masks into sorted indices with
nonzero and prefix sums, no TopK or sorting. GPU gather uses padded rows, while
FlashAttention consumes true per-row lengths; padding is excluded. Complete KV
is preallocated to32768, with the prior cache lifetime repair preserved.

Mask generation, conversion, copies, gather and attention are included in decode
and batch wall time. Record actual and expected pairs separately. Binomial
count variance is diagnostic, not a hard budget. Report accuracy over1000
outputs and per-rollout500, not pass@2. The ratio and distribution differ from
historical Random40; this is not an isolated sampler-speed comparison.

Frozen source/model/input manifests, correctness.log, smoke/summary.json,
submission.json and random/summary.json provide reproducibility and status.
''')
write(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
print(root)
