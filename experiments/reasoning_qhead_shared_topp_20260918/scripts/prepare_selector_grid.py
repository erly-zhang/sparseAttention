import datetime,json,shutil,sys
from pathlib import Path
from selector_grid import sha,write,read,rows
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve()
root.mkdir();(root/'source/scripts').mkdir(parents=True);(root/'logs').mkdir();(root/'inputs').mkdir()
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('selector_grid.py','prepare_selector_grid.py'):shutil.copy2(E/'scripts'/name,root/'source/scripts'/name)
mathref=E/'results/full_kv_bernoulli_r020_recent64_20260928';hmmtref=E/'results/compact_multibench_20260926'
shutil.copy2(mathref/'model_assets_sha256.json',root/'model_assets_sha256.json')
for name,src,inp in [('math',Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl'),mathref/'inputs.jsonl'),('hmmt',hmmtref/'data/hmmt/test.jsonl',hmmtref/'inputs/hmmt.jsonl')]:
    (root/'data'/name).mkdir(parents=True);shutil.copy2(src,root/'data'/name/'test.jsonl');shutil.copy2(inp,root/'inputs'/f'{name}.jsonl')
base=read(mathref/'config.json')
cfg={k:base[k] for k in ('model','model_geometry','seed','rep_seed','batch_size','rollouts','max_tokens','generation_temperature','generation_top_p','packages','timing')}
cfg.update(created=datetime.datetime.now().astimezone().isoformat(),experiment='controlled_shared_selector_grid_v1',full_run_authorized=True,auto_full_submission=True,max_concurrent_gpus=8,smoke_sample_ids=[0,1,2],mask_seed=42,score_temperature=1.0,historical_query_causal_mask=False,actual_attention_causal_mask=True,representative='private CPU generator stable_hash(0x47524550,rep_seed0,sample_idx,layer,decode_step); same representative across W1/W16,TopK/TopP,protection and storage at equal step; shared across rollouts',random='independent float32 Uniform per candidate, private CUDA seed42/sample/layer/step/original_rollout; independent secondary random key then position breaks finite precision ties; same score law as random_pp, different sharing and tie/RNG details',protection='prompt + latest64 original reasoning positions, or no forced positions; prompt/recent included in fixed total K',top_p='conditional probability mass over unprotected candidates reaches0.95, then union protected; no fixed K cap',compaction='after attention at decode steps divisible by64; fixed TopK only if current count>K; chronological gather all KV heads using same positions then swap K/V buffers; permanent deletion; absolute RoPE positions unchanged',full_kv='append complete KV, choose and attend temporary gathered subset every decode step; no permanent deletion',caveats=['Compare random and representative TopK within matched storage/protection/K to isolate scorer; Top-p has variable budget.','Compact selects after attention every64 forwards; full KV selects before attention every forward. Cross-storage comparison includes update frequency and irreversible deletion.','W16 importance uses historical queries against all currently available keys without old-query causal masking, matching prior window estimator. Actual attention is causal.','Accuracy is mean of two sampled outputs per question, not pass@2; generation may diverge from identical initial input.','This shared random control is not official per-KV-head independent random_pp.','Compacted TopK length is K immediately after events, temporarily up to K+64 between events.'],cells=[])
smoke=[];formal=[]
for task,count,K,shards in [('math',500,1024,16),('hmmt',60,4096,4)]:
    assert len(rows(root/'inputs'/f'{task}.jsonl'))==count
    assert max(r['input_tokens'] for r in rows(root/'inputs'/f'{task}.jsonl'))+64<=K
    for storage in ('compact','full'):
        for scorer,selector in [('random','topk'),('w1','topk'),('w16','topk'),('w1','topp'),('w16','topp')]:
            for protection in ('protected','none'):
                method=f'grid_{storage}_{scorer}_{selector}_{protection}';ci=len(cfg['cells'])
                cfg['cells'].append(dict(name=f'{task}_{method}',task=task,count=count,K=K,shards=shards,method=method,storage=storage,scorer=scorer,selector=selector,protection=protection,window=16 if scorer=='w16' else 1))
                smoke.append(dict(cell=ci,shard=0,num_shards=1))
                formal.extend(dict(cell=ci,shard=i,num_shards=shards) for i in range(shards))
write(root/'config.json',cfg);write(root/'smoke_work.json',smoke);write(root/'formal_work.json',formal)
common='''#!/bin/bash -l
#SBATCH --account=yl258
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
'''
body='''set -euo pipefail
module purge
module load wulver
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN_ROOT="${1:?}"
/project/yl258/jz648/miniconda3/envs/random-attention/bin/python -u "$RUN_ROOT/source/scripts/selector_grid.py" '''
for stage,n in [('smoke',len(smoke)),('formal',len(formal))]:
    s=common+f'#SBATCH --partition=gpu\n#SBATCH --gres=gpu:a100:1\n#SBATCH --time=2-00:00:00\n#SBATCH --job-name=selgrid-{stage}\n#SBATCH --array=0-{n-1}%8\n'+body+f'{stage} "$RUN_ROOT" "${{SLURM_ARRAY_TASK_ID:?}}"\n'
    (root/'source/scripts'/f'{stage}.sbatch').write_text(s)
for stage in ('gate','collect'):
    s=common+f'#SBATCH --partition=general\n#SBATCH --time=08:00:00\n#SBATCH --job-name=selgrid-{stage}\n'+body+f'{stage} "$RUN_ROOT"\n'
    (root/'source/scripts'/f'{stage}.sbatch').write_text(s)
(root/'README.md').write_text('''# Shared random vs representative selectors: MATH500 / HMMT60

40 cells = 5 selectors x 2 protection policies x 2 storage policies x 2 datasets.
Selectors: Uniform TopK, W1/W16 probability-mean TopK, W1/W16 Top-p0.95.
Fixed total K is1024 for MATH and4096 for HMMT, INCLUDING protection.
Protected: all prompt plus latest64 reasoning positions. Unprotected: no forced
position; prompt/current/recent may still be selected by their score.
Prefill always dense. Same initial input IDs as historical baselines, checked
before generation and again by collector. All new generations, two sampled
rollouts each, seed42, BF16 Qwen3-4B, total length32768.

Every layer shares selected positions across its eight KV heads (32 Q heads).
K/V values remain head-specific. Representative head drawn independently per
layer and decode step, same draw for W1/W16 and TopK/Top-p at equal step.
The random selector does not consume Q/K scores; representative identity has
no effect on its independent Uniform scores. Private RNG never changes token
sampling RNG. Float32 Uniform primary scores use independent secondary random
bits and position as tie breaks; this is not a bitwise random_pp reproduction.

W1/W16 scoring: correct GQA Q->KV mapping, independent scaled-dot-product
softmax for each recent real decode query over all currently available keys,
then equal probability mean. Historical window estimator is preserved: no
historical-query causal mask for importance estimation. Actual model attention
is exactly causal. Head-score ties prefer earlier original token position.
Top-p normalizes over UNPROTECTED candidates, retains the smallest ranked prefix
reaching0.95 of that mass, then unions protection. Top-p has no K cap.

Full storage: every decode forward temporarily gathers selected KV; persistent
history is unchanged. Compact: attention first uses all surviving KV plus the
current token; after every64 decode forwards select, gather chronologically
into another buffer and swap. For TopK skip eviction while current size<=K.
Unselected KV is permanently removed. Absolute position/RoPE never renumbered.
Compacted fixed-K buffers grow logically to at most K+64 between events.
Top-p compact buffers grow on demand; actual kept lengths can differ by rollout
and layer. Padding is never attended. Decode timing includes all selector work.

40 three-question full-generation smokes gate400 formal GPU shards (MATH16
shards/cell, HMMT4). At most8 array GPUs run concurrently. No existing experiment
is stopped or overwritten. CPU gate checks all40 smoke cells; any failure blocks
formal jobs. GPU workers only generate and technically validate. Separate CPU
collection grades with30-second per-answer timeout, reports unresolved grades
instead of silently treating them as incorrect, and checks all expected IDs,
input hashes, causal pair totals and analytic fixed-budget selected pair counts.
Variable Top-p pairs are actual observed counts, not an assumed fixed budget.

Compare random/W1/W16 TopK within each matched storage/protection cell. Top-p
comparisons include budget changes. Storage comparisons also include per-step
vs every64 selection frequency and irreversible deletion. Report accuracy,
output lengths/caps, decode wall time and tokens/s, pairs/sparsity and memory.
See config.json, manifest.json, correctness.json, submission.json, smoke_gate.json,
cells/*/{smoke,formal}/summary.json and final summary.json.
''')
write(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
print(json.dumps(dict(root=str(root),cells=len(cfg['cells']),smoke_jobs=len(smoke),formal_jobs=len(formal))))
