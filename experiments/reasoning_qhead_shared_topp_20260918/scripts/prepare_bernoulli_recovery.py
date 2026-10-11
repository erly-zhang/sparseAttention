"""Audit completed samples using the existing validator, then freeze recovery."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1]
old=E/'results/full_kv_bernoulli_r020_recent64_20260928'
root=E/'results/full_kv_bernoulli_r020_recent64_recovery_20260929'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for rel,h in read(old/'manifest.json').items():assert sha(old/rel)==h,rel
completed={};missing={};history_hashes={}
for shard in range(8):
    d=old/f'random/shard_{shard}'
    rows=[json.loads(s) for s in (d/'cache_metrics.jsonl').read_text().splitlines()]
    ids={x['idx'] for x in rows};assert len(ids)==len(rows)
    for run in (0,1):
        files=list((d/f'run_{run}').glob('completions*.jsonl'));assert len(files)==1
        cr=[json.loads(s) for s in files[0].read_text().splitlines()]
        assert {x['idx'] for x in cr}==ids and len(cr)==len(ids)
    completed[str(shard)]=sorted(ids)
    absent=sorted(set(range(shard,500,8))-ids)
    if absent:missing[str(shard)]=absent
    assert read(d/'exit.json')['exit_code']==(1 if absent else 0)
    for f in d.rglob('*'):
        if f.is_file() and f.name in ('cache_metrics.jsonl','completions.jsonl','exit.json','hardware.json','preflight.json','command.json'):
            history_hashes[str(f.relative_to(old))]=sha(f)
assert sum(map(len,completed.values()))==471 and sum(map(len,missing.values()))==29
# Reuse the entire independent input/pair/cache validator before its grading
# section. Permit the two known failed job exits ONLY for this partial audit.
code=(old/'source/scripts/check_bernoulli_math500.py').read_text().split('sys.path[:0]=')[0]
needle="ids=set(cfg['smoke_sample_ids']) if a.phase=='smoke' else set(expected)"
assert needle in code
code=code.replace(needle,'ids='+repr(set(x for v in completed.values() for x in v)))
needle="    assert json.loads((d/'exit.json').read_text())['exit_code']==0,str(d)"
assert needle in code
code=code.replace(needle,"    assert json.loads((d/'exit.json').read_text())['exit_code']==(1 if shard in (6,7) else 0),str(d)")
saved=sys.argv;sys.argv=['partial_audit',str(old),'--phase','formal']
try:exec(compile(code,'historical_partial_validator','exec'),{})
finally:sys.argv=saved
root.mkdir();(root/'repair').mkdir();(root/'random').mkdir();(root/'logs').mkdir()
shutil.copytree(old/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('inputs.jsonl','model_assets_sha256.json','correctness.json','correctness.log','smoke_gate.json'):
    shutil.copy2(old/name,root/name)
shutil.copytree(old/'smoke',root/'smoke')
for name in ('bernoulli_recovery.py','prepare_bernoulli_recovery.py'):
    shutil.copy2(E/'scripts'/name,root/'source/scripts'/name)
(root/'source/scripts/historical_partial_validator.py').write_text(code)
write(root/'historical_output_hashes.json',history_hashes)
write(root/'inference_source_hashes.json',{rel:h for rel,h in read(old/'manifest.json').items() if rel.startswith(('source/ra_harness/','source/tests/'))})
plan=dict(original_root=str(old),created=datetime.datetime.now().astimezone().isoformat(),completed_by_shard=completed,missing_by_shard=missing,original_complete_questions=471,recovery_questions=29,recovery_rollouts=58,excluded_nodes=['n0068','n0090'],partial_audit='passed/input hashes, both outputs, pair counts, cache lengths and memory release',inference_source='byte-identical to original frozen source',source_manifest_sha256=sha(old/'manifest.json'))
write(root/'recovery_plan.json',plan)
cfg=read(old/'config.json');cfg.update(scope='Recover29 missing Bernoulli MATH500 questions; merge471 complete original questions',all_samples_fresh=False,historical_outputs_reused=942,recovered_questions=29)
write(root/'config.json',cfg)
base=(old/'source/scripts/bernoulli_array.sbatch').read_text()
base=base.replace('#SBATCH --array=0-7%8','#SBATCH --array=6,7%2').replace('--job-name=bern20-full','--job-name=bern20-recover').replace('bernoulli_math500.py','bernoulli_recovery.py')
base=base.replace('#SBATCH --gres=gpu:a100:1','#SBATCH --gres=gpu:a100:1\n#SBATCH --exclude=n0068,n0090')
(root/'source/scripts/bernoulli_recovery_array.sbatch').write_text(base)
collect=(old/'source/scripts/bernoulli_collect.sbatch').read_text().replace('bernoulli_math500.py','bernoulli_recovery.py').replace('--job-name=bern20-grade','--job-name=bern20-rec-grade')
(root/'source/scripts/bernoulli_recovery_collect.sbatch').write_text(collect)
(root/'README.md').write_text('''# Bernoulli20 complete-KV recovery

The original run saved471 complete questions (942 outputs); both outputs,
input hashes, pair counts, cache lengths and memory release passed the existing
validator. Original outputs remain untouched. Only29 missing questions are
regenerated in repair/shard_6 (10 questions) and repair/shard_7 (19 questions).

Inference and tests are byte-identical to the passed original run. Same model,
inputs, prompt, seed42, private per-layer/step/rollout Bernoulli RNG, batch2 and
32768 cap. Only orchestration changes. Exclude n0068,n0090 from Slurm allocation.

A dependent CPU collector checks both recovery workers, merges each sample
exactly once into a NEW random/ directory, records sample and hardware provenance,
and runs the original full500 input/pair/length/memory validator and grader.
The new summary combines471 original timings and29 recovery timings; it is not
a new500-question generation run. Original errors and logs remain preserved.
''')
write(root/'manifest.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
print(json.dumps(dict(root=str(root),validated_questions=471,missing_by_shard=missing),indent=2))
