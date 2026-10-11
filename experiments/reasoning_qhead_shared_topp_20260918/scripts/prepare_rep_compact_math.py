"""Freeze the W1 TopK selector trial without changing running frozen experiments."""
import datetime,difflib,json,shutil,sys
from pathlib import Path
from rep_compact_math_common import sha,rows
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve();baseline=E/'results/compact_kv_random_reason1024_recent64_20260926'
assert json.loads((baseline/'collection_status.json').read_text())['phase']=='completed'
for rel,h in json.loads((baseline/'inference_source_hashes.json').read_text()).items():assert sha(baseline/rel)==h
root.mkdir();(root/'logs').mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in (E/'scripts').glob('*rep_compact_math*'):shutil.copyfile(p,root/'source/scripts'/p.name)
shutil.copyfile(baseline/'model_assets_sha256.json',root/'model_assets_sha256.json')
(root/'inputs').mkdir();shutil.copyfile(baseline/'inputs.jsonl',root/'inputs/math.jsonl')
(root/'data/math').mkdir(parents=True);dataset=Path('/project/yl258/jz648/Random-Attention/data/math/test.jsonl')
cfg=json.loads((baseline/'config.json').read_text());assert sha(dataset)==cfg['dataset_sha256'];shutil.copyfile(dataset,root/'data/math/test.jsonl')
assert len(rows(root/'inputs/math.jsonl'))==len(rows(root/'data/math/test.jsonl'))==500
# Ensure storage/gather/attention and the random sampler are unchanged. The only
# existing inference files allowed to change add factory/config/metrics branches.
allowed={'kvcompress/engine/cache_utils.py','kvcompress/harness/my_utils.py','kvcompress/harness/eval_hf.py'};diff=[];changed=[]
for old in sorted((baseline/'source/ra_harness').rglob('*.py')):
 rel=old.relative_to(baseline/'source/ra_harness');new=root/'source/ra_harness'/rel
 if sha(old)!=sha(new):
  assert str(rel) in allowed,str(rel);changed.append(str(rel))
  diff.extend(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='random/'+str(rel),tofile='representative/'+str(rel)))
(root/'inference_adapter.diff').write_text(''.join(diff))
(root/'implementation_alignment.json').write_text(json.dumps(dict(status='passed',baseline=str(baseline),changed_existing_files=changed,new_selector='kvcompress/engine/representative_compact_recent_cache.py',inherited_storage_gather_attention_identical=True),indent=2)+'\n')
for phase,ref in [('formal',baseline/'random'),('smoke',E/'results/compact_kv_random_reason1024_recent64_smoke_20260926/smoke')]:
 dest=root/'reference_random'/phase;dest.mkdir(parents=True)
 for name in ('summary.json','grades.jsonl'):shutil.copyfile(ref/name,dest/name)
 for i in (0,1):
  assert {(r['run'],r['idx']) for r in rows(dest/'grades.jsonl')}=={(r,j) for r in (0,1) for j in range(500 if phase=='formal' else 3)}
for key in ('random_selection','num_shards','reference_inputs','smoke_source'):cfg.pop(key,None)
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='qhead_shared_representative_compact_recent_gpu',query_window=1,selection_p=None,representative_policy='one private-stream Q-head draw per layer and compaction; same head across active rollouts',selection='W1 post-RoPE Q; correct GQA K; FP32 probability softmax over all surviving keys; TopK960 older reasoning; stable position tie-break',seed_coordinates='representative private device generator seeded by stable_hash(rep_seed0,sample_idx,layer); advanced only at compaction; generation seed42 unchanged',tasks=[dict(name='math',count=500,shards=8)],versions={'math':'MATH500 canonical test500'},full_run_authorized=True,auto_full_submission=True,formal_after_all_smoke_validation=True,smoke_sample_ids=[0,1,2],max_concurrent_gpus=4,array_tasks=8,scope='W1 representative TopK compact MATH500; three full-input smoke before formal500; two rollouts per question; no other query windows',comparison_caveat='Compare to completed random compact with identical budget/cadence/inputs. Masks, output lengths and A100 nodes differ; total latency and accuracy differences are single-seed observations.',implementation_reference=str(baseline))
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'formal_work.json').write_text(json.dumps([dict(task='math',shard=i,num_shards=8) for i in range(8)],indent=2)+'\n')
(root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},indent=2)+'\n')
print(json.dumps(dict(root=str(root),method=cfg['method'],changed_existing_inference_files=changed,scope=cfg['scope'])),flush=True)
