import datetime,json,shutil,sys
from pathlib import Path
from rep_compact_math_common import sha,rows
E=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve();ref=E/'results/w1_optimized_pair_20260927'
root.mkdir();(root/'logs').mkdir();(root/'source/scripts').mkdir(parents=True)
for name in ('ra_harness','tests'):shutil.copytree(E/name,root/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in (E/'scripts').glob('*w8_trial*'):
 if p.is_file():shutil.copyfile(p,root/'source/scripts'/p.name)
shutil.copyfile(E/'scripts/rep_compact_math_common.py',root/'source/scripts/rep_compact_math_common.py')
s=(E/'scripts/check_rep_compact_math.py').read_text();fn=s[s.index('def check_metric('):s.index('\ndef check(root,')]
fn=fn.replace("r['query_window']==1", "r['query_window']==cfg['query_window']")
(root/'source/scripts/check_w8_metrics.py').write_text(fn)
shutil.copyfile(ref/'model_assets_sha256.json',root/'model_assets_sha256.json')
refrows=rows(ref/'inputs.jsonl');(root/'inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in refrows if x['idx']<20))
(root/'data/math').mkdir(parents=True);shutil.copyfile(ref/'data/math/test.jsonl',root/'data/math/test.jsonl')
cfg=json.loads((ref/'config.json').read_text())
for key in ('pair_order','optimizations','unchanged','smoke_sample_ids','formal_after_all_smoke_validation'):cfg.pop(key,None)
cfg.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='qhead_shared_representative_window_compact_optimized_gpu',baseline_method='qhead_shared_representative_compact_optimized_gpu',query_window=8,sample_ids=list(range(20)),array_tasks=4,max_concurrent_gpus=4,walltime_per_worker_hours=4,tasks=[dict(name='math',count=20,shards=4)],full_run_authorized=False,auto_full_submission=False,scope='First20 MATH500 questions, two sampled rollouts each, paired optimized W1 and W8 on the same GPU per shard; order reversed on odd shards',selection='One Q representative per compaction; independent BF16 QK / FP32 softmax for each of the latest8 post-RoPE Q; arithmetic probability mean; stable TopK960 older reasoning',historical_query_causal_mask=False,selection_key_scope='All currently surviving keys for every query in the scoring window; historical window selector semantics',actual_attention_causal_mask=True,query_capture='Only last8 queries before each scheduled compaction, chronological buffer; interval64 >= window8',gather_backend='reasoning-only fused Triton gather, two full-capacity buffers swapped; prompt copied once',comparison_caveat='20 questions / 40 outputs per method, not full MATH500. Sampled generation can differ; pair counts validated per realized lengths, not required equal between methods. Same GPU for each paired shard, order balanced; report lengths and tokens/s alongside wall time.',implementation_reference=str(ref))
(root/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},indent=2)+'\n')
print(root,flush=True)
