"""Compare identical smoke inputs; preserve timing and algorithm caveats."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();E=root.parents[1];ids={0,1,2}
rows=[]
for method,paths in [('random_recent_gpu',[root/'smoke/cache_metrics.jsonl'])]+[(w,sorted((E/'results/full_kv_w16_w64_20260922'/w).glob('shard_*/cache_metrics.jsonl'))) for w in ('w16','w64')]:
    records={}
    for path in paths:
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['idx'] in ids:
                assert r['idx'] not in records
                records[r['idx']]=(r,path)
    assert set(records)==ids
    total_t=total_q=total_sel=total_causal=0
    for idx,(r,path) in sorted(records.items()):
        total_t+=r['decode_seconds'];total_q+=r['decode_query_tokens'];total_sel+=r['decode_selected_pairs'];total_causal+=r['decode_causal_pairs']
        hardware=json.loads((path.parent/'hardware.json').read_text()) if (path.parent/'hardware.json').exists() else None
        rows.append(dict(method=method,idx=idx,generated_tokens=r['generated_tokens_per_rollout'],eos_reached=r['eos_reached'],decode_seconds=r['decode_seconds'],decode_tokens_per_second=r['decode_query_tokens']/r['decode_seconds'],forward_steps_per_second=r['decode_forward_calls']/r['decode_seconds'],decode_sparsity=1-r['decode_selected_pairs']/r['decode_causal_pairs'],input_tokens=r['input_tokens'],input_ids_sha256=r['input_ids_sha256'],hardware=hardware,source=str(path)))
    rows.append(dict(method=method,idx='aggregate',decode_seconds=total_t,mean_decode_seconds=total_t/3,decode_tokens_per_second=total_q/total_t,decode_sparsity=1-total_sel/total_causal))
for idx in ids:
    rr=[r for r in rows if r['idx']==idx]
    assert len({(r['input_tokens'],r['input_ids_sha256']) for r in rr})==1
history=json.loads((E/'audits/reasoning_speed_history_20260924/summary.json').read_text())
dense=history['methods']['dense_500']['first_three']
dense_time=sum(x['batch_seconds'] for x in dense)
dense_tokens=sum(sum(x['generated_lengths']) for x in dense)
legacy_dense=dict(rows=dense,total_batch_seconds=dense_time,mean_batch_seconds=dense_time/3,generated_tokens_per_batch_second=dense_tokens/dense_time,caveat='Historical whole-batch time, no synchronized decode-only metric; lengths exclude EOS; retained as context, not an isolated kernel speed comparison.')
out=dict(legacy_dense_context=legacy_dense,input_alignment='passed/3 identical inputs',rows=rows,caveat='Different masks and output lengths; end-to-end throughput is not an isolated sampler/kernel benchmark. Old random40 was stopped before its first completion. Three questions do not establish full MATH500 accuracy.')
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
