"""Merge disjoint source-data shards into reference (length-sorted) identity order."""
import json
import numpy as np
import schedule as s


def merge(method,paths):
    dest=s.ROOT/'merged'/method/'kv_retrieval'
    reference=s.rows(s.REFERENCE)
    ref_ids=[s.identity(r) for r in reference]
    if (dest/'summary.json').exists():
        rr=s.rows(dest/'online_metrics.jsonl')
        assert [s.identity(r) for r in rr]==ref_ids
        return
    assert not dest.exists(),f'Refusing to overwrite partial merge {dest}'
    chunks=[s.validate(p,b-a) for p,(a,b) in zip(paths,s.SHARDS)]
    unordered=[r for _,rr,_ in chunks for r in rr]
    indexed={s.identity(r):r for r in unordered}
    assert len(unordered)==len(indexed)==len(reference)==497
    assert set(indexed)==set(ref_ids)
    rr=[indexed[i] for i in ref_ids]
    # lm_eval samples are in source-document order; metrics are length-sorted.
    samples=[r for _,_,pp in chunks for r in pp]
    assert len(samples)==len({r['doc_hash'] for r in samples})==497
    # Preserve prediction identities and shard origin rather than assume zip order.
    predictions=[dict(r,source_shard=str(p)) for p,(_,_,pp) in zip(paths,chunks) for r in pp]
    correct=sum(r['get_score_one_kv_retrieval'] for r in predictions)
    selected=sum(r['selected_token_pairs'] for r in rr)
    causal=sum(r['causal_token_pairs'] for r in rr)
    assert causal==sum(1024*r['input_tokens']*(r['input_tokens']+1)//2 for r in rr)
    for i,r in enumerate(rr):r['call_index']=i
    prefill=np.array([r['prefill_latency_sec'] for r in rr])
    runtime={'count':497,'correct':correct,'accuracy':correct/497,
        'avg_prefill_latency_sec':float(prefill.mean()),'median_prefill_latency_sec':float(np.median(prefill)),
        'p90_prefill_latency_sec':float(np.quantile(prefill,.9)),
        'avg_decode_latency_sec':float(np.mean([r['decode_latency_sec'] for r in rr])),
        'avg_total_latency_sec':float(np.mean([r['total_latency_sec'] for r in rr])),
        'selected_token_pairs':selected,'causal_token_pairs':causal,'global_token_sparsity':1-selected/causal,
        'max_peak_memory_bytes':max(r.get('peak_memory_bytes',0) for r in rr)}
    summary={**chunks[0][0],'runtime':runtime,'source_shards':[str(p) for p in paths],
        'input_alignment':{'status':'passed','alignment_scope':'full','count':497,'reference_count':497,
            'reference_metrics':str(s.REFERENCE),'compared_fields':['input_ids_sha256','input_tokens'],
            'merge_order':'reference identity order; not shard concatenation'},
        'official_lm_eval_results':{'kv_retrieval':{'get_score_one_kv_retrieval,none':correct/497}},
        'predictions_order':'source shard then lm_eval sample order; original doc_hash retained'}
    dest.mkdir(parents=True)
    (dest/'online_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    s.status(event='identity_order_merged',method=method,count=497,correct=correct)


def completed_paths(method):
    found=[]
    for a,b in s.SHARDS:
        candidates=sorted((s.ROOT/'attempts/formal'/f'{method}_{a}_{b}').glob('attempt*'))
        passed=None
        for attempt in candidates:
            path=attempt/method/'kv_retrieval'
            try:s.validate(path,b-a)
            except (FileNotFoundError,AssertionError,KeyError,json.JSONDecodeError):continue
            passed=path;break
        if passed is None:return None
        found.append(passed)
    return found
