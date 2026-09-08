"""Aggregate formal scores separately from the six profiling generations."""
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())


def profile_summary(path):
    metrics=[json.loads(x) for x in (path/'online_metrics.jsonl').read_text().splitlines()]
    overall={};members={};layers={};heads={};phase={};samples=[]
    def add(dst,a,fields):
        dst['rows']=dst.get('rows',0)+len(a)
        for i,name in enumerate(fields):
            valid=np.isfinite(a[:,i]);dst[name+'_sum']=dst.get(name+'_sum',0.)+float(a[valid,i].sum(dtype=np.float64))
            dst[name+'_count']=dst.get(name+'_count',0)+int(valid.sum())
    def finish(dst):
        out=dict(dst)
        for key,value in dst.items():
            if key.endswith('_sum'):
                name=key[:-4];den=dst.get(name+'_count',0)
                out[name+'_mean']=value/den if den else None
        for tag in ('shared10240','shared_core','final'):
            inter=dst.get(tag+'_intersection_pairs_sum',0)
            out[tag+'_pooled_causal_pair_recall']=inter/dst['perhead10240_causal_pairs_sum']
            out[tag+'_pooled_causal_pair_jaccard']=inter/dst[tag+'_union_pairs_sum']
        for evidence in ('value','kv_evidence'):
            for tag in ('shared10240','core','final'):
                for unit in ('slots','pairs'):
                    den=dst[evidence+'_legal_'+unit+'_sum']
                    out[evidence+'_'+tag+'_'+unit+'_coverage']=dst[evidence+'_'+tag+'_'+unit+'_sum']/den if den else None
        return out
    for r in metrics:
        p=Path(r['profiling_file']);meta=read(p.with_suffix('.json'));one={}
        assert r['input_ids_sha256']==meta['input_ids_sha256']
        with np.load(p) as arrays:
            for item in meta['mapping']:
                a=arrays[item['array']];fields=meta['fields']
                for dst in (overall,one,layers.setdefault(str(item['layer']),{}),heads.setdefault(f"{item['layer']}:{item['head']}",{})):
                    add(dst,a,fields)
                if item['head']!=item['representative']:add(members,a,fields)
        samples.append({'input_ids_sha256':r['input_ids_sha256'],'statistics':finish(one)})
        for k,v in meta['phase_gpu_stream_sec'].items():phase[k]=phase.get(k,0.)+v
    result={'count':3,'scope':'independent three-input full-generation profiling, not 497-score statistics',
        'overall':finish(overall),'nonrepresentative_members':finish(members),
        'per_layer':{k:finish(v) for k,v in layers.items()},'per_layer_head':{k:finish(v) for k,v in heads.items()},
        'per_sample':samples,'mean_phase_gpu_stream_sec':{k:v/3 for k,v in phase.items()},
        'phase_note':'GPU-event stream durations include host submission gaps; not a CPU-time decomposition; phases are profiling-only',
        'mass_definition':'member full128-query probability; member2q proxy fields separate; each method own forward trajectory',
        'mask_reference':'protected Per-head compact10240 on the same current Q/K; old Shared10240 also recomputed on current Q/K',
        'raw_per_layer_head_tile':[r['profiling_file'] for r in metrics]}
    (path/'profiling_summary.json').write_text(json.dumps(result,indent=2));return result


def summarize(root,merged,profiles):
    work=Path('/home/ubuntu/work')
    older=work/'experiments/outputs/llama_kv_probmean_compact_20260906/merged'
    controls=[('Shared10240',older/'probmean_shared_compact10240/kv_retrieval/summary.json'),
              ('Historical random9216+1024',Path('/local/results/random_residual_allmembers_20260908/merged/probmean_shared9216_random1024_allmembers_seed42/kv_retrieval/summary.json')),
              ('Per-head10240',older/'probmean_perhead_compact10240/kv_retrieval/summary.json')]
    records=[{'label':label,'source':str(path),'summary':read(path),'timing_comparable_to_new':False,
              'reason':'historical implementations include additional GPU diagnostics; do not attribute total time difference to budget ratio alone'} for label,path in controls]
    for label,p in zip(('A Shared6144+random4096','B Shared6144+member2q4096'),merged):
        s=read(p/'summary.json');assert s['input_alignment']['status']=='passed' and s['runtime']['count']==497
        records.append({'label':label,'source':str(p/'summary.json'),'summary':s,'timing_comparable_to_new':True})
    prof=[profile_summary(p) for p in profiles]
    report={'formal_results':records,'profiling':prof,'baseline_accuracy':[],
        'interpretation':{'budget_ratio':'A vs historical random changes both 10/90 to 40/60 residual allocation and index/statistics implementation, not a strict ratio-only ablation',
            'selection_signal':'A vs B shares budget, core rule, implementation and input; B adds two true member queries. Hidden trajectories and thus actual core sets may differ',
            'optimization':'short-index construction and removing extra formal diagnostics differ from historical timing; no matched old/new implementation-only full ablation was run'},
        'limitations':['single full random seed42','profiling covers only same three inputs','pattern overlap does not establish causal evidence usefulness',
            'probability masses from current member Q/K, not globally comparable across different model trajectories']}
    for method in ('flexprefill','minference'):
        path=work/f'experiments/outputs/infinitebench_multimodel_topk8192_20260818/llama31/{method}/kv_retrieval/summary.json'
        if path.exists():
            s=read(path);report['baseline_accuracy'].append({'method':method,'source':str(path),
                'score':s['official_lm_eval_results'],'hardware':s.get('hardware'),'latency_comparison':False})
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# Shared60% + Residual40%: Llama KV Retrieval','',
        'Both formal methods: 497/497 unique inputs, passed/full alignment, exact per-sample selected/causal pair equality to Shared10240.','',
        '| Method | Correct / 497 | Accuracy | Prefill mean (s) | Median | P90 | Decode | Total | Pair sparsity |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        r=record['summary']['runtime']
        lines.append(f"| {record['label']} | {r['correct']} | {100*r['accuracy']:.2f}% | {r['avg_prefill_latency_sec']:.3f} | {r.get('median_prefill_latency_sec',float('nan')):.3f} | {r.get('p90_prefill_latency_sec',float('nan')):.3f} | {r['avg_decode_latency_sec']:.3f} | {r['avg_total_latency_sec']:.3f} | {100*r['global_token_sparsity']:.2f}% |")
    lines+=['','Historical times include different GPU diagnostic overhead and are not pure algorithm comparisons.',
            'Raw pair totals, GPU peak memory, hardware metadata, profiling-only layer/head/tile statistics and phase timings are retained in comparison.json and referenced artifacts.','']
    for k,v in report['interpretation'].items():lines.append(f'- {k}: {v}')
    lines+=['']+report['limitations']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
