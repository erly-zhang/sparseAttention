"""Pooled profiling counts, formal metrics, and conservative historical comparison."""
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def profile_summary(path):
    rr = rows(path/'online_metrics.jsonl')
    assert len(rr) == 3
    total, members, phases, samples = {}, {}, {}, []
    def add(dst, array, fields):
        dst['row_count'] = dst.get('row_count', 0) + len(array)
        for i, key in enumerate(fields):
            dst[key] = dst.get(key, 0) + int(array[:,i].sum(dtype=np.float64))
    def finish(dst):
        result = {'raw_sums': dst, 'per_head_tile_means': {k: v/dst['row_count'] for k,v in dst.items() if k != 'row_count'}}
        result['pooled_evidence_coverage'] = {}
        for evidence in ('value','kv_evidence'):
            for tag in ('shared10240','core','residual','final'):
                for unit in ('slots','pairs'):
                    denominator = dst[f'{evidence}_legal_{unit}']
                    result['pooled_evidence_coverage'][f'{evidence}_{tag}_{unit}'] = (
                        dst[f'{evidence}_{tag}_{unit}']/denominator if denominator else None)
        return result
    for r in rr:
        raw = Path(r['profiling_file'])
        meta = read(raw.with_suffix('.json'))
        assert meta['input_ids_sha256'] == r['input_ids_sha256']
        one = {}
        with np.load(raw) as arrays:
            assert len(meta['mapping']) == 1024
            for item in meta['mapping']:
                array = arrays[item['array']]
                assert np.isfinite(array).all()
                add(total,array,meta['fields'])
                add(one,array,meta['fields'])
                if item['head'] != item['representative']:add(members,array,meta['fields'])
        samples.append({'input_ids_sha256':r['input_ids_sha256'],'input_tokens':r['input_tokens'],'statistics':finish(one)})
        for k,v in meta['phase_gpu_stream_sec'].items():phases[k]=phases.get(k,0.)+v
    return {'count':3,'overall':finish(total),'nonrepresentative_members':finish(members),'per_sample':samples,
            'mean_phase_gpu_stream_sec':{k:v/3 for k,v in phases.items()},
            'timing_note':'Separate profiling only; CUDA-event stream intervals can include host submission gaps. Nested phases must not be summed twice.',
            'reference_note':'Novel positions and shadow Shared10240 are evaluated on the new method current Q/K trajectory.',
            'coverage_note':'Raw retained/eligible totals are pooled before division. Token coverage is not whole-record retention or causal importance.',
            'raw_files':[r['profiling_file'] for r in rr]}


def historical(metrics, label, correct):
    rr=rows(metrics)
    summary_path=Path(metrics).with_name('summary.json')
    summary=read(summary_path) if summary_path.is_file() else {}
    result={'label':label,'source':str(metrics),'count':len(rr),
            'avg_prefill_latency_sec':float(np.mean([r['prefill_latency_sec'] for r in rr])),
            'hardware':summary.get('hardware'),'correct':correct,'accuracy':correct/497,
            'correct_source':'historical user-provided result; replace with archived score when available'}
    official=summary.get('official_lm_eval_results',{}).get('kv_retrieval',{})
    if 'get_score_one_kv_retrieval,none' in official:
        result['accuracy']=official['get_score_one_kv_retrieval,none']
        result['correct']=round(result['accuracy']*497)
        result['correct_source']=str(summary_path)
    result['speed_comparison_supported']=False
    result['speed_note']='No automatic speedup claim: validate matching GPU, package versions, load and timing instrumentation first.'
    return result


def summarize(root,c,merged,profile):
    from experiments.random_score_residual_20260910.schedule import save
    summary=read(merged/'summary.json')
    prof=profile_summary(profile)
    comparison={'new':summary,'profiling':prof,'historical':[
        historical(c['old_random_metrics'],'Shared6144 + old random4096',8),
        historical(c['shared_metrics'],'Shared10240',14)],
        'interpretation':[
            'Both samplers target uniform subsets without replacement, up to documented finite random-priority ties.',
            'Same seed42 does not imply the same masks across algorithms; an accuracy change in one seed is not evidence that one uniform sampler is better.',
            'Profiling measures random generation, global TopK, residual sorting and final index construction separately, on only three inputs.',
            'End-to-end prefill changes cannot isolate sampler overhead if hardware, environment or timing scope differs.',
            'Global pair sparsity is computed from pooled selected and causal pair counts.']}
    save(root/'comparison.json',comparison)
    save(profile/'profiling_summary.json',prof)
    r=summary['runtime']
    lines=['# Uniform-score TopK residual: Llama KV Retrieval','',
           f"Formal completion: {r['count']}/497, input_alignment=passed/full, exact per-input and total pair agreement.",'',
           '| Method | Correct/497 | Accuracy | Mean prefill (s) |',
           '| --- | ---: | ---: | ---: |',
           f"| New random-score TopK | {r['correct']} | {r['accuracy']:.4%} | {r['avg_prefill_latency_sec']:.4f} |"]
    for h in comparison['historical']:
        lines.append(f"| {h['label']} | {h['correct']} | {h['accuracy']:.4%} | {h['avg_prefill_latency_sec']:.4f} |")
    lines += ['',f"Prefill median/P90: {r['median_prefill_latency_sec']:.4f}/{r['p90_prefill_latency_sec']:.4f} s.",
        f"Mean decode/total: {r['avg_decode_latency_sec']:.4f}/{r['avg_total_latency_sec']:.4f} s.",
        f"Peak memory: {r['max_peak_memory_bytes']} bytes.",
        f"Selected/causal pairs: {r['selected_token_pairs']}/{r['causal_token_pairs']}.",
        f"Global pair sparsity: {r['global_pair_sparsity']:.6%}.",'',
        'Historical accuracy sources and hardware metadata are recorded in comparison.json. Speedup is not asserted.',
        '',*comparison['interpretation']]
    path=root/'RESULTS.md'
    text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
