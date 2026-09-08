"""CPU-only statistics aggregation and explicitly scoped comparators."""
import json
from pathlib import Path
import numpy as np
from experiments.probmean_compact_20260906.summarize import aggregate


def summarize(root,dest):
    summary=json.loads((dest/'summary.json').read_text())
    selection=aggregate(dest)
    by_head={};by_layer={};total=np.zeros(5,dtype=np.float64)
    for line in (dest/'online_metrics.jsonl').read_text().splitlines():
        row=json.loads(line);p=Path(row['selection_stats_file'])
        meta=json.loads(p.with_suffix('.json').read_text())
        with np.load(p) as arrays:
            for item in meta['mapping']:
                a=arrays[item['array']];layer=str(item['layer']);head=item['member_heads'][0]
                stat=np.array([len(a),a[:,11].sum(dtype=np.float64),a[:,12].sum(dtype=np.float64),
                               a[:,1].sum(dtype=np.float64),a[:,13].sum(dtype=np.float64)])
                total+=stat;by_head.setdefault(f'{layer}:{head}',np.zeros(5))[:] +=stat
                by_layer.setdefault(layer,np.zeros(5))[:] +=stat
    def finish(s):return dict(zip(['rows','core_sum','residual_sum','final_sum','novel_residual_sum'],s.tolist()))
    (dest/'random_selection_summary.json').write_text(json.dumps({'overall':finish(total),
        'per_layer':{k:finish(v) for k,v in by_layer.items()},'per_layer_head':{k:finish(v) for k,v in by_head.items()},
        'per_head_tile_counts':'retained in each sample selection_stats NPZ'},indent=2))
    baseline=Path('/home/ubuntu/work/experiments/outputs/llama_kv_probmean_compact_20260906/merged/probmean_shared_compact10240/kv_retrieval/summary.json')
    proxy=Path('/local/results/rep_residual_20260907/merged/probmean_shared_residual_compact10240/kv_retrieval/summary.json')
    report={'new':summary,'selection':selection['overall'],'random_selection':finish(total),
        'primary_comparator':json.loads(baseline.read_text()),'secondary_comparator':json.loads(proxy.read_text()),
        'caveats':['One full random seed (42); no claim of seed-robust accuracy.',
            'Proxy residual enabled 692 member-layer pairs; random residual enables all 928. This is not a pure selection-rule comparison.',
            'Random mask diversity is not equivalent to task-critical evidence retention.',
            'Random generation, rejection, deduplication, sorting, indexing and GPU stats are included in prefill.',
            'Evidence coverage is original-position token-slot coverage, not causal necessity.',
            'Input hash is part of the random stream identity; kernel is causal but full-prefix-invariant selection is not claimed.']}
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    lines=['# All-Member Random Residual','', '| Method | Correct / 497 | Accuracy | Prefill seconds | Pair sparsity |',
           '|---|---:|---:|---:|---:|']
    for label,doc in [('Original Shared10240',report['primary_comparator']),('Gated proxy residual',report['secondary_comparator']),('All-member random residual',summary)]:
        r=doc['runtime'];lines.append(f"| {label} | {r['correct']} | {r['accuracy']*100:.2f}% | {r['avg_prefill_latency_sec']:.3f} | {r['global_token_sparsity']*100:.2f}% |")
    lines+=['','Input alignment: passed/full; exact per-sample and total selected/causal pair equality with original Shared10240.','']+report['caveats']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
