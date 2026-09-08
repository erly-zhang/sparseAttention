"""Aggregate actual costs and compare only fully aligned benchmark cells."""
import json
from pathlib import Path
import numpy as np
from experiments.probmean_compact_20260906.summarize import aggregate


def summarize(root):
    outputs=[]
    for path in sorted((root/'merged').glob('*/kv_retrieval/summary.json')):
        summary=json.loads(path.read_text())
        selection=aggregate(path.parent)
        counts={'rows':0,'novel_sum':0.,'dropped_sum':0.,'baseline_mass_sum':0.,'final_mass_sum':0.}
        for line in (path.parent/'online_metrics.jsonl').read_text().splitlines():
            record=json.loads(line);npz=Path(record['selection_stats_file'])
            meta=json.loads(npz.with_suffix('.json').read_text())
            with np.load(npz) as arrays:
                for item in meta['mapping']:
                    array=arrays[item['array']];weight=len(item['member_heads'])
                    counts['rows']+=len(array)*weight
                    for column,name in ((13,'novel_sum'),(14,'dropped_sum'),(12,'baseline_mass_sum'),(3,'final_mass_sum')):
                        counts[name]+=float(array[:,column].sum(dtype=np.float64))*weight
        outputs.append({'summary_path':str(path),'summary':summary,'selection':selection['overall'],
                        'residual':counts,'role':'new calibrated residual'})
    oldroot=Path('/home/ubuntu/work/experiments/outputs/llama_kv_probmean_compact_20260906/merged')
    for name in ('probmean_shared_compact10240','probmean_perhead_compact10240',
                 'probmean_shared_compact_topp99','probmean_perhead_compact_topp99'):
        path=oldroot/name/'kv_retrieval/summary.json'
        summary=json.loads(path.read_text())
        assert summary['input_alignment']['status']=='passed'
        assert summary['input_alignment']['count']==497
        outputs.append({'summary_path':str(path),'summary':summary,'role':'existing probability-mean comparator'})
    historical=Path('/home/ubuntu/work/diagnostics_review_20260906/verified_historical.json')
    if historical.exists():
        for row in json.loads(historical.read_text())['comparisons']:
            if row['count']==497 and any(s in row['path'] for s in ('/llama31/flexprefill/','/llama31/minference/','/llama31/shareprefill_ae3_token_block_auto/')):
                outputs.append({'role':'historical different hardware; no speed ranking','verified_record':row})
    (root/'comparison.json').write_text(json.dumps({'results':outputs,'caveats':[
        'Quota fitting uses only two excluded records; one additional excluded record gates adoption.',
        'Calibration uses 32 sampled query tiles; online evaluates all query tiles.',
        'Same-QK equal-count replacement does not imply same end-to-end Top-p sparsity after hidden states change.',
        'Top-p=.99 defines the original target; matched replacement may reduce final representative probability mass.',
        'Online records representative probability mass, not uncomputed member-head mass.',
        'Same pair budget does not make list merging, sorting, deduplication or index construction free.',
        'Accuracy is full 497-item generation; evidence coverage is a diagnostic, not ground truth importance.']},indent=2))


if __name__=='__main__':summarize(Path('/local/results/rep_residual_20260907'))
