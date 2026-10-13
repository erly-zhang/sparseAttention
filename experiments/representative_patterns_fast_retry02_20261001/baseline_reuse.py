"""Read-only comparison of historical baseline provenance and fixed evaluation IDs."""
import json
from collections import Counter
from protocol import RUN, PRIOR, ORIGIN, sha, save


def inspect_baselines():
    evaluation=json.loads((ORIGIN/'inputs/manifest.json').read_text())['records']
    hashes={r['input_ids_sha256'] for r in evaluation}
    results=json.loads((PRIOR/'results.json').read_text())
    old_manifest=json.loads((PRIOR/'inputs/manifest.json').read_text())
    data={}
    for method in ('flexprefill','minference'):
        rows=[r for r in results if r['method']==method and r['status']=='complete']
        overlap=[r['source_index'] for r in rows if r['input_ids_sha256'] in hashes]
        data[method]=dict(available_complete_records=len(rows),task='KV Retrieval only',
            identical_input_overlap=len(overlap),overlap_source_indices=overlap,
            generation_budget_in_old_runner=128,generation_budget_in_current_kv=50,
            old_configuration=rows[0]['metadata'],reuse=False,
            reasons=['32 old KV records do not cover four tasks x 50 fixed evaluation examples',
                'Only one input hash matches; old decode budget 128 differs from current KV cap 50',
                'New common 32-layer adapter requires freshly verified per-example provenance'])
    return dict(status='rerun_required',baselines=data,evaluation_counts=dict(Counter(r['task'] for r in evaluation)),
        evaluated_source_files={str(p):sha(p) for p in [PRIOR/'results.json',PRIOR/'quality.py',
            PRIOR/'inputs/manifest.json',ORIGIN/'inputs/manifest.json']},
        additional_historical_result='psc_kv_layer_outputs_20260925/result_summary.json: incomplete correctness gate',
        model=old_manifest.get('model'),decision_scope='available local baseline records; no experiment executed')


if __name__=='__main__':
    result=inspect_baselines()
    save(RUN/'baseline-reuse-audit.json',result)
    print(json.dumps(result,indent=2))
