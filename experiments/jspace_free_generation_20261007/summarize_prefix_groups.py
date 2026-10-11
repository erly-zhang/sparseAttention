"""Separate fixed-input sparse effects from divergence of own token histories."""
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ARMS = ['full_random','full_w1','compact_random','compact_w1']


def main():
    cases = []
    for idx in [0,1,2]:
        out = HERE/'captures'/f'sample_{idx}'
        v = json.loads((out/'validation.json').read_text())
        ds = {a['arm']:a for a in v['arms']}
        rows = [json.loads(s) for s in (out/'readouts.jsonl').read_text().splitlines()]
        # Restrict to checkpoints where EVERY method still consumed the same
        # token prefix, and BOTH storage paths have already become sparse.
        last_shared = min(ds[a]['common_prefix_tokens_vs_dense'] for a in ARMS)
        first_shared_sparse = ds['compact_random']['first_sparse_affected_output_token']-1
        for arm in ARMS:
            strata = {
                'all_methods_same_prefix_and_both_storage_paths_sparse':[
                    r for r in rows if r['arm']==arm and r['reasoning'] and r['layer']<35
                    and first_shared_sparse<=r['step']<=last_shared],
                'this_method_has_diverged_own_prefix':[
                    r for r in rows if r['arm']==arm and r['reasoning'] and r['layer']<35
                    and not r['same_token_prefix_as_dense']]
            }
            for stratum,rs in strata.items():
                assert all(r['recall25_vs_dense_at_same_step'] is not None for r in rs)
                cases.append(dict(idx=idx,arm=arm,stratum=stratum,cells=len(rs),
                                  steps=sorted({r['step'] for r in rs}),
                                  recall25=statistics.mean(r['recall25_vs_dense_at_same_step'] for r in rs) if rs else None,
                                  js_nats=statistics.mean(r['js_nats_vs_dense_at_same_step'] for r in rs) if rs else None))
    aggregate = []
    for arm in ARMS:
        for stratum in sorted({c['stratum'] for c in cases}):
            matched = [c for c in cases if c['arm']==arm and c['stratum']==stratum and c['cells']]
            aggregate.append(dict(arm=arm,stratum=stratum,samples=len(matched),
                recall25=statistics.mean(c['recall25'] for c in matched) if matched else None,
                js_nats=statistics.mean(c['js_nats'] for c in matched) if matched else None))
    result = dict(status='completed',per_case=cases,equal_weight_per_case=aggregate,
                  caveats=['Same-prefix window is a small endogenous window ending at first sampled divergence; not the entire reasoning trajectory.',
                           'Different-prefix readouts mix attention effects and changing text history and cannot establish isolated sparse fidelity.',
                           'No workspace-capacity or accuracy claim from these three questions.'])
    (HERE/'prefix_stratified_jlens_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='completed',equal_weight_per_case=aggregate)),flush=True)


if __name__=='__main__':main()
