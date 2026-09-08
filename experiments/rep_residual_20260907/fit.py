"""Fit on excluded records 0/1, gate once on excluded record 2."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path('/local/results/rep_residual_20260907')
GROUP=Path('/home/ubuntu/work/experiments/outputs/infinitebench_multimodel_topk8192_20260818/calibration/llama31_8b_instruct/shareprefill_ae_k3_head_groups.json')


def main():
    config=json.loads(GROUP.read_text())
    policies={kind:{'layers':{},'enabled':0,'total_members':928} for kind in ('fixed','topp')}
    for layer in range(32):
        docs=[json.loads((ROOT/'calibration'/str(i)/f'layer{layer:02d}.json').read_text()) for i in range(3)]
        maps=[{(r['kind'],r['head']):r for r in d['records']} for d in docs]
        for kind,policy in policies.items():
            layer_rules={}
            for g,group in enumerate(config['layers'][str(layer)]):
                for head in group['members']:
                    rule={'quarter':-1,'own_group':g,'sources':[j for j in range(3) if j!=g]}
                    if head!=int(group['representative']):
                        rows=[m[(kind,head)] for m in maps]
                        # Rank by fit count gain, then member probability mass gain.
                        best=max(range(5),key=lambda x:(sum(r['scores'][str(x)][0] for r in rows[:2]),
                                                       sum(r['scores'][str(x)][1] for r in rows[:2]),-abs(x-2)))
                        scores=[r['scores'][str(best)] for r in rows]
                        stable=all(s[0]>0 for s in scores)
                        # Reject a >1 percentage-point mean member-mass loss on any record.
                        mass_ok=all(s[1]/len(d['tile_indices'])>=-.01 for s,d in zip(scores,docs))
                        rule.update(candidate_quarter=best,per_record_scores=scores,
                                    count_gain_positive_each_record=stable,member_mass_guard_passed=mass_ok)
                        if stable and mass_ok:
                            rule['quarter']=best;policy['enabled']+=1
                    layer_rules[str(head)]=rule
            policy['layers'][str(layer)]=layer_rules
    output={'schema':1,'residual_tokens':1024,'fractions':[0,.25,.5,.75,1],
            'fit_records':[0,1],'gate_records':[2],'tiles_per_record':32,
            'quota_selection':'max fit micro target-count gain; member mass tie-break',
            'gate':'positive net target-count gain on each record; mean member-mass loss <=0.01',
            'representative_heads':'unchanged','top_p_mode':'equal-count replacement; final mass may be below .99',
            'group_sha256':hashlib.sha256(GROUP.read_bytes()).hexdigest(),'policies':policies}
    path=ROOT/'policies.json'
    if path.exists() and json.loads(path.read_text())!=output:
        raise RuntimeError('Refusing to overwrite different fitted policy')
    path.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:{'enabled':v['enabled'],'total_members':v['total_members']} for k,v in policies.items()}))


if __name__=='__main__':main()
