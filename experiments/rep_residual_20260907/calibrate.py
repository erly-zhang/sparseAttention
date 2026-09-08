"""Same-QK calibration on excluded data; no accuracy-derived fitting."""
import json
import os
import sys
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
# Import the existing isolated runner without shadowing its selector module.
sys.path.insert(0,str(HERE.parent/'probmean_compact_20260906'))
import run as base
from selector import ProbabilitySelector
sys.path.insert(0,str(HERE))
from policy import select, prepare_group, apply_ratio, RATIOS

ROOT=Path(os.environ['CALIBRATION_OUTPUT'])
original_call=ProbabilitySelector.__call__
old_probability=base.probability_mean
import selector as original_selector


def call(self,q,k,v,*args,**kwargs):
    if not self.collect:
        return original_call(self,q,k,v,*args,**kwargs)
    layer=self.current_layer.get(); n=q.shape[1]
    row_ids=torch.linspace(0,(n-1)//128,min(32,(n+127)//128),device=q.device).round().long().unique()
    captured={}
    def probability(q0,k0,h,kh,*aa,**kk):
        p=old_probability(q0,k0,h,kh,*aa,**kk)
        captured[h]=p.index_select(0,row_ids)
        return p
    original_selector.probability_mean=probability
    try:
        index=original_call(self,q,k,v,*args,**kwargs)
    finally:
        original_selector.probability_mean=old_probability
    assert len(captured)==q.shape[2]
    groups=self.layers[str(layer)]; starts=row_ids*128
    output=[]
    for kind in ('fixed','topp'):
        ps=[captured[int(g['representative'])] for g in groups]
        selections=[select(p,kind,starts) for p in ps]
        for g,group in enumerate(groups):
            bundle=prepare_group(ps,selections,g,kind)
            candidates=[apply_ratio(bundle,x)[0] for x in RATIOS]
            for h in group['members']:
                if h==int(group['representative']): continue
                p=captured[h]; _, teacher, _=select(p,kind,starts)
                old=bundle['old']
                baseline_tp=(teacher&old).sum().double()
                baseline_mass=(p*old).sum().double()
                rows=[]
                for mask in candidates:
                    rows.append(torch.stack(((teacher&mask).sum().double()-baseline_tp,
                        (p*mask).sum().double()-baseline_mass,(teacher&mask).sum().double(),
                        teacher.sum().double(),baseline_tp,baseline_mass,(p*mask).sum().double())))
                scores=torch.stack(rows).cpu().tolist()
                output.append({'kind':kind,'head':h,'group':g,'sources':bundle['sources'],
                    'scores':{str(r):s for r,s in zip(RATIOS,scores)}})
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/f'layer{layer:02d}.json'
    if path.exists(): raise RuntimeError(f'Refusing to overwrite calibration layer {path}')
    path.write_text(json.dumps({'layer':layer,'n':n,'tile_indices':row_ids.cpu().tolist(),
        'teacher_trajectory':'perhead_probability_mean_compact10240',
        'fields':['net_target_count','net_member_mass_sum','target_intersection',
                  'target_count','baseline_intersection','baseline_member_mass_sum','final_member_mass_sum'],
        'records':output},indent=2))
    print(f'CALIBRATION layer={layer} n={n} completed',flush=True)
    return index


ProbabilitySelector.__call__=call
if __name__=='__main__': base.runner.main()
