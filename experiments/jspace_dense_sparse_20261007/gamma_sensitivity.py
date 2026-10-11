"""Explicit RMSNorm-scale-aware sensitivity, keeping primary results intact."""
import json
import statistics
import torch
import torch.nn.functional as F
from capture import HERE, ARMS, dense_control, initialize, dump
from analysis_math import nonnegative_gradient_pursuit
import jlens


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=True
    model,tok=initialize()
    lens=jlens.JacobianLens.load(str(next((HERE/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))))
    rows=[json.loads(s) for s in (HERE/'analysis/jspace_decomposition.jsonl').read_text().splitlines()]
    captures={i:{a:torch.load(HERE/'captures'/f'sample_{i}'/f'{a}.pt',map_location='cpu',weights_only=True) for a in ARMS} for i in (0,1,2)}
    out=[]
    gamma=model.model.norm.weight.float()
    for layer in sorted({r['layer'] for r in rows}):
        original=[r for r in rows if r['layer']==layer]
        targets=torch.stack([captures[r['idx']][r['arm']]['residuals'][captures[r['idx']][r['arm']]['steps'].tolist().index(r['step']),layer] for r in original]).cuda().float()
        J=lens.jacobians[layer].cuda()
        dictionary=(model.lm_head.weight.float()*gamma[None,:])@J
        dictionary/=dictionary.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        fit=nonnegative_gradient_pursuit(targets,dictionary,k=25)
        keys={(r['idx'],r['arm'],r['step']):j for j,r in enumerate(original)}
        for j,r in enumerate(original):
            di=keys[r['idx'],dense_control(r['arm']),r['step']]
            gain=float(fit['gains'][j,-1]);random_gain=r['random_control_gain']
            out.append(dict(idx=r['idx'],arm=r['arm'],layer=layer,step=r['step'],reasoning=r['reasoning'],regime=r['regime'],
                            dictionary='unit rows of W_U diag(final_RMSNorm_gamma) J_l',
                            raw_reconstruction_gain=gain,random_control_gain=random_gain,excess_gain=gain-random_gain,
                            random_control='reused primary control on identical targets',
                            component_cosine_to_dense=float(F.cosine_similarity(fit['reconstruction'][j:j+1],fit['reconstruction'][di:di+1])),
                            gain_curve=fit['gains'][j].cpu().tolist(),
                            token_ids=fit['ids'][j].cpu().tolist(),coefficients=fit['coefficients'][j].cpu().tolist()))
        del J,dictionary,targets,fit
        torch.cuda.empty_cache()
        print(json.dumps(dict(event='gamma_sensitivity_layer',layer=layer)),flush=True)
    (HERE/'analysis/gamma_sensitivity.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))
    groups=[]
    for arm in ARMS:
        group=[r for r in out if r['arm']==arm and r['reasoning'] and r['regime']=='both_sparse']
        groups.append(dict(arm=arm,**{m:statistics.mean(statistics.mean(r[m] for r in group if r['idx']==i) for i in {r['idx'] for r in group}) for m in ('raw_reconstruction_gain','random_control_gain','excess_gain','component_cosine_to_dense')}))
    dump(HERE/'analysis/gamma_sensitivity_summary.json',dict(status='completed',formula='W_U diag(gamma) J_l',
          primary_result_preserved=True,groups=groups,
          gamma_min=float(gamma.min()),gamma_max=float(gamma.max()),gamma_mean=float(gamma.mean()),
          caveat='Sensitivity of the measurement frame, not an additional sparse run or exact paper replication.'))
    print('GAMMA_SENSITIVITY_COMPLETED',flush=True)


if __name__=='__main__':
    main()
