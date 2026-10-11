"""Fixed-coordinate J-lens readouts and explicitly approximate J-space coding."""
from __future__ import annotations
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F

from capture import HERE, ARMS, SPARSE_ARMS, dense_control, initialize, dump, _stable_hash
from analysis_math import topk_recall, js_divergence, nonnegative_gradient_pursuit, selfcheck
import jlens
from jlens.vis import _meaningful_token_mask

GP_LAYERS = [0,8,17,26,34]
GP_K = 25


def phase_labels(trajectory, steps, tok):
    ids = trajectory['continuation_ids']
    close = tok.convert_tokens_to_ids('</think>')
    opening = tok.convert_tokens_to_ids('<think>')
    end = ids.index(close) if close in ids else len(ids)
    start = ids.index(opening)+1 if opening in ids else 0
    p = trajectory['input_tokens']
    first_full = max(1,2048-p+1)
    first_compact_visible = ((first_full+63)//64)*64+1
    return [dict(step=s,reasoning=start<s<=end,
                 absolute_position=p+s-1,
                 regime='pre_sparse' if s<first_full else ('full_only' if s<first_compact_visible else 'both_sparse'),
                 full_selection_active=s>=first_full,compact_eviction_visible=s>=first_compact_visible)
            for s in steps]


@torch.inference_mode()
def readout(model, residuals, J):
    h=residuals.cuda().float()
    if J is not None:
        h=h@J.T
    return model.lm_head(model.model.norm(h.to(torch.bfloat16))).float()


def tensor_rows(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def summarize_readouts(rows):
    metrics=['raw_residual_cosine','relative_residual_l2','recall25','meaningful_recall25','js_divergence_nats']
    groups=[]
    for arm in SPARSE_ARMS:
        for regime in ('pre_sparse','both_sparse'):
            selected=[r for r in rows if r['arm']==arm and r['reasoning'] and r['regime']==regime and r['layer']<35]
            # Equal weight per sample; sampled positions and layers are correlated.
            by_sample=[]
            for idx in sorted({r['idx'] for r in selected}):
                rs=[r for r in selected if r['idx']==idx]
                by_sample.append(dict(idx=idx,points=len(rs),**{m:statistics.mean(r[m] for r in rs) for m in metrics}))
            groups.append(dict(arm=arm,regime=regime,samples=len(by_sample),per_sample=by_sample,
                               **{m:statistics.mean(r[m] for r in by_sample) if by_sample else None for m in metrics}))
    return groups


@torch.inference_mode()
def analyze():
    selfcheck()
    torch.backends.cuda.matmul.allow_tf32=True
    model,tok=initialize()
    lens_path=next((HERE/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))
    lens=jlens.JacobianLens.load(str(lens_path))
    assert lens.d_model==model.config.hidden_size and lens.source_layers==list(range(35))
    samples=[]
    for directory in sorted((HERE/'captures').glob('sample_*')):
        complete=json.loads((directory/'complete.json').read_text())
        assert complete['status']=='completed'
        trajectory=json.loads((directory/'dense_trajectory.json').read_text())
        arms={a:torch.load(directory/f'{a}.pt',map_location='cpu',weights_only=True) for a in ARMS}
        dense=arms['dense']
        for a in arms.values():
            assert a['prefix_sha256']==dense['prefix_sha256']
            assert torch.equal(a['steps'],dense['steps'])
        labels=phase_labels(trajectory,dense['steps'].tolist(),tok)
        samples.append(dict(idx=trajectory['idx'],trajectory=trajectory,arms=arms,labels=labels))
    assert samples
    meaningful=_meaningful_token_mask(tok,model.config.vocab_size,torch.device('cuda'))
    rows=[]
    for layer in range(36):
        J=lens.jacobians[layer].cuda() if layer<35 else None
        for sample in samples:
            for arm in ['dense_full',*SPARSE_ARMS]:
                baseline=dense_control(arm)
                hd=sample['arms'][baseline]['residuals'][:,layer].float().cuda()
                zd=readout(model,hd,J)
                dense_top=zd.topk(25,dim=-1).indices
                dense_words=zd.masked_fill(~meaningful,-float('inf')).topk(25,dim=-1).indices
                h=sample['arms'][arm]['residuals'][:,layer].float().cuda()
                z=readout(model,h,J)
                top=z.topk(25,dim=-1).indices
                words=z.masked_fill(~meaningful,-float('inf')).topk(25,dim=-1).indices
                recall=topk_recall(dense_top,top)
                mrecall=topk_recall(dense_words,words)
                js=js_divergence(zd,z)
                cosine=F.cosine_similarity(hd,h,dim=-1)
                relative=(h-hd).norm(dim=-1)/hd.norm(dim=-1).clamp_min(1e-12)
                ranks_lower=[]
                ranks_upper=[]
                for candidate in range(5):
                    values=z.gather(1,dense_words[:,candidate:candidate+1])
                    ranks_lower.append((z>values).sum(-1)+1)
                    ranks_upper.append((z>=values).sum(-1))
                dense_persist=topk_recall(dense_top[:-1],dense_top[1:])
                sparse_persist=topk_recall(top[:-1],top[1:])
                for pos,label in enumerate(sample['labels']):
                    rows.append(dict(idx=sample['idx'],arm=arm,baseline=baseline,layer=layer,**label,
                                     raw_residual_cosine=float(cosine[pos]),relative_residual_l2=float(relative[pos]),
                                     recall25=float(recall[pos]),meaningful_recall25=float(mrecall[pos]),
                                     js_divergence_nats=float(js[pos]),
                                     lag=label['step']-sample['labels'][pos-1]['step'] if pos else None,
                                     dense_persistence25=float(dense_persist[pos-1]) if pos else None,
                                     sparse_persistence25=float(sparse_persist[pos-1]) if pos else None,
                                     dense_selected_word_rank_lower=[int(r[pos]) for r in ranks_lower],
                                     dense_selected_word_rank_upper=[int(r[pos]) for r in ranks_upper],
                                     dense_top_words=[tok.decode([int(t)]) for t in dense_words[pos,:5]],
                                     sparse_top_words=[tok.decode([int(t)]) for t in words[pos,:5]]))
                del h,z,hd,zd
        del J
        print(json.dumps(dict(event='readout_layer_completed',layer=layer)),flush=True)
    tensor_rows(HERE/'analysis/readouts.jsonl',rows)

    # Predeclared five depths; sparse coding is separate from token ranking.
    gp_rows=[]
    for layer in GP_LAYERS:
        J=lens.jacobians[layer].cuda()
        # Paper dictionary W_U J; RMSNorm scale is intentionally not silently
        # folded into this primary decomposition. Readouts above use full norm.
        dictionary=model.lm_head.weight.float()@J
        dictionary/=dictionary.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        generator=torch.Generator(device='cuda').manual_seed(_stable_hash(42,0x4A5350,layer))
        control=torch.randn(dictionary.shape,device='cuda',generator=generator)
        control/=control.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        all_targets=[]
        metadata=[]
        for sample in samples:
            # At most eight aligned reasoning positions, including pre/post.
            candidates=[i for i,l in enumerate(sample['labels']) if l['reasoning'] and
                        (l['step']%512==0 or i==len(sample['labels'])-1)]
            candidates=candidates[-8:]
            for arm in ARMS:
                for i in candidates:
                    all_targets.append(sample['arms'][arm]['residuals'][i,layer])
                    metadata.append(dict(idx=sample['idx'],arm=arm,layer=layer,**sample['labels'][i]))
        if not all_targets:
            continue
        targets=torch.stack(all_targets).cuda().float()
        got=nonnegative_gradient_pursuit(targets,dictionary,k=GP_K)
        random=nonnegative_gradient_pursuit(targets,control,k=GP_K)
        by_key={(r['idx'],r['arm'],r['step']):i for i,r in enumerate(metadata)}
        for i,meta in enumerate(metadata):
            baseline=dense_control(meta['arm'])
            di=by_key[meta['idx'],baseline,meta['step']]
            ids=got['ids'][i].cpu().tolist()
            coef=got['coefficients'][i].cpu().tolist()
            gain=got['gains'][i,-1].item()
            random_gain=random['gains'][i,-1].item()
            gp_rows.append(dict(**meta,baseline=baseline,k=GP_K,raw_reconstruction_gain=gain,
                                random_control_gain=random_gain,excess_gain=gain-random_gain,
                                active_atoms=int(got['nonzero_atoms'][i]),
                                component_cosine_to_dense=float(F.cosine_similarity(got['reconstruction'][i:i+1],got['reconstruction'][di:di+1])),
                                component_l2_to_dense=float((got['reconstruction'][i]-got['reconstruction'][di]).norm()),
                                token_ids=ids,coefficients=coef,words=[tok.decode([t]) for t in ids],
                                gain_curve=got['gains'][i].cpu().tolist(),
                                random_gain_curve=random['gains'][i].cpu().tolist()))
        del J,dictionary,control,targets,got,random
        torch.cuda.empty_cache()
        tensor_rows(HERE/'analysis/jspace_decomposition.jsonl',gp_rows)
        print(json.dumps(dict(event='jspace_layer_completed',layer=layer)),flush=True)
    summary=dict(status='completed',samples=len(samples),sample_ids=[x['idx'] for x in samples],
                 budget=2048,protection='prompt+recent64_inside_total_K',lens_n_prompts=lens.n_prompts,
                 input_alignment='passed/full',fixed_prefix_alignment='passed',
                 pair_accounting='passed/fixed_budget_analytic',readout_groups=summarize_readouts(rows),
                 gp_layers=GP_LAYERS,gp_k=GP_K,
                 inference_gpu=torch.cuda.get_device_name(),accuracy_benchmark=False,
                 caveats=['Small fixed-prefix pilot; descriptive, not causal workspace evidence.',
                          'Full and compact paths have matched dense kernel controls; cross-kernel numerical drift is separately recorded.',
                          'Sparse arms are teacher forced on dense prefixes, not their own generations.',
                          'Raw reconstruction gain is uncentered; random excess is not the paper variance metric.',
                          'Gradient pursuit is our explicit approximation; no assumption of Claude workspace depths.',
                          'Diagnostics are not formal accuracy or runtime comparisons.'])
    dump(HERE/'analysis/summary.json',summary)
    print('ANALYSIS_COMPLETED',flush=True)


if __name__=='__main__':
    analyze()
