"""Export checkpoint-based Figure 5 data, without generating any new tokens.

Uses jlens' Apache-2.0 meaningful-token display filter (Anthropic PBC).
Ranks refer to the full vocabulary, with exact finite-precision tie intervals.
Every displayed token is trackable at every exported position/layer/arm.
"""
from __future__ import annotations
import base64
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from free_generate import HERE, PILOT, ARMS, initialize
from analyze import phase_labels, readout
import jlens
from jlens.vis import _meaningful_token_mask

OUT = HERE / 'figure5_html_en'
N = 8


def bounds(sorted_logits, values):
    """1-based full-vocabulary rank interval; a tie has no arbitrary order."""
    v = sorted_logits.shape[-1]
    lo = v - torch.searchsorted(sorted_logits, values.contiguous(), right=True) + 1
    hi = v - torch.searchsorted(sorted_logits, values.contiguous(), right=False)
    return lo, hi


@torch.inference_mode()
def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--display',choices=['wordlike','english'],default='english')
    args=parser.parse_args()
    OUT=args.output
    OUT.mkdir(exist_ok=True)
    # Match the A100 own-generation readout calculation, which used full FP32
    # transport rather than TF32. Check every Top8 against those saved readouts.
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tok = initialize()
    lens = jlens.JacobianLens.load(str(next((PILOT/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))))
    mask = _meaningful_token_mask(tok, model.config.vocab_size, torch.device('cuda'))
    if args.display=='english':
        # Display scope is explicitly ASCII letters/numbers, not language inference.
        strings=[tok.decode([t],clean_up_tokenization_spaces=False) for t in range(model.config.vocab_size)]
        mask=torch.tensor([s.isascii() and bool(re.search('[A-Za-z0-9]',s)) for s in strings],device='cuda')
    display='top 8 ASCII letter/number-containing tokens selected from the full vocabulary' if args.display=='english' else 'top 8 letter/number-containing tokens; ranking uses the full vocabulary'
    manifest = dict(schema=1, source='saved residual checkpoints; no new model generation',
                    trajectory_mode='own-prefix free generation',teacher_forced=False,
                    rank='[1 + count(logit > token_logit), count(logit >= token_logit)]',
                    display=display,display_filter=args.display,
                    shape_order=['arm','checkpoint','layer','top_token'], samples=[],
                    run_config=json.loads((HERE/'run_config.json').read_text()),
                    lens_fit=json.loads((PILOT/'run_config.json').read_text())['lens_n_prompts'],
                    export_gpu=torch.cuda.get_device_name(),tf32_transport=False)
    summary = json.loads((HERE/'summary.json').read_text())
    parity = dict(status='passed',samples=[])
    for directory in sorted((HERE/'captures').glob('sample_*')):
        trajectories = {a:json.loads((directory/f'{a}.json').read_text()) for a in ARMS}
        trajectory = trajectories['dense']
        idx = trajectory['idx']
        old = [json.loads(x) for x in (directory/'readouts.jsonl').read_text().splitlines()]
        old = {(r['idx'], r['arm'], r['step'], r['layer']): r for r in old}
        captures = {a: torch.load(directory/f'{a}.pt', map_location='cpu', weights_only=True) for a in ARMS}
        assert all(torch.equal(c['steps'], captures['dense']['steps']) and c['input_ids_sha256']==captures['dense']['input_ids_sha256'] for c in captures.values())
        assert len({c['prefix_sha256'] for c in captures.values()})==5
        steps = captures['dense']['steps'].tolist()
        p = len(steps)
        shape = (len(ARMS), p, 36, N)
        ids = np.zeros(shape, dtype=np.int32)
        low = np.zeros(shape, dtype=np.int32)
        high = np.zeros(shape, dtype=np.int32)
        probs = np.zeros(shape, dtype=np.float32)
        for layer in range(36):
            J = lens.jacobians[layer].cuda() if layer<35 else None
            for ai, arm in enumerate(ARMS):
                z = readout(model, captures[arm]['residuals'][:,layer], J)
                top = z.masked_fill(~mask, -torch.inf).topk(N, dim=-1).indices
                vals = z.gather(1,top)
                sorted_z = z.sort(dim=-1).values
                lower, upper = bounds(sorted_z, vals)
                ids[ai,:,layer] = top.cpu().numpy()
                low[ai,:,layer] = lower.cpu().numpy()
                high[ai,:,layer] = upper.cpu().numpy()
                probs[ai,:,layer] = torch.exp(vals-z.logsumexp(-1,keepdim=True)).cpu().numpy()
                for pi, step in enumerate(steps):
                    saved=old[(idx,arm,step,layer)]['top8']
                    assert ids[ai,pi,layer].tolist()==[t['token_id'] for t in saved]
                    assert low[ai,pi,layer].tolist()==[t['rank_lower'] for t in saved]
                    assert high[ai,pi,layer].tolist()==[t['rank_upper'] for t in saved]
                    expected=np.array([t['probability'] for t in saved],dtype=np.float32)
                    assert np.allclose(probs[ai,pi,layer],expected,rtol=2e-5,atol=1e-7)
                    probs[ai,pi,layer]=expected
            if layer%6==0:
                print(json.dumps(dict(event='figure5_top', idx=idx, layer=layer)),flush=True)
        tracked = sorted(set(ids.ravel().tolist()))
        ranks = np.zeros((len(ARMS),p,36,len(tracked),2),dtype=np.uint32)
        tracked_tensor = torch.tensor(tracked,device='cuda')
        for layer in range(36):
            J = lens.jacobians[layer].cuda() if layer<35 else None
            for ai,arm in enumerate(ARMS):
                z = readout(model,captures[arm]['residuals'][:,layer],J)
                lo,hi = bounds(z.sort(-1).values,z[:,tracked_tensor])
                ranks[ai,:,layer,:,0] = lo.cpu().numpy()
                ranks[ai,:,layer,:,1] = hi.cpu().numpy()
                # Independent direct comparisons verify ties for representative pins.
                if layer in (0,17,35):
                    for ti in (0,len(tracked)//2,len(tracked)-1):
                        value = z[:,tracked_tensor[ti]:tracked_tensor[ti]+1]
                        assert torch.equal(lo[:,ti], (z>value).sum(-1)+1)
                        assert torch.equal(hi[:,ti], (z>=value).sum(-1))
            if layer%6==0:
                print(json.dumps(dict(event='figure5_ranks',idx=idx,layer=layer,tracked=len(tracked))),flush=True)
        lookup = {tid:i for i,tid in enumerate(tracked)}
        ti = np.vectorize(lookup.__getitem__)(ids)
        gathered = np.take_along_axis(ranks, ti[...,None], axis=3)
        assert np.array_equal(gathered[...,0],low) and np.array_equal(gathered[...,1],high)
        encoded = {str(tid):base64.b64encode(gzip.compress(ranks[:,:,:,i,:].astype('<u4').tobytes(),compresslevel=9,mtime=0)).decode() for i,tid in enumerate(tracked)}
        vocab = {str(t):tok.decode([t],clean_up_tokenization_spaces=False) for t in tracked}
        labels_by_arm=[];snippets_by_arm=[]
        for arm in ARMS:
            tr=trajectories[arm]
            labels=phase_labels(tr,steps,tok)
            context=tr['prompt_ids']+tr['continuation_ids']
            snippets=[]
            for label in labels:
                r=old[(idx,arm,label['step'],0)]
                label['reasoning']=r['reasoning']
                label['same_token_prefix_as_dense']=r['same_token_prefix_as_dense']
                absolute=label['absolute_position']
                snippets.append([tok.decode(context[max(0,absolute-25):absolute]),tok.decode([context[absolute]]),tok.decode(context[absolute+1:absolute+26])])
            labels_by_arm.append(labels);snippets_by_arm.append(snippets)
        metrics=[]
        for arm in ARMS:
            arm_metrics=[]
            for step in steps:
                for layer in range(36):
                    r=old[(idx,arm,step,layer)]
                    arm_metrics.append([round(r['recall25_vs_dense_at_same_step'],5),round(r['js_nats_vs_dense_at_same_step'],7)])
            metrics.append(arm_metrics)
        defaults=[]
        for word in ('answer','therefore','sum','check','probably','perhaps'):
            candidates=[t for t in tracked if vocab[str(t)].strip().lower()==word]
            if candidates:
                # Prefer the most frequently displayed spelling/token identity.
                defaults.append(max(candidates,key=lambda t:int((ids==t).sum())))
        if len(defaults)<3:
            defaults += [int(t) for t in np.unique(ids[0,:,17:35,0])[:3] if int(t) not in defaults]
        if args.display=='english':
            assert all(s.isascii() and re.search('[A-Za-z0-9]',s) for s in vocab.values())
        data=dict(idx=idx,P=p,L=36,N=N,arms=ARMS,vocab_size=model.config.vocab_size,display_filter=args.display,
                  labels=labels_by_arm[0],labels_by_arm=labels_by_arm,
                  prompt_tokens=trajectory['input_tokens'],continuation_tokens=len(trajectory['continuation_ids']),
                  prompt=tok.decode(trajectory['prompt_ids']),continuations=[trajectories[a]['continuation_text'] for a in ARMS],
                  snippets_by_arm=snippets_by_arm,
                  trajectory_stats=[next(r for r in summary['results'] if r['idx']==idx and r['arm']==a) for a in ARMS],
                  pairs=[r for r in summary['pairs'] if r['idx']==idx],
                  prefix_sha256_by_arm=[captures[a]['prefix_sha256'] for a in ARMS],
                  top_ids=ids.ravel().tolist(),top_low=low.ravel().tolist(),top_high=high.ravel().tolist(),
                  top_probs=[float(f'{x:.6g}') for x in probs.ravel()],metrics=metrics,
                  vocab=vocab,tracked=tracked,pins=defaults[:3],ranks=encoded,
                  input_ids_sha256=trajectory['input_ids_sha256'],prefix_sha256=captures['dense']['prefix_sha256'])
        body='window.FIGURE5_CASES=window.FIGURE5_CASES||{};window.FIGURE5_CASES['+str(idx)+']='+json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+';\n'
        path=OUT/f'case_{idx}.js'
        path.write_text(body)
        manifest['samples'].append(dict(idx=idx,checkpoints=p,tracked_tokens=len(tracked),file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         input_ids_sha256=trajectory['input_ids_sha256'],prefix_sha256=captures['dense']['prefix_sha256'],rank_validation='passed: direct full-vocabulary comparisons and all top8 cells'))
        print(json.dumps(dict(event='figure5_exported',**manifest['samples'][-1])),flush=True)
        parity['samples'].append(dict(idx=idx,top8_entries_exact_ids_and_rank_bounds=len(ARMS)*p*36*N,
                                      probability_source='original saved readouts',own_prefix_hashes_distinct=5))
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'original_readout_parity.json').write_text(json.dumps(parity,indent=2)+'\n')


if __name__=='__main__':
    main()
