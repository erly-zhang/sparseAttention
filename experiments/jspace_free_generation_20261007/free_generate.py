"""Own-prefix free-generation follow-up to the fixed-prefix J-lens pilot.

Frozen selectors and fixed lens; no dense-token forcing. One diagnostic
rollout per original input/method. Captures include a final, unsampled forward
of the final emitted token, explicitly separated from generation timings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'jspace_dense_sparse_20261007'
sys.path.insert(0, str(PILOT))
import torch
from capture import (FROZEN, SelectedRecorder, _stable_hash, canonical_prompt_ids,
                     checkpoints, dump, ids_sha, initialize, make_cache, references,
                     wrap_evict_attn_forward)
from transformers.generation.logits_process import TopPLogitsWarper
from analyze import readout
from analysis_math import js_divergence, topk_recall
import jlens

ARMS = ['dense', 'full_random', 'full_w1', 'compact_random', 'compact_w1']
LABELS = dict(dense='Dense', full_random='Random, full KV', full_w1='W1, full KV',
              compact_random='Random, compact KV', compact_w1='W1, compact KV')
CAP = 4096
K = 2048


def source_hashes():
    paths = [Path(__file__), PILOT/'capture.py', PILOT/'analyze.py',
             FROZEN/'source/ra_harness/kvcompress/engine/selector_grid_cache.py',
             FROZEN/'inputs/hmmt.jsonl',
             next((PILOT/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def repeated_fraction(ids, size=16):
    if len(ids) < size:
        return 0.0
    counts = Counter(tuple(ids[i:i+size]) for i in range(len(ids)-size+1))
    return sum(n-1 for n in counts.values()) / (len(ids)-size+1)


def periodic_suffix(ids):
    """Exact consecutive repetition at the end, period <=64, at least3 copies."""
    best = dict(period=None, copies=0, tokens=0)
    for p in range(1, min(64, len(ids)//3)+1):
        pattern = ids[-p:]
        copies = 1
        while (copies+1)*p <= len(ids) and ids[-(copies+1)*p:-copies*p] == pattern:
            copies += 1
        if copies >= 3 and copies*p > best['tokens']:
            best = dict(period=p, copies=copies, tokens=copies*p)
    return best


def count_pairs(p, steps, arm, diagnostic_final=False):
    factor = 36 * 32
    n = p
    selected = p*(p+1)//2
    causal = selected
    events = 0
    for step in range(1, steps+1):
        n += 1
        causal += p+step
        selected += min(K, n) if arm.startswith('full') else n
        if arm.startswith('full'):
            events += 1
        elif arm.startswith('compact') and step % 64 == 0 and n > K:
            events += 1
            n = K
    return factor*selected, factor*causal, events, n


@torch.inference_mode()
def generate(model, tok, ref, arm, cap=CAP, capture=True):
    ids = canonical_prompt_ids(tok, ref)
    p = len(ids)
    # Keep smoke/study buffer geometry identical even for a short smoke trace.
    cache = make_cache(model, arm, K, ref['idx'], p+max(cap,CAP)+128)
    # Separate CUDA stream of random numbers from all selector RNGs. Same
    # seed/identity as the old diagnostic Dense generator, reset for every arm.
    rng = torch.Generator(device=model.device).manual_seed(_stable_hash(42, ref['idx']))
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state().clone()
    warper = TopPLogitsWarper(top_p=.95, min_tokens_to_keep=1)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    recorder = SelectedRecorder(model)
    points = set(checkpoints(p, cap, K)) if capture else set()
    trace, saved, steps, tops = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    generation_seconds = None
    try:
        # After each forward, sample the next input from THIS arm's logits.
        # One additional final forward captures state after its last token.
        for step in range(cap+1):
            final = step == cap or (bool(trace) and trace[-1] in eos)
            recorder.enabled = capture and (step in points or final)
            token_ids = ids if step == 0 else [trace[-1]]
            absolute = 0 if step == 0 else p+step-1
            inp = torch.tensor([token_ids], device=model.device)
            positions = torch.arange(absolute, absolute+len(token_ids), device=model.device)
            out = model.model(input_ids=inp, past_key_values=cache, use_cache=True,
                              position_ids=positions[None], cache_position=positions)
            logits = model.lm_head(out.last_hidden_state[:, -1]).float()
            if recorder.enabled:
                saved.append(torch.stack([recorder.values[l] for l in range(36)]))
                steps.append(step)
                tops.append(logits.cpu()[0].topk(25).indices)
            if final:
                break
            sampling_logits = warper(inp, logits/.6)
            token = int(torch.multinomial(sampling_logits.softmax(-1), 1, generator=rng)[0, 0])
            trace.append(token)
            if len(trace) == cap or token in eos:
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter()-start
            if len(trace) % 512 == 0:
                print(json.dumps(dict(event='own_prefix_generate',idx=ref['idx'],arm=arm,
                                      generated=len(trace),elapsed=round(time.perf_counter()-start,1))),flush=True)
        assert len(trace) >= 1
        selected = sum(x.selected_pairs for x in cache.layers)
        causal = sum(x.causal_pairs for x in cache.layers)
        ex_selected, ex_causal, events, final_length = count_pairs(p, len(trace), arm)
        assert selected == ex_selected and causal == ex_causal, (arm, selected, ex_selected)
        assert all(x.cumulative_length == p+len(trace) for x in cache.layers)
        assert all(x._lengths == [final_length] for x in cache.layers)
        if arm != 'dense':
            assert all(x.selection_events == events for x in cache.layers)
        else:
            assert all(x.selection_events == 0 for x in cache.layers)
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_state, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_state, rtol=0, atol=0)
        text = tok.decode(trace, clean_up_tokenization_spaces=False)
        close = tok.convert_tokens_to_ids('</think>')
        close_pos = trace.index(close)+1 if close in trace else None
        trajectory = dict(status='completed',idx=ref['idx'],arm=arm,label=LABELS[arm],
                          prompt_ids=ids,input_tokens=p,input_ids_sha256=ids_sha(ids),
                          continuation_ids=trace,continuation_ids_sha256=ids_sha(trace),
                          prefix_sha256=ids_sha(ids+trace),continuation_text=text,
                          gold=ref['gold'],generated_tokens=len(trace),max_new_tokens=cap,
                          stop_reason='eos' if trace[-1] in eos else 'max_new_tokens',
                          think_closed=close_pos is not None,think_close_token=close_pos,
                          total_K=K,protection='prompt+recent64_inside_total_K',
                          compact_interval=64,query_window=1,teacher_forced=False,
                          sampling=dict(seed=42,temperature=.6,top_p=.95,top_k=None,batch_size=1,
                                        rng='private CUDA generator; identical seed/input stream per arm'),
                          duplicate_16gram_fraction=repeated_fraction(trace),
                          tail1024_duplicate_16gram_fraction=repeated_fraction(trace[-1024:]),
                          exact_periodic_suffix=periodic_suffix(trace),
                          gpu=torch.cuda.get_device_name(),node=os.environ.get('SLURMD_NODENAME'),
                          slurm_job=os.environ.get('SLURM_JOB_ID'),
                          wall_generation_seconds_with_capture=generation_seconds,
                          wall_all_seconds_with_final_capture=time.perf_counter()-start,
                          peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                          timing_scope='diagnostic including sampling, residual capture and CPU transfers; not formal decode timing',
                          diagnostic_final_forward=True,
                          selected_pairs_including_final_forward=selected,
                          causal_pairs_including_final_forward=causal,
                          global_pair_sparsity_including_final_forward=1-selected/causal,
                          selection_events_per_layer=[x.selection_events for x in cache.layers],
                          final_cache_lengths=[x.cache_seqlens.tolist() for x in cache.layers],
                          pair_accounting='passed/analytic',global_rng_unchanged=True,
                          accuracy_benchmark=False)
        record = dict(idx=ref['idx'],arm=arm,steps=torch.tensor(steps),
                      residuals=torch.stack(saved) if saved else torch.empty(0),
                      model_top25=torch.stack(tops) if tops else torch.empty(0,dtype=torch.long),
                      input_ids_sha256=ids_sha(ids),prefix_sha256=ids_sha(ids+trace))
        return trajectory, record
    finally:
        recorder.close()
        del cache, recorder
        gc.collect()
        torch.cuda.empty_cache()


def validate_sample(out):
    trajectories = {a:json.loads((out/f'{a}.json').read_text()) for a in ARMS}
    dense = trajectories['dense']
    result = dict(status='passed',idx=dense['idx'],teacher_forced=False,
                  input_alignment='passed/full',pairs='passed/analytic',checks=[],arms=[])
    for arm,t in trajectories.items():
        assert t['input_ids_sha256'] == dense['input_ids_sha256']
        assert t['global_rng_unchanged'] and t['pair_accounting']=='passed/analytic'
        common = lcp(dense['continuation_ids'], t['continuation_ids'])
        first_full = max(1,K-dense['input_tokens']+1)
        first_visible = first_full if arm.startswith('full') else ((first_full+63)//64)*64+1
        # The first sparse-affected output is token first_visible+1. All
        # previously generated tokens must match under the common sampler.
        if arm != 'dense':
            assert common >= min(first_visible, len(t['continuation_ids']), len(dense['continuation_ids'])), (arm,common,first_visible)
        record = torch.load(out/f'{arm}.pt',map_location='cpu',weights_only=True)
        dr = torch.load(out/'dense.pt',map_location='cpu',weights_only=True)
        dm = {int(s):i for i,s in enumerate(dr['steps'])}
        checked = 0
        for i,s in enumerate(record['steps'].tolist()):
            if s in dm and s < first_visible:
                torch.testing.assert_close(record['residuals'][i],dr['residuals'][dm[s]],rtol=0,atol=0)
                checked += 1
        assert checked > 0
        first_diff = common+1 if common < min(len(t['continuation_ids']),len(dense['continuation_ids'])) else None
        result['arms'].append(dict(arm=arm,common_prefix_tokens_vs_dense=common,
                                  first_different_generated_token=first_diff,
                                  first_sparse_affected_output_token=first_visible+1 if arm!='dense' else None,
                                  pre_sparse_equal_residual_checkpoints=checked,
                                  own_prefix_sha256=t['prefix_sha256']))
    old = json.loads((PILOT/'captures'/out.name/'dense_trajectory.json').read_text())
    result['dense_common_prefix_tokens_vs_old_pilot'] = lcp(old['continuation_ids'],dense['continuation_ids'])
    result['old_pilot_gpu_difference'] = 'Old pilot used L40; new GPU is recorded per trajectory. Exact old-token replay is not imposed.'
    result['checks'] = ['five unique methods','canonical input token hashes','private sampler and selector RNG',
                        'exact common generation prefix before sparsity','bitwise pre-sparse residual equality',
                        'actual selected/causal pairs','no forced Dense continuation','EOS or explicit cap']
    dump(out/'validation.json',result)
    return result


@torch.inference_mode()
def analyze_sample(model,tok,out):
    lens = jlens.JacobianLens.load(str(next((PILOT/'assets').rglob('Qwen3-4B_jacobian_lens.pt'))))
    assert lens.d_model == 2560 and lens.source_layers == list(range(35))
    trajectories = {a:json.loads((out/f'{a}.json').read_text()) for a in ARMS}
    captures = {a:torch.load(out/f'{a}.pt',map_location='cpu',weights_only=True) for a in ARMS}
    strings = [tok.decode([i],clean_up_tokenization_spaces=False) for i in range(model.config.vocab_size)]
    import re
    english = torch.tensor([s.isascii() and bool(re.search('[A-Za-z0-9]',s)) for s in strings],device='cuda')
    rows = []
    dr = captures['dense']
    dmap = {int(s):i for i,s in enumerate(dr['steps'])}
    for layer in range(36):
        J = lens.jacobians[layer].cuda() if layer<35 else None
        dz = readout(model,dr['residuals'][:,layer],J)
        for arm in ARMS:
            c = captures[arm]
            z = dz if arm=='dense' else readout(model,c['residuals'][:,layer],J)
            probs = z.softmax(-1)
            top = z.masked_fill(~english,-float('inf')).topk(8,dim=-1).indices
            vals = z.gather(1,top)
            # ranks preserve exact finite precision tie intervals in full vocab.
            lo = (z[:,:,None]>vals[:,None,:]).sum(1)+1
            hi = (z[:,:,None]>=vals[:,None,:]).sum(1)
            tp = probs.gather(1,top)
            at = c['steps'].tolist()
            matched = [i for i,s in enumerate(at) if s in dmap]
            di = [dmap[at[i]] for i in matched]
            recall = topk_recall(dz[di].topk(25,dim=-1).indices,z[matched].topk(25,dim=-1).indices)
            js = js_divergence(dz[di],z[matched])
            pair = {i:(float(recall[j]),float(js[j])) for j,i in enumerate(matched)}
            common = lcp(trajectories['dense']['continuation_ids'],trajectories[arm]['continuation_ids'])
            close = trajectories[arm]['think_close_token']
            for i,s in enumerate(at):
                tids = top[i].cpu().tolist()
                rows.append(dict(idx=trajectories[arm]['idx'],arm=arm,step=s,layer=layer,
                                 reasoning=s>0 and (close is None or s<close),
                                 same_token_prefix_as_dense=s<=common,
                                 comparison_scope='same-prefix' if s<=common else 'different-own-prefixes; mixed attention and text-history effects',
                                 recall25_vs_dense_at_same_step=pair[i][0] if i in pair else None,
                                 js_nats_vs_dense_at_same_step=pair[i][1] if i in pair else None,
                                 top8=[dict(token_id=t,token=strings[t],rank_lower=int(lo[i,k]),
                                            rank_upper=int(hi[i,k]),probability=float(tp[i,k])) for k,t in enumerate(tids)]))
            if arm != 'dense':del z
        del dz,J
    with (out/'readouts.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(out/'readout_complete.json',dict(status='completed',rows=len(rows),
         lens_sha256=hashlib.sha256(next((PILOT/'assets').rglob('Qwen3-4B_jacobian_lens.pt')).read_bytes()).hexdigest(),
         fixed_lens=True,display_filter='ASCII letters/numbers; all ranks and probabilities full vocabulary',
         own_prefixes=True,causal_interpretation_after_divergence=False))


def run(idx):
    model,tok = initialize()
    assert json.loads((PILOT/'smoke.json').read_text())['status']=='passed'
    wrap_evict_attn_forward('/scratch/yl258/jz648/Qwen3-4B')
    ref = next(r for r in references() if r['idx']==idx)
    out = HERE/'captures'/f'sample_{idx}'
    out.mkdir(parents=True,exist_ok=True)
    # A new sampling-loop correctness check on the actual execution GPU:
    # complete 64-token own-generation paths must match before pruning starts.
    if not (out/'smoke.json').exists():
        traces = {}
        for arm in ARMS:
            t,_ = generate(model,tok,ref,arm,cap=64,capture=False)
            traces[arm] = t['continuation_ids']
        assert all(traces[a]==traces['dense'] for a in ARMS)
        dump(out/'smoke.json',dict(status='passed',new_sampler_before_sparse=True,
             private_global_rng_unchanged=True,all_five_methods_equal=True,
             generated_tokens=len(traces['dense']),gpu=torch.cuda.get_device_name()))
        print('FREE_GENERATION_SMOKE_PASSED',flush=True)
    for arm in ARMS:
        path = out/f'{arm}.json'
        if path.exists():
            assert (out/f'{arm}.pt').exists()
            continue
        trajectory,record = generate(model,tok,ref,arm)
        torch.save(record,out/f'{arm}.pt')
        (out/f'{arm}.txt').write_text(trajectory['continuation_text'])
        dump(path,trajectory)
        print(json.dumps(dict(event='arm_completed',idx=idx,arm=arm,tokens=trajectory['generated_tokens'],
                              stop=trajectory['stop_reason'],think_closed=trajectory['think_closed'])),flush=True)
    validation = validate_sample(out)
    if not (out/'readout_complete.json').exists():analyze_sample(model,tok,out)
    dump(out/'complete.json',dict(status='completed',sample_id=idx,arms=ARMS,
                                 input_alignment=validation['input_alignment'],teacher_forced=False,
                                 source_sha256=source_hashes()))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample',type=int,choices=[0,1,2],required=True)
    args = ap.parse_args()
    run(args.sample)
