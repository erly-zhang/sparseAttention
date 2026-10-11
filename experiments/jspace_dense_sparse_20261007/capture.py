"""Paired, fixed-token-prefix diagnosis of the frozen HMMT sparse algorithms.

All caches use the original selector implementation. Dense uses that same
attention kernel with a budget larger than the whole diagnostic trajectory;
it never selects or deletes positions. This is not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FROZEN = PROJECT / 'sparseAttention/experiments/reasoning_qhead_shared_topp_20260918/results/hmmt_k1024_k2048_full_compact_k4096_full_20261001'
sys.path[:0] = [str(FROZEN / 'source/ra_harness'), str(PROJECT / 'jacobian-lens')]
from kvcompress.engine.cache_utils import EvictCache, _stable_hash
from kvcompress.engine.modify_forward import wrap_evict_attn_forward
from kvcompress.engine.selector_grid_cache import SelectorGridLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

SPARSE_ARMS = ['full_random', 'full_w1', 'compact_random', 'compact_w1']
ARMS = ['dense', 'dense_full', *SPARSE_ARMS]


def dense_control(arm):
    return 'dense_full' if arm.startswith('full') else 'dense'


class DenseControlLayer(SelectorGridLayer):
    """All-history dense with matching pre-budget buffer shapes.

    Only the diagnostic dense control is customized. Sparse layer classes
    and their math/RNG stay unchanged. Buffer shapes can change FlashAttention
    split/reduction choices even when logical keys are identical.
    """
    def lazy_initialization(self,k,value_states=None):
        budget=self.token_budget
        if self.storage=='compact':
            self.token_budget=self.match_budget
        try:
            super().lazy_initialization(k,value_states)
        finally:
            self.token_budget=budget
        if self.storage=='full':
            self._backing_temp_k=self._temp_k
            self._backing_temp_v=self._temp_v

    def attention(self,q,k,v,scale):
        if not self.is_initialized:
            self.lazy_initialization(k,v)
        required=max(self._lengths)+k.shape[1]
        if self.storage=='full':
            width=min(self.capacity,max(self.match_budget,required))
            self._temp_k=self._backing_temp_k[:,:width]
            self._temp_v=self._backing_temp_v[:,:width]
        elif required>self.capacity:
            cap=min(self.max_cache_length,max(required,2*self.capacity))
            for name in ('k_cache','v_cache'):
                old=getattr(self,name)
                new=old.new_empty((old.shape[0],cap,*old.shape[2:]))
                new[:,:self.capacity].copy_(old)
                setattr(self,name,new)
            positions=torch.full((self.slot_positions.shape[0],cap),-1,device=self.device,dtype=torch.long)
            positions[:,:self.capacity].copy_(self.slot_positions)
            self.slot_positions=positions
            self._temp_k=torch.empty_like(self.k_cache)
            self._temp_v=torch.empty_like(self.v_cache)
            self.capacity=cap
        return super().attention(q,k,v,scale)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def ids_sha(ids):
    return hashlib.sha256(torch.as_tensor(ids).cpu().numpy().astype('<i8').tobytes()).hexdigest()


def make_cache(model, arm, budget, sample_id, capacity):
    if arm.startswith('dense'):
        storage,scorer=('full' if arm=='dense_full' else 'compact'),'random'
    else:
        storage,scorer=arm.split('_')
    cache=EvictCache(model.config, token_budget=capacity if arm.startswith('dense') else budget,
                      residual_length=0, eviction_mode=f'grid_{storage}_{scorer}_topk_protected',
                      rkv_lambda=0., smooth=False, verbose=False, top_p=.95,
                      temperature=1., rep_seed=0, mask_seed=42, sample_id=sample_id,
                      eviction_interval=64 if storage == 'compact' else 1,
                      query_window=1, max_cache_length=capacity, track_positions=False)
    if arm.startswith('dense'):
        for layer in cache.layers:
            layer.__class__=DenseControlLayer
            layer.match_budget=budget
    return cache


class SelectedRecorder:
    def __init__(self, model):
        self.enabled = False
        self.values = {}
        self.handles = []
        for layer, block in enumerate(model.model.layers):
            def hook(module, inp, out, layer=layer):
                if self.enabled:
                    h = out if torch.is_tensor(out) else out[0]
                    self.values[layer] = h[0, -1].detach().cpu()
            self.handles.append(block.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def replay(model, prompt_ids, continuation, arm, budget, idx, checkpoints):
    device = model.device
    p = len(prompt_ids)
    capacity = p + len(continuation) + 128
    cache = make_cache(model, arm, budget, idx, capacity)
    recorder = SelectedRecorder(model)
    saved = []
    output_top = []
    kept_steps = []
    tic = time.perf_counter()
    try:
        for step in range(len(continuation) + 1):
            recorder.enabled = step in checkpoints
            token_ids = prompt_ids if step == 0 else [continuation[step - 1]]
            absolute = 0 if step == 0 else p + step - 1
            inp = torch.tensor([token_ids], device=device)
            positions = torch.arange(absolute, absolute + len(token_ids), device=device)
            output = model.model(input_ids=inp, past_key_values=cache, use_cache=True,
                                 position_ids=positions[None], cache_position=positions)
            if recorder.enabled:
                kept_steps.append(step)
                saved.append(torch.stack([recorder.values[l] for l in range(len(model.model.layers))]))
                output_top.append(model.lm_head(output.last_hidden_state[:, -1]).float().cpu()[0].topk(25).indices)
            if step and step % 512 == 0:
                print(json.dumps(dict(event='replay', idx=idx, arm=arm, step=step)), flush=True)
        layers = cache.layers
        selected = sum(x.selected_pairs for x in layers)
        causal = sum(x.causal_pairs for x in layers)
        factor = model.config.num_hidden_layers * model.config.num_attention_heads
        expected_causal = factor * (p * (p + 1) // 2 + sum(p + step for step in range(1, len(continuation) + 1)))
        assert causal == expected_causal
        assert all(x.cumulative_length == p + len(continuation) for x in layers)
        expected_selected = 0
        n = p
        for step in range(1, len(continuation) + 1):
            n += 1
            expected_selected += min(budget, n) if arm.startswith('full') else n
            if arm.startswith('compact') and step % 64 == 0 and n > budget:
                n = budget
        if arm.startswith('dense'):
            assert selected == causal
            if arm=='dense':
                assert sum(x.selection_events for x in layers) == 0
        else:
            assert selected == factor * (p * (p + 1) // 2 + expected_selected)
        stats = dict(selected_pairs=selected, causal_pairs=causal,
                     global_pair_sparsity=1-selected/causal,
                     selection_events_per_layer=[x.selection_events for x in layers],
                     final_cache_lengths=[x.cache_seqlens.tolist() for x in layers],
                     wall_seconds_with_capture=time.perf_counter()-tic,
                     pair_accounting='passed/fixed_budget_analytic')
        return dict(arm=arm, idx=idx, budget=budget, input_ids_sha256=ids_sha(prompt_ids),
                    prefix_sha256=ids_sha(prompt_ids + continuation),
                    steps=torch.tensor(kept_steps), residuals=torch.stack(saved),
                    model_top25=torch.stack(output_top), stats=stats)
    finally:
        recorder.close()
        del recorder, cache
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def dense_continuation(model, prompt_ids, idx, max_new_tokens):
    """Same sampling law as the frozen runner; a private batch1 diagnostic RNG."""
    capacity = len(prompt_ids) + max_new_tokens + 128
    cache = make_cache(model, 'dense', capacity, idx, capacity)
    gen = torch.Generator(device=model.device).manual_seed(_stable_hash(42, idx))
    warper = TopPLogitsWarper(top_p=.95, min_tokens_to_keep=1)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    trace = []
    try:
        inp = torch.tensor([prompt_ids], device=model.device)
        for step in range(max_new_tokens):
            absolute = 0 if step == 0 else len(prompt_ids) + step - 1
            positions = torch.arange(absolute, absolute + inp.shape[1], device=model.device)
            out = model.model(input_ids=inp, past_key_values=cache, use_cache=True,
                              position_ids=positions[None], cache_position=positions)
            logits = model.lm_head(out.last_hidden_state[:, -1]).float() / .6
            token = int(torch.multinomial(warper(inp, logits).softmax(-1), 1, generator=gen)[0, 0])
            trace.append(token)
            if token in eos:
                break
            inp = torch.tensor([[token]], device=model.device)
            if step and step % 512 == 0:
                print(json.dumps(dict(event='dense_generate', idx=idx, step=step)), flush=True)
        return trace
    finally:
        del cache
        gc.collect()
        torch.cuda.empty_cache()


def checkpoints(p, n, budget):
    transition = max(1, budget - p + 1)
    first_event = ((transition + 63) // 64) * 64
    points = set(range(0, n + 1, 128)) | {n}
    for boundary in (transition, first_event, first_event + 64):
        points.update(boundary + offset for offset in (-2, -1, 0, 1, 2, 8, 32))
    return sorted(s for s in points if 0 <= s <= n)


def initialize():
    assert torch.cuda.device_count() == 1 and os.environ.get('SLURM_JOB_ID')
    model_path = '/scratch/yl258/jz648/Qwen3-4B'
    for rel, expected in json.loads((FROZEN/'model_assets_sha256.json').read_text()).items():
        assert hashlib.sha256((Path(model_path)/rel).read_bytes()).hexdigest() == expected, rel
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                attn_implementation='flash_attention_2',
                                                local_files_only=True).cuda().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def references():
    return [json.loads(x) for x in (FROZEN / 'inputs/hmmt.jsonl').read_text().splitlines()]


def canonical_prompt_ids(tok, reference):
    # prompt_text was decoded with skip_special_tokens=True for completion
    # alignment. Retokenizing it would lose chat control tokens. Rebuild the
    # original single-user chat and require its exact canonical token hash.
    text = reference['prompt_text']
    assert text.startswith('user\n') and text.endswith('\nassistant\n')
    content = text[len('user\n'):-len('\nassistant\n')]
    formatted = tok.apply_chat_template([dict(role='user',content=content)],
                                       tokenize=False,add_generation_prompt=True)
    ids = tok(formatted,add_special_tokens=False).input_ids
    assert len(ids) == reference['input_tokens'] and ids_sha(ids) == reference['input_ids_sha256']
    return ids


def smoke(model, tok):
    reference = references()[0]
    ids = canonical_prompt_ids(tok, reference)
    inp = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        original_logits = model(inp, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
    wrap_evict_attn_forward('/scratch/yl258/jz648/Qwen3-4B')
    # K=128 here is only a kernel/compaction correctness case, not a study arm.
    trace = dense_continuation(model, ids, reference['idx'], 256)
    if len(trace) < 192:
        # Fixed teacher-forced carrier for exercising compaction; never graded.
        trace = (tok(' Let us check this intermediate calculation carefully.', add_special_tokens=False).input_ids * 80)[:256]
    steps = sorted(set(range(0, len(trace)+1, 16)) | {len(trace)})
    captures = {arm: replay(model, ids, trace, arm, 128, reference['idx'], steps) for arm in ARMS}
    dense = captures['dense']
    # Compare original HF dense prefill to the shared-kernel dense control.
    h = dense['residuals'][0, -1].cuda().float()
    with torch.inference_mode():
        logits = model.lm_head(model.model.norm(h.to(torch.bfloat16))).float().cpu()
    prefill_error = float((original_logits-logits).abs().max())
    assert prefill_error < .25, prefill_error
    for arm, capture in captures.items():
        before = capture['steps'] <= 128-len(ids)
        if arm in SPARSE_ARMS:
            baseline=captures[dense_control(arm)]
            error=float((capture['residuals'][before]-baseline['residuals'][before]).abs().max())
            print(json.dumps(dict(event='before_sparse_check',arm=arm,baseline=dense_control(arm),max_abs_error=error)),flush=True)
            if error:
                torch.save(captures,HERE/f"smoke_debug_{os.environ['SLURM_JOB_ID']}.pt")
            torch.testing.assert_close(capture['residuals'][before], baseline['residuals'][before], rtol=0, atol=0)
            assert min(capture['stats']['selection_events_per_layer']) > 0
    dump(HERE/'smoke.json', dict(status='passed', original_hf_prefill_max_logit_error=prefill_error,
                               before_sparse_residuals='bitwise_identical_to_matched_dense_kernel',
                               dense_kernel_calibration_max_residual_error=float((captures['dense_full']['residuals']-dense['residuals']).abs().max()),
                               input_alignment='passed', all_six_paths=True,
                               pair_accounting='passed/fixed_budget_analytic',
                               gpu=torch.cuda.get_device_name(),
                               slurm_job=os.environ['SLURM_JOB_ID']))
    print('SMOKE_PASSED', flush=True)


def capture_study(model, tok, max_new_tokens, sample_ids):
    assert json.loads((HERE/'smoke.json').read_text())['status'] == 'passed'
    wrap_evict_attn_forward('/scratch/yl258/jz648/Qwen3-4B')
    for ref in references():
        if ref['idx'] not in sample_ids:
            continue
        out = HERE/'captures'/f"sample_{ref['idx']}"
        out.mkdir(parents=True, exist_ok=True)
        ids = canonical_prompt_ids(tok, ref)
        trajectory_path = out/'dense_trajectory.json'
        if trajectory_path.exists():
            trajectory = json.loads(trajectory_path.read_text())
            assert trajectory['input_ids_sha256'] == ids_sha(ids)
            continuation = trajectory['continuation_ids']
        else:
            continuation = dense_continuation(model, ids, ref['idx'], max_new_tokens)
            dump(trajectory_path, dict(idx=ref['idx'], input_tokens=len(ids), prompt_ids=ids,
                                       input_ids_sha256=ids_sha(ids), continuation_ids=continuation,
                                       continuation_text=tok.decode(continuation),
                                       max_new_tokens=max_new_tokens, sampling=dict(seed=42,temperature=.6,top_p=.95),
                                       diagnostic_batch_size=1, accuracy_benchmark=False))
        points = checkpoints(len(ids), len(continuation), 2048)
        for arm in ARMS:
            path = out/f'{arm}.pt'
            if path.exists():
                continue
            record = replay(model, ids, continuation, arm, 2048, ref['idx'], points)
            if arm in SPARSE_ARMS:
                dense_record = torch.load(out/f'{dense_control(arm)}.pt',map_location='cpu',weights_only=True)
                before = record['steps'] <= 2048-len(ids)
                torch.testing.assert_close(record['residuals'][before],dense_record['residuals'][before],rtol=0,atol=0)
                record['before_sparse_residuals'] = 'bitwise_identical'
                record['paired_dense_control'] = dense_control(arm)
            record['prompt_tokens'] = len(ids)
            record['continuation_tokens'] = len(continuation)
            record['gpu'] = torch.cuda.get_device_name()
            torch.save(record, path)
            dump(out/f'{arm}_stats.json', record['stats'])
            print(json.dumps(dict(event='arm_completed',idx=ref['idx'],arm=arm,points=len(points))),flush=True)
        dump(out/'complete.json',dict(status='completed',input_alignment='passed/full',
                                      fixed_prefix_alignment='passed',arms=ARMS,
                                      sparse_history_exercised=len(ids)+len(continuation)>2048+64,
                                      accuracy_benchmark=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke','capture'])
    parser.add_argument('--max-new-tokens',type=int,default=4096)
    parser.add_argument('--samples',type=int,nargs='+',default=[0,1,2])
    args = parser.parse_args()
    model, tok = initialize()
    if args.phase == 'smoke':
        smoke(model,tok)
    else:
        capture_study(model,tok,args.max_new_tokens,args.samples)
