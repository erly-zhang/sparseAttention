"""Where does a hybrid W1 decode step spend its time? Real Qwen3-4B, teacher forcing.

Every variant decodes the identical saved token sequences (HMMT idx51, both
rollouts of the formal hybrid K1024 run), so lengths and contexts are matched.
The per-step loop mirrors batch_exist_generate: forward, logits clone, top-p
sampling (result discarded), token/mask concatenation and one .item() sync.

Variants (timing diagnostics only; ablations are NOT valid inference methods):
  dense            HF DynamicCache + FlashAttention2
  hybrid_v2        formal hybrid1000 K1024, packed_qk_topk backend
  hybrid_v3        fused v3 candidate, same backend
  no_reselect      v2, but reuse the previous selection IDs (no scoring/TopK)
  pool_attention   attend directly to the bounded pool; select only at compaction
"""
import argparse, json, os, platform, sys, time
from pathlib import Path
import torch

E = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(E/'ra_harness'), str(E/'ra_harness/kvcompress/harness')]
MODEL = '/scratch/yl258/jz648/Qwen3-4B'
TOKENS = E/'results/hmmt_w1_raw_qk_fast_full_hybrid_k1024_20261012_packed32/cells/hmmt_k1024_w1_raw_qk_fast_hybrid_protected/formal/shard_11/token_ids.json'
HYBRID = 'grid_hybrid1000_w1_topk_protected'


def install(variant):
    import kvcompress.engine.selector_grid_cache as grid
    from kvcompress.engine.w1_interval1000_cache import install as install_interval
    grid.SelectorGridLayer, grid.decode_mode = ORIGINAL
    if variant == 'hybrid_v6':
        from kvcompress.engine.w1_hybrid_v6_cache import install as v6
        v6('packed_qk_topk')
        return
    if variant == 'hybrid_v5':
        from kvcompress.engine.w1_hybrid_v5_cache import install as v5
        v5('packed_qk_topk')
        return
    if variant == 'hybrid_v4':
        from kvcompress.engine.w1_hybrid_gpu_select_cache import install as v4
        v4('packed_qk_topk')
        return
    if variant == 'hybrid_v3':
        from kvcompress.engine.w1_qk_fused_cache import install as fused
        fused('packed_qk_topk')
        return
    from kvcompress.engine.w1_qk_fast_cache import install as fast, FastHybridW1Layer
    fast('packed_qk_topk')
    if variant in ('no_reselect', 'pool_attention'):
        base = grid.SelectorGridLayer

        class NoReselect(FastHybridW1Layer):
            def select_indices(self):
                step = self.cumulative_length-self.prompt_len
                # Reselect at and right after each compaction so reused slots stay valid.
                if self._ids is None or self._lengths[0] <= self.token_budget or step % 1000 in (0, 1):
                    return super().select_indices()
                return self._ids, [self._ids.shape[1]]*len(self._lengths)

        class PoolAttention(NoReselect):
            def attention(self, q, k, v, scale):
                from flash_attn import flash_attn_with_kvcache
                if not self.cumulative_length or (self.cumulative_length+1-self.prompt_len) % 1000 == 0:
                    return super().attention(q, k, v, scale)
                b = k.shape[0]
                out = flash_attn_with_kvcache(q, self.k_cache, self.v_cache, k=k, v=v,
                    cache_seqlens=self.cache_seqlens, causal=True, softmax_scale=scale)
                self.slot_positions[:, self._lengths[0]] = self.cumulative_length
                self.cumulative_length += 1
                self._lengths = [x+1 for x in self._lengths]
                self.cache_seqlens += 1
                return out
        cls = NoReselect if variant == 'no_reselect' else PoolAttention

        def factory(*args, **kwargs):
            if args[3] == HYBRID:
                return cls(*args, rank_backend='packed_qk_topk', **kwargs)
            return base(*args, **kwargs)
        grid.SelectorGridLayer = factory


def run(model, variant, prompt, teacher, steps, windows, profile_at):
    from transformers.generation.logits_process import TopPLogitsWarper
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from kvcompress.engine.cache_utils import EvictCache
    if variant == 'dense':
        Qwen3Attention.forward = ORIGINAL_FORWARD
        cache = DynamicCache()
    else:
        Qwen3Attention.forward = EVICT_FORWARD
        install(variant)
        mode = 'grid_hybrid1000_w1_topk_protected' if variant == 'hybrid_v5' else HYBRID
        cache = EvictCache(config=model.config, token_budget=1024, residual_length=0, eviction_mode=mode,
                           rkv_lambda=0., smooth=False, verbose=False, top_p=.95, query_window=1,
                           eviction_interval=1000, rep_seed=0, sample_id=51, max_cache_length=prompt.shape[1]+steps+2)
    warper = TopPLogitsWarper(top_p=.95, min_tokens_to_keep=1)
    generated = prompt.clone()
    mask = torch.ones_like(prompt)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(prompt, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    records, profile = [], None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(steps):
        if step == profile_at:
            torch.cuda.synchronize()
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA])
            profile.__enter__()
        token = teacher[:, step:step+1]
        with torch.no_grad():
            out = model(token, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :].clone().float()/.6
        probs = torch.softmax(warper(token, logits), dim=-1)
        torch.multinomial(probs, num_samples=1)
        cache = out.past_key_values
        generated = torch.cat([generated, token], dim=1)
        mask = torch.cat([mask, torch.ones((mask.size(0), 1), device=mask.device)], dim=1)
        (torch.zeros(2, dtype=torch.bool, device=token.device)).sum().item()
        if profile is not None and step == profile_at+49:
            torch.cuda.synchronize()
            profile.__exit__(None, None, None)
        if (step+1) % windows == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            records.append(dict(end_step=step+1, context=prompt.shape[1]+step+1,
                                ms_per_step=1000*(now-start)/windows))
            start = time.perf_counter()
    summary = None
    if profile is not None:
        events = profile.key_averages()
        def dev(e): return getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0))
        launches = sum(e.count for e in events if e.key in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC'))
        kernel_us = sum(dev(e) for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
        top = sorted(events, key=lambda e: dev(e), reverse=True)[:25]
        top_cpu = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:25]
        summary = dict(window_steps=50, launches_per_step=launches/50, gpu_kernel_ms_per_step=kernel_us/1000/50,
                       top_gpu=[dict(name=e.key, calls_per_step=e.count/50, ms_per_step=dev(e)/1000/50) for e in top],
                       top_cpu_self=[dict(name=e.key, calls_per_step=e.count/50, ms_per_step=e.self_cpu_time_total/1000/50) for e in top_cpu])
    return records, summary


def main():
    global ORIGINAL, ORIGINAL_FORWARD, EVICT_FORWARD
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--window', type=int, default=1000)
    p.add_argument('--profile-at', type=int, default=8000)
    p.add_argument('--variants', default='dense,hybrid_v2,hybrid_v3,no_reselect,pool_attention')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    import kvcompress.engine.selector_grid_cache as grid
    ORIGINAL = (grid.SelectorGridLayer, grid.decode_mode)
    ORIGINAL_FORWARD = Qwen3Attention.forward
    from kvcompress.engine.modify_forward import wrap_evict_attn_forward
    wrap_evict_attn_forward('Qwen3-4B')
    EVICT_FORWARD = Qwen3Attention.forward
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map='cuda',
                                                 attn_implementation='flash_attention_2').eval()
    record = next(r for r in json.loads(TOKENS.read_text()) if r['idx'] == 51)
    ids = torch.tensor(record['token_ids'], device='cuda')
    P = 70
    prompt, teacher = ids[:, :P], ids[:, P:P+a.steps]
    assert teacher.shape[1] == a.steps and not torch.isin(teacher, torch.tensor([151645, 151643], device='cuda')).any()
    results = dict(node=platform.node(), gpu=torch.cuda.get_device_name(), job=os.environ.get('SLURM_JOB_ID'),
                   question=51, prompt_tokens=P, steps=a.steps, batch=2, variants={})
    # Warm every code path once (short run) before timed runs.
    for variant in a.variants.split(','):
        run(model, variant, prompt, teacher, 1100, 1100, -1)
    for variant in a.variants.split(','):
        torch.cuda.empty_cache()
        records, profile = run(model, variant, prompt, teacher, a.steps, a.window, a.profile_at)
        results['variants'][variant] = dict(windows=records, profile=profile,
            mean_ms_per_step=sum(r['ms_per_step'] for r in records)/len(records))
        print(json.dumps(dict(variant=variant, mean_ms_per_step=results['variants'][variant]['mean_ms_per_step'],
                              profile=None if profile is None else {k: profile[k] for k in ('launches_per_step', 'gpu_kernel_ms_per_step')})), flush=True)
        (a.out/'profile.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
