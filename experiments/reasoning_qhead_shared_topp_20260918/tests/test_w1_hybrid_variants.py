"""Hybrid variants: interval1000/random == v4; top4096 pool == independent reference."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvcompress.engine.w1_interval1000_cache import HYBRID_MODE
from kvcompress.engine.w1_qk_fast_cache import exact_qk_topk
from kvcompress.engine.w1_hybrid_gpu_select_cache import GpuSelectHybridW1Layer
from kvcompress.engine.w1_hybrid_variants_cache import HybridVariantLayer, mode_name, EXTRA


def make(cls, mode, interval, n, budget, absolute, prompt=137, batch=2):
    """Synthetic pool: prompt slots, then the latest n-prompt reasoning positions."""
    kw = dict(query_window=1, eviction_interval=interval, rep_seed=0, sample_id=19,
              max_cache_length=32768, track_positions=False)
    layer = cls(3, budget, 0, mode, 0., False, False, **kw)
    layer.lazy_initialization(torch.zeros(batch, prompt, 8, 128, device='cuda', dtype=torch.bfloat16))
    g = torch.Generator(device='cuda').manual_seed(932)
    layer.k_cache[:, :n].normal_(generator=g)
    layer.v_cache[:, :n].normal_(generator=g)
    positions = torch.cat((torch.arange(prompt, device='cuda'),
                           torch.arange(absolute-(n-prompt), absolute, device='cuda')))
    layer.slot_positions[:, :n] = positions
    layer.cumulative_length = absolute
    layer._lengths = [n]*batch
    layer.cache_seqlens.fill_(n)
    layer._num_query_heads = 32
    return layer


def step_inputs(g, b):
    q = torch.randn(b, 1, 32, 128, device='cuda', generator=g).bfloat16()
    k = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
    v = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
    return q, k, v


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def test_interval1000_random_equals_v4(self):
        for budget in (1024, 2048):
            pool = budget+EXTRA
            old = make(GpuSelectHybridW1Layer, HYBRID_MODE, 1000, pool+500, budget, 137+7500)
            new = make(HybridVariantLayer, mode_name(1000, 'random'), 1000, pool+500, budget, 137+7500)
            g = torch.Generator(device='cuda').manual_seed(5)
            for i in range(2600):
                if i == 1300:
                    for c in (old, new):
                        c.batch_select_indices(torch.tensor([1], device='cuda'))
                q, k, v = step_inputs(g, old.k_cache.shape[0])
                outs = []
                for c in (old, new):
                    c.update_queries(q.transpose(1, 2))
                    outs.append(c.attention(q, k, v, 128**-.5))
                self.assertTrue(torch.equal(*outs), (budget, i))
                self.assertTrue(torch.equal(old.slot_positions, new.slot_positions))
            self.assertEqual(old.compaction_events, new.compaction_events)
            self.assertGreaterEqual(new.compaction_events, 2)

    def test_top_and_random_pool_updates_every_interval(self):
        for budget in (1024, 2048):
            for interval in (64, 1024):
                for residual in ('top', 'random'):
                    pool = budget+EXTRA
                    # Pool length = K+4096 + steps since the last aligned update.
                    # Absolute history must exceed the pool so original positions stay increasing.
                    since = 10
                    aligned = interval*(-(-(pool+since)//interval)+1)
                    layer = make(HybridVariantLayer, mode_name(interval, residual), interval,
                                 pool+since, budget, 137+aligned+since)
                    self.assertEqual(layer.capacity, pool+interval)
                    g = torch.Generator(device='cuda').manual_seed(17)
                    events = 0
                    for i in range(2*interval+30):
                        q, k, v = step_inputs(g, 2)
                        layer.update_queries(q.transpose(1, 2))
                        n = layer._lengths[0]
                        before_positions = layer.slot_positions[:, :n+1].clone()
                        before_positions[:, n] = layer.cumulative_length  # appended this step
                        layer.attention(q, k, v, 128**-.5)
                        step = layer.cumulative_length-layer.prompt_len
                        if step % interval == 0:
                            events += 1
                            self.assertEqual(layer._lengths, [pool]*2)
                            core = before_positions.gather(1, layer._ids)
                            kept = layer.slot_positions[:, :pool]
                            # Chronological, unique, prompt + recent64 + whole core retained.
                            self.assertTrue(torch.all(kept[:, 1:] > kept[:, :-1]).item())
                            self.assertTrue(torch.equal(kept[:, :137], torch.arange(137, device='cuda').expand(2, -1)))
                            for row in range(2):
                                self.assertTrue(torch.isin(core[row], kept[row]).all().item())
                            if residual == 'top':
                                scores = layer._pool_scores
                                n1 = n+1
                                count = pool-137-64
                                ref = exact_qk_topk(scores, count, 137, n1-64, backend='stable_sort')+137
                                ref_ids = torch.cat((torch.arange(137, device='cuda').expand(2, -1), ref,
                                                     torch.arange(n1-64, n1, device='cuda').expand(2, -1)), 1)
                                self.assertTrue(torch.equal(kept, before_positions.gather(1, ref_ids)))
                        self.assertLessEqual(layer._lengths[0], pool+interval)
                    self.assertEqual(layer.compaction_events, events)
                    self.assertGreaterEqual(events, 2)
                    del layer

    def test_tiny_qwen_generation_all_modes(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_hybrid_gpu_select_cache import install as v4_install
        from kvcompress.engine.w1_hybrid_variants_cache import install
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=8192, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        inputs = torch.tensor([[1, 5, 11, 7, 23]]*2, device='cuda')
        outputs, metrics = {}, {}
        runs = [('v4', HYBRID_MODE, 1000)] + [(f'{i}_{r}', mode_name(i, r), i)
                                             for i in (1000, 64, 1024) for r in ('random', 'top')]
        try:
            for name, mode, interval in runs:
                grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                (v4_install if name == 'v4' else install)('packed_qk_topk')
                cc = dict(token_budget=80, residual_length=0, eviction_mode=mode, rkv_lambda=0.,
                          smooth=False, verbose=False, top_p=.95, query_window=1,
                          eviction_interval=interval, rep_seed=0, sample_id=19)
                torch.manual_seed(9182)
                torch.cuda.manual_seed_all(9182)
                metrics[name] = {}
                outputs[name] = batch_exist_generate(model, input_ids=inputs,
                    attention_mask=torch.ones_like(inputs), max_length=4500, do_sample=True,
                    metrics=metrics[name], cache_implementation='evict', cache_config=cc)
            self.assertTrue(torch.equal(outputs['v4'], outputs['1000_random']))
            pool = 80+EXTRA
            for name, mode, interval in runs[1:]:
                m = metrics[name]
                cached, events = 5, 0
                for step in range(1, max(m['generated_tokens_per_rollout'])):
                    cached += 1
                    if step % interval == 0 and cached > pool:
                        cached, events = pool, events+1
                if 'compaction_events' in m:  # recorded by the frozen generation_utils
                    self.assertEqual(m['compaction_events'], 2*events, name)
                self.assertEqual(m['w1_implementation'], f"hybrid_w1_raw_qk_gpu_select_v4_i{interval}_{name.split('_')[1]}4096")
            self.assertFalse(torch.equal(outputs['64_top'], outputs['64_random']))
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LongPromptChecks(unittest.TestCase):
    def test_boundary_prompt_equals_v4(self):
        # P+64 == K: the whole prompt stays attention-protected, exactly v4.
        budget, prompt = 1024, 960
        pool = budget+EXTRA
        old = make(GpuSelectHybridW1Layer, HYBRID_MODE, 1000, pool+500, budget, prompt+7500, prompt=prompt)
        new = make(HybridVariantLayer, mode_name(1000, 'random'), 1000, pool+500, budget, prompt+7500, prompt=prompt)
        self.assertEqual(new._protect_len, prompt)
        g = torch.Generator(device='cuda').manual_seed(8)
        for i in range(1100):
            q, k, v = step_inputs(g, 2)
            outs = []
            for c in (old, new):
                c.update_queries(q.transpose(1, 2))
                outs.append(c.attention(q, k, v, 128**-.5))
            self.assertTrue(torch.equal(*outs), i)
            self.assertTrue(torch.equal(old.slot_positions, new.slot_positions))
        self.assertEqual(old.compaction_events, new.compaction_events)
        self.assertGreaterEqual(new.compaction_events, 1)

    def test_v4_still_rejects_long_prompt(self):
        with self.assertRaises(ValueError):
            make(GpuSelectHybridW1Layer, HYBRID_MODE, 1000, 2000, 1024, 3000, prompt=1567)

    def test_long_prompt_attention_cap_and_pool_keeps_prompt(self):
        for budget, prompt in ((1024, 1567), (2048, 2792)):
            pool = budget+EXTRA
            cap = budget-65
            for interval in (64, 1024):
                for residual in ('random', 'top'):
                    since = 10
                    aligned = interval*(-(-(pool+since)//interval)+1)
                    layer = make(HybridVariantLayer, mode_name(interval, residual), interval,
                                 pool+since, budget, prompt+aligned+since, prompt=prompt)
                    self.assertEqual(layer._protect_len, cap)
                    g = torch.Generator(device='cuda').manual_seed(23)
                    events = 0
                    for i in range(2*interval+20):
                        q, k, v = step_inputs(g, 2)
                        layer.update_queries(q.transpose(1, 2))
                        n = layer._lengths[0]
                        layer.attention(q, k, v, 128**-.5)
                        ids = layer._ids
                        # Attention: capped prompt prefix + W1 picks + recent64, exactly K.
                        self.assertEqual(ids.shape[1], budget)
                        self.assertTrue(torch.equal(ids[:, :cap], torch.arange(cap, device='cuda').expand(2, -1)))
                        self.assertTrue(torch.equal(ids[:, -64:], torch.arange(n+1-64, n+1, device='cuda').expand(2, -1)))
                        self.assertTrue(torch.all(ids[:, 1:] > ids[:, :-1]).item())
                        step = layer.cumulative_length-layer.prompt_len
                        if step % interval == 0:
                            events += 1
                            kept = layer.slot_positions[:, :pool]
                            self.assertEqual(layer._lengths, [pool]*2)
                            self.assertTrue(torch.all(kept[:, 1:] > kept[:, :-1]).item())
                            # The full prompt is never evicted from the pool.
                            self.assertTrue(torch.equal(kept[:, :prompt], torch.arange(prompt, device='cuda').expand(2, -1)))
                    self.assertEqual(layer.compaction_events, events)
                    self.assertGreaterEqual(events, 2)
                    del layer


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
