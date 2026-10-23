"""Independent exact-order, masks, private RNG, compaction and attention checks."""
import unittest
from unittest.mock import patch
import torch

from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer
from kvcompress.engine.w1_interval1000_cache import W1Hybrid1000Layer, HYBRID_MODE, uniform_residual_ids
from kvcompress.engine.w1_qk_fast_cache import (
    FastFullW1Layer, FastHybridW1Layer, packed_qk_keys, exact_qk_topk,
    fast_uniform_residual_ids)


def layer(cls, device, n, budget=1024, prompt=137, batch=2, hkv=2, dim=16, absolute=None):
    hybrid = issubclass(cls, W1Hybrid1000Layer)
    mode = HYBRID_MODE if hybrid else 'grid_full_w1_topk_protected'
    kw = dict(query_window=1, eviction_interval=1000 if hybrid else 1,
              rep_seed=0, sample_id=19, max_cache_length=32768, track_positions=True)
    if cls is FullW1OptimizedLayer:
        kw['score_mode'] = 'qk'
    result = cls(3, budget, 0, mode, 0., False, False, **kw)
    result.lazy_initialization(torch.zeros(batch, prompt, hkv, dim, device=device, dtype=torch.bfloat16))
    generator = torch.Generator(device=device).manual_seed(932)
    result.k_cache[:, :n].normal_(generator=generator)
    result.v_cache[:, :n].normal_(generator=generator)
    absolute = n if absolute is None else absolute
    positions = torch.cat((torch.arange(prompt, device=device),
                           torch.arange(absolute-(n-prompt), absolute, device=device)))
    result.slot_positions[:, :n] = positions
    result.cumulative_length = absolute
    result._lengths = [n]*batch
    result.cache_seqlens.fill_(n)
    q = torch.randn(batch, hkv*4, 1, dim, device=device, generator=generator).bfloat16()
    result.update_queries(q)
    return result


class CPUChecks(unittest.TestCase):
    def test_exhaustive_bf16_order_including_nan_infinity_and_signed_zero(self):
        # Every BF16 bit pattern, deliberately permuted. NaNs tie; +/-0 tie.
        bits = torch.randperm(65536, generator=torch.Generator().manual_seed(15)).to(torch.int16)
        values = bits.view(torch.bfloat16)[None]
        expected = values.argsort(dim=1, descending=True, stable=True)
        actual = packed_qk_keys(values).argsort(dim=1, descending=True)
        self.assertTrue(torch.equal(expected, actual))
        for count in (0, 1, 1024, 32768, 65536):
            for backend in ('packed_qk_topk', 'packed_qk32_topk'):
                self.assertTrue(torch.equal(exact_qk_topk(values, count, backend=backend), expected[:, :count].sort(dim=1).values))

    def test_full_and_hybrid_masks_raw_scores_protection_short_and_shrink(self):
        for old_cls, fast_cls in ((FullW1OptimizedLayer, FastFullW1Layer),
                                  (W1Hybrid1000Layer, FastHybridW1Layer)):
            for n in (81, 95, 96, 97, 320):
                absolute = n+4000 if old_cls is W1Hybrid1000Layer else n
                old = layer(old_cls, 'cpu', n, budget=96, prompt=17, absolute=absolute)
                new = layer(fast_cls, 'cpu', n, budget=96, prompt=17, absolute=absolute)
                before = torch.random.get_rng_state().clone()
                with patch.object(torch.Tensor, 'softmax', side_effect=AssertionError('selector softmax')):
                    old_ids, old_counts = old.select_indices()
                    new_ids, new_counts = new.select_indices()
                self.assertTrue(torch.equal(old_ids, new_ids))
                self.assertTrue(torch.equal(old._last_scores, new._last_scores))
                self.assertEqual(old_counts, new_counts)
                self.assertEqual(old._rep_head, new._rep_head)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                old.batch_select_indices(torch.tensor([1]))
                new.batch_select_indices(torch.tensor([1]))
                self.assertTrue(torch.equal(old.select_indices()[0], new.select_indices()[0]))

    def test_short_history_skips_unused_matmul(self):
        new = layer(FastFullW1Layer, 'cpu', 81, budget=96, prompt=17)
        new.trace = False
        new.queries = None
        ids, counts = new.select_indices()
        self.assertEqual(counts, [81, 81])
        self.assertEqual(ids.tolist(), [list(range(81))]*2)

    def test_random_residual_cpu_preserves_draws_and_masks(self):
        for n, count in ((9, 2), (9, 4096), (5800, 4096)):
            core = torch.tensor([0, 2, 5]) if n == 9 else torch.arange(1024)
            a = torch.Generator().manual_seed(47)
            b = torch.Generator().manual_seed(47)
            expected = uniform_residual_ids(n, core, count, a)
            actual = fast_uniform_residual_ids(n, core, count, b).sort().values
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(a.get_state(), b.get_state()))


@unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
class GPUChecks(unittest.TestCase):
    def test_cuda_packing_exhaustive_bf16_and_partial_range(self):
        bits = torch.randperm(65536, generator=torch.Generator().manual_seed(15)).to(torch.int16)
        values = bits.view(torch.bfloat16)[None].repeat(2, 1).cuda()
        for start, end, count in ((0, 65536, 1024), (137, 65500, 961), (0, 65536, 32768),
                                  (7, 87, 0), (7, 87, 16), (7, 87, 80)):
            oracle = values[:, start:end].argsort(dim=1, descending=True, stable=True)[:, :count].sort(dim=1).values
            for backend in ('packed_qk_topk', 'packed_qk32_topk'):
                self.assertTrue(torch.equal(exact_qk_topk(values, count, start, end, backend=backend), oracle))
        for backend, dtype in (('packed_qk_topk', torch.int64), ('packed_qk32_topk', torch.int32)):
            keys = torch.empty_like(values, dtype=dtype)
            exact_qk_topk(values, 1024, keys=keys, backend=backend)
            self.assertTrue(torch.equal(keys.argsort(dim=1, descending=True),
                                        values.argsort(dim=1, descending=True, stable=True)))

    def test_cuda_layer_masks_gqa_attention_exact_and_oracle(self):
        for n in (300, 6120, 8192, 32700):
            old = layer(FullW1OptimizedLayer, 'cuda', n, hkv=8, dim=128)
            new = layer(FastFullW1Layer, 'cuda', n, hkv=8, dim=128)
            if n >= 8192:
                new.rank_backend = 'packed_qk32_topk'
            self.assertTrue(torch.equal(old.select_indices()[0], new.select_indices()[0]))
            self.assertTrue(torch.equal(old._last_scores, new._last_scores))
            # Append the same token and use the actual fused KV/attention path.
            k, v = old.k_cache[:, n-1:n].contiguous(), old.v_cache[:, n-1:n].contiguous()
            q = old.queries.transpose(1, 2)
            expected = old.attention(q, k, v, 128**-.5)
            actual = new.attention(q, k, v, 128**-.5)
            self.assertTrue(torch.equal(expected, actual))
            ks = new._temp_k[:, :min(n+1, 1024)].repeat_interleave(4, dim=2).float()
            vs = new._temp_v[:, :min(n+1, 1024)].repeat_interleave(4, dim=2).float()
            logits = torch.einsum('bqhd,bkhd->bhqk', q.float(), ks)*128**-.5
            oracle = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), vs)
            torch.testing.assert_close(actual.float(), oracle, atol=.018, rtol=.035)
            del old, new

    def test_cuda_random_keys_and_rng_exact(self):
        for n in (5121, 5800, 6120):
            for seed in (0, 42, 79013):
                core = torch.randperm(n, device='cuda', generator=torch.Generator(device='cuda').manual_seed(1))[:1024].sort().values
                a = torch.Generator(device='cuda').manual_seed(seed)
                b = torch.Generator(device='cuda').manual_seed(seed)
                before = torch.cuda.get_rng_state().clone()
                expected = uniform_residual_ids(n, core, 4096, a)
                actual = fast_uniform_residual_ids(n, core, 4096, b).sort().values
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(a.get_state(), b.get_state()))
                self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))

    def test_two_candidate_compactions_and_batch_shrink_exact(self):
        old = layer(W1Hybrid1000Layer, 'cuda', 5800, absolute=6137)
        new = layer(FastHybridW1Layer, 'cuda', 5800, absolute=6137)
        for event in range(2):
            old_ids = old.select_indices()[0].clone()
            new_ids = new.select_indices()[0].clone()
            self.assertTrue(torch.equal(old_ids, new_ids))
            old.compact_candidate_pool(old_ids)
            new.compact_candidate_pool(new_ids)
            self.assertTrue(torch.equal(old.slot_positions, new.slot_positions))
            self.assertTrue(torch.equal(old.k_cache[:, :5120], new.k_cache[:, :5120]))
            self.assertTrue(torch.equal(old.v_cache[:, :5120], new.v_cache[:, :5120]))
            self.assertEqual(old.compaction_events, new.compaction_events)
            if event == 0:
                # A real next interval contributes 1000 chronological new slots.
                for cache in (old, new):
                    g = torch.Generator(device='cuda').manual_seed(301)
                    cache.k_cache[:, 5120:6120].normal_(generator=g)
                    cache.v_cache[:, 5120:6120].normal_(generator=g)
                    cache.slot_positions[:, 5120:6120] = torch.arange(6137, 7137, device='cuda')
                    cache.cumulative_length = 7137
                    cache._lengths = [6120]*2
                    cache.cache_seqlens.fill_(6120)
                    cache.batch_select_indices(torch.tensor([1], device='cuda'))
        self.assertEqual(new.compaction_events, 2)

    def test_tiny_qwen_sampled_generation_exact(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.full_w1_optimized_cache import install as old_install
        from kvcompress.engine.w1_qk_fast_cache import install as fast_install
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=256, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        inputs = torch.tensor([[1, 5, 11, 7, 23]]*2, device='cuda')
        cc = dict(token_budget=80, residual_length=0, eviction_mode='grid_full_w1_topk_protected',
                  rkv_lambda=0., smooth=False, verbose=False, top_p=.95, query_window=1,
                  eviction_interval=1, rep_seed=0, sample_id=19)
        outputs = {}
        try:
            for name in ('baseline', 'stable_sort', 'packed_qk_topk', 'packed_qk32_topk'):
                grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                if name == 'baseline':
                    old_install(score_mode='qk')
                else:
                    fast_install(name)
                torch.manual_seed(9182)
                torch.cuda.manual_seed_all(9182)
                outputs[name] = batch_exist_generate(model, input_ids=inputs,
                    attention_mask=torch.ones_like(inputs), max_length=165, do_sample=True,
                    metrics={}, cache_implementation='evict', cache_config=cc)
            self.assertTrue(torch.equal(outputs['baseline'], outputs['stable_sort']))
            self.assertTrue(torch.equal(outputs['baseline'], outputs['packed_qk_topk']))
            self.assertTrue(torch.equal(outputs['baseline'], outputs['packed_qk32_topk']))
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
