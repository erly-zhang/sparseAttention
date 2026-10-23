"""Full-KV W1 interval64: refresh masks, appended tokens, attention, RNG, shrink."""
import unittest
import torch

from kvcompress.engine.selector_grid_cache import SelectorGridLayer
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer
from kvcompress.engine.w1_qk_fast_cache import FastFullW1Layer
from kvcompress.engine.w1_full_interval_cache import FullW1Interval64Layer, MODE, INTERVAL


class ReferenceInterval(FullW1OptimizedLayer):
    """Independent plain-torch reference: historical v1 raw-QK selection at
    steps 1,65,...; otherwise the previous IDs plus every newer position.
    Gathers with torch.gather into a scratch of the same shape as the tested layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, score_mode='qk', rank_backend='stable_sort', **kwargs)
        self.ref_ids = None

    def lazy_initialization(self, k, value_states=None):
        super().lazy_initialization(k, value_states)
        b, _, h, d = k.shape
        size = min(self.token_budget+INTERVAL, self.capacity)
        self.ref_k = k.new_zeros((b, size, h, d))
        self.ref_v = torch.zeros_like(self.ref_k)

    def attention(self, q, k, v, scale):
        from flash_attn import flash_attn_with_kvcache
        if not self.cumulative_length:
            return SelectorGridLayer.attention(self, q, k, v, scale)
        b, _, hkv, d = k.shape
        old = self.cumulative_length
        self.k_cache[:, old:old+1].copy_(k)
        self.v_cache[:, old:old+1].copy_(v)
        self.slot_positions[:, old] = old
        self.cache_seqlens.fill_(old+1)
        self.cumulative_length = old+1
        self._lengths = [old+1]*b
        step = old+1-self.prompt_len
        if self.ref_ids is None or (step-1) % INTERVAL == 0:
            self.ref_ids = self.select_indices()[0].clone()
        else:
            self.ref_ids = torch.cat((self.ref_ids, torch.full((b, 1), old, device=self.device)), 1)
        count = self.ref_ids.shape[1]
        ix = self.ref_ids[:, :, None, None].expand(-1, -1, hkv, d)
        self.ref_k[:, :count].copy_(self.k_cache.gather(1, ix))
        self.ref_v[:, :count].copy_(self.v_cache.gather(1, ix))
        lengths = torch.full((b,), count, device=self.device, dtype=torch.int32)
        return flash_attn_with_kvcache(q, self.ref_k, self.ref_v, cache_seqlens=lengths,
                                       causal=True, softmax_scale=scale)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.ref_k = self.ref_k[indices].contiguous()
        self.ref_v = self.ref_v[indices].contiguous()
        if self.ref_ids is not None:
            self.ref_ids = self.ref_ids[indices].contiguous()


def make(cls, n, budget, prompt=137, batch=2, hkv=8, dim=128, seed=932):
    mode = MODE if cls is FullW1Interval64Layer else 'grid_full_w1_topk_protected'
    kw = dict(query_window=1, eviction_interval=INTERVAL if cls is FullW1Interval64Layer else 1,
              rep_seed=0, sample_id=19, max_cache_length=32768, track_positions=False)
    result = cls(3, budget, 0, mode, 0., False, False, **kw)
    result.lazy_initialization(torch.zeros(batch, prompt, hkv, dim, device='cuda', dtype=torch.bfloat16))
    g = torch.Generator(device='cuda').manual_seed(seed)
    result.k_cache[:, :n].normal_(generator=g)
    result.v_cache[:, :n].normal_(generator=g)
    result.slot_positions[:, :n] = torch.arange(n, device='cuda')
    result.cumulative_length = n
    result._lengths = [n]*batch
    result.cache_seqlens.fill_(n)
    result._num_query_heads = hkv*4
    return result


def expected_counts(prompt, n0, steps, budget):
    """Attention size per decode call, from the absolute refresh schedule."""
    counts, current = [], None
    for i in range(steps):
        n = n0+i+1
        step = n-prompt
        if current is None or (step-1) % INTERVAL == 0:
            current = min(budget, n)
        else:
            current += 1
        counts.append(current)
    return counts


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def run_sequence(self, n0, budget, steps, shrink_at=None):
        new = make(FullW1Interval64Layer, n0, budget)
        ref = make(ReferenceInterval, n0, budget)
        per_step = make(FastFullW1Layer, n0, budget)
        g = torch.Generator(device='cuda').manual_seed(77)
        before = torch.cuda.get_rng_state().clone()
        cpu_before = torch.random.get_rng_state().clone()
        b = 2
        for i in range(steps):
            if shrink_at is not None and i == shrink_at:
                for layer in (new, ref, per_step):
                    layer.batch_select_indices(torch.tensor([1], device='cuda'))
                b = 1
            q = torch.randn(b, 1, 32, 128, device='cuda', generator=g).bfloat16()
            k = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
            v = torch.randn(b, 1, 8, 128, device='cuda', generator=g).bfloat16()
            step = new.cumulative_length+1-new.prompt_len
            refresh = (step-1) % INTERVAL == 0 or i == 0
            outs = []
            for layer in (new, ref, per_step):
                layer.update_queries(q.transpose(1, 2))
                outs.append(layer.attention(q, k, v, 128**-.5))
            out, expected, every_step = outs
            self.assertTrue(torch.equal(out, expected), (i, step))
            count = new._attend_count
            self.assertEqual(count, ref.ref_ids.shape[1])
            self.assertTrue(torch.equal(new._temp_positions[:, :count], ref.ref_ids))
            self.assertTrue(torch.equal(new._attend_lengths, torch.full((b,), count, device='cuda', dtype=torch.int32)))
            self.assertTrue(torch.all(ref.ref_ids[:, 1:] > ref.ref_ids[:, :-1]).item())
            if refresh:
                # A refresh equals the per-step Full-KV W1 decision at that step.
                self.assertTrue(torch.equal(out, every_step), (i, step))
                self.assertTrue(torch.equal(new._ids, per_step._ids))
                p = new.prompt_len
                if new.cumulative_length > budget:
                    self.assertTrue(torch.equal(new._ids[:, :p], torch.arange(p, device='cuda').expand(b, -1)))
                    tail = torch.arange(new.cumulative_length-64, new.cumulative_length, device='cuda')
                    self.assertTrue(torch.equal(new._ids[:, -64:], tail.expand(b, -1)))
            # Full history retained exactly.
            n = new.cumulative_length
            self.assertTrue(torch.equal(new.k_cache[:, :n], ref.k_cache[:, :n]))
            self.assertTrue(torch.equal(new.v_cache[:, :n], ref.v_cache[:, :n]))
            self.assertEqual(new.cache_seqlens.tolist(), [n]*b)
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))
        counts = expected_counts(new.prompt_len, n0, steps, budget)
        self.assertEqual(new.attention_token_sum, sum(c*(2 if i < (shrink_at or steps) else 1) for i, c in enumerate(counts)))
        self.assertEqual(new.selected_pairs, 32*new.attention_token_sum)
        self.assertEqual(new.max_selected_tokens, max(counts))
        self.assertLessEqual(max(counts), budget+INTERVAL-1)
        refreshes = sum(1 for i in range(steps) if i == 0 or (n0+i+1-new.prompt_len-1) % INTERVAL == 0)
        self.assertEqual(new.selection_events, refreshes)
        return new, counts

    def test_long_history_k1024_refresh_append_shrink(self):
        # n0-prompt = 64*20, so the first call is an aligned refresh step.
        self.run_sequence(137+64*20, 1024, 200, shrink_at=97)

    def test_k2048_and_misaligned_first_call(self):
        self.run_sequence(137+64*40+17, 2048, 150)

    def test_short_history_grows_past_budget(self):
        # Starts with all positions visible, crosses K between refreshes.
        new, counts = self.run_sequence(137+64*13, 1024, 260)
        self.assertEqual(counts[0], 137+64*13+1)
        self.assertGreater(max(counts), 1024)

    def test_attention_matches_fp32_oracle(self):
        new = make(FullW1Interval64Layer, 137+64*20, 1024)
        g = torch.Generator(device='cuda').manual_seed(5)
        for _ in range(70):
            q = torch.randn(2, 1, 32, 128, device='cuda', generator=g).bfloat16()
            k = torch.randn(2, 1, 8, 128, device='cuda', generator=g).bfloat16()
            v = torch.randn(2, 1, 8, 128, device='cuda', generator=g).bfloat16()
            new.update_queries(q.transpose(1, 2))
            out = new.attention(q, k, v, 128**-.5)
            c = new._attend_count
            ids = new._temp_positions[:, :c]
            ks = new.k_cache.gather(1, ids[:, :, None, None].expand(-1, -1, 8, 128)).repeat_interleave(4, 2).float()
            vs = new.v_cache.gather(1, ids[:, :, None, None].expand(-1, -1, 8, 128)).repeat_interleave(4, 2).float()
            logits = torch.einsum('bqhd,bkhd->bhqk', q.float(), ks)*128**-.5
            oracle = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), vs)
            torch.testing.assert_close(out.float(), oracle, atol=.018, rtol=.035)

    def test_tiny_qwen_sampled_generation_matches_reference(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        from kvcompress.harness.generation_utils import batch_exist_generate
        import kvcompress.engine.selector_grid_cache as grid
        from kvcompress.engine.w1_full_interval_cache import install
        original_factory, original_decode = grid.SelectorGridLayer, grid.decode_mode
        original_forward = Qwen3Attention.forward
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                            num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                            head_dim=16, max_position_embeddings=512, eos_token_id=128, pad_token_id=0)
        config._attn_implementation = 'flash_attention_2'
        torch.manual_seed(914)
        model = Qwen3ForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        model.generation_config.temperature = .6
        model.generation_config.top_p = .95
        wrap_evict_attn_forward('Qwen3-tiny-controlled')
        inputs = torch.tensor([[1, 5, 11, 7, 23]]*2, device='cuda')
        outputs, metrics = {}, {}
        try:
            for name in ('reference', 'interval64', 'per_step'):
                grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
                install('stable_sort')
                cc = dict(token_budget=80, residual_length=0, eviction_mode=MODE, rkv_lambda=0.,
                          smooth=False, verbose=False, top_p=.95, query_window=1,
                          eviction_interval=INTERVAL, rep_seed=0, sample_id=19)
                if name == 'reference':
                    installed = grid.SelectorGridLayer

                    def factory(*args, **kwargs):
                        if args[3] == MODE:
                            args = (*args[:3], 'grid_full_w1_topk_protected', *args[4:])
                            kwargs.update(eviction_interval=1)
                            return ReferenceInterval(*args, **kwargs)
                        return installed(*args, **kwargs)
                    grid.SelectorGridLayer = factory
                if name == 'per_step':
                    cc.update(eviction_mode='grid_full_w1_topk_protected', eviction_interval=1)
                torch.manual_seed(9182)
                torch.cuda.manual_seed_all(9182)
                metrics[name] = {}
                outputs[name] = batch_exist_generate(model, input_ids=inputs,
                    attention_mask=torch.ones_like(inputs), max_length=400, do_sample=True,
                    metrics=metrics[name], cache_implementation='evict', cache_config=cc)
            self.assertTrue(torch.equal(outputs['reference'], outputs['interval64']))
            m = metrics['interval64']
            self.assertEqual(m['w1_implementation'], 'full_w1_raw_qk_interval64_v1')
            self.assertEqual(m['grid_storage'], 'full')
            self.assertEqual(m['cache_semantics'], 'full_history_temporary_mask')
            self.assertEqual(m.get('compaction_events', 0), 0)
            self.assertLessEqual(m['max_selected_tokens'], 80+INTERVAL-1)
            self.assertLess(m['selected_pairs'], m['causal_pairs'])
            longest = max(m['generated_tokens_per_rollout'])
            self.assertEqual(m['selection_events'], 2*len(range(1, longest, INTERVAL)))
        finally:
            grid.SelectorGridLayer, grid.decode_mode = original_factory, original_decode
            Qwen3Attention.forward = original_forward


class CPUChecks(unittest.TestCase):
    def test_mode_and_interval_contract(self):
        with self.assertRaises(ValueError):
            FullW1Interval64Layer(3, 1024, 0, MODE, 0., False, False, query_window=1,
                                  eviction_interval=1, rep_seed=0, sample_id=1)
        with self.assertRaises(ValueError):
            FullW1Interval64Layer(3, 1024, 0, 'grid_full_w1_topk_protected', 0., False, False,
                                  query_window=1, eviction_interval=INTERVAL, rep_seed=0, sample_id=1)
        layer = FullW1Interval64Layer(3, 1024, 0, MODE, 0., False, False, query_window=1,
                                      eviction_interval=INTERVAL, rep_seed=0, sample_id=1)
        self.assertEqual((layer.MODE, layer.storage, layer.eviction_interval), (MODE, 'full', 64))
        self.assertTrue(layer.uses_temporary_mask)

    def test_expected_count_schedule(self):
        self.assertEqual(expected_counts(10, 10, 3, 100), [11, 12, 13])
        c = expected_counts(10, 10, 130, 50)
        self.assertEqual(c[:2], [11, 12])
        self.assertEqual(c[64], 50)
        self.assertEqual(c[127], 113)
        self.assertEqual(c[128], 50)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
