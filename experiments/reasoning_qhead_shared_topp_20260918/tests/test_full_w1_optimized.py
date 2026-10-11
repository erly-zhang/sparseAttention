import math
import unittest
import torch
from kvcompress.engine.selector_grid_cache import SelectorGridLayer, choose, representative
from kvcompress.engine.full_w1_optimized_cache import FullW1OptimizedLayer, chronological_topk


def make(cls, k=80, protection='protected', capacity=256, backend='stable_sort', trace=True):
    args = (3, k, 0, f'grid_full_w1_topk_{protection}', 0., False, False)
    kw = dict(top_p=.95, query_window=1, eviction_interval=1, rep_seed=0,
              sample_id=19, max_cache_length=capacity, track_positions=trace)
    if cls is FullW1OptimizedLayer:
        kw['rank_backend'] = backend
    return cls(*args, **kw)


class FullW1Tests(unittest.TestCase):
    def test_stable_rank_cpu_ties_and_zero_budget(self):
        scores = torch.tensor([[0., .2, .2, .1, .2, .1, 0.], [1., 1., 1., 1., 1., 1., 1.]])
        for start, end in [(0, 7), (1, 6)]:
            for count in range(end-start+1):
                ids = chronological_topk(scores, count, start, end)
                for row in range(2):
                    expected = torch.argsort(scores[row, start:end], descending=True, stable=True)[:count].sort().values + start
                    self.assertTrue(torch.equal(ids[row], expected))

    def test_reject_other_algorithms(self):
        for mode in ['grid_compact_w1_topk_protected', 'grid_full_w16_topk_protected', 'grid_full_w1_topp_none']:
            with self.assertRaises(ValueError):
                FullW1OptimizedLayer(0, 80 if '_topk_' in mode else 0, 0, mode, 0., False, False,
                                     query_window=16 if '_w16_' in mode else 1,
                                     eviction_interval=64 if '_compact_' in mode else 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
    def test_packed_rank_exact_ties_subnormals_and_ranges(self):
        gen = torch.Generator(device='cuda').manual_seed(431)
        for n in [127, 1024, 8192, 32700]:
            values = torch.randint(0, 17, (2, n), generator=gen, device='cuda').float() / 16
            values[:, :5] = torch.tensor([0., torch.finfo(torch.float32).tiny, 1e-43, .5, 1.], device='cuda')
            for start, end in [(0, n), (17, n-64)]:
                for count in [0, 1, min(80, end-start), end-start]:
                    expected = chronological_topk(values, count, start, end)
                    actual = chronological_topk(values, count, start, end, 'packed_topk')
                    self.assertTrue(torch.equal(expected, actual), (n, start, count))

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
    def test_real_geometry_masks_scores_gqa_and_rng_exact(self):
        gen = torch.Generator(device='cuda').manual_seed(713)
        for k in [1024, 2048, 4096]:
            for n in [k-1, k, k+1, 8192, 32700]:
                for protection in ['protected', 'none']:
                    for backend in ['stable_sort', 'packed_topk']:
                        base = make(SelectorGridLayer, k, protection, 32768)
                        fast = make(FullW1OptimizedLayer, k, protection, 32768, backend)
                        initial = torch.zeros(2, 137, 8, 128, device='cuda', dtype=torch.bfloat16)
                        base.lazy_initialization(initial);fast.lazy_initialization(initial)
                        base.k_cache[:, :n].normal_(generator=gen)
                        fast.k_cache[:, :n].copy_(base.k_cache[:, :n])
                        q = torch.randn(2, 32, 1, 128, device='cuda', dtype=torch.bfloat16, generator=gen)
                        for layer in [base, fast]:
                            layer.cumulative_length = n;layer._lengths = [n, n]
                            layer.cache_seqlens.fill_(n)
                            layer.slot_positions[:, :n] = torch.arange(n, device='cuda')
                            layer.update_queries(q)
                        before_cpu = torch.random.get_rng_state().clone()
                        before_cuda = torch.cuda.get_rng_state().clone()
                        old, counts = base.select_indices();new, new_counts = fast.select_indices()
                        self.assertTrue(torch.equal(old, new), (k, n, protection, backend))
                        self.assertEqual(counts, new_counts)
                        self.assertTrue(torch.equal(base._last_scores, fast._last_scores))
                        self.assertEqual(base.event_trace, fast.event_trace)
                        self.assertEqual(fast._rep_head, representative(0, 19, 3, n-137, 32))
                        self.assertTrue(torch.equal(before_cpu, torch.random.get_rng_state()))
                        self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))
                        del base, fast

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU required')
    def test_attention_full_kv_exact_and_independent_oracle_with_batch_shrink(self):
        gen = torch.Generator(device='cuda').manual_seed(643)
        for protection in ['protected', 'none']:
            for backend in ['stable_sort', 'packed_topk']:
                base = make(SelectorGridLayer, protection=protection)
                fast = make(FullW1OptimizedLayer, protection=protection, backend=backend)
                for step in range(151):
                    b = 2 if step < 130 else 1
                    n = 5 if step == 0 else 1
                    q = torch.randn(b, n, 8, 16, generator=gen, device='cuda', dtype=torch.bfloat16)
                    k = torch.randn(b, n, 2, 16, generator=gen, device='cuda', dtype=torch.bfloat16)
                    v = torch.randn(k.shape, generator=gen, device='cuda', dtype=torch.bfloat16)
                    base.update_queries(q.transpose(1, 2));fast.update_queries(q.transpose(1, 2))
                    old = base.attention(q, k, v, .25);new = fast.attention(q, k, v, .25)
                    self.assertTrue(torch.equal(old, new), (step, protection, backend))
                    for name in ['cumulative_length', '_lengths', 'selected_pairs', 'causal_pairs', 'selection_events',
                                 'max_selected_tokens', 'max_visible_tokens', 'prefill_pairs', 'attention_token_sum',
                                 'attention_token_observations', 'selection_token_sum', 'selection_token_observations']:
                        self.assertEqual(getattr(base, name), getattr(fast, name), (step, name))
                    extent = fast.cumulative_length
                    for name in ['k_cache', 'v_cache', 'slot_positions']:
                        self.assertTrue(torch.equal(getattr(base, name)[:, :extent], getattr(fast, name)[:, :extent]))
                    self.assertTrue(torch.equal(base.cache_seqlens, fast.cache_seqlens))
                    if step:
                        self.assertTrue(torch.equal(base._last_selected_ids, fast._last_selected_ids))
                    if step in [1, 76, 127, 150]:
                        selected = fast._last_selected_ids
                        ks = fast.k_cache.gather(1, selected[:, :, None, None].expand(-1, -1, 2, 16)).repeat_interleave(4, 2).float()
                        vs = fast.v_cache.gather(1, selected[:, :, None, None].expand(-1, -1, 2, 16)).repeat_interleave(4, 2).float()
                        logits = torch.einsum('bqhd,bkhd->bhqk', q.float(), ks) / math.sqrt(16)
                        ref = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), vs)
                        torch.testing.assert_close(new.float(), ref, atol=.018, rtol=.035)
                    if step == 129:
                        ix = torch.tensor([1], device='cuda')
                        base.batch_select_indices(ix);fast.batch_select_indices(ix)
                        self.assertEqual(fast._batch_ids, [1])


if __name__ == '__main__':
    unittest.main()
