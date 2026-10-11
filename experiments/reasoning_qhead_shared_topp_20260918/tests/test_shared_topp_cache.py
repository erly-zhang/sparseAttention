"""Regression tests exercise the real cache, with an independent scalar selector reference."""
import math
import os
import unittest
import torch
from kvcompress.engine.cache_utils import EvictLayer

os.environ['QHEAD_DEBUG'] = '1'
torch.set_num_threads(2)


def layer(device='cpu', p=.95, budget=0, interval=1, prompt=2, lengths=(7, 11)):
    obj = EvictLayer(0, budget, 0, 'qhead_shared_topp_eviction', rkv_lambda=0.,
                     smooth=False, verbose=False, top_p=p, rep_seed=0, sample_id=5,
                     eviction_interval=interval)
    k = torch.zeros(len(lengths), prompt, 2, 8, device=device)
    obj.lazy_initialization(k)
    obj.prompt_len = prompt
    obj.cumulative_length = max(lengths)
    obj.cache_seqlens.copy_(torch.tensor(lengths, device=device))
    gen = torch.Generator(device=device).manual_seed(91)
    obj.k_cache.normal_(generator=gen)
    obj.v_cache.normal_(generator=gen)
    obj.update_queries(torch.randn(len(lengths), 4, 64, 8, generator=gen, device=device))
    return obj


def independent_ids(q, k, prompt, threshold):
    # Scalar row reference in float64, unique deliberately nontied random scores.
    probs = torch.softmax((q.double() @ k.double().T) / math.sqrt(k.shape[-1]), -1).mean(0)[prompt:]
    if not len(probs):
        return list(range(prompt))
    probs /= probs.sum()
    order = sorted(range(len(probs)), key=lambda i: -float(probs[i]))
    chosen = []; mass = 0.
    for i in order:
        chosen.append(prompt+i); mass += float(probs[i])
        if mass >= threshold:
            break
    return sorted(list(range(prompt))+chosen)


class CacheTests(unittest.TestCase):
    def test_ragged_selection_and_stale_slots(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for lengths in [(7, 19), (19, 7), (2, 11)]:
                obj = layer(device, p=.7, lengths=lengths)
                before_k = obj.k_cache.clone(); before_v = obj.v_cache.clone()
                # Invalid stale slots must not influence softmax, even if NaN.
                for b, n in enumerate(lengths):
                    obj.k_cache[b, n:] = float('nan')
                obj._run_eviction(max(lengths))
                for b, n in enumerate(lengths):
                    expected = independent_ids(obj.queries[b, obj._rep_head], before_k[b, :n, obj._rep_kv], 2, .7)
                    actual = obj._last_selected_ids[b].tolist()
                    self.assertEqual(actual, expected)
                    kept = len(expected)
                    self.assertEqual(obj.cache_seqlens[b].item(), kept)
                    torch.testing.assert_close(obj.k_cache[b, :kept], before_k[b, expected])
                    torch.testing.assert_close(obj.v_cache[b, :kept], before_v[b, expected])

    def test_padding_reproduction(self):
        obj = layer(p=.8, lengths=(5, 5))
        # First row ranking: token 4 first, then 2. Repeated padded 2 used to displace 4.
        obj.k_cache.zero_(); obj.queries.fill_(0.)
        obj.queries[:, :, :, 0] = math.sqrt(8)
        obj.k_cache[0, 2:5, :, 0] = torch.tensor([.3, .1, .6]).log()[:, None]
        obj.k_cache[1, 2:5, :, 0] = 0.
        obj._run_eviction(5)
        self.assertEqual(obj._last_selected_ids[0].tolist(), [0, 1, 2, 4])
        self.assertEqual(obj._last_selected_ids[1].tolist(), [0, 1, 2, 3, 4])

    def test_grows_beyond_1024_and_preserves_prompt(self):
        obj = layer(p=1., prompt=1100, lengths=(1100, 1100))
        before = obj.k_cache.clone()
        new = torch.ones(2, 1, 2, 8)
        obj.prepare(new, new)
        self.assertGreater(obj.k_cache.shape[1], 1100)
        torch.testing.assert_close(obj.k_cache[:, :1100], before)
        obj.k_cache[:, 1100] = new[:, 0]; obj.v_cache[:, 1100] = new[:, 0]
        obj.post_attention_update(1)
        self.assertEqual(obj.cache_seqlens.tolist(), [1101, 1101])
        self.assertEqual(obj.get_seq_length(), 1101)
        torch.testing.assert_close(obj.k_cache[:, :1100], before)

    def test_p095_reasoning_can_exceed_1024(self):
        obj = layer(p=.95, prompt=16, lengths=(16, 16))
        incoming = torch.zeros(2, 1200, 2, 8)
        obj.prepare(incoming, incoming)
        obj.k_cache.zero_(); obj.v_cache.zero_(); obj.queries.zero_()
        obj.cache_seqlens.fill_(1216)
        obj.cumulative_length = 1216
        obj._run_eviction(1216)
        for n in obj.cache_seqlens.tolist():
            self.assertGreater(n - 16, 1024)
            self.assertIn(n - 16, [1140, 1141])
        self.assertTrue(all(m >= .95 - 1e-5 for m in obj._sel_mass))

    def test_per_row_threshold_and_batch_shrink(self):
        obj = layer(p=.7, budget=16, lengths=(7, 15))
        obj.post_attention_update(1)
        self.assertEqual(obj.cache_seqlens[0].item(), 8)
        self.assertLess(obj.cache_seqlens[1].item(), 16)
        retained = obj.k_cache[1].clone()
        length = obj.cache_seqlens[1].item()
        obj.batch_select_indices(torch.tensor([1]))
        self.assertEqual(obj._batch_ids, [1])
        torch.testing.assert_close(obj.k_cache[0], retained)
        self.assertEqual(obj.cache_seqlens.tolist(), [length])

    def test_cadence_short_prompt_and_pair_counts(self):
        obj = EvictLayer(0, 0, 0, 'qhead_shared_topp_eviction', rkv_lambda=0., smooth=False,
                         verbose=False, top_p=1., eviction_interval=2)
        q = torch.ones(2, 4, 1, 8); k = torch.ones(2, 1, 2, 8)
        for _ in range(3):
            obj.update_queries(q)
            obj.prepare(k, k)
            for b, n in enumerate(obj.cache_seqlens.tolist()):
                obj.k_cache[b, n] = k[b, 0]; obj.v_cache[b, n] = k[b, 0]
            obj.post_attention_update(1)
        self.assertEqual(obj._q_valid, 3)
        self.assertEqual(obj.prompt_len, 1)
        self.assertEqual(obj.selection_events, 1)
        self.assertEqual(obj.selected_pairs, 2*4*(1+2+3))
        self.assertEqual(obj.selected_pairs, obj.causal_pairs)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA FlashAttention')
    def test_flash_attention_ragged_append_reference(self):
        from flash_attn import flash_attn_with_kvcache
        obj = layer('cuda', p=.7, lengths=(7, 19))
        obj.k_cache = obj.k_cache.bfloat16(); obj.v_cache = obj.v_cache.bfloat16()
        obj.queries = obj.queries.bfloat16()
        obj._run_eviction(19)
        lengths = obj.cache_seqlens.tolist()
        gen = torch.Generator(device='cuda').manual_seed(123)
        q = torch.randn(2, 1, 4, 8, device='cuda', dtype=torch.bfloat16, generator=gen)
        k = torch.randn(2, 1, 2, 8, device='cuda', dtype=torch.bfloat16, generator=gen)
        v = torch.randn(2, 1, 2, 8, device='cuda', dtype=torch.bfloat16, generator=gen)
        kc, vc, lens = obj.prepare(k, v)
        out = flash_attn_with_kvcache(q, kc, vc, k=k, v=v, cache_seqlens=lens, causal=True)
        for b, n in enumerate(lengths):
            for h in range(4):
                weights = torch.softmax(q[b, 0, h].float() @ kc[b, :n+1, h//2].float().T / math.sqrt(8), -1)
                expected = weights @ vc[b, :n+1, h//2].float()
                torch.testing.assert_close(out[b, 0, h].float(), expected, atol=.012, rtol=.02)
        obj.post_attention_update(1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
