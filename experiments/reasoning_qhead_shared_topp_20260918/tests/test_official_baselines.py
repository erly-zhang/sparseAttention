"""Paper baselines through our EvictCache == the unmodified upstream EvictCache, bitwise.

Real Qwen3-4B geometry (36 layers, 32 Q / 8 KV heads, head dim 128) on synthetic
decode streams long enough for several 64-step eviction rounds. The reference is the
vendored upstream EvictCache/EvictLayer at commit 64db9688 driven directly, with the
same private per-(sample, layer) RNG stream the adapters use around each eviction.
"""
import os
import unittest
from pathlib import Path
import torch

STATS = os.environ.get('MB_TRIATTN_STATS', str(Path(__file__).resolve().parents[1]/'data/triattn/qwen3-4b.pt'))


def geometry():
    """Model geometry of the run's TriAttention stats (Qwen3-4B 36/32/8, Phi-4 40/40/10)."""
    if Path(STATS).exists():
        meta = torch.load(STATS, map_location='cpu', weights_only=False)['metadata']
        return int(meta['n_layers']), int(meta['Hq']), int(meta['Hkv']), int(meta['head_dim'])
    return 36, 32, 8, 128


def qwen_config():
    from transformers import Qwen3Config
    layers, hq, hkv, d = geometry()
    return Qwen3Config(num_hidden_layers=layers, num_attention_heads=hq, num_key_value_heads=hkv, head_dim=d,
                       hidden_size=hq*d, intermediate_size=4*hq*d, vocab_size=1024)


def kwargs_for(mode, K):
    return dict(attn=dict(rkv_lambda=None, smooth=True, n_large=-1),
                attn_rkv=dict(rkv_lambda=0.5, smooth=True, n_large=-1),
                range_sink_sample_attn=dict(rkv_lambda=None, smooth=True, n_large=K//4),
                triattn_ph=dict(rkv_lambda=None, smooth=False, n_large=256))[mode]


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def run_pair(self, ours_mode, up_mode, K=256, prompt=137, steps=400, sample_id=7):
        import kvcompress.engine.official_random_attention.cache_utils as up
        from kvcompress.engine.cache_utils import EvictCache, _stable_hash
        cfg = qwen_config()
        kw = kwargs_for(up_mode, K)
        ours = EvictCache(cfg, K, 64, ours_mode, verbose=False, temperature=1.0, top_p=.95, query_window=1,
                          eviction_interval=1, rep_seed=0, sample_id=sample_id, mask_seed=42, **kw)
        ref = up.EvictCache(cfg, K, 64, up_mode, verbose=False, temperature=1.0, **kw)
        gens = {}

        def ref_evict(layer, cur_len, original=up.EvictLayer._run_eviction):
            # Identical private stream to OfficialRandomPPLayer._run_eviction.
            if layer.layer_idx not in gens:
                gens[layer.layer_idx] = torch.Generator(device='cuda').manual_seed(
                    _stable_hash(42, sample_id, layer.layer_idx))
            g = gens[layer.layer_idx]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.cuda.set_rng_state(g.get_state(), torch.cuda.current_device())
                original(layer, cur_len)
                g.set_state(torch.cuda.get_rng_state(torch.cuda.current_device()))
        from flash_attn import flash_attn_with_kvcache
        layers, hq, hkv, d = geometry()
        g = torch.Generator(device='cuda').manual_seed(3)
        before = torch.cuda.get_rng_state().clone()
        for step in range(steps):
            n = prompt if step == 0 else 1
            for li in range(layers):
                q = torch.randn(2, n, hq, d, device='cuda', generator=g).bfloat16()
                k = torch.randn(2, n, hkv, d, device='cuda', generator=g).bfloat16()
                v = torch.randn(2, n, hkv, d, device='cuda', generator=g).bfloat16()
                a, b = ours.layers[li], ref.layers[li]
                a.update_queries(q.transpose(1, 2))
                b.update_queries(q.transpose(1, 2))
                out_a = a.attention(q, k, v, d**-.5)
                kc, vc, lengths = b.prepare(k, v)
                out_b = flash_attn_with_kvcache(q, kc, vc, k=k, v=v, cache_seqlens=lengths, causal=True,
                                                softmax_scale=d**-.5, window_size=(-1, -1))
                if not hasattr(b, '_patched'):
                    b._run_eviction = (lambda layer: (lambda cur_len: ref_evict(layer, cur_len)))(b)
                    b._patched = True
                b.post_attention_update(n)  # upstream trigger rule, private-stream eviction
                self.assertTrue(torch.equal(out_a, out_b), (ours_mode, step, li))
                m = int(a.cache_seqlens[0])
                self.assertEqual(m, int(b.cache_seqlens[0]))
                self.assertTrue(torch.equal(a.k_cache[:, :m], b.k_cache[:, :m]), (ours_mode, step, li))
                self.assertTrue(torch.equal(a.v_cache[:, :m], b.v_cache[:, :m]), (ours_mode, step, li))
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        events = sum(l.selection_events for l in ours.layers)
        self.assertGreater(events, layers*3)
        return ours

    def test_snapkv(self):
        self.run_pair('official_snapkv', 'attn')

    def test_rkv(self):
        self.run_pair('official_rkv', 'attn_rkv')

    def test_vase(self):
        self.run_pair('official_vase', 'range_sink_sample_attn')

    @unittest.skipUnless(Path(STATS).exists(), 'TriAttention stats missing')
    def test_triattn_cross_layer_and_memo_reset(self):
        os.environ['TRIATTN_STATS'] = STATS
        os.environ['TRIATTN_FIX_MEMO'] = '1'
        import kvcompress.engine.official_random_attention.cache_utils as up
        self.run_pair('official_triattn', 'triattn_ph')
        # A new cache (= new generation) must clear the vendored cross-layer memo.
        up._triattn_ph_shared = 'stale'
        from kvcompress.engine.cache_utils import EvictCache
        cache = EvictCache(qwen_config(), 256, 64, 'official_triattn', rkv_lambda=None, smooth=False, n_large=256,
                           verbose=False, temperature=1.0, sample_id=1, mask_seed=42)
        self.assertIsNone(up._triattn_ph_shared)
        self.assertIs(cache.layers[5]._sibling_layers, cache.layers)


if __name__ == '__main__':
    unittest.main(verbosity=2)
