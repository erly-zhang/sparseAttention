"""baseline_scoring.select == upstream EvictLayer._run_eviction (Random Attention engine 64db9688).

Run in the HF environment (the vendored upstream module needs transformers 5). Qwen3-4B geometry
(32 Q / 8 KV heads, D=128), BF16, candidates of several lengths; the kept K/V of the upstream layer
must equal our selection gathered from the same tensors, in the same order, for SnapKV (attn),
R-KV (attn_rkv, lambda 0.5) and VaSE (range_sink_sample_attn, n_large=K/4, same RNG stream).
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/mmfs1/project/yl258/jz648/sparseAttention/experiments/reasoning_qhead_shared_topp_20260918/ra_harness')
import baseline_scoring as bs  # noqa: E402
import kvcompress.engine.official_random_attention.cache_utils as up  # noqa: E402


class Equivalence(unittest.TestCase):
    def run_case(self, mode, K, cur_len, device, seed):
        hq, hkv, d, res = 32, 8, 128, 64
        g = torch.Generator(device=device).manual_seed(seed)
        layer = up.EvictLayer(0, K, res, mode, rkv_lambda=0.5, smooth=True, verbose=False, n_large=K//4, temperature=1.0)
        prompt = torch.randn(1, 100, hkv, d, device=device, generator=g).bfloat16()
        layer.lazy_initialization(prompt, prompt)
        n = min(cur_len, layer.k_cache.shape[1])
        layer.k_cache[:, :n] = torch.randn(1, n, hkv, d, device=device, generator=g).bfloat16()
        layer.v_cache[:, :n] = torch.randn(1, n, hkv, d, device=device, generator=g).bfloat16()
        layer.queries = torch.randn(1, hq, res, d, device=device, generator=g).bfloat16()
        layer._q_write_pos = 17
        # same memory layout as upstream ([B, n, Hkv, D] storage viewed as [B, Hkv, n, D]), as in the vLLM path
        k_cand = layer.k_cache[:, :n-res].clone().transpose(1, 2)
        v_cand = layer.v_cache[:, :n-res].clone().transpose(1, 2)
        queries = layer.queries.clone()
        # ours, with an explicit generator in the same state as upstream's global stream
        gen = torch.Generator(device=device).manual_seed(1234+seed)
        ids = bs.select(bs.MODES[mode] if mode in bs.MODES else mode, k_cand, v_cand, queries, K-res,
                        rkv_lambda=0.5, smooth=True, n_large=K//4, temperature=1.0, generator=gen)
        if device == 'cuda':
            torch.cuda.manual_seed(1234+seed)
        else:
            torch.manual_seed(1234+seed)
        layer.cache_seqlens.fill_(n)
        layer._run_eviction(n)
        exp_k = layer.k_cache[:, :K-res].transpose(1, 2)
        exp_v = layer.v_cache[:, :K-res].transpose(1, 2)
        # upstream sorts every keep-set chronologically before the gather (as the vLLM selector does)
        idx = torch.sort(ids, dim=-1).values[..., None].expand(-1, -1, -1, d)
        self.assertTrue(torch.equal(torch.gather(k_cand, 2, idx), exp_k), (mode, K, n, device))
        self.assertTrue(torch.equal(torch.gather(v_cand, 2, idx), exp_v), (mode, K, n, device))
        self.assertEqual(int(layer.cache_seqlens[0]), K)

    def test_all(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in ('attn', 'attn_rkv', 'range_sink_sample_attn'):
                for K, cur in ((1024, 1088), (2048, 2112)):
                    for seed in (0, 1):
                        self.run_case(mode, K, cur, device, seed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
