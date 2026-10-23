"""vLLM glue for SnapKV / R-KV / VaSE: query capture + paged selector == upstream selection on dense tensors.

Run in the vLLM venv with PYTHONPATH=<tri_baselines_fix>. (baseline_scoring itself is checked against the
upstream EvictLayer by test_baseline_scoring.py.)
"""
import os
import types
import unittest

import numpy as np
import torch

import triattention.vllm.runtime.baseline_scoring as bs
import triattention.vllm.runtime.baselines_vllm as bv


class FakeBatch:
    def __init__(self, req_ids, prompts):
        self.req_ids = list(req_ids)
        self.num_reqs = len(req_ids)
        self.num_prompt_tokens = np.asarray(prompts)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glue(unittest.TestCase):
    def setUp(self):
        bv.RINGS.clear()
        bv.SLOT_OF.clear()

    def capture(self, req_ids, qlens, positions_end, prompts, hq=32, d=128, layers=2, seed=0):
        """Drive the patched forward for one step; returns the query tensor written."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        qsl = np.concatenate([[0], np.cumsum(qlens)])
        pos = np.concatenate([np.arange(e-q, e) for q, e in zip(qlens, positions_end)])
        bv.RUNNER = types.SimpleNamespace(input_batch=FakeBatch(req_ids, prompts), query_start_loc=types.SimpleNamespace(np=qsl),
                                          positions=torch.as_tensor(pos, device="cuda"),
                                          scheduler_config=types.SimpleNamespace(max_num_seqs=4))
        T = int(qsl[-1])
        q = torch.randn(T, hq, d, device="cuda", generator=g).bfloat16()
        md = types.SimpleNamespace(num_actual_tokens=T)
        bv._ORIG_FORWARD = lambda *a, **k: None
        impl = types.SimpleNamespace(num_heads=hq, head_size=d)
        for li in range(layers):
            layer = types.SimpleNamespace(layer_name=f"model.layers.{li}.self_attn.attn")
            bv._patched_forward(impl, layer, q, None, None, None, md, None)
        return q, qsl, pos

    def test_ring_holds_last_window(self):
        # step 1: two prefills (prompt 100 and 30 tokens); step 2: both decode one token
        q1, qsl1, pos1 = self.capture(["a-12345678", "b-12345678"], [100, 30], [100, 30], [100, 30], seed=1)
        q2, qsl2, pos2 = self.capture(["b-12345678", "a-12345678"], [1, 1], [31, 101], [30, 100], seed=2)  # rows reordered
        W = bv.WINDOW
        for rid, toks in (("a-12345678", [(q1, list(range(0, 100)), 0), (q2, [100], 1)]),
                          ("b-12345678", [(q1, list(range(100, 130)), 0), (q2, [30], 0)])):
            ring = bv.RINGS[1][bv.SLOT_OF[rid]]
            expect = torch.zeros_like(ring)
            if rid == "a-12345678":   # upstream: last 64 prompt queries in slots 0..63, decode round-robin from 0
                for t in range(36, 100):
                    expect[t-36] = q1[t]
                expect[0] = q2[1]
            else:
                for t in range(30):
                    expect[t % W] = q1[100+t]
                expect[30 % W] = q2[0]
            self.assertTrue(torch.equal(ring, expect), rid)

    def run_selector(self, kind, K, total, seed):
        os.environ["TRIATTN_RUNTIME_SELECTOR"] = kind
        hkv, hq, d, bs_ = 8, 32, 128, 16
        g = torch.Generator(device="cuda").manual_seed(seed)
        nblocks = (total+bs_-1)//bs_+5
        kv = torch.randn(2, nblocks+20, bs_, hkv, d, device="cuda", generator=g).bfloat16()
        block_ids = torch.randperm(nblocks+20, device="cuda", generator=g)[:nblocks].tolist()
        rid = "7-abcdefgh"
        bv.SLOT_OF[rid] = 2
        bv.RINGS[5] = torch.randn(4, bv.WINDOW, hq, d, device="cuda", generator=g).bfloat16()
        cfg = types.SimpleNamespace(kv_budget=K, window_size=bv.WINDOW)
        runner = types.SimpleNamespace(_triattention_active_req_id=rid)
        sel, _, _ = bv.build_baseline_selector(cfg, runner)
        budget_total = K-4
        out = sel(kv_cache=kv, block_ids=block_ids, block_size=bs_, total_tokens=total, prefill_len=100,
                  protect_prefill=False, layer_idx=5, round_start=total, budget_total=budget_total)
        # reference: dense tensors in the upstream layout
        j = torch.arange(total, device="cuda")
        blk = torch.as_tensor(block_ids, device="cuda")[j//bs_]
        dense_k = kv[0][blk, j % bs_]          # [total, Hkv, D]
        dense_v = kv[1][blk, j % bs_]
        n_cand = total-bv.WINDOW
        kc = dense_k[:n_cand].unsqueeze(0).transpose(1, 2)
        vc = dense_v[:n_cand].unsqueeze(0).transpose(1, 2)
        queries = bv.RINGS[5][2].permute(1, 0, 2).contiguous().unsqueeze(0)
        gen = torch.Generator(device="cuda").manual_seed(bv._stable_seed(1234, "7", total, 5))
        ids = bs.select(bs.MODES[kind], kc, vc, queries, budget_total-bv.WINDOW, n_large=K//4, generator=gen)[0]
        expect = torch.sort(torch.cat([ids, torch.arange(n_cand, total, device="cuda").expand(hkv, -1)], -1), -1).values
        self.assertEqual(out["mode"], "per_head")
        self.assertTrue(torch.equal(out["indices"], expect), (kind, K, total))
        self.assertEqual(out["indices"].shape, (hkv, budget_total))

    def test_selectors(self):
        for kind in ("snapkv", "rkv", "vase"):
            for K, total in ((1024, 1084), (2048, 2108)):
                self.run_selector(kind, K, total, seed=K+total)

    def test_vllm_paths_layout(self):
        # the runtime's paged gather returns [n, Hkv, D]; the transposed view must match upstream strides
        kv = torch.randn(2, 10, 16, 8, 128, device="cuda").bfloat16()
        k = kv[0][torch.tensor([3, 3]), torch.tensor([0, 1])].transpose(0, 1).unsqueeze(0)
        self.assertEqual(k.stride()[1:], (128, 1024, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
