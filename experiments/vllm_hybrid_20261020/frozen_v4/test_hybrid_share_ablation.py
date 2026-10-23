"""Share ablation (A1) kernels and selector.

1. Per-KV-head path (NH = H_kv): each KV head's scores come from its own representative Q head (inside its
   GQA group), its selection is exact, and every Q head attends over its KV head's own positions.
2. Shared path (NH = 1) is bitwise identical to the archived v2 module (indices, scores and outputs).
3. Per-head compaction selector: every head keeps its own core + full prompt + newer slots, exactly
   budget_total positions, sorted, deterministic; returned as mode=per_head.
Run with PYTHONPATH=<runtime copy containing the new hybrid files> (for 3).
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hybrid_attention as ha  # noqa: E402
from test_hybrid_kernels import logical, make_cache, ref_select  # noqa: E402

spec = importlib.util.spec_from_file_location("ha_v2", HERE.parents[1]/"archive/hybrid_attention_v2_shared_only.py")
ha_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha_v2)


def build(mod, K, lens, prompts, nh, hq, hkv, seed, mixed=False):
    d, bs = 128, 16
    kv = make_cache(hkv=hkv, seed=seed)
    kc, vc = kv.unbind(0)
    B = len(lens)
    maxb = (max(lens)+bs-1)//bs
    perm = torch.randperm(kv.shape[1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed+1))
    bt = perm[:B*maxb].view(B, maxb).int().contiguous()
    seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
    qrow = np.arange(B)*(3 if mixed else 1)
    T = int(qrow[-1])+1
    q = torch.randn(T, hq, d, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed+2)).bfloat16()
    rows = np.arange(B)
    pl = np.array([p if p+64 <= K else K-65 for p in prompts])
    keys = [7*r+1 for r in rows]
    if nh == 1:
        rep = ha.representative_heads(keys, np.array(lens)-1, hq)
    else:
        rep = ha.representative_heads_per_kv(keys, np.array(lens)-1, hq, hkv).reshape(ha.MAX_LAYERS, -1)
    packed = np.zeros((3+ha.MAX_LAYERS, B*nh), dtype=np.int32)
    packed[0, :B], packed[1, :B], packed[2, :B] = rows, qrow, pl
    packed[3:] = rep
    dev = torch.from_numpy(packed).cuda()
    plan = mod.StepPlan()
    plan.rows, plan.qrow, plan.pl, plan.rep = dev[0, :B], dev[1, :B], dev[2, :B], dev[3:]
    plan.nb, plan.max_n = B, max(lens)
    if hasattr(type(plan), "nh") or "nh" in getattr(mod.StepPlan, "__slots__", ()):
        plan.nh = nh
    plan.row_of_req = {f"r{r}": r for r in rows}
    plan.n_cpu = np.array(lens)
    plan.scores = torch.empty((B*nh, ha.SELECT_BLOCK), device="cuda", dtype=torch.bfloat16)
    plan.ml = plan.acc = None
    out = torch.full((T, hq, d), float("nan"), device="cuda").bfloat16()
    return dict(kc=kc, vc=vc, bt=bt, seq=seq, q=q, plan=plan, out=out, qrow=qrow, pl=pl, rep=rep, d=d)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ShareAblation(unittest.TestCase):
    def check_per_head(self, K, lens, prompts, hq=32, hkv=8, seed=0, mixed=False):
        c = build(ha, K, lens, prompts, hkv, hq, hkv, seed, mixed)
        layer = 7
        idx, cnt = ha.sparse_attention(c["q"], c["kc"], c["vc"], c["bt"], c["seq"], c["plan"], layer, c["d"]**-.5,
                                       c["out"], hq, hkv, c["d"], K)
        torch.cuda.synchronize()
        grp = hq//hkv
        self.assertEqual(tuple(idx.shape), (len(lens), hkv, K))
        rep = c["rep"][layer].reshape(len(lens), hkv)
        for i, n in enumerate(lens):
            keys, vals = logical(c["kc"], c["bt"][i], n), logical(c["vc"], c["bt"][i], n)
            sels = []
            for h in range(hkv):
                r = int(rep[i, h])
                self.assertEqual(r//grp, h)                      # representative inside its own group
                got = c["plan"].scores[i*hkv+h, :n]
                ref = (keys[:, h].float() @ c["q"][c["qrow"][i], r].float()).bfloat16()
                if n > K:
                    self.assertTrue(bool(((got.float()-ref.float()).abs() <= ref.float().abs()*2**-7+1e-3).all()))
                sel = ref_select(got, n, int(c["pl"][i]), K)
                cn = int(cnt[i*hkv+h])
                self.assertEqual(cn, min(n, K))
                self.assertTrue(torch.equal(idx[i, h, :cn].long(), sel), (i, h))
                sels.append(sel)
            middle, extra = n-64-int(c["pl"][i]), K-64-int(c["pl"][i])
            if n > K and 0 < extra < middle:   # a real choice exists: heads really select differently
                self.assertTrue(any(not torch.equal(sels[0], s) for s in sels[1:]))
            for qh in range(hq):
                h = qh//grp
                kk, vv = keys[sels[h], h].float(), vals[sels[h], h].float()
                p = torch.softmax(kk @ c["q"][c["qrow"][i], qh].float()*c["d"]**-.5, 0)
                err = (c["out"][c["qrow"][i], qh].float()-p @ vv).abs().max().item()
                self.assertLess(err, 2e-2, (i, qh, err))

    def test_per_head_k1024(self):
        self.check_per_head(1024, [500, 1025, 3000, 6144], [100, 1000, 1024, 300])

    def test_per_head_k2048_mixed(self):
        self.check_per_head(2048, [2049, 4100, 7168], [1024, 1984, 2000], seed=3, mixed=True)

    def test_per_head_phi_geometry(self):
        self.check_per_head(1024, [1500, 5000], [800, 960], hq=40, hkv=10, seed=5)

    def test_shared_path_bitwise_equal_to_v2(self):
        for K, lens, prompts in ((1024, [500, 1025, 3000, 6144, 7168], [100, 1000, 1024, 300, 2000]),
                                 (2048, [2049, 4100, 8192], [1024, 1984, 2000])):
            a = build(ha, K, lens, prompts, 1, 32, 8, seed=11)
            b = build(ha_v2, K, lens, prompts, 1, 32, 8, seed=11)
            ia, ca = ha.sparse_attention(a["q"], a["kc"], a["vc"], a["bt"], a["seq"], a["plan"], 3, 0.088, a["out"],
                                         32, 8, 128, K)
            ib, cb = ha_v2.sparse_attention(b["q"], b["kc"], b["vc"], b["bt"], b["seq"], b["plan"], 3, 0.088,
                                            b["out"], 32, 8, 128, K)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(ca, cb))
            ia = ia.view(len(lens), K)
            for r, cn in enumerate(cb.tolist()):   # entries beyond the count are never written
                self.assertTrue(torch.equal(ia[r, :cn], ib[r, :cn]), r)
            for r, n in enumerate(lens):   # scores are written for the live positions only
                self.assertTrue(torch.equal(a["plan"].scores[r, :n], b["plan"].scores[r, :n]), r)
            self.assertTrue(torch.equal(a["out"], b["out"]))

    def test_per_head_compaction_selector(self):
        try:
            import triattention.vllm.runtime.hybrid_attention as rha
            import triattention.vllm.runtime.selector_hybrid as sh
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"runtime package not importable: {e}")
        import os
        os.environ["HYBRID_K"] = "1024"
        hkv, K, P, n_prev = 8, 1024, 300, 6143
        total, budget = n_prev+1, 5116
        g = torch.Generator(device="cuda").manual_seed(0)
        core = torch.stack([torch.cat([torch.arange(P), torch.sort(torch.randperm(n_prev-64-P, generator=torch.Generator().manual_seed(h))[:K-P-64]).values+P,
                                       torch.arange(n_prev-64, n_prev)]) for h in range(hkv)]).int().cuda()
        rha.REGISTRY.clear()
        rha.REGISTRY[4] = ({"9-abcdefgh": 0}, core[None], np.array([n_prev]), 1)
        runner = types.SimpleNamespace(_triattention_active_req_id="9-abcdefgh")
        sel, _, _ = sh.build_hybrid_selector(types.SimpleNamespace(), runner)
        kw = dict(kv_cache=torch.zeros(1, device="cuda"), block_ids=None, block_size=16, total_tokens=total,
                  prefill_len=P, protect_prefill=True, layer_idx=4, round_start=total, budget_total=budget)
        out1, out2 = sel(**kw), sel(**kw)
        self.assertEqual(out1["mode"], "per_head")
        ind = out1["indices"]
        self.assertEqual(tuple(ind.shape), (hkv, budget))
        self.assertTrue(torch.equal(ind, out2["indices"]))
        for h in range(hkv):
            s = set(ind[h].tolist())
            self.assertTrue(set(core[h].tolist()) <= s and set(range(P)) <= s and n_prev in s)
            self.assertTrue(bool((ind[h, 1:] > ind[h, :-1]).all()))
        self.assertFalse(torch.equal(ind[0], ind[1]))     # independent per-head fill


if __name__ == "__main__":
    unittest.main(verbosity=2)
