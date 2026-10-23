"""Hybrid vLLM kernels vs a PyTorch reference on a random paged KV cache.

Scores: fp32 dot rounded to BF16 (tolerance: accumulation order may differ by 1 ulp).
Selection: exact equality with a stable-sort reference fed the kernel's own scores.
Attention: fp32 softmax reference over the selected positions.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hybrid_attention as ha  # noqa: E402


def make_cache(nblocks=3000, bs=16, hkv=8, d=128, seed=0, layout="NHD"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "NHD":
        kv = torch.randn(2, nblocks, bs, hkv, d, device="cuda", generator=g).bfloat16()
    else:  # HND memory layout seen through an NHD-shaped view
        kv = torch.randn(2, nblocks, hkv, bs, d, device="cuda", generator=g).bfloat16().transpose(2, 3)
    return kv


def logical(cache, bt_row, n):
    bs = cache.shape[1]
    j = torch.arange(n, device="cuda")
    return cache[bt_row[j//bs].long(), j % bs]  # [n, hkv, d]


def ref_select(scores_row, n, pl, K):
    if n <= K:
        return torch.arange(n, device="cuda")
    start, end = pl, max(pl, n-64)
    extra = K-start-(n-end)
    mid = scores_row[start:end].float()
    order = torch.sort(mid, descending=True, stable=True).indices[:extra]
    chosen = torch.sort(order).values+start
    return torch.cat([torch.arange(start, device="cuda"), chosen, torch.arange(end, n, device="cuda")])


class KernelChecks(unittest.TestCase):
    def run_case(self, K, lens, prompts, hq=32, hkv=8, layout="NHD", mixed=False, seed=0):
        d, bs = 128, 16
        kv = make_cache(hkv=hkv, layout=layout, seed=seed)
        kc, vc = kv.unbind(0)
        B = len(lens)
        nblocks = kv.shape[1]
        maxb = (max(lens)+bs-1)//bs
        perm = torch.randperm(nblocks, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed+1))
        bt = perm[:B*maxb].view(B, maxb).int().contiguous()
        seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
        qrow = np.arange(B)*(3 if mixed else 1)
        T = int(qrow[-1])+1
        q = torch.randn(T, hq, d, device="cuda").bfloat16()
        rows = np.arange(B)
        pl = np.array([p if p+64 <= K else K-65 for p in prompts])
        rep = ha.representative_heads([7*r+1 for r in rows], np.array(lens)-1, hq)
        packed = np.concatenate([rows[None], qrow[None], pl[None], rep]).astype(np.int32)
        dev = torch.from_numpy(packed).cuda()
        plan = ha.StepPlan()
        plan.rows, plan.qrow, plan.pl, plan.rep = dev[0], dev[1], dev[2], dev[3:]
        plan.nb, plan.max_n = B, max(lens)
        plan.row_of_req = {f"r{r}": r for r in rows}
        plan.n_cpu = np.array(lens)
        plan.scores = torch.empty((B, ha.SELECT_BLOCK), device="cuda", dtype=torch.bfloat16)
        plan.ml = plan.acc = None
        out = torch.full((T, hq, d), float("nan"), device="cuda").bfloat16()
        layer = 5
        idx, cnt = ha.sparse_attention(q, kc, vc, bt, seq, plan, layer, d**-.5, out, hq, hkv, d, K)
        torch.cuda.synchronize()
        grp = hq//hkv
        for i, n in enumerate(lens):
            r = int(rep[layer, i])
            keys = logical(kc, bt[i], n)
            vals = logical(vc, bt[i], n)
            ref_scores = (keys[:, r//grp].float() @ q[qrow[i], r].float()).bfloat16()
            got = plan.scores[i, :n]
            if n > K:
                diff = (got.float()-ref_scores.float()).abs()
                tol = ref_scores.float().abs()*2**-7+1e-3
                self.assertTrue(bool((diff <= tol).all()), (i, float(diff.max())))
            sel = ref_select(got, n, int(pl[i]), K)
            c = int(cnt[i])
            self.assertEqual(c, min(n, K))
            self.assertTrue(torch.equal(idx[i, 0, :c].long(), sel), i)
            self.assertTrue(bool((sel[1:] > sel[:-1]).all()))
            kk, vv = keys[sel].float(), vals[sel].float()  # [c, hkv, d]
            for h in range(hq):
                p = torch.softmax(kk[:, h//grp] @ q[qrow[i], h].float()*d**-.5, 0)
                expect = p @ vv[:, h//grp]
                err = (out[qrow[i], h].float()-expect).abs().max().item()
                self.assertLess(err, 2e-2, (i, h, err))
        if mixed:  # untouched rows stay NaN
            others = sorted(set(range(T))-set(qrow.tolist()))
            self.assertTrue(bool(torch.isnan(out[others].float()).all()))

    def test_k1024_mixed_lengths(self):
        self.run_case(1024, [500, 1024, 1025, 3000, 6144, 7168], [100, 1024, 1000, 1024, 2000, 300])

    def test_k2048_long_prompt_and_rows(self):
        self.run_case(2048, [2049, 4100, 7168, 8192], [1024, 3000, 2000, 1984], mixed=True, seed=3)

    def test_hnd_layout_and_phi_geometry(self):
        self.run_case(1024, [1500, 5000], [800, 960], hq=40, hkv=10, layout="HND", seed=5)

    def test_ties_prefer_earlier(self):
        # Constant keys -> all scores tie; selection must be the earliest middle slots.
        K, n, P = 1024, 4000, 200
        kv = torch.ones(2, 300, 16, 8, 128, device="cuda").bfloat16()
        kc, vc = kv.unbind(0)
        bt = torch.arange(250, device="cuda", dtype=torch.int32)[None]
        seq = torch.tensor([n], device="cuda", dtype=torch.int32)
        dev = torch.zeros(3+ha.MAX_LAYERS, 1, dtype=torch.int32, device="cuda")
        dev[2] = P
        plan = ha.StepPlan()
        plan.rows, plan.qrow, plan.pl, plan.rep = dev[0], dev[1], dev[2], dev[3:]
        plan.nb, plan.max_n, plan.row_of_req, plan.n_cpu = 1, n, {}, np.array([n])
        plan.scores = torch.empty((1, ha.SELECT_BLOCK), device="cuda", dtype=torch.bfloat16)
        plan.ml = plan.acc = None
        q = torch.ones(1, 32, 128, device="cuda").bfloat16()
        out = torch.empty_like(q)
        idx, _ = ha.sparse_attention(q, kc, vc, bt, seq, plan, 0, 1.0, out, 32, 8, 128, K)
        extra = K-P-64
        expect = torch.cat([torch.arange(P+extra), torch.arange(n-64, n)]).cuda()
        self.assertTrue(torch.equal(idx[0, 0].long(), expect))


if __name__ == "__main__":
    unittest.main(verbosity=2)
