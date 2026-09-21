"""CPU routing checks and optional real FlexPrefill CUDA numerical checks."""
import copy
import importlib
import math
import unittest

import torch

from experiments.shared_flexprefill_20260911.adapter import representative_indices, validate_groups


GROUPS = [{"representative": 3, "members": [1, 3, 7]},
          {"representative": 5, "members": [2, 4, 5]},
          {"representative": 0, "members": [0, 6]}]


def dense_reference(q, k, v, masks, block_size):
    n, heads, dim = q.shape[1:]
    blocks = math.ceil(n / block_size)
    positions = torch.arange(n, device=q.device)
    result = torch.empty_like(q)
    for h in range(heads):
        block_mask = torch.zeros(blocks * blocks, device=q.device, dtype=torch.bool)
        block_mask[masks[0][h].long()] = True
        mask = block_mask.view(blocks, blocks)[positions[:, None] // block_size,
                                              positions[None, :] // block_size]
        mask &= positions[:, None] >= positions[None, :]
        kv = h // (heads // k.shape[2])
        score = q[0, :, h].float() @ k[0, :, kv].float().T / math.sqrt(dim)
        prob = score.masked_fill(~mask, -torch.inf).softmax(-1)
        result[0, :, h] = (prob @ v[0, :, kv].float()).to(q.dtype)
    return result


class RoutingTests(unittest.TestCase):
    def test_partition_validation(self):
        validate_groups({"layers": {"0": GROUPS}}, 1, 8)
        bad = copy.deepcopy(GROUPS)
        bad[0]["members"].append(0)
        with self.assertRaises(ValueError):
            validate_groups({"layers": {"0": bad}}, 1, 8)

    def test_gqa_broadcast_and_parameters(self):
        q = torch.arange(8.).view(1, 1, 8, 1).expand(1, 17, 8, 1)
        k = torch.tensor([10., 20.]).view(1, 1, 2, 1).expand(1, 17, 2, 1)
        for interleave in (False, True):
            def selector(rq, rk, rv, **kwargs):
                self.assertEqual(rq[0, 0, :, 0].tolist(), [3., 5., 0.])
                self.assertEqual(rk[0, 0, :, 0].tolist(),
                                 [20., 20., 10.] if interleave else [10., 20., 10.])
                torch.testing.assert_close(rv, rk + 100)
                self.assertFalse(kwargs["gqa_interleave"])
                self.assertEqual(kwargs["gamma"], .9)
                return [[torch.tensor([i]) for i in range(3)]]
            masks = representative_indices(selector, q, k, k + 100, GROUPS,
                                            dict(gqa_interleave=interleave, gamma=.9))
            self.assertEqual([m.item() for m in masks[0]], [2, 0, 1, 0, 1, 1, 2, 0])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/FlexPrefill required")
class KernelTests(unittest.TestCase):
    def test_official_masks_and_member_kernel(self):
        ops = importlib.import_module("flex_prefill.ops.flex_prefill_attention")
        torch.manual_seed(42)
        for n in (512, 641):
            q = torch.randn(1, n, 8, 32, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, n, 2, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            for tau in (0., 1.):
                params = dict(block_size=128, gamma=.9, min_budget=1,
                              max_budget=math.ceil(n / 128), tau=tau)
                original = ops.get_active_blocks(q, k, v, **params)
                masks = representative_indices(ops.get_active_blocks, q, k, v, GROUPS, params)
                for group in GROUPS:
                    expected = original[0][group["representative"]].sort().values
                    for h in group["members"]:
                        torch.testing.assert_close(masks[0][h].sort().values, expected)
                got = ops.triton_block_wise_attention(q, k, v, masks, block_size=128)
                ref = dense_reference(q, k, v, masks, 128)
                torch.testing.assert_close(got, ref, atol=.03, rtol=.03)
                self.assertTrue(torch.isfinite(got).all())
                # Test kernel causality with a FROZEN mask: selectors use tail queries.
                changed = v.clone()
                changed[:, n // 2:] += 10
                out = ops.triton_block_wise_attention(q, k, changed, masks, block_size=128)
                torch.testing.assert_close(out[:, :n // 2], got[:, :n // 2], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

