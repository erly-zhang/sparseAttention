"""Independent prefix-by-prefix equivalence checks; CPU or allocated GPU."""
import unittest
import torch
from experiments.learned_mask_nnonly_20260914.model import (
    Config, Predictor, VARIANTS, shared_masks, final_mask, mapped_k,
    exact_probability_mean)
from .batched import (protection, fixed_mask, summaries, content_scores,
                      encode_masks, mask_scores, indices)


class Equivalence(unittest.TestCase):
    def test_all_variants(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            c = Config(head_dim=8, tile_size=8, sink=2, budget=12, core_budget=10)
            groups = {'0': [{'representative': 0, 'members': [0, 1, 2, 3]}]}
            q = torch.randn(25, 4, 8, device=device)
            k = torch.randn(25, 2, 8, device=device)
            starts, ends = [0, 8, 16, 24], [8, 16, 24, 25]
            end = torch.tensor(ends, device=device)
            start = torch.tensor(starts, device=device)
            p = torch.stack([exact_probability_mean(q[a:b, 0], k[:, 0], a)
                             for a, b in zip(starts, ends)])
            self.assertTrue(torch.allclose(p.sum(-1), torch.ones(4, device=device)))
            protect = protection(end, start, 25, c.sink)
            core = fixed_mask(p, protect, end, c.core_budget)
            full = fixed_mask(p, protect, end, c.budget)
            for variant in VARIANTS:
                model = Predictor(groups, variant, c).to(device).eval()
                for h in (1, 2, 3):
                    net = model.networks[model.routing[0, h]]
                    if variant == 'full_mask':
                        scores = mask_scores(net, encode_masks(net, full, end), h)
                    else:
                        scores = content_scores(net, net.key(mapped_k(k, 4, h)),
                            summaries(q[:, h], starts, ends, c.query_segments), p, core, end, h)
                    selected = fixed_mask(scores, protect if variant == 'full_mask' else core,
                                          end, c.budget)
                    for r, (a, b) in enumerate(zip(starts, ends)):
                        pr, co, fu = shared_masks(p[r, :b], a, b, c)
                        expected_score = model(q[a:b, h], mapped_k(k[:b], 4, h),
                                               p[r, :b], fu if variant == 'full_mask' else co, 0, h)
                        torch.testing.assert_close(scores[r, :b], expected_score, atol=3e-6, rtol=3e-5)
                        expected = final_mask(expected_score, co, fu, pr, b, c, variant)
                        self.assertTrue(torch.equal(selected[r, :b], expected))
                        self.assertFalse(selected[r, b:].any())
                    self.assertTrue(torch.equal(selected.sum(-1), end.clamp(max=c.budget)))
                    self.assertFalse((protect & ~selected).any())
                    ids = indices(selected, c.budget, 25)
                    self.assertTrue(((ids[:, 1:] > ids[:, :-1]) | (ids[:, 1:] == 25)).all())

    def test_ties_negative_and_future(self):
        s = torch.tensor([[-1., -1., -2., -1., 100.]])
        fixed = torch.tensor([[False, False, True, False, False]])
        actual = fixed_mask(s, fixed, torch.tensor([4]), 3)
        self.assertEqual(actual.tolist(), [[True, True, True, False, False]])


if __name__ == '__main__':
    unittest.main()
