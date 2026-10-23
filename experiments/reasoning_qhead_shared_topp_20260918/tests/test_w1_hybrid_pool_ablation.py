"""Pool-step ablations I and III on the v6 engine.

1. Below the pool size (no removal yet) both are bitwise identical to v6 (outputs, ids, pool, counters).
2. I: once full, every step removes exactly one position, never one selected by that step's W1 core
   (prompt and recent 64 included) and never the new token; the pool stays chronological at M = 4096.
3. III: once full, every step first removes exactly one position outside the full prompt and the 63 newest
   slots (W1 core not protected), then appends; the pool stays at M; W1 still selects prompt + recent 64.
"""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvcompress.engine.w1_hybrid_pool_ablation_cache import (HybridPoolStepLayer, POOL_STEP, POOL_STEP_PRE,
                                                            POOL_TOTAL)
from kvcompress.engine.w1_hybrid_v6_cache import HybridV6Layer
from kvcompress.engine.w1_hybrid_variants_cache import mode_name
from test_w1_hybrid_variants import make, step_inputs


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CUDAChecks(unittest.TestCase):
    def test_identical_to_v6_before_the_pool_fills(self):
        for mode in (POOL_STEP, POOL_STEP_PRE):
            ref = make(HybridV6Layer, mode_name(1024, 'random'), 1024, 1500, 1024, 137+3000)
            new = make(HybridPoolStepLayer, mode, 1, 1500, 1024, 137+3000)
            g = torch.Generator(device='cuda').manual_seed(3)
            for i in range(300):
                q, k, v = step_inputs(g, 2)
                outs = []
                for c in (ref, new):
                    c.update_queries(q.transpose(1, 2))
                    outs.append(c.attention(q, k, v, 128**-.5))
                self.assertTrue(torch.equal(*outs), (mode, i))
                self.assertTrue(torch.equal(ref._ids, new._ids), (mode, i))
                n = new._lengths[0]   # capacities differ (v6 pool K+4096+1024 vs 4096+1); compare live slots
                self.assertEqual(ref._lengths, new._lengths)
                self.assertTrue(torch.equal(ref.slot_positions[:, :n], new.slot_positions[:, :n]), (mode, i))
                self.assertTrue(torch.equal(ref.k_cache[:, :n], new.k_cache[:, :n]), (mode, i))
            self.assertEqual(new.compaction_events, 0)
            self.assertEqual(ref.selected_pairs, new.selected_pairs)

    def run_full(self, mode, steps=300, prompt=137, budget=1024):
        M = POOL_TOTAL
        layer = make(HybridPoolStepLayer, mode, 1, M, budget, prompt+20000, prompt=prompt)
        self.assertEqual(layer.capacity, M+1)
        g = torch.Generator(device='cuda').manual_seed(11)
        removed_all = []
        for i in range(steps):
            q, k, v = step_inputs(g, 2)
            before = layer.slot_positions[:, :M].clone()
            new_pos = layer.cumulative_length
            layer.update_queries(q.transpose(1, 2))
            layer.attention(q, k, v, 128**-.5)
            after = layer.slot_positions[:, :M]
            self.assertEqual(layer._lengths, [M, M])
            self.assertTrue(bool((after[:, 1:] > after[:, :-1]).all()))           # chronological
            self.assertTrue(torch.equal(after[:, :prompt], torch.arange(prompt, device='cuda').expand(2, -1)))
            self.assertTrue(bool((after[:, -1] == new_pos).all()))                  # new token kept
            for row in range(2):
                b, a = set(before[row].tolist()), set(after[row].tolist())
                removed = b-a
                self.assertEqual(len(removed), 1, (mode, i, row))
                self.assertEqual(a-b, {new_pos})
                r = removed.pop()
                removed_all.append(r)
                ordered = before[row].tolist()
                if mode == POOL_STEP:
                    # the removed token was not selected in this step (core = selected slots of the M+1 pool)
                    pool = ordered+[new_pos]
                    core = {pool[s] for s in layer._last_selected_ids[row].tolist()} if hasattr(layer, '_last_selected_ids') else set()
                    self.assertNotIn(r, core, (i, row))
                else:
                    self.assertGreaterEqual(r, prompt)
                    self.assertNotIn(r, ordered[-63:])
            ids = layer._ids
            self.assertEqual(ids.shape[1], budget)
        self.assertEqual(layer.compaction_events, steps)
        self.assertGreater(len(set(removed_all)), steps)   # removals spread over the pool
        return layer

    def test_pool_step_after_attention_I(self):
        layer = self.run_full(POOL_STEP)
        self.assertEqual(layer.pre_evictions, 0)
        self.assertIn('v6_single_launch', layer.implementation)

    def test_pool_step_before_selection_III(self):
        layer = self.run_full(POOL_STEP_PRE)
        self.assertEqual(layer.pre_evictions, 300)
        self.assertIn('v6_single_launch', layer.implementation)

    def test_rejects_other_intervals(self):
        with self.assertRaises(ValueError):
            make(HybridPoolStepLayer, POOL_STEP, 64, 1500, 1024, 137+3000)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
