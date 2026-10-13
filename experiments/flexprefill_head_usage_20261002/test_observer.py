import sys
from types import SimpleNamespace
import unittest
from observer import HeadObserver


class Tensor:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def cpu(self):
        return self
    def float(self):
        return self
    def tolist(self):
        return self.values


class ObserverTests(unittest.TestCase):
    def make_modules(self):
        def selector(q, k, v):
            tau, seq_len = .1, 80000
            block_sparse_mask = Tensor([[True, False, True, False]])
            kl_div = Tensor([[.01, .4, .08, float('nan')]])
            return q
        flex = SimpleNamespace(get_active_blocks=selector)
        def baseline(q, k, v, method, layer, options):
            return flex.get_active_blocks(q, k, v), {'layer': layer}
        return flex, SimpleNamespace(baseline_attention=baseline)

    def test_observes_actual_branch_preserves_output_and_restores_functions(self):
        flex, baselines = self.make_modules()
        original_selector, original_baseline = flex.get_active_blocks, baselines.baseline_attention
        profiler = sys.getprofile()
        output = object()
        with HeadObserver(flex, baselines, heads=4) as observer:
            result, stats = baselines.baseline_attention(output, None, None, 'flexprefill', 17, {})
            self.assertIs(result, output)
            self.assertIs(sys.getprofile(), profiler)
            row = observer.records[17]
            self.assertEqual(row['query_aware_heads'], [0, 2])
            self.assertEqual(row['vertical_slash_heads'], [1, 3])
            self.assertEqual(row['nonfinite_divergence_heads'], [3])
            self.assertIsNone(row['sqrt_js_divergence'][3])
        self.assertIs(flex.get_active_blocks, original_selector)
        self.assertIs(baselines.baseline_attention, original_baseline)

    def test_restores_on_failure_and_rejects_dense_fallback(self):
        flex, baselines = self.make_modules()
        baselines.baseline_attention = lambda *args: (object(), {})
        original_selector, original_baseline = flex.get_active_blocks, baselines.baseline_attention
        with self.assertRaises(AssertionError):
            with HeadObserver(flex, baselines, heads=4):
                baselines.baseline_attention(None, None, None, 'flexprefill', 1, {})
        self.assertIs(flex.get_active_blocks, original_selector)
        self.assertIs(baselines.baseline_attention, original_baseline)


if __name__ == '__main__':
    unittest.main()
