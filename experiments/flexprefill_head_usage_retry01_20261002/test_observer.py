from types import SimpleNamespace
import unittest
from observer import HeadObserver


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, 80000, 4, 128)
    def __lt__(self, threshold):
        return Tensor([[value < threshold for value in row] for row in self.values])
    def detach(self):
        return self
    def cpu(self):
        return self
    def float(self):
        return self
    def tolist(self):
        return self.values


class ObserverTests(unittest.TestCase):
    def make_modules(self, calls=2):
        flex = SimpleNamespace(square_root_js_divergence=lambda *args: Tensor([[.01, .4, .08, float('nan')]]),
                               score_cover_idx=lambda *args: args[0])
        def baseline(q, k, v, method, layer, options):
            flex.square_root_js_divergence(None, None)
            for _ in range(calls):
                flex.score_cover_idx(object(), .95)
            return q, {'layer': layer}
        return flex, SimpleNamespace(baseline_attention=baseline)

    def test_observes_exact_threshold_and_confirms_actual_branch_calls(self):
        flex, baselines = self.make_modules()
        originals = flex.square_root_js_divergence, flex.score_cover_idx, baselines.baseline_attention
        q = Tensor([[1]])
        with HeadObserver(flex, baselines, heads=4) as observer:
            result, stats = baselines.baseline_attention(q, None, None, 'flexprefill', 17, {'flexprefill': {'tau': .1}})
            self.assertIs(result, q)
            record = observer.records[17]
            self.assertEqual(record['query_aware_heads'], [0, 2])
            self.assertEqual(record['vertical_slash_heads'], [1, 3])
            self.assertEqual(record['actual_query_aware_branch_calls'], 2)
            self.assertEqual(record['nonfinite_divergence_heads'], [3])
        self.assertEqual(originals, (flex.square_root_js_divergence, flex.score_cover_idx, baselines.baseline_attention))

    def test_rejects_mismatch_with_actual_branch_and_restores_on_error(self):
        flex, baselines = self.make_modules(calls=1)
        original = baselines.baseline_attention
        with self.assertRaises(AssertionError):
            with HeadObserver(flex, baselines, heads=4):
                baselines.baseline_attention(Tensor([[1]]), None, None, 'flexprefill', 0, {'flexprefill': {'tau': .1}})
        self.assertIs(baselines.baseline_attention, original)


if __name__ == '__main__':
    unittest.main()
