"""Observe returned JS values and actual query-aware branch calls, without tracing."""
import math


class HeadObserver:
    def __init__(self, flex_module, baseline_module, heads=32):
        self.flex, self.baselines, self.heads = flex_module, baseline_module, heads
        self.records, self.layer, self.divergence, self.query_aware_calls = {}, None, None, 0
        self.original_js = flex_module.square_root_js_divergence
        self.original_cover = flex_module.score_cover_idx
        self.original_baseline = baseline_module.baseline_attention

    def __enter__(self):
        def divergence(*args, **kwargs):
            result = self.original_js(*args, **kwargs)
            assert self.layer is not None and self.divergence is None
            # Keep only the 32 values; perform no tensor operations inside selection.
            self.divergence = result
            return result

        def cover(*args, **kwargs):
            self.query_aware_calls += 1
            return self.original_cover(*args, **kwargs)

        def baseline(q, k, v, method, layer, options):
            assert method == 'flexprefill' and self.layer is None
            self.layer, self.divergence, self.query_aware_calls = layer, None, 0
            try:
                result = self.original_baseline(q, k, v, method, layer, options)
                assert self.divergence is not None, 'Unexpected dense fallback'
                tau = options[method]['tau']
                # Identical tensor comparison to the official branch, after its attention output.
                mask = (self.divergence < tau).detach().cpu().tolist()[0]
                values = self.divergence.detach().float().cpu().tolist()[0]
                assert len(mask) == len(values) == self.heads
                qa = [head for head, selected in enumerate(mask) if selected]
                vs = [head for head, selected in enumerate(mask) if not selected]
                assert len(qa) == self.query_aware_calls, 'Mismatch with actual query-aware branch calls'
                self.records[layer] = {'vertical_slash_heads': vs, 'vertical_slash_count': len(vs),
                    'query_aware_heads': qa, 'query_aware_count': len(qa),
                    'sqrt_js_divergence': [float(x) if math.isfinite(x) else None for x in values],
                    'nonfinite_divergence_heads': [i for i, x in enumerate(values) if not math.isfinite(x)],
                    'actual_query_aware_branch_calls': self.query_aware_calls,
                    'tau': tau, 'input_tokens': int(q.shape[1])}
                return result
            finally:
                self.layer, self.divergence = None, None

        self.flex.square_root_js_divergence = divergence
        self.flex.score_cover_idx = cover
        self.baselines.baseline_attention = baseline
        return self

    def __exit__(self, *args):
        self.flex.square_root_js_divergence = self.original_js
        self.flex.score_cover_idx = self.original_cover
        self.baselines.baseline_attention = self.original_baseline
