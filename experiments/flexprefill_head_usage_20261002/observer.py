"""Read the official selector's local decision at return without changing math."""
import math
import sys


class HeadObserver:
    def __init__(self, flex_module, baseline_module, heads=32):
        self.flex = flex_module
        self.baselines = baseline_module
        self.heads = heads
        self.records = {}
        self.layer = None
        self.original_selector = flex_module.get_active_blocks
        self.original_baseline = baseline_module.baseline_attention

    def capture(self, values):
        mask = values['block_sparse_mask'].detach().cpu().tolist()
        divergence = values['kl_div'].detach().float().cpu().tolist()
        assert len(mask) == len(divergence) == 1
        mask, divergence = mask[0], divergence[0]
        assert len(mask) == len(divergence) == self.heads
        assert self.layer is not None and self.layer not in self.records
        qa = [i for i, selected in enumerate(mask) if selected]
        vs = [i for i, selected in enumerate(mask) if not selected]
        self.records[self.layer] = {
            'vertical_slash_heads': vs, 'vertical_slash_count': len(vs),
            'query_aware_heads': qa, 'query_aware_count': len(qa),
            'sqrt_js_divergence': [float(x) if math.isfinite(x) else None for x in divergence],
            'nonfinite_divergence_heads': [i for i, x in enumerate(divergence) if not math.isfinite(x)],
            'tau': float(values['tau']), 'input_tokens': int(values['seq_len'])}

    def __enter__(self):
        def selector(*args, **kwargs):
            previous = sys.getprofile()
            def profile(frame, event, arg):
                if event == 'return' and frame.f_code is self.original_selector.__code__:
                    self.capture(frame.f_locals)
            sys.setprofile(profile)
            try:
                return self.original_selector(*args, **kwargs)
            finally:
                sys.setprofile(previous)

        def baseline(q, k, v, method, layer, options):
            assert method == 'flexprefill' and self.layer is None
            self.layer = layer
            try:
                result = self.original_baseline(q, k, v, method, layer, options)
                assert layer in self.records, 'Unexpected dense fallback or missing selector observation'
                return result
            finally:
                self.layer = None

        self.flex.get_active_blocks = selector
        self.baselines.baseline_attention = baseline
        return self

    def __exit__(self, *args):
        self.flex.get_active_blocks = self.original_selector
        self.baselines.baseline_attention = self.original_baseline

