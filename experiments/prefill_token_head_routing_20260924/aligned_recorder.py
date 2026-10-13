"""Keep canonical primary metrics; flush routing JSON outside its timer.
Static alignment revision: not runtime validated.
"""
from functools import wraps


class PairStats:
    def __init__(self):
        self.selected_token_pairs = 0
        self.causal_token_pairs = 0

    def snapshot(self):
        return dict(selected_token_pairs=self.selected_token_pairs,
                    causal_token_pairs=self.causal_token_pairs)


def make_recorder(base, bridge_provider):
    class AlignedRecorder(base):
        def install(self):
            if self._installed:
                return
            bridge = bridge_provider()
            bridge.rows.clear()
            bridge.pending_row = None  # Canonical warmup must not enter results.
            super().install()
            timed_generate = self.model.generate

            @wraps(timed_generate)
            def generate(*args, **kwargs):
                result = timed_generate(*args, **kwargs)
                bridge.flush_pending()
                return result
            self.model.generate = generate
    return AlignedRecorder
