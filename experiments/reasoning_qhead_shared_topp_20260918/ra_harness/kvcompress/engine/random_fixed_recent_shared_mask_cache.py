"""Full prompt + fixed reasoning budget, with budgeted recent protection.

The generation runner supplies a full-history storage bound. Sampling,
private RNG, temporary fused gather and append-only full KV are inherited.
"""
from kvcompress.engine.random_recent_shared_mask_cache import RandomRecentSharedMaskLayer


class RandomFixedRecentSharedMaskLayer(RandomRecentSharedMaskLayer):
    def __init__(self, *args, reasoning_budget, recent_window=64, random_keep_ratio=None, max_cache_length=None, **kwargs):
        if not isinstance(reasoning_budget, int) or reasoning_budget < 1:
            raise ValueError('reasoning_budget must be a positive integer')
        if not 0 <= recent_window <= reasoning_budget or random_keep_ratio is not None:
            raise ValueError('recent protection must fit in the fixed budget; ratio is not used')
        # Parent validates its ratio; selection below overrides the count rule.
        super().__init__(*args, random_keep_ratio=1.0, recent_window=recent_window, **kwargs)
        if max_cache_length is not None and (not isinstance(max_cache_length, int) or max_cache_length < 1):
            raise ValueError("max_cache_length must be a positive integer")
        self.max_cache_length = max_cache_length
        self.reasoning_budget = reasoning_budget
        self.random_keep_ratio = None

    def reasoning_keep_count(self, reasoning):
        return min(self.reasoning_budget, reasoning)

    def lazy_initialization(self, key_states, value_states=None):
        if self.max_cache_length is not None and key_states.shape[1] > self.max_cache_length:
            raise ValueError("prompt exceeds full-history storage bound")
        super().lazy_initialization(key_states, value_states)
        if self.max_cache_length is not None:
            # Reserve the known generation limit once; no geometric growth or
            # copying the full history during decoding. Unwritten slots are
            # never read: attention uses cache_seqlens and selection uses n.
            shape = (key_states.shape[0], self.max_cache_length, *key_states.shape[2:])
            self.k_cache = key_states.new_empty(shape)
            self.v_cache = key_states.new_empty(shape)
            if self._track_positions:
                self.positions = self.positions.new_zeros(
                    (shape[0], shape[2], self.max_cache_length))

    def prepare(self, key_states, value_states):
        required = self.cumulative_length + key_states.shape[1]
        if self.max_cache_length is not None and required > self.max_cache_length:
            raise ValueError("append exceeds full-history storage bound; no KV is evicted")
        return super().prepare(key_states, value_states)
