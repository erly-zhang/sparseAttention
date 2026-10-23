"""Paper baselines (SnapKV, R-KV, VaSE, TriAttention) on the unmodified upstream engine.

The upstream Random-Attention engine (commit 64db9688, byte-identical vendored copy)
implements every baseline of the paper as an eviction mode; we call its trigger,
scoring, selection, sort and compaction verbatim. The adapters are the same as for
official random_pp: a private RNG stream around each eviction (VaSE samples, so the
generation RNG must not move), logical attention-pair counters, and, for TriAttention,
the per-generation cross-layer wiring the upstream EvictCache performs.

Paper configurations (scripts/run_cell.sh, Appendix F):
  SnapKV        attn                    --smooth
  R-KV          attn_rkv  rkv_lambda=0.5 --smooth
  VaSE          range_sink_sample_attn  n_large=K/4 --smooth
  TriAttention  triattn_ph  n_large=256, TRIATTN_STATS=<model stats>, TRIATTN_FIX_MEMO=1
All: total budget K including the residual (recent) window r=64, eviction every 64 steps.
"""
import sys

import kvcompress.engine.official_random_attention.cache_utils as upstream
from kvcompress.engine.official_random_pp_cache import OfficialRandomPPLayer

MODES = {
    'official_snapkv': 'attn',
    'official_rkv': 'attn_rkv',
    'official_vase': 'range_sink_sample_attn',
    'official_triattn': 'triattn_ph',
}


class OfficialBaselineLayer(OfficialRandomPPLayer):
    """Same adapters as OfficialRandomPPLayer; only the upstream mode differs."""

    def __init__(self, layer_idx, token_budget, residual_length, eviction_mode,
                 rkv_lambda, smooth, verbose, sample_id=None, mask_seed=42,
                 n_large=-1, temperature=1.0, top_p=.95, query_window=64,
                 eviction_interval=1, rep_seed=0, **kwargs):
        if eviction_mode not in MODES or token_budget <= residual_length or residual_length <= 0:
            raise ValueError(f'invalid official baseline configuration: {eviction_mode}')
        upstream.EvictLayer.__init__(self, layer_idx, token_budget, residual_length, MODES[eviction_mode],
                                     rkv_lambda, smooth, verbose, n_large=n_large, temperature=temperature, **kwargs)
        self.official_mode = eviction_mode
        self.sample_id = sample_id
        self.mask_seed = mask_seed
        self.query_window = None
        self.selected_pairs = self.causal_pairs = self.prefill_pairs = self.selection_events = 0
        self.max_visible_tokens = self.max_selected_tokens = 0
        self._cached_tokens = 0
        self._mask_gen = None


def wire_cache(eviction_mode, layers):
    """Upstream EvictCache per-generation state, for the vendored module."""
    if MODES[eviction_mode] == 'triattn_ph':
        for layer in layers:
            layer._sibling_layers = layers
    # The upstream cross-layer keep-set memo lives in the vendored module.
    sys.modules[upstream.__name__]._triattn_ph_shared = None
