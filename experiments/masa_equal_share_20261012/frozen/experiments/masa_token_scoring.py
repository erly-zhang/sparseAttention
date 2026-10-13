"""MASA (arXiv:2610.10871v1, Eqs. 7--11) for AE3 token-mask sharing.

Each key is one candidate unit spanning four profile query rows. We retain
E2's query tile, probes, fixed token budget and group broadcast, using equal
query-probe weights by default.
Ranking is the weighted sum of *per-probe* error reductions, not a softmax
of averaged logits. The mean-query row is a synthetic profile row at the
tile end; this is profile-resolution MASA, not full-query dense MASA.
"""

from __future__ import annotations

import contextvars
import math

import torch

from experiments.jsd_grouped_sparse import _block_mean
from experiments.token_compacted_sparse import RepresentativeTokenFirstBlockSelector


def value_probe(values, *, norm_alpha=0.0, clip_percentile=None, eps=1e-8):
    """Build X from [batch, group, key, dim] V; never change kernel V.

    Clipping threshold is per batch/group over the full profile key region.
    If both controls are requested, apply clipping before soft balancing.
    """
    if not math.isfinite(norm_alpha) or not 0 <= norm_alpha <= 1:
        raise ValueError("MASA norm_alpha must be in [0, 1]")
    if clip_percentile is not None and (
        not math.isfinite(clip_percentile) or not 0 < clip_percentile <= 100
    ):
        raise ValueError("MASA clip_percentile must be in (0, 100]")
    x = values.float()
    if clip_percentile is not None:
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        threshold = torch.quantile(
            norms.squeeze(-1), clip_percentile / 100, dim=-1, keepdim=True
        ).unsqueeze(-1)
        x = x * (threshold / (norms + eps)).clamp(max=1)
    if norm_alpha:
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        x = x / (norms + eps).pow(norm_alpha)
    return x


def masa_pair_scores(attention, probe):
    """Eq. 8: A [B,G,Q,K], X [B,G,K,D] -> D [B,G,Q,K]."""
    a = attention.float()
    x = probe.float()
    output = a @ x
    alignment = output @ x.transpose(-1, -2)
    norm_squared = x.square().sum(-1).unsqueeze(-2)
    return 2 * a * alignment - a.square() * norm_squared


class RepresentativeMASATokenSelector(RepresentativeTokenFirstBlockSelector):
    """Replace E2 ranking only; superclass constructs and shares the CSR mask."""

    def __init__(self, group_config, *, masa_norm_alpha=0.0,
                 masa_clip_percentile=None, masa_query_chunk_size=16, **kwargs):
        kwargs.setdefault("probe_weights", (0.25, 0.25, 0.25, 0.25))
        super().__init__(group_config, **kwargs)
        # Validate controls even before a model forward.
        value_probe(torch.zeros(1, 1, 1, 1), norm_alpha=masa_norm_alpha,
                    clip_percentile=masa_clip_percentile)
        if masa_query_chunk_size <= 0:
            raise ValueError("MASA query chunk size must be positive")
        if (self.project_target_to_blocks or self.target_token_top_p is not None
                or self.block_probability_top_p is not None
                or self.measure_fixed_target_probability_mass
                or self.member_vertical_slash or self.residual_mode != "none"
                or self.query_score_mode != "four_probe_weighted"
                or self.force_sink_block or self.force_diagonal_block
                or self.force_sink_target_tokens or self.force_diagonal_target_tokens
                or self.final_token_budget is not None):
            raise ValueError("MASA token sharing requires the unprotected E2 fixed-TopK path")
        self.masa_norm_alpha = masa_norm_alpha
        self.masa_clip_percentile = masa_clip_percentile
        self.masa_query_chunk_size = masa_query_chunk_size
        self._profile_values = contextvars.ContextVar("masa_profile_values", default=None)

    def __call__(self, q, k, v, block_size, gamma, min_budget, max_budget,
                 tau=0, gqa_interleave=False):
        layer = self.current_layer.get()
        if layer is None:
            raise RuntimeError("MASA selector requires a layer context")
        reps = torch.tensor([int(g["representative"]) for g in self.layers[str(layer)]],
                            device=q.device, dtype=torch.long)
        if q.shape[2] % k.shape[2] or v.shape[2] != k.shape[2]:
            raise ValueError("Invalid GQA Q/K/V head counts")
        kv_indices = reps % k.shape[2] if gqa_interleave else reps // (q.shape[2] // k.shape[2])
        token = self._profile_values.set(v.index_select(2, kv_indices))
        try:
            return super().__call__(q, k, v, block_size, gamma, min_budget,
                                    max_budget, tau, gqa_interleave)
        finally:
            self._profile_values.reset(token)

    def _score_representative_keys(self, representative_q, representative_k, *, block_size):
        values = self._profile_values.get()
        if values is None:
            raise RuntimeError("MASA scoring requires matching representative V")
        batch, seq_len, groups, dim = representative_q.shape
        mean_q = _block_mean(representative_q, block_size).float()
        tiles = mean_q.shape[1]
        starts = torch.arange(tiles, device=representative_q.device) * block_size
        ends = (starts + block_size).clamp(max=seq_len)
        lengths = ends - starts
        positions = starts[:, None] + torch.stack(
            ((lengths - 1) // 3, 2 * (lengths - 1) // 3, lengths - 1), dim=-1
        )
        keys = representative_k.permute(0, 2, 1, 3).float()
        x = value_probe(values.permute(0, 2, 1, 3),
                        norm_alpha=self.masa_norm_alpha,
                        clip_percentile=self.masa_clip_percentile)
        key_ids = torch.arange(seq_len, device=keys.device)
        scores = torch.empty(batch, groups, tiles, seq_len, device=keys.device,
                             dtype=torch.float32)
        for start in range(0, tiles, self.masa_query_chunk_size):
            end = min(start + self.masa_query_chunk_size, tiles)
            total = torch.zeros(batch, groups, end - start, seq_len,
                                device=keys.device, dtype=torch.float32)
            for p, weight in enumerate(self.probe_weights):
                if weight == 0:
                    continue
                if p < 3:
                    query = representative_q.index_select(1, positions[start:end, p]).float()
                    causal_end = positions[start:end, p] + 1
                else:
                    query = mean_q[:, start:end]
                    causal_end = ends[start:end]
                logits = (query.permute(0, 2, 1, 3) @ keys.transpose(-1, -2)) / math.sqrt(dim)
                logits.masked_fill_(key_ids[None, :] >= causal_end[:, None], float("-inf"))
                attention = torch.softmax(logits, dim=-1)
                total.add_(masa_pair_scores(attention, x), alpha=weight)
            scores[:, :, start:end] = total
        scores.masked_fill_(key_ids[None, :] >= ends[:, None], float("-inf"))
        return scores, starts, ends, lengths
