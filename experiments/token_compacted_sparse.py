"""Representative-head block sparse attention with token compaction.

The outer selector keeps the existing 128-token block-top-p policy. Sink and
diagonal key blocks can be protected in full, while tokens in other selected
blocks are ranked by the representative head and compacted into a shared index
list. The Triton kernel gathers only those K/V tokens before QK and PV.
"""

from __future__ import annotations

import contextvars
import math
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import triton
import triton.language as tl

from experiments.jsd_grouped_sparse import _block_mean, validate_group_config


@dataclass
class TokenCompactedIndex:
    """CSR-like token indices shared by every head in an offline group."""

    row_starts: torch.Tensor
    row_ends: torch.Tensor
    token_indices: torch.Tensor
    head_to_group: torch.Tensor
    num_groups: int
    num_query_blocks: int
    query_block_size: int


class TokenCompactedSelectorStats:
    """Accumulate macro-block and effective token-pair sparsity."""

    def __init__(self) -> None:
        self.calls = 0
        self.selected_blocks = 0
        self.causal_blocks = 0
        self.selected_token_pairs = 0
        self.causal_token_pairs = 0
        self.compacted_key_tokens = 0
        self.candidate_key_tokens = 0
        self.target_mask_tokens = 0
        self.covered_target_tokens = 0
        self.chosen_block_size_rows: Dict[int, int] = {}
        self.selection_latency_ms = 0.0
        self.kernel_latency_ms = 0.0
        self._selection_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._kernel_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def record(
        self,
        *,
        selected_blocks: int,
        causal_blocks: int,
        selected_token_pairs: int,
        causal_token_pairs: int,
        compacted_key_tokens: int,
        candidate_key_tokens: int,
        timing_events: tuple[torch.cuda.Event, torch.cuda.Event],
        target_mask_tokens: int = 0,
        covered_target_tokens: int = 0,
        chosen_block_size_rows: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.calls += 1
        self.selected_blocks += selected_blocks
        self.causal_blocks += causal_blocks
        self.selected_token_pairs += selected_token_pairs
        self.causal_token_pairs += causal_token_pairs
        self.compacted_key_tokens += compacted_key_tokens
        self.candidate_key_tokens += candidate_key_tokens
        self.target_mask_tokens += target_mask_tokens
        self.covered_target_tokens += covered_target_tokens
        if chosen_block_size_rows:
            for size, count in chosen_block_size_rows.items():
                self.chosen_block_size_rows[int(size)] = (
                    self.chosen_block_size_rows.get(int(size), 0) + int(count)
                )
        self._selection_events.append(timing_events)

    def record_kernel_events(
        self, start: torch.cuda.Event, end: torch.cuda.Event
    ) -> None:
        self._kernel_events.append((start, end))

    @staticmethod
    def _resolve_events(
        events: List[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> float:
        elapsed = 0.0
        for start, end in events:
            end.synchronize()
            elapsed += start.elapsed_time(end)
        events.clear()
        return elapsed

    def snapshot(self) -> Dict[str, Any]:
        self.selection_latency_ms += self._resolve_events(self._selection_events)
        self.kernel_latency_ms += self._resolve_events(self._kernel_events)
        block_keep = (
            self.selected_blocks / self.causal_blocks
            if self.causal_blocks
            else float("nan")
        )
        pair_keep = (
            self.selected_token_pairs / self.causal_token_pairs
            if self.causal_token_pairs
            else float("nan")
        )
        compacted_keep = (
            self.compacted_key_tokens / self.candidate_key_tokens
            if self.candidate_key_tokens
            else float("nan")
        )
        target_coverage = (
            self.covered_target_tokens / self.target_mask_tokens
            if self.target_mask_tokens
            else float("nan")
        )
        return {
            "calls": self.calls,
            "selected_blocks": self.selected_blocks,
            "causal_blocks": self.causal_blocks,
            "mean_block_keep_ratio": block_keep,
            "mean_block_sparsity": 1.0 - block_keep,
            "selected_token_pairs": self.selected_token_pairs,
            "causal_token_pairs": self.causal_token_pairs,
            "mean_token_pair_keep_ratio": pair_keep,
            "mean_token_pair_sparsity": 1.0 - pair_keep,
            "compacted_key_tokens": self.compacted_key_tokens,
            "candidate_key_tokens": self.candidate_key_tokens,
            "mean_intra_block_token_keep_ratio": compacted_keep,
            "target_mask_tokens": self.target_mask_tokens,
            "covered_target_tokens": self.covered_target_tokens,
            "mean_target_mask_coverage_ratio": target_coverage,
            "chosen_block_size_rows": dict(self.chosen_block_size_rows),
            "selection_latency_sec": self.selection_latency_ms / 1000.0,
            "kernel_latency_sec": self.kernel_latency_ms / 1000.0,
        }


class RepresentativeTokenCompactedSelector:
    """Select macro blocks, then compact arbitrary tokens in ordinary blocks."""

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        token_top_p: float,
        min_tokens_per_selected_block: int,
        force_sink_block: bool,
        force_diagonal_block: bool,
    ) -> None:
        if not 0.0 < token_top_p <= 1.0:
            raise ValueError("token_top_p must be in (0, 1]")
        if min_tokens_per_selected_block <= 0:
            raise ValueError("min_tokens_per_selected_block must be positive")
        self.layers = group_config["layers"]
        self.token_top_p = float(token_top_p)
        self.min_tokens_per_selected_block = int(min_tokens_per_selected_block)
        self.force_sink_block = bool(force_sink_block)
        self.force_diagonal_block = bool(force_diagonal_block)
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("token_compacted_sparse_layer", default=None)
        )
        self.stats = TokenCompactedSelectorStats()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        gamma: float,
        min_budget: int,
        max_budget: int,
        tau: float = 0,
        gqa_interleave: bool = False,
    ) -> TokenCompactedIndex:
        del v, tau
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_start.record()
        layer_idx = self.current_layer.get()
        if layer_idx is None:
            raise RuntimeError(
                "Token-compacted selector was called without layer context"
            )
        if block_size != 128:
            raise ValueError(
                "Token-compacted attention currently requires block_size=128"
            )

        groups = self.layers[str(layer_idx)]
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        num_share_q_heads = num_q_heads // num_kv_heads

        representatives = torch.tensor(
            [int(group["representative"]) for group in groups],
            dtype=torch.long,
            device=q.device,
        )
        if gqa_interleave:
            kv_indices = representatives % num_kv_heads
        else:
            kv_indices = representatives // num_share_q_heads
        representative_q = q.index_select(2, representatives)
        representative_k = k.index_select(2, kv_indices)

        pooled_q = _block_mean(representative_q, block_size)
        pooled_k = _block_mean(representative_k, block_size)
        macro_scores = torch.einsum("bqhd,bkhd->bhqk", pooled_q, pooled_k) / math.sqrt(
            head_dim
        )
        num_blocks = macro_scores.shape[-1]
        causal_blocks = torch.tril(
            torch.ones(
                (num_blocks, num_blocks),
                dtype=torch.bool,
                device=q.device,
            )
        )
        macro_probabilities = torch.softmax(
            macro_scores.masked_fill(~causal_blocks[None, None, :, :], float("-inf")),
            dim=-1,
            dtype=torch.float32,
        )

        min_blocks = min(num_blocks, max(1, int(min_budget)))
        max_blocks = min(num_blocks, max(1, int(max_budget)))
        sorted_macro_values, sorted_macro_indices = torch.sort(
            macro_probabilities, dim=-1, descending=True
        )
        macro_cumulative = torch.cumsum(sorted_macro_values, dim=-1)
        macro_keep_counts = (macro_cumulative < gamma).sum(dim=-1) + 1
        available_blocks = torch.arange(1, num_blocks + 1, device=q.device).view(
            1, 1, num_blocks
        )
        macro_keep_counts = torch.maximum(
            macro_keep_counts,
            macro_keep_counts.new_full((), min_blocks),
        )
        macro_keep_counts = torch.minimum(
            macro_keep_counts,
            macro_keep_counts.new_full((), max_blocks),
        )
        macro_keep_counts = torch.minimum(macro_keep_counts, available_blocks)
        macro_ranks = torch.arange(num_blocks, device=q.device).view(
            1, 1, 1, num_blocks
        )
        keep_sorted_macro = macro_ranks < macro_keep_counts.unsqueeze(-1)
        macro_selected = torch.zeros_like(keep_sorted_macro)
        macro_selected.scatter_(-1, sorted_macro_indices, keep_sorted_macro)
        macro_selected &= causal_blocks[None, None, :, :]
        if self.force_sink_block:
            macro_selected[:, :, :, 0] = True
        if self.force_diagonal_block:
            diagonal = torch.arange(num_blocks, device=q.device)
            macro_selected[:, :, diagonal, diagonal] = True

        # A single pooled representative query scores all 128 individual tokens
        # in every key block. This avoids computing a second 128x128 attention.
        token_scores = torch.einsum(
            "bqhd,bthd->bhqt", pooled_q, representative_k
        ) / math.sqrt(head_dim)
        padded_seq_len = num_blocks * block_size
        if seq_len < padded_seq_len:
            token_scores = torch.nn.functional.pad(
                token_scores,
                (0, padded_seq_len - seq_len),
                value=float("-inf"),
            )
        token_scores = token_scores.view(
            batch_size,
            len(groups),
            num_blocks,
            num_blocks,
            block_size,
        )
        valid_token = (torch.arange(padded_seq_len, device=q.device) < seq_len).view(
            num_blocks, block_size
        )
        token_scores = token_scores.masked_fill(
            ~valid_token[None, None, None, :, :], float("-inf")
        )
        token_probabilities = torch.softmax(token_scores, dim=-1, dtype=torch.float32)
        sorted_values, sorted_indices = torch.sort(
            token_probabilities, dim=-1, descending=True
        )
        cumulative = torch.cumsum(sorted_values, dim=-1)
        keep_counts = ((cumulative < self.token_top_p).sum(dim=-1) + 1).clamp(
            min=min(self.min_tokens_per_selected_block, block_size),
            max=block_size,
        )
        ranks = torch.arange(block_size, device=q.device).view(1, 1, 1, 1, block_size)
        keep_sorted = ranks < keep_counts.unsqueeze(-1)
        token_keep = torch.zeros_like(keep_sorted)
        token_keep.scatter_(-1, sorted_indices, keep_sorted)
        token_keep &= valid_token[None, None, None, :, :]
        token_keep &= macro_selected.unsqueeze(-1)

        # Protected structural blocks retain every valid token. Token-level
        # causality inside the diagonal block is applied by the Triton kernel.
        if self.force_sink_block:
            token_keep[:, :, :, 0, :] = valid_token[0]
        if self.force_diagonal_block:
            diagonal = torch.arange(num_blocks, device=q.device)
            token_keep[:, :, diagonal, diagonal, :] = valid_token[diagonal]

        selected_mask = token_keep.flatten(start_dim=-2)
        selected_mask = selected_mask[..., :seq_len]
        rows = selected_mask.reshape(-1, seq_len)
        row_counts = rows.sum(dim=-1, dtype=torch.int32)
        row_ends_flat = torch.cumsum(row_counts, dim=0, dtype=torch.int32)
        row_starts_flat = row_ends_flat - row_counts
        token_indices = torch.nonzero(rows, as_tuple=False)[:, 1].to(torch.int32)
        row_starts = row_starts_flat.view(batch_size, len(groups), num_blocks)
        row_ends = row_ends_flat.view(batch_size, len(groups), num_blocks)

        head_to_group = torch.empty((num_q_heads,), dtype=torch.int32, device=q.device)
        for group_idx, group in enumerate(groups):
            member_indices = torch.tensor(
                [int(member) for member in group["members"]],
                dtype=torch.long,
                device=q.device,
            )
            head_to_group[member_indices] = group_idx

        group_sizes = torch.tensor(
            [len(group["members"]) for group in groups],
            dtype=torch.int64,
            device=q.device,
        )
        selected_blocks = int(
            (macro_selected.sum(dim=(-1, -2), dtype=torch.int64) * group_sizes[None, :])
            .sum()
            .item()
        )
        total_causal_blocks = (
            batch_size * num_q_heads * num_blocks * (num_blocks + 1) // 2
        )

        # Exact effective token-pair count after applying token-level causality.
        prefix = torch.cumsum(selected_mask, dim=-1, dtype=torch.int64)
        query_positions = torch.arange(padded_seq_len, device=q.device).view(
            num_blocks, block_size
        )
        valid_queries = query_positions < seq_len
        gather_positions = query_positions.clamp(max=seq_len - 1)
        gather_positions = gather_positions.view(1, 1, num_blocks, block_size).expand(
            batch_size, len(groups), -1, -1
        )
        selected_per_query = torch.gather(prefix, dim=-1, index=gather_positions)
        selected_per_query *= valid_queries[None, None, :, :].to(
            selected_per_query.dtype
        )
        selected_pairs_per_group = selected_per_query.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        selected_token_pairs = int(
            (selected_pairs_per_group * group_sizes[None, :]).sum().item()
        )
        total_causal_token_pairs = (
            batch_size * num_q_heads * seq_len * (seq_len + 1) // 2
        )

        protected = torch.zeros_like(token_keep)
        if self.force_sink_block:
            protected[:, :, :, 0, :] = valid_token[0]
        if self.force_diagonal_block:
            diagonal = torch.arange(num_blocks, device=q.device)
            protected[:, :, diagonal, diagonal, :] = valid_token[diagonal]
        ordinary_selected = macro_selected.unsqueeze(-1) & ~protected
        ordinary_selected &= valid_token[None, None, None, :, :]
        candidate_tokens_per_group = ordinary_selected.sum(
            dim=(-1, -2, -3), dtype=torch.int64
        )
        compacted_tokens_per_group = (token_keep & ordinary_selected).sum(
            dim=(-1, -2, -3), dtype=torch.int64
        )
        candidate_key_tokens = int(
            (candidate_tokens_per_group * group_sizes[None, :]).sum().item()
        )
        compacted_key_tokens = int(
            (compacted_tokens_per_group * group_sizes[None, :]).sum().item()
        )
        self.stats.record(
            selected_blocks=selected_blocks,
            causal_blocks=total_causal_blocks,
            selected_token_pairs=selected_token_pairs,
            causal_token_pairs=total_causal_token_pairs,
            compacted_key_tokens=compacted_key_tokens,
            candidate_key_tokens=candidate_key_tokens,
            timing_events=(timing_start, timing_end),
        )
        timing_end.record()

        return TokenCompactedIndex(
            row_starts=row_starts.contiguous(),
            row_ends=row_ends.contiguous(),
            token_indices=token_indices.contiguous(),
            head_to_group=head_to_group.contiguous(),
            num_groups=len(groups),
            num_query_blocks=num_blocks,
            query_block_size=block_size,
        )


def _allocate_block_token_budgets(
    block_mass: torch.Tensor,
    selected_blocks: torch.Tensor,
    protected_blocks: torch.Tensor,
    block_capacities: torch.Tensor,
    total_token_budget: int,
) -> torch.Tensor:
    """Allocate a fixed row budget proportionally, respecting block capacity."""

    capacities = block_capacities.view(1, 1, 1, -1).expand_as(selected_blocks)
    protected_blocks = protected_blocks & selected_blocks
    ordinary_blocks = selected_blocks & ~protected_blocks
    allocation = torch.where(
        protected_blocks, capacities, torch.zeros_like(capacities)
    )
    # Every selected ordinary block receives one token before proportional
    # allocation. With at most 256 blocks this is always below the 8192 budget.
    allocation += ordinary_blocks.to(allocation.dtype)
    available_tokens = (selected_blocks.to(capacities.dtype) * capacities).sum(-1)
    target = torch.minimum(
        available_tokens,
        available_tokens.new_full((), total_token_budget),
    )
    remaining = (target - allocation.sum(-1)).clamp_min(0)
    mass = block_mass.masked_fill(~ordinary_blocks, 0.0).clamp_min(0.0)

    # Capped proportional apportionment. Bulk floor allocation handles most
    # tokens; a largest-priority pass resolves integer remainders each round.
    num_blocks = selected_blocks.shape[-1]
    rank_template = torch.arange(num_blocks, device=selected_blocks.device).view(
        1, 1, 1, num_blocks
    )
    for _ in range(16):
        eligible = ordinary_blocks & (allocation < capacities)
        eligible &= remaining.unsqueeze(-1) > 0
        weights = mass.masked_fill(~eligible, 0.0)
        weight_sum = weights.sum(-1, keepdim=True)
        uniform = eligible.to(weights.dtype)
        weights = torch.where(weight_sum > 0, weights, uniform)
        weight_sum = weights.sum(-1, keepdim=True).clamp_min(1e-12)
        proposed = torch.floor(
            weights / weight_sum * remaining.unsqueeze(-1)
        ).to(allocation.dtype)
        proposed = torch.minimum(proposed, capacities - allocation)
        proposed = torch.where(eligible, proposed, torch.zeros_like(proposed))
        allocation += proposed
        remaining = (target - allocation.sum(-1)).clamp_min(0)

        eligible = ordinary_blocks & (allocation < capacities)
        eligible &= remaining.unsqueeze(-1) > 0
        priority = torch.where(
            eligible,
            mass / (allocation.to(mass.dtype) + 1.0),
            mass.new_full((), float("-inf")),
        )
        order = torch.argsort(priority, dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, rank_template.expand_as(order))
        one_count = torch.minimum(
            remaining,
            eligible.sum(-1, dtype=remaining.dtype),
        )
        add_one = eligible & (ranks < one_count.unsqueeze(-1))
        allocation += add_one.to(allocation.dtype)
        remaining = (target - allocation.sum(-1)).clamp_min(0)

    return allocation


class RepresentativeHISAStyleSelector:
    """Two-stage fixed-budget selection inspired by HISA.

    Stage 1 selects a fixed number of causal key blocks from pooled block
    scores. Stage 2 merges their tokens and applies one global top-k using
    three representative query positions plus the block-mean query.
    """

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        candidate_block_count: int,
        final_token_budget: int,
        force_sink_block: bool,
        force_diagonal_block: bool,
        token_selection_mode: str = "global_topk_max",
        probe_weights: tuple[float, float, float, float] = (
            0.2,
            0.3,
            0.4,
            0.1,
        ),
    ) -> None:
        if candidate_block_count <= 0:
            raise ValueError("candidate_block_count must be positive")
        if final_token_budget <= 0:
            raise ValueError("final_token_budget must be positive")
        self.layers = group_config["layers"]
        self.candidate_block_count = int(candidate_block_count)
        self.final_token_budget = int(final_token_budget)
        self.force_sink_block = bool(force_sink_block)
        self.force_diagonal_block = bool(force_diagonal_block)
        if token_selection_mode not in {
            "global_topk_max",
            "block_mass_weighted_average",
        }:
            raise ValueError(f"Unknown HISA token selection mode: {token_selection_mode}")
        if len(probe_weights) != 4 or any(weight < 0 for weight in probe_weights):
            raise ValueError("probe_weights must contain four non-negative values")
        if not math.isclose(sum(probe_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probe_weights must sum to 1")
        self.token_selection_mode = token_selection_mode
        self.probe_weights = tuple(float(weight) for weight in probe_weights)
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("hisa_style_sparse_layer", default=None)
        )
        self.stats = TokenCompactedSelectorStats()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        gamma: float,
        min_budget: int,
        max_budget: int,
        tau: float = 0,
        gqa_interleave: bool = False,
    ) -> TokenCompactedIndex:
        del v, gamma, min_budget, max_budget, tau
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_start.record()
        layer_idx = self.current_layer.get()
        if layer_idx is None:
            raise RuntimeError("HISA-style selector was called without layer context")
        if block_size != 128:
            raise ValueError("HISA-style attention currently requires block_size=128")

        groups = self.layers[str(layer_idx)]
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        num_share_q_heads = num_q_heads // num_kv_heads

        representatives = torch.tensor(
            [int(group["representative"]) for group in groups],
            dtype=torch.long,
            device=q.device,
        )
        if gqa_interleave:
            kv_indices = representatives % num_kv_heads
        else:
            kv_indices = representatives // num_share_q_heads
        representative_q = q.index_select(2, representatives)
        representative_k = k.index_select(2, kv_indices)

        pooled_q = _block_mean(representative_q, block_size)
        pooled_k = _block_mean(representative_k, block_size)
        macro_scores = torch.einsum(
            "bqhd,bkhd->bhqk", pooled_q, pooled_k
        ) / math.sqrt(head_dim)
        num_blocks = macro_scores.shape[-1]
        causal_blocks = torch.tril(
            torch.ones(
                (num_blocks, num_blocks), dtype=torch.bool, device=q.device
            )
        )
        macro_scores.masked_fill_(
            ~causal_blocks[None, None, :, :], float("-inf")
        )
        macro_probabilities = None
        if self.token_selection_mode == "block_mass_weighted_average":
            macro_probabilities = torch.softmax(
                macro_scores, dim=-1, dtype=torch.float32
            )

        # Structural blocks are part of, rather than additions to, the fixed
        # candidate budget.
        if self.force_sink_block:
            macro_scores[:, :, :, 0] = float("inf")
        if self.force_diagonal_block:
            diagonal = torch.arange(num_blocks, device=q.device)
            macro_scores[:, :, diagonal, diagonal] = float("inf")
        top_m = min(self.candidate_block_count, num_blocks)
        _, top_block_indices = torch.topk(
            macro_scores, k=top_m, dim=-1, sorted=False
        )
        available_blocks = torch.arange(1, num_blocks + 1, device=q.device).view(
            1, 1, num_blocks, 1
        )
        block_ranks = torch.arange(top_m, device=q.device).view(1, 1, 1, top_m)
        keep_top_blocks = block_ranks < torch.minimum(
            available_blocks,
            available_blocks.new_full((), top_m),
        )
        macro_selected = torch.zeros_like(macro_scores, dtype=torch.bool)
        macro_selected.scatter_(
            -1, top_block_indices, keep_top_blocks.expand_as(top_block_indices)
        )
        macro_selected &= causal_blocks[None, None, :, :]

        padded_seq_len = num_blocks * block_size
        valid_token = (torch.arange(padded_seq_len, device=q.device) < seq_len).view(
            num_blocks, block_size
        )
        candidate_mask = (
            macro_selected.unsqueeze(-1)
            & valid_token[None, None, None, :, :]
        ).flatten(start_dim=-2)[..., :seq_len]

        # The representative head supplies three real query probes and one
        # mean-query probe.
        query_starts = torch.arange(num_blocks, device=q.device) * block_size
        query_lengths = (seq_len - query_starts).clamp(min=1, max=block_size)
        probe_positions = (
            query_starts[:, None]
            + torch.stack(
                (
                    (query_lengths - 1) // 3,
                    2 * (query_lengths - 1) // 3,
                    query_lengths - 1,
                ),
                dim=-1,
            )
        )
        mean_scores = torch.einsum(
            "bqhd,bthd->bhqt", pooled_q, representative_k
        ) / math.sqrt(head_dim)
        if self.token_selection_mode == "block_mass_weighted_average":
            token_scores = mean_scores * self.probe_weights[3]
        else:
            token_scores = mean_scores
        for probe_idx in range(3):
            probe_q = representative_q.index_select(
                1, probe_positions[:, probe_idx]
            )
            probe_scores = torch.einsum(
                "bqhd,bthd->bhqt", probe_q, representative_k
            ) / math.sqrt(head_dim)
            if self.token_selection_mode == "block_mass_weighted_average":
                token_scores.add_(probe_scores, alpha=self.probe_weights[probe_idx])
            else:
                torch.maximum(token_scores, probe_scores, out=token_scores)
            del probe_scores

        if self.token_selection_mode == "block_mass_weighted_average":
            if macro_probabilities is None:
                raise RuntimeError("Macro attention mass was not computed")
            protected_blocks = torch.zeros_like(macro_selected)
            if self.force_sink_block:
                protected_blocks[:, :, :, 0] = True
            if self.force_diagonal_block:
                diagonal = torch.arange(num_blocks, device=q.device)
                protected_blocks[:, :, diagonal, diagonal] = True
            protected_blocks &= macro_selected
            block_capacities = valid_token.sum(-1, dtype=torch.int64)
            allocations = _allocate_block_token_budgets(
                macro_probabilities,
                macro_selected,
                protected_blocks,
                block_capacities,
                self.final_token_budget,
            )
            if seq_len < padded_seq_len:
                token_scores = torch.nn.functional.pad(
                    token_scores,
                    (0, padded_seq_len - seq_len),
                    value=float("-inf"),
                )
            block_token_scores = token_scores.view(
                batch_size,
                len(groups),
                num_blocks,
                num_blocks,
                block_size,
            )
            block_token_scores.masked_fill_(
                ~valid_token[None, None, None, :, :], float("-inf")
            )
            sorted_token_indices = torch.argsort(
                block_token_scores, dim=-1, descending=True
            )
            token_ranks = torch.arange(block_size, device=q.device).view(
                1, 1, 1, 1, block_size
            )
            keep_sorted_tokens = token_ranks < allocations.unsqueeze(-1)
            token_keep = torch.zeros_like(keep_sorted_tokens)
            token_keep.scatter_(-1, sorted_token_indices, keep_sorted_tokens)
            token_keep &= macro_selected.unsqueeze(-1)
            token_keep &= valid_token[None, None, None, :, :]
            token_keep |= (
                protected_blocks.unsqueeze(-1)
                & valid_token[None, None, None, :, :]
            )
            selected_mask = token_keep.flatten(start_dim=-2)[..., :seq_len]
        else:
            token_scores.masked_fill_(~candidate_mask, float("-inf"))
            protected = torch.zeros_like(candidate_mask)
            if self.force_sink_block:
                protected[..., : min(block_size, seq_len)] = True
            if self.force_diagonal_block:
                diagonal_tokens = (
                    torch.arange(num_blocks, device=q.device)[:, None] * block_size
                    + torch.arange(block_size, device=q.device)[None, :]
                )
                diagonal_valid = diagonal_tokens < seq_len
                diagonal_tokens = diagonal_tokens.clamp(max=seq_len - 1)
                diagonal_tokens = diagonal_tokens.view(
                    1, 1, num_blocks, block_size
                ).expand(batch_size, len(groups), -1, -1)
                diagonal_valid = diagonal_valid.view(
                    1, 1, num_blocks, block_size
                ).expand(batch_size, len(groups), -1, -1)
                protected.scatter_(-1, diagonal_tokens, diagonal_valid)
            protected &= candidate_mask
            token_scores.masked_fill_(protected, float("inf"))

            top_k = min(self.final_token_budget, seq_len)
            _, top_token_indices = torch.topk(
                token_scores, k=top_k, dim=-1, sorted=False
            )
            available_candidate_tokens = candidate_mask.sum(
                dim=-1, dtype=torch.int64
            )
            keep_counts = torch.minimum(
                available_candidate_tokens,
                available_candidate_tokens.new_full((), top_k),
            )
            token_ranks = torch.arange(top_k, device=q.device).view(1, 1, 1, top_k)
            keep_top_tokens = token_ranks < keep_counts.unsqueeze(-1)
            selected_mask = torch.zeros_like(candidate_mask)
            selected_mask.scatter_(-1, top_token_indices, keep_top_tokens)
            selected_mask |= protected

        rows = selected_mask.reshape(-1, seq_len)
        row_counts = rows.sum(dim=-1, dtype=torch.int32)
        row_ends_flat = torch.cumsum(row_counts, dim=0, dtype=torch.int32)
        row_starts_flat = row_ends_flat - row_counts
        token_indices = torch.nonzero(rows, as_tuple=False)[:, 1].to(torch.int32)
        row_starts = row_starts_flat.view(batch_size, len(groups), num_blocks)
        row_ends = row_ends_flat.view(batch_size, len(groups), num_blocks)

        head_to_group = torch.empty((num_q_heads,), dtype=torch.int32, device=q.device)
        for group_idx, group in enumerate(groups):
            member_indices = torch.tensor(
                [int(member) for member in group["members"]],
                dtype=torch.long,
                device=q.device,
            )
            head_to_group[member_indices] = group_idx
        group_sizes = torch.tensor(
            [len(group["members"]) for group in groups],
            dtype=torch.int64,
            device=q.device,
        )

        selected_blocks = int(
            (macro_selected.sum(dim=(-1, -2), dtype=torch.int64) * group_sizes[None, :])
            .sum()
            .item()
        )
        total_causal_blocks = (
            batch_size * num_q_heads * num_blocks * (num_blocks + 1) // 2
        )
        prefix = torch.cumsum(selected_mask, dim=-1, dtype=torch.int64)
        query_positions = torch.arange(padded_seq_len, device=q.device).view(
            num_blocks, block_size
        )
        valid_queries = query_positions < seq_len
        gather_positions = query_positions.clamp(max=seq_len - 1)
        gather_positions = gather_positions.view(
            1, 1, num_blocks, block_size
        ).expand(batch_size, len(groups), -1, -1)
        selected_per_query = torch.gather(prefix, dim=-1, index=gather_positions)
        selected_per_query *= valid_queries[None, None, :, :].to(
            selected_per_query.dtype
        )
        selected_pairs_per_group = selected_per_query.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        selected_token_pairs = int(
            (selected_pairs_per_group * group_sizes[None, :]).sum().item()
        )
        total_causal_token_pairs = (
            batch_size * num_q_heads * seq_len * (seq_len + 1) // 2
        )
        candidate_tokens_per_group = candidate_mask.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        compacted_tokens_per_group = selected_mask.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        candidate_key_tokens = int(
            (candidate_tokens_per_group * group_sizes[None, :]).sum().item()
        )
        compacted_key_tokens = int(
            (compacted_tokens_per_group * group_sizes[None, :]).sum().item()
        )
        self.stats.record(
            selected_blocks=selected_blocks,
            causal_blocks=total_causal_blocks,
            selected_token_pairs=selected_token_pairs,
            causal_token_pairs=total_causal_token_pairs,
            compacted_key_tokens=compacted_key_tokens,
            candidate_key_tokens=candidate_key_tokens,
            timing_events=(timing_start, timing_end),
        )
        timing_end.record()

        return TokenCompactedIndex(
            row_starts=row_starts.contiguous(),
            row_ends=row_ends.contiguous(),
            token_indices=token_indices.contiguous(),
            head_to_group=head_to_group.contiguous(),
            num_groups=len(groups),
            num_query_blocks=num_blocks,
            query_block_size=block_size,
        )


def _cover_target_mask_with_key_blocks(
    target_mask: torch.Tensor,
    causal_tokens: torch.Tensor,
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    *,
    key_block_size: int,
    final_token_budget: int,
    minimum_block_coverage_ratio: float,
    force_sink_block: bool,
    force_diagonal_block: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project token targets to whole K blocks for one candidate size."""

    batch_size, num_groups, num_query_blocks, seq_len = target_mask.shape
    num_key_blocks = triton.cdiv(seq_len, key_block_size)
    padded_key_len = num_key_blocks * key_block_size
    padded_target_mask = torch.nn.functional.pad(
        target_mask, (0, padded_key_len - seq_len), value=False
    )
    target_by_block = padded_target_mask.view(
        batch_size,
        num_groups,
        num_query_blocks,
        num_key_blocks,
        key_block_size,
    )
    target_count = target_by_block.sum(-1, dtype=torch.int32)

    key_block_starts = torch.arange(
        num_key_blocks, device=target_mask.device
    ) * key_block_size
    key_block_ends = (key_block_starts + key_block_size).clamp(max=seq_len)
    causal_key_blocks = key_block_starts.view(1, num_key_blocks) < query_ends.view(
        num_query_blocks, 1
    )
    protected_blocks = torch.zeros(
        (batch_size, num_groups, num_query_blocks, num_key_blocks),
        dtype=torch.bool,
        device=target_mask.device,
    )
    if force_sink_block:
        protected_blocks[..., 0] = True
    if force_diagonal_block:
        local_overlap = (
            key_block_starts.view(1, num_key_blocks)
            < query_ends.view(num_query_blocks, 1)
        ) & (
            key_block_ends.view(1, num_key_blocks)
            > query_starts.view(num_query_blocks, 1)
        )
        protected_blocks |= local_overlap.view(
            1, 1, num_query_blocks, num_key_blocks
        )
    protected_blocks &= causal_key_blocks.view(
        1, 1, num_query_blocks, num_key_blocks
    )

    minimum_target_tokens = math.ceil(
        key_block_size * minimum_block_coverage_ratio
    )
    eligible_blocks = target_count >= minimum_target_tokens
    eligible_blocks &= causal_key_blocks.view(
        1, 1, num_query_blocks, num_key_blocks
    )
    eligible_blocks |= protected_blocks
    block_priority = target_count.to(torch.float32)
    block_priority.masked_fill_(~eligible_blocks, float("-inf"))
    block_priority.masked_fill_(protected_blocks, float("inf"))
    maximum_selected_blocks = max(1, final_token_budget // key_block_size)
    top_blocks = min(maximum_selected_blocks, num_key_blocks)
    top_values, top_indices = torch.topk(
        block_priority, k=top_blocks, dim=-1, sorted=False
    )
    keep_blocks = top_values > float("-inf")
    selected_key_blocks = torch.zeros_like(eligible_blocks)
    selected_key_blocks.scatter_(-1, top_indices, keep_blocks)
    selected_key_blocks &= causal_key_blocks.view(
        1, 1, num_query_blocks, num_key_blocks
    )

    valid_key_tokens = (
        torch.arange(padded_key_len, device=target_mask.device) < seq_len
    ).view(num_key_blocks, key_block_size)
    selected_mask = (
        selected_key_blocks.unsqueeze(-1)
        & valid_key_tokens.view(1, 1, 1, num_key_blocks, key_block_size)
    ).flatten(start_dim=-2)[..., :seq_len]
    selected_mask &= causal_tokens
    covered_target_tokens = (target_mask & selected_mask).sum(
        -1, dtype=torch.int64
    )
    selected_block_counts = selected_key_blocks.sum(-1, dtype=torch.int64)
    causal_block_counts = causal_key_blocks.sum(-1, dtype=torch.int64).view(
        1, 1, num_query_blocks
    ).expand(batch_size, num_groups, -1)
    return (
        selected_mask,
        selected_block_counts,
        causal_block_counts,
        covered_target_tokens,
    )


class RepresentativeTokenFirstBlockSelector:
    """Create a token mask first, then cover it with fixed-size K blocks."""

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        logical_key_block_size: int | tuple[int, ...],
        target_token_budget: int,
        final_token_budget: int,
        minimum_block_coverage_ratio: float,
        force_sink_block: bool,
        force_diagonal_block: bool,
        probe_weights: tuple[float, float, float, float] = (
            0.2,
            0.3,
            0.4,
            0.1,
        ),
        adaptive_coverage_tolerance: float = 0.0,
    ) -> None:
        if isinstance(logical_key_block_size, int):
            logical_key_block_sizes = (logical_key_block_size,)
        else:
            logical_key_block_sizes = tuple(logical_key_block_size)
        if not logical_key_block_sizes or any(
            size not in {32, 64, 128} for size in logical_key_block_sizes
        ):
            raise ValueError("Logical K block sizes must be drawn from 32, 64, 128")
        if len(set(logical_key_block_sizes)) != len(logical_key_block_sizes):
            raise ValueError("Logical K block sizes must be unique")
        if target_token_budget <= 0 or final_token_budget <= 0:
            raise ValueError("Token budgets must be positive")
        if not 0.0 < minimum_block_coverage_ratio <= 1.0:
            raise ValueError("minimum_block_coverage_ratio must be in (0, 1]")
        if not 0.0 <= adaptive_coverage_tolerance <= 1.0:
            raise ValueError("adaptive_coverage_tolerance must be in [0, 1]")
        if len(probe_weights) != 4 or any(weight < 0 for weight in probe_weights):
            raise ValueError("probe_weights must contain four non-negative values")
        if not math.isclose(sum(probe_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probe_weights must sum to 1")
        self.layers = group_config["layers"]
        self.logical_key_block_sizes = tuple(sorted(logical_key_block_sizes))
        self.target_token_budget = int(target_token_budget)
        self.final_token_budget = int(final_token_budget)
        self.minimum_block_coverage_ratio = float(minimum_block_coverage_ratio)
        self.force_sink_block = bool(force_sink_block)
        self.force_diagonal_block = bool(force_diagonal_block)
        self.probe_weights = tuple(float(weight) for weight in probe_weights)
        self.adaptive_coverage_tolerance = float(adaptive_coverage_tolerance)
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("token_first_block_sparse_layer", default=None)
        )
        self.stats = TokenCompactedSelectorStats()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        gamma: float,
        min_budget: int,
        max_budget: int,
        tau: float = 0,
        gqa_interleave: bool = False,
    ) -> TokenCompactedIndex:
        del v, gamma, min_budget, max_budget, tau
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)
        timing_start.record()
        layer_idx = self.current_layer.get()
        if layer_idx is None:
            raise RuntimeError("Token-first selector was called without layer context")
        if block_size != 128:
            raise ValueError("The execution query tile must remain 128 tokens")

        groups = self.layers[str(layer_idx)]
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        num_share_q_heads = num_q_heads // num_kv_heads

        representatives = torch.tensor(
            [int(group["representative"]) for group in groups],
            dtype=torch.long,
            device=q.device,
        )
        if gqa_interleave:
            kv_indices = representatives % num_kv_heads
        else:
            kv_indices = representatives // num_share_q_heads
        representative_q = q.index_select(2, representatives)
        representative_k = k.index_select(2, kv_indices)

        pooled_q = _block_mean(representative_q, block_size)
        num_query_blocks = pooled_q.shape[1]
        query_starts = torch.arange(
            num_query_blocks, device=q.device
        ) * block_size
        query_ends = (query_starts + block_size).clamp(max=seq_len)
        query_lengths = (query_ends - query_starts).clamp_min(1)
        probe_positions = query_starts[:, None] + torch.stack(
            (
                (query_lengths - 1) // 3,
                2 * (query_lengths - 1) // 3,
                query_lengths - 1,
            ),
            dim=-1,
        )

        token_scores = torch.einsum(
            "bqhd,bthd->bhqt", pooled_q, representative_k
        ) / math.sqrt(head_dim)
        token_scores.mul_(self.probe_weights[3])
        for probe_idx in range(3):
            probe_q = representative_q.index_select(
                1, probe_positions[:, probe_idx]
            )
            probe_scores = torch.einsum(
                "bqhd,bthd->bhqt", probe_q, representative_k
            ) / math.sqrt(head_dim)
            token_scores.add_(probe_scores, alpha=self.probe_weights[probe_idx])

        key_positions = torch.arange(seq_len, device=q.device)
        causal_tokens = key_positions.view(1, 1, 1, seq_len) < query_ends.view(
            1, 1, num_query_blocks, 1
        )
        token_scores.masked_fill_(~causal_tokens, float("-inf"))
        target_k = min(self.target_token_budget, seq_len)
        _, target_indices = torch.topk(
            token_scores, k=target_k, dim=-1, sorted=False
        )
        target_keep_counts = torch.minimum(
            query_ends,
            query_ends.new_full((), target_k),
        )
        target_ranks = torch.arange(target_k, device=q.device).view(
            1, 1, 1, target_k
        )
        keep_target = target_ranks < target_keep_counts.view(
            1, 1, num_query_blocks, 1
        )
        target_mask = torch.zeros_like(token_scores, dtype=torch.bool)
        target_mask.scatter_(-1, target_indices, keep_target.expand_as(target_indices))
        target_mask &= causal_tokens

        target_tokens_per_row = target_mask.sum(-1, dtype=torch.int64)
        selected_mask = None
        chosen_block_counts = None
        chosen_causal_block_counts = None
        chosen_block_sizes = None
        best_coverage = None
        for key_block_size in self.logical_key_block_sizes:
            (
                candidate_mask,
                candidate_block_counts,
                candidate_causal_block_counts,
                candidate_covered_targets,
            ) = _cover_target_mask_with_key_blocks(
                target_mask,
                causal_tokens,
                query_starts,
                query_ends,
                key_block_size=key_block_size,
                final_token_budget=self.final_token_budget,
                minimum_block_coverage_ratio=(
                    self.minimum_block_coverage_ratio
                ),
                force_sink_block=self.force_sink_block,
                force_diagonal_block=self.force_diagonal_block,
            )
            candidate_coverage = (
                candidate_covered_targets.to(torch.float32)
                / target_tokens_per_row.clamp_min(1).to(torch.float32)
            )
            if selected_mask is None:
                selected_mask = candidate_mask
                chosen_block_counts = candidate_block_counts
                chosen_causal_block_counts = candidate_causal_block_counts
                chosen_block_sizes = torch.full_like(
                    candidate_block_counts, key_block_size, dtype=torch.int32
                )
                best_coverage = candidate_coverage
                continue

            if best_coverage is None:
                raise RuntimeError("Adaptive coverage state was not initialized")
            choose_larger = candidate_coverage >= (
                best_coverage - self.adaptive_coverage_tolerance
            )
            selected_mask = torch.where(
                choose_larger.unsqueeze(-1), candidate_mask, selected_mask
            )
            chosen_block_counts = torch.where(
                choose_larger, candidate_block_counts, chosen_block_counts
            )
            chosen_causal_block_counts = torch.where(
                choose_larger,
                candidate_causal_block_counts,
                chosen_causal_block_counts,
            )
            chosen_block_sizes = torch.where(
                choose_larger,
                torch.full_like(
                    chosen_block_sizes, key_block_size, dtype=torch.int32
                ),
                chosen_block_sizes,
            )
            best_coverage = torch.maximum(best_coverage, candidate_coverage)

        if (
            selected_mask is None
            or chosen_block_counts is None
            or chosen_causal_block_counts is None
            or chosen_block_sizes is None
        ):
            raise RuntimeError("No logical K block candidate was evaluated")

        rows = selected_mask.reshape(-1, seq_len)
        row_counts = rows.sum(dim=-1, dtype=torch.int32)
        row_ends_flat = torch.cumsum(row_counts, dim=0, dtype=torch.int32)
        row_starts_flat = row_ends_flat - row_counts
        token_indices = torch.nonzero(rows, as_tuple=False)[:, 1].to(torch.int32)
        row_starts = row_starts_flat.view(
            batch_size, len(groups), num_query_blocks
        )
        row_ends = row_ends_flat.view(batch_size, len(groups), num_query_blocks)

        head_to_group = torch.empty(
            (num_q_heads,), dtype=torch.int32, device=q.device
        )
        for group_idx, group in enumerate(groups):
            members = torch.tensor(
                [int(member) for member in group["members"]],
                dtype=torch.long,
                device=q.device,
            )
            head_to_group[members] = group_idx
        group_sizes = torch.tensor(
            [len(group["members"]) for group in groups],
            dtype=torch.int64,
            device=q.device,
        )

        selected_blocks = int(
            (
                chosen_block_counts.sum(dim=-1, dtype=torch.int64)
                * group_sizes[None, :]
            ).sum().item()
        )
        total_causal_blocks = int(
            (
                chosen_causal_block_counts.sum(dim=-1, dtype=torch.int64)
                * group_sizes[None, :]
            ).sum().item()
        )
        prefix = torch.cumsum(selected_mask, dim=-1, dtype=torch.int64)
        query_positions = torch.arange(
            num_query_blocks * block_size, device=q.device
        ).view(num_query_blocks, block_size)
        valid_queries = query_positions < seq_len
        gather_positions = query_positions.clamp(max=seq_len - 1).view(
            1, 1, num_query_blocks, block_size
        ).expand(batch_size, len(groups), -1, -1)
        selected_per_query = torch.gather(prefix, -1, gather_positions)
        selected_per_query *= valid_queries.view(
            1, 1, num_query_blocks, block_size
        ).to(selected_per_query.dtype)
        selected_pairs_per_group = selected_per_query.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        selected_token_pairs = int(
            (selected_pairs_per_group * group_sizes[None, :]).sum().item()
        )
        total_causal_token_pairs = (
            batch_size * num_q_heads * seq_len * (seq_len + 1) // 2
        )
        selected_keys_per_group = selected_mask.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        causal_keys_per_row = query_ends.sum(dtype=torch.int64)
        compacted_key_tokens = int(
            (selected_keys_per_group * group_sizes[None, :]).sum().item()
        )
        candidate_key_tokens = int(
            batch_size * num_q_heads * causal_keys_per_row.item()
        )
        target_per_group = target_mask.sum(dim=(-1, -2), dtype=torch.int64)
        covered_per_group = (target_mask & selected_mask).sum(
            dim=(-1, -2), dtype=torch.int64
        )
        target_mask_tokens = int(
            (target_per_group * group_sizes[None, :]).sum().item()
        )
        covered_target_tokens = int(
            (covered_per_group * group_sizes[None, :]).sum().item()
        )
        chosen_block_size_rows = {
            size: int(
                (
                    (chosen_block_sizes == size).sum(dim=-1, dtype=torch.int64)
                    * group_sizes[None, :]
                ).sum().item()
            )
            for size in self.logical_key_block_sizes
        }
        self.stats.record(
            selected_blocks=selected_blocks,
            causal_blocks=total_causal_blocks,
            selected_token_pairs=selected_token_pairs,
            causal_token_pairs=total_causal_token_pairs,
            compacted_key_tokens=compacted_key_tokens,
            candidate_key_tokens=candidate_key_tokens,
            target_mask_tokens=target_mask_tokens,
            covered_target_tokens=covered_target_tokens,
            chosen_block_size_rows=chosen_block_size_rows,
            timing_events=(timing_start, timing_end),
        )
        timing_end.record()

        return TokenCompactedIndex(
            row_starts=row_starts.contiguous(),
            row_ends=row_ends.contiguous(),
            token_indices=token_indices.contiguous(),
            head_to_group=head_to_group.contiguous(),
            num_groups=len(groups),
            num_query_blocks=num_query_blocks,
            query_block_size=block_size,
        )


@triton.jit
def _token_compacted_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    row_starts_ptr,
    row_ends_ptr,
    token_indices_ptr,
    head_to_group_ptr,
    num_heads,
    num_kv_heads,
    num_share_q_heads,
    num_groups,
    num_query_blocks,
    q_len,
    k_len,
    softmax_scale,
    gqa_interleave: tl.constexpr,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    if gqa_interleave:
        pid_kh = pid_h % num_kv_heads
    else:
        pid_kh = pid_h // num_share_q_heads

    group_idx = tl.load(head_to_group_ptr + pid_h).to(tl.int32)
    row_offset = (pid_b * num_groups + group_idx) * num_query_blocks + pid_q
    row_start = tl.load(row_starts_ptr + row_offset).to(tl.int32)
    row_end = tl.load(row_ends_ptr + row_offset).to(tl.int32)
    row_length = row_end - row_start
    num_chunks = tl.cdiv(row_length, BLOCK_SIZE_K)

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(q_len, BLOCK_SIZE_D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    query_offsets = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    key_lanes = tl.arange(0, BLOCK_SIZE_K)
    dim_offsets = tl.arange(0, BLOCK_SIZE_D)

    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)

    for chunk_idx in range(0, num_chunks):
        token_offset = chunk_idx * BLOCK_SIZE_K
        token_ptrs = row_start + token_offset + key_lanes
        valid_lane = token_offset + key_lanes < row_length
        token_ids = tl.load(
            token_indices_ptr + token_ptrs,
            mask=valid_lane,
            other=0,
        ).to(tl.int32)
        valid_key = valid_lane & (token_ids < k_len)

        k_ptrs = (
            k_ptr
            + pid_b * stride_kb
            + pid_kh * stride_kh
            + dim_offsets[:, None] * stride_kd
            + token_ids[None, :] * stride_kn
        )
        k_tile = tl.load(
            k_ptrs,
            mask=valid_key[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k_tile) * softmax_scale
        causal = (
            valid_key[None, :]
            & (query_offsets[:, None] < q_len)
            & (query_offsets[:, None] >= token_ids[None, :])
        )
        qk = tl.where(causal, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_scale = tl.math.exp2(m_i - m_ij)
        acc_o *= acc_scale[:, None]

        v_ptrs = (
            v_ptr
            + pid_b * stride_vb
            + pid_kh * stride_vh
            + token_ids[:, None] * stride_vn
            + dim_offsets[None, :] * stride_vd
        )
        v_tile = tl.load(
            v_ptrs,
            mask=valid_key[:, None],
            other=0.0,
        )
        acc_o += tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)

    acc_o *= tl.math.exp2(m_i - lse_i)[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(q_len, BLOCK_SIZE_D),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(
        o_ptrs,
        acc_o.to(tl.bfloat16),
        boundary_check=(0, 1),
    )


def triton_token_compacted_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index: TokenCompactedIndex,
    *,
    token_chunk_size: int,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Run causal prefill over compacted arbitrary key-token indices."""

    batch_size, q_len, num_q_heads, head_dim = q.shape
    _, k_len, num_kv_heads, _ = k.shape
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Token-compacted kernel requires BF16 Q/K/V")
    if head_dim != 128:
        raise ValueError("Token-compacted kernel currently requires head_dim=128")
    if token_chunk_size not in {16, 32}:
        raise ValueError("token_chunk_size must be 16 or 32")
    if index.query_block_size != 128:
        raise ValueError("Token-compacted kernel requires 128-token query blocks")
    if index.num_query_blocks != triton.cdiv(q_len, 128):
        raise ValueError("Token index/query length mismatch")
    if num_q_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")

    num_share_q_heads = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    scale *= math.log2(math.e)
    output = torch.empty_like(q)
    _token_compacted_prefill_kernel[(batch_size * num_q_heads, index.num_query_blocks)](
        q,
        k,
        v,
        output,
        index.row_starts,
        index.row_ends,
        index.token_indices,
        index.head_to_group,
        num_q_heads,
        num_kv_heads,
        num_share_q_heads,
        index.num_groups,
        index.num_query_blocks,
        q_len,
        k_len,
        scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_SIZE_Q=128,
        BLOCK_SIZE_K=token_chunk_size,
        BLOCK_SIZE_D=128,
        num_warps=8,
        num_stages=2,
    )
    return output


@dataclass
class TokenCompactedFlexPrefillPatch:
    selector: Any
    ops_module: Any
    original_get_active_blocks: Any
    original_block_wise_attention: Any

    def restore(self) -> None:
        self.ops_module.get_active_blocks = self.original_get_active_blocks
        self.ops_module.triton_block_wise_attention = self.original_block_wise_attention


def install_grouped_token_compacted_flexprefill(
    model,
    group_config: Mapping[str, Any],
    *,
    block_size: int,
    gamma: float,
    tau: float,
    min_budget: int,
    max_budget: Optional[int],
    token_top_p: float,
    min_tokens_per_selected_block: int,
    token_chunk_size: int,
    force_sink_block: bool,
    force_diagonal_block: bool,
    selection_mode: str = "per_block_top_p",
    candidate_block_count: int = 256,
    final_token_budget: int = 8192,
    probe_weights: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.1),
    logical_key_block_size: int = 128,
    minimum_block_coverage_ratio: float = 0.125,
    adaptive_coverage_tolerance: float = 0.0,
) -> TokenCompactedFlexPrefillPatch:
    """Patch long prefill with block-selected, token-compacted attention."""

    from flex_prefill import patch_model
    import flex_prefill.ops.flex_prefill_attention as ops_module

    validate_group_config(
        group_config,
        num_layers=int(model.config.num_hidden_layers),
        num_heads=int(model.config.num_attention_heads),
    )
    if block_size != 128:
        raise ValueError("Token compaction currently requires block_size=128")
    patch_model(
        model,
        "flex_prefill",
        {
            "block_size": block_size,
            "flex_prefill_gamma": gamma,
            "flex_prefill_tau": tau,
            "flex_prefill_min_budget": min_budget,
            "flex_prefill_max_budget": max_budget,
        },
    )

    if selection_mode == "per_block_top_p":
        selector = RepresentativeTokenCompactedSelector(
            group_config,
            token_top_p=token_top_p,
            min_tokens_per_selected_block=min_tokens_per_selected_block,
            force_sink_block=force_sink_block,
            force_diagonal_block=force_diagonal_block,
        )
    elif selection_mode == "hisa_style_global_topk":
        selector = RepresentativeHISAStyleSelector(
            group_config,
            candidate_block_count=candidate_block_count,
            final_token_budget=final_token_budget,
            force_sink_block=force_sink_block,
            force_diagonal_block=force_diagonal_block,
        )
    elif selection_mode == "hisa_style_mass_allocated":
        selector = RepresentativeHISAStyleSelector(
            group_config,
            candidate_block_count=candidate_block_count,
            final_token_budget=final_token_budget,
            force_sink_block=force_sink_block,
            force_diagonal_block=force_diagonal_block,
            token_selection_mode="block_mass_weighted_average",
            probe_weights=probe_weights,
        )
    elif selection_mode == "token_first_block_cover":
        selector = RepresentativeTokenFirstBlockSelector(
            group_config,
            logical_key_block_size=logical_key_block_size,
            target_token_budget=final_token_budget,
            final_token_budget=final_token_budget,
            minimum_block_coverage_ratio=minimum_block_coverage_ratio,
            force_sink_block=force_sink_block,
            force_diagonal_block=force_diagonal_block,
            probe_weights=probe_weights,
            adaptive_coverage_tolerance=adaptive_coverage_tolerance,
        )
    else:
        raise ValueError(f"Unknown token selection mode: {selection_mode}")
    original_get_active_blocks = ops_module.get_active_blocks
    original_block_wise_attention = ops_module.triton_block_wise_attention

    def token_compacted_dispatch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_idx,
        block_size: int,
        grid_offset: int = 0,
        softmax_scale: Optional[float] = None,
        gqa_interleave: bool = False,
    ) -> torch.Tensor:
        if isinstance(block_idx, TokenCompactedIndex):
            if grid_offset:
                raise ValueError("Token-compacted kernel does not use grid_offset")
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_start.record()
            output = triton_token_compacted_prefill_attention(
                q,
                k,
                v,
                block_idx,
                token_chunk_size=token_chunk_size,
                softmax_scale=softmax_scale,
                gqa_interleave=gqa_interleave,
            )
            timing_end.record()
            selector.stats.record_kernel_events(timing_start, timing_end)
            return output
        return original_block_wise_attention(
            q,
            k,
            v,
            block_idx,
            block_size,
            grid_offset,
            softmax_scale,
            gqa_interleave,
        )

    ops_module.get_active_blocks = selector
    ops_module.triton_block_wise_attention = token_compacted_dispatch

    wrapped_layers = 0
    for module in model.modules():
        layer_idx = getattr(module, "layer_idx", None)
        if (
            layer_idx is None
            or not hasattr(module, "q_proj")
            or not hasattr(module, "k_proj")
        ):
            continue
        original_forward = module.forward

        def layer_forward(
            self,
            *args,
            __forward=original_forward,
            __layer_idx=int(layer_idx),
            **kwargs,
        ):
            token = selector.current_layer.set(__layer_idx)
            try:
                return __forward(*args, **kwargs)
            finally:
                selector.current_layer.reset(token)

        module.forward = types.MethodType(layer_forward, module)
        wrapped_layers += 1

    if wrapped_layers != int(model.config.num_hidden_layers):
        ops_module.get_active_blocks = original_get_active_blocks
        ops_module.triton_block_wise_attention = original_block_wise_attention
        raise RuntimeError(
            f"Wrapped {wrapped_layers} layers, expected "
            f"{model.config.num_hidden_layers}"
        )

    return TokenCompactedFlexPrefillPatch(
        selector=selector,
        ops_module=ops_module,
        original_get_active_blocks=original_get_active_blocks,
        original_block_wise_attention=original_block_wise_attention,
    )
