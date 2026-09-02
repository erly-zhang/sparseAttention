"""Representative-head sparse attention with block and token selection.

This module supports both block-first token compaction and token-first block
projection. Sink and diagonal key blocks can be protected in full. The Triton
kernel gathers only the selected K/V tokens before QK and PV.
"""

from __future__ import annotations

import contextvars
import math
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
        self.target_probability_mass = 0.0
        self.covered_target_probability_mass = 0.0
        self.chosen_block_size_rows: Dict[int, int] = {}
        self.layer_selected_key_tokens: Dict[int, int] = {}
        self.layer_candidate_key_tokens: Dict[int, int] = {}
        self.layer_target_mask_tokens: Dict[int, int] = {}
        self.layer_covered_target_tokens: Dict[int, int] = {}
        self.layer_target_probability_mass: Dict[int, float] = {}
        self.layer_covered_target_probability_mass: Dict[int, float] = {}
        self.layer_selected_blocks: Dict[int, int] = {}
        self.layer_selected_token_pairs: Dict[int, int] = {}
        self.layer_chosen_block_size_rows: Dict[int, Dict[int, int]] = {}
        self.candidate_kernel_pairs_by_size: Dict[int, int] = {}
        self.layer_candidate_kernel_pairs_by_size: Dict[int, Dict[int, int]] = {}
        self.layer_selector_rows: Dict[int, int] = {}
        self.layer_target_selection_mode_rows: Dict[int, Dict[str, int]] = {}
        self.watched_key_range_stats: Dict[str, Dict[str, float]] = {}
        self.layer_watched_key_range_stats: Dict[
            int, Dict[str, Dict[str, float]]
        ] = {}
        self.member_mask_fidelity_stats: Dict[
            int, Dict[int, Dict[str, Any]]
        ] = {}
        self.vertical_slash_candidate_blocks = 0
        self.vertical_slash_added_blocks = 0
        self.vertical_slash_added_key_tokens = 0
        self.layer_vertical_slash_stats: Dict[int, Dict[str, int]] = {}
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
        target_probability_mass: float = 0.0,
        covered_target_probability_mass: float = 0.0,
        chosen_block_size_rows: Optional[Mapping[int, int]] = None,
        candidate_kernel_pairs_by_size: Optional[Mapping[int, int]] = None,
        layer_idx: Optional[int] = None,
        selector_rows: int = 0,
        target_selection_mode: Optional[str] = None,
        watched_key_range_stats: Optional[
            Mapping[str, Mapping[str, float]]
        ] = None,
        member_mask_fidelity_stats: Optional[
            Mapping[int, Mapping[str, Any]]
        ] = None,
        vertical_slash_candidate_blocks: int = 0,
        vertical_slash_added_blocks: int = 0,
        vertical_slash_added_key_tokens: int = 0,
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
        self.target_probability_mass += target_probability_mass
        self.covered_target_probability_mass += covered_target_probability_mass
        self.vertical_slash_candidate_blocks += int(
            vertical_slash_candidate_blocks
        )
        self.vertical_slash_added_blocks += int(vertical_slash_added_blocks)
        self.vertical_slash_added_key_tokens += int(
            vertical_slash_added_key_tokens
        )
        if chosen_block_size_rows:
            for size, count in chosen_block_size_rows.items():
                self.chosen_block_size_rows[int(size)] = (
                    self.chosen_block_size_rows.get(int(size), 0) + int(count)
                )
        if candidate_kernel_pairs_by_size:
            for size, count in candidate_kernel_pairs_by_size.items():
                self.candidate_kernel_pairs_by_size[int(size)] = (
                    self.candidate_kernel_pairs_by_size.get(int(size), 0)
                    + int(count)
                )
        if watched_key_range_stats:
            for range_name, values in watched_key_range_stats.items():
                aggregate = self.watched_key_range_stats.setdefault(
                    str(range_name), {}
                )
                for name, value in values.items():
                    aggregate[str(name)] = aggregate.get(str(name), 0.0) + float(
                        value
                    )
        if layer_idx is not None:
            layer_idx = int(layer_idx)
            self.layer_selected_key_tokens[layer_idx] = (
                self.layer_selected_key_tokens.get(layer_idx, 0)
                + int(compacted_key_tokens)
            )
            self.layer_candidate_key_tokens[layer_idx] = (
                self.layer_candidate_key_tokens.get(layer_idx, 0)
                + int(candidate_key_tokens)
            )
            self.layer_target_mask_tokens[layer_idx] = (
                self.layer_target_mask_tokens.get(layer_idx, 0)
                + int(target_mask_tokens)
            )
            self.layer_covered_target_tokens[layer_idx] = (
                self.layer_covered_target_tokens.get(layer_idx, 0)
                + int(covered_target_tokens)
            )
            self.layer_target_probability_mass[layer_idx] = (
                self.layer_target_probability_mass.get(layer_idx, 0.0)
                + float(target_probability_mass)
            )
            self.layer_covered_target_probability_mass[layer_idx] = (
                self.layer_covered_target_probability_mass.get(layer_idx, 0.0)
                + float(covered_target_probability_mass)
            )
            self.layer_selected_blocks[layer_idx] = (
                self.layer_selected_blocks.get(layer_idx, 0)
                + int(selected_blocks)
            )
            self.layer_selected_token_pairs[layer_idx] = (
                self.layer_selected_token_pairs.get(layer_idx, 0)
                + int(selected_token_pairs)
            )
            layer_sizes = self.layer_chosen_block_size_rows.setdefault(layer_idx, {})
            if chosen_block_size_rows:
                for size, count in chosen_block_size_rows.items():
                    layer_sizes[int(size)] = layer_sizes.get(int(size), 0) + int(count)
            layer_candidate_costs = self.layer_candidate_kernel_pairs_by_size.setdefault(
                layer_idx, {}
            )
            if candidate_kernel_pairs_by_size:
                for size, count in candidate_kernel_pairs_by_size.items():
                    layer_candidate_costs[int(size)] = (
                        layer_candidate_costs.get(int(size), 0) + int(count)
                    )
            self.layer_selector_rows[layer_idx] = (
                self.layer_selector_rows.get(layer_idx, 0)
                + int(selector_rows)
            )
            if target_selection_mode is not None:
                layer_modes = self.layer_target_selection_mode_rows.setdefault(
                    layer_idx, {}
                )
                layer_modes[str(target_selection_mode)] = (
                    layer_modes.get(str(target_selection_mode), 0)
                    + int(selector_rows)
                )
            if watched_key_range_stats:
                layer_ranges = self.layer_watched_key_range_stats.setdefault(
                    layer_idx, {}
                )
                for range_name, values in watched_key_range_stats.items():
                    aggregate = layer_ranges.setdefault(str(range_name), {})
                    for name, value in values.items():
                        aggregate[str(name)] = aggregate.get(
                            str(name), 0.0
                        ) + float(value)
            if member_mask_fidelity_stats:
                layer_members = self.member_mask_fidelity_stats.setdefault(
                    layer_idx, {}
                )
                for head, values in member_mask_fidelity_stats.items():
                    member_totals = layer_members.setdefault(int(head), {})
                    for name, value in values.items():
                        if name == "watched_key_ranges":
                            watched_totals = member_totals.setdefault(name, {})
                            for range_name, range_values in value.items():
                                range_totals = watched_totals.setdefault(
                                    str(range_name), {}
                                )
                                for field, field_value in range_values.items():
                                    range_totals[str(field)] = (
                                        range_totals.get(str(field), 0.0)
                                        + float(field_value)
                                    )
                        elif name in {
                            "group_index",
                            "representative_head",
                            "kv_head",
                            "is_representative",
                        }:
                            member_totals[str(name)] = int(value)
                        else:
                            member_totals[str(name)] = (
                                member_totals.get(str(name), 0.0)
                                + float(value)
                            )
            layer_vs = self.layer_vertical_slash_stats.setdefault(
                layer_idx,
                {
                    "candidate_blocks": 0,
                    "added_blocks": 0,
                    "added_key_tokens": 0,
                },
            )
            layer_vs["candidate_blocks"] += int(
                vertical_slash_candidate_blocks
            )
            layer_vs["added_blocks"] += int(vertical_slash_added_blocks)
            layer_vs["added_key_tokens"] += int(
                vertical_slash_added_key_tokens
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
        target_probability_coverage = (
            self.covered_target_probability_mass / self.target_probability_mass
            if self.target_probability_mass
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
            "target_probability_mass": self.target_probability_mass,
            "covered_target_probability_mass": (
                self.covered_target_probability_mass
            ),
            "mean_target_probability_coverage_ratio": (
                target_probability_coverage
            ),
            "vertical_slash_candidate_blocks": (
                self.vertical_slash_candidate_blocks
            ),
            "vertical_slash_added_blocks": self.vertical_slash_added_blocks,
            "vertical_slash_added_key_tokens": (
                self.vertical_slash_added_key_tokens
            ),
            "layer_vertical_slash_stats": {
                layer: dict(values)
                for layer, values in self.layer_vertical_slash_stats.items()
            },
            "chosen_block_size_rows": dict(self.chosen_block_size_rows),
            "layer_selected_key_tokens": dict(
                self.layer_selected_key_tokens
            ),
            "layer_candidate_key_tokens": dict(
                self.layer_candidate_key_tokens
            ),
            "layer_target_mask_tokens": dict(self.layer_target_mask_tokens),
            "layer_covered_target_tokens": dict(
                self.layer_covered_target_tokens
            ),
            "layer_target_probability_mass": dict(
                self.layer_target_probability_mass
            ),
            "layer_covered_target_probability_mass": dict(
                self.layer_covered_target_probability_mass
            ),
            "layer_selected_blocks": dict(self.layer_selected_blocks),
            "layer_selected_token_pairs": dict(
                self.layer_selected_token_pairs
            ),
            "layer_chosen_block_size_rows": {
                layer: dict(values)
                for layer, values in self.layer_chosen_block_size_rows.items()
            },
            "candidate_kernel_pairs_by_size": dict(
                self.candidate_kernel_pairs_by_size
            ),
            "layer_candidate_kernel_pairs_by_size": {
                layer: dict(values)
                for layer, values in self.layer_candidate_kernel_pairs_by_size.items()
            },
            "layer_selector_rows": dict(self.layer_selector_rows),
            "layer_target_selection_mode_rows": {
                layer: dict(values)
                for layer, values in self.layer_target_selection_mode_rows.items()
            },
            "watched_key_range_stats": {
                range_name: dict(values)
                for range_name, values in self.watched_key_range_stats.items()
            },
            "layer_watched_key_range_stats": {
                layer: {
                    range_name: dict(values)
                    for range_name, values in ranges.items()
                }
                for layer, ranges in self.layer_watched_key_range_stats.items()
            },
            "member_mask_fidelity_stats": {
                layer: {
                    head: {
                        name: (
                            {
                                range_name: dict(range_values)
                                for range_name, range_values in value.items()
                            }
                            if name == "watched_key_ranges"
                            else value
                        )
                        for name, value in values.items()
                    }
                    for head, values in members.items()
                }
                for layer, members in self.member_mask_fidelity_stats.items()
            },
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


def _causal_block_pair_costs(
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    key_block_starts: torch.Tensor,
    key_block_ends: torch.Tensor,
) -> torch.Tensor:
    """Return exact causal QK-pair costs for every query-tile/K-block pair."""

    key_lengths = (key_block_ends - key_block_starts).to(torch.int64)

    def clamped_ramp_prefix(relative_end: torch.Tensor) -> torch.Tensor:
        positive = relative_end.clamp_min(0).to(torch.int64)
        ramp = torch.minimum(positive, key_lengths.view(1, -1))
        triangular = ramp * (ramp + 1) // 2
        tail = (positive - key_lengths.view(1, -1)).clamp_min(0)
        return triangular + tail * key_lengths.view(1, -1)

    relative_start = query_starts.to(torch.int64).view(-1, 1) - key_block_starts.to(
        torch.int64
    ).view(1, -1)
    relative_end = query_ends.to(torch.int64).view(-1, 1) - key_block_starts.to(
        torch.int64
    ).view(1, -1)
    return clamped_ramp_prefix(relative_end) - clamped_ramp_prefix(relative_start)


def _cover_target_mask_with_key_blocks(
    target_mask: torch.Tensor,
    causal_tokens: torch.Tensor,
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    *,
    key_block_size: int,
    final_token_budget: Optional[int],
    maximum_selected_blocks: Optional[int],
    fill_final_token_budget: bool,
    minimum_block_coverage_ratio: float,
    target_probabilities: Optional[torch.Tensor] = None,
    minimum_block_target_probability_ratio: Optional[float] = None,
    block_probability_top_p: Optional[float] = None,
    block_target_probability_top_p: Optional[float] = None,
    block_target_count_coverage_ratio: Optional[float] = None,
    block_target_cost_aware: bool = False,
    block_f_beta: Optional[float] = None,
    forced_key_ranges: tuple[tuple[int, int], ...] = (),
    force_sink_block: bool,
    force_diagonal_block: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
    if target_probabilities is not None:
        padded_probabilities = torch.nn.functional.pad(
            target_probabilities, (0, padded_key_len - seq_len), value=0.0
        )
        probability_by_block = padded_probabilities.view(
            batch_size,
            num_groups,
            num_query_blocks,
            num_key_blocks,
            key_block_size,
        )
        target_probability_by_block = torch.where(
            target_by_block, probability_by_block, torch.zeros_like(probability_by_block)
        )
        target_probability_mass = target_probability_by_block.sum(
            -1, dtype=torch.float32
        )
        block_probability_mass = probability_by_block.sum(-1, dtype=torch.float32)
    else:
        target_probability_mass = target_count.to(torch.float32)
        block_probability_mass = target_probability_mass

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
    for forced_start, forced_end in forced_key_ranges:
        clipped_start = min(max(int(forced_start), 0), seq_len)
        clipped_end = min(max(int(forced_end), clipped_start), seq_len)
        if clipped_end <= clipped_start:
            continue
        first_block = clipped_start // key_block_size
        last_block = (clipped_end - 1) // key_block_size
        protected_blocks[..., first_block : last_block + 1] = True
    protected_blocks &= causal_key_blocks.view(
        1, 1, num_query_blocks, num_key_blocks
    )

    block_selection_score = target_count.new_zeros(
        target_count.shape[:-1], dtype=torch.float32
    )
    if block_f_beta is not None:
        causal = causal_key_blocks.view(1, 1, num_query_blocks, num_key_blocks)
        valid_token_count = (
            torch.minimum(
                query_ends.view(num_query_blocks, 1),
                key_block_ends.view(1, num_key_blocks),
            )
            - key_block_starts.view(1, num_key_blocks)
        ).clamp(min=0, max=key_block_size).view(
            1, 1, num_query_blocks, num_key_blocks
        )
        valid_token_count = valid_token_count.expand(
            batch_size, num_groups, -1, -1
        )
        false_positive_count = valid_token_count - target_count
        density = target_count.to(torch.float32) / valid_token_count.clamp_min(1)
        ordinary_density = density.masked_fill(
            ~causal | protected_blocks | (target_count == 0), float("-inf")
        )
        sorted_density, sorted_indices = torch.sort(
            ordinary_density, dim=-1, descending=True
        )
        sorted_tp = torch.gather(target_count.to(torch.float32), -1, sorted_indices)
        sorted_fp = torch.gather(
            false_positive_count.to(torch.float32), -1, sorted_indices
        )
        finite = torch.isfinite(sorted_density)
        sorted_tp = torch.where(finite, sorted_tp, torch.zeros_like(sorted_tp))
        sorted_fp = torch.where(finite, sorted_fp, torch.zeros_like(sorted_fp))

        protected_float = protected_blocks.to(torch.float32)
        initial_tp = (target_count.to(torch.float32) * protected_float).sum(
            -1, keepdim=True
        )
        initial_fp = (false_positive_count.to(torch.float32) * protected_float).sum(
            -1, keepdim=True
        )
        prefix_tp = initial_tp + torch.cumsum(sorted_tp, dim=-1)
        prefix_fp = initial_fp + torch.cumsum(sorted_fp, dim=-1)
        total_targets = target_count.sum(-1, keepdim=True, dtype=torch.float32)
        beta_squared = block_f_beta * block_f_beta

        def f_beta(tp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
            fn = (total_targets - tp).clamp_min(0.0)
            numerator = (1.0 + beta_squared) * tp
            denominator = numerator + beta_squared * fn + fp
            return torch.where(
                denominator > 0, numerator / denominator, torch.zeros_like(numerator)
            )

        prefix_scores = f_beta(prefix_tp, prefix_fp)
        initial_score = f_beta(initial_tp, initial_fp)
        all_scores = torch.cat((initial_score, prefix_scores), dim=-1)
        best_prefix_length = torch.argmax(all_scores, dim=-1)
        block_selection_score = torch.gather(
            all_scores, -1, best_prefix_length.unsqueeze(-1)
        ).squeeze(-1)
        sorted_rank = torch.arange(num_key_blocks, device=target_mask.device).view(
            1, 1, 1, num_key_blocks
        )
        keep_sorted = sorted_rank < best_prefix_length.unsqueeze(-1)
        keep_sorted &= finite
        ordinary_selected = torch.zeros_like(protected_blocks)
        ordinary_selected.scatter_(-1, sorted_indices, keep_sorted)
        selected_key_blocks = protected_blocks | ordinary_selected
        eligible_blocks = causal
    elif block_target_probability_top_p is not None:
        if target_probabilities is None:
            raise ValueError(
                "block_target_probability_top_p requires token probabilities"
            )
        causal = causal_key_blocks.view(1, 1, num_query_blocks, num_key_blocks)
        ordinary_mask = causal & ~protected_blocks & (target_count > 0)
        causal_pair_cost = _causal_block_pair_costs(
            query_starts,
            query_ends,
            key_block_starts,
            key_block_ends,
        ).view(1, 1, num_query_blocks, num_key_blocks)
        if block_target_cost_aware:
            ordinary_score = (
                target_probability_mass
                / causal_pair_cost.clamp_min(1).to(torch.float32)
            ).masked_fill(~ordinary_mask, float("-inf"))
        else:
            ordinary_score = target_probability_mass.masked_fill(
                ~ordinary_mask, float("-inf")
            )
        sorted_score, sorted_indices = torch.sort(
            ordinary_score, dim=-1, descending=True
        )
        sorted_mass = torch.gather(
            target_probability_mass, -1, sorted_indices
        )
        finite = torch.isfinite(sorted_score)
        finite_mass = torch.where(
            finite, sorted_mass, torch.zeros_like(sorted_mass)
        )
        protected_mass = (
            target_probability_mass
            * protected_blocks.to(target_probability_mass.dtype)
        ).sum(-1, keepdim=True, dtype=torch.float32)
        target_total = target_probability_mass.sum(
            -1, keepdim=True, dtype=torch.float32
        )
        target_count_total = target_count.sum(
            -1, keepdim=True, dtype=torch.int64
        )
        required_mass = target_total * block_target_probability_top_p
        remaining_mass = (required_mass - protected_mass).clamp_min(0.0)
        cumulative_before = torch.cumsum(
            finite_mass, dim=-1, dtype=torch.float32
        ) - finite_mass
        keep_sorted = (cumulative_before < remaining_mass) & finite
        protected_block_count = protected_blocks.sum(
            -1, keepdim=True, dtype=torch.int64
        )
        if maximum_selected_blocks is not None:
            ordinary_slots = (
                maximum_selected_blocks - protected_block_count
            ).clamp_min(0)
            sorted_rank = torch.arange(
                num_key_blocks, device=target_mask.device
            ).view(1, 1, 1, num_key_blocks)
            keep_sorted &= sorted_rank < ordinary_slots
        mass_selected = torch.zeros_like(protected_blocks)
        mass_selected.scatter_(-1, sorted_indices, keep_sorted)

        count_selected = torch.zeros_like(protected_blocks)
        if block_target_count_coverage_ratio is not None:
            protected_target_count = (
                target_count.to(torch.int64)
                * protected_blocks.to(torch.int64)
            ).sum(-1, keepdim=True, dtype=torch.int64)
            mass_target_count = (
                target_count.to(torch.int64)
                * mass_selected.to(torch.int64)
            ).sum(-1, keepdim=True, dtype=torch.int64)
            required_count = torch.ceil(
                target_count_total.to(torch.float32)
                * block_target_count_coverage_ratio
            ).to(torch.int64)
            remaining_count = (
                required_count - protected_target_count - mass_target_count
            ).clamp_min(0)
            count_candidates = ordinary_mask & ~mass_selected
            count_score = (
                target_count.to(torch.float32)
                / causal_pair_cost.clamp_min(1).to(torch.float32)
            ).masked_fill(~count_candidates, float("-inf"))
            sorted_count_score, count_sorted_indices = torch.sort(
                count_score, dim=-1, descending=True
            )
            finite_count = torch.isfinite(sorted_count_score)
            sorted_target_count = torch.gather(
                target_count.to(torch.int64), -1, count_sorted_indices
            )
            sorted_target_count = torch.where(
                finite_count,
                sorted_target_count,
                torch.zeros_like(sorted_target_count),
            )
            cumulative_count_before = torch.cumsum(
                sorted_target_count, dim=-1, dtype=torch.int64
            ) - sorted_target_count
            keep_count_sorted = (
                cumulative_count_before < remaining_count
            ) & finite_count
            if maximum_selected_blocks is not None:
                mass_block_count = mass_selected.sum(
                    -1, keepdim=True, dtype=torch.int64
                )
                count_slots = (
                    maximum_selected_blocks
                    - protected_block_count
                    - mass_block_count
                ).clamp_min(0)
                keep_count_sorted &= sorted_rank < count_slots
            count_selected.scatter_(
                -1, count_sorted_indices, keep_count_sorted
            )

        selected_key_blocks = protected_blocks | mass_selected | count_selected
        eligible_blocks = causal
        covered_mass = (
            target_probability_mass
            * selected_key_blocks.to(target_probability_mass.dtype)
        ).sum(-1, dtype=torch.float32)
        block_selection_score = covered_mass / target_total.squeeze(-1).clamp_min(
            1e-12
        )
    elif block_probability_top_p is not None:
        if target_probabilities is None:
            raise ValueError("block_probability_top_p requires token probabilities")
        causal = causal_key_blocks.view(1, 1, num_query_blocks, num_key_blocks)
        ordinary_mass = block_probability_mass.masked_fill(
            ~causal | protected_blocks, float("-inf")
        )
        sorted_mass, sorted_indices = torch.sort(
            ordinary_mass, dim=-1, descending=True
        )
        finite_mass = torch.where(
            torch.isfinite(sorted_mass), sorted_mass, torch.zeros_like(sorted_mass)
        )
        protected_mass = (
            block_probability_mass * protected_blocks.to(block_probability_mass.dtype)
        ).sum(-1, keepdim=True, dtype=torch.float32)
        remaining_mass = (block_probability_top_p - protected_mass).clamp_min(0.0)
        cumulative_before = torch.cumsum(
            finite_mass, dim=-1, dtype=torch.float32
        ) - finite_mass
        keep_sorted = (cumulative_before < remaining_mass) & (finite_mass > 0)
        ordinary_selected = torch.zeros_like(protected_blocks)
        ordinary_selected.scatter_(-1, sorted_indices, keep_sorted)
        selected_key_blocks = protected_blocks | ordinary_selected
        eligible_blocks = causal
        block_selection_score = (
            block_probability_mass * selected_key_blocks.to(torch.float32)
        ).sum(-1)
    elif minimum_block_target_probability_ratio is None:
        minimum_target_tokens = math.ceil(
            key_block_size * minimum_block_coverage_ratio
        )
        block_priority = target_count.to(torch.float32)
        if fill_final_token_budget:
            # A matched-budget run treats the density threshold as a priority
            # boundary rather than a hard exclusion. Blocks above the boundary
            # are ranked first because they contain more target tokens; lower-
            # density blocks then fill the same whole-key budget as fixed top-k.
            eligible_blocks = causal_key_blocks.view(
                1, 1, num_query_blocks, num_key_blocks
            ).expand(batch_size, num_groups, -1, -1).clone()
            if target_probabilities is not None:
                row_max_mass = target_probability_mass.amax(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                block_priority = block_priority + (
                    0.5 * target_probability_mass / row_max_mass
                )
        else:
            eligible_blocks = target_count >= minimum_target_tokens
    else:
        # The threshold is relative to the probability mass of the top-p target
        # set, not to the full pre-softmax token set. Protected blocks bypass it.
        target_probability_total = target_probability_mass.sum(
            -1, keepdim=True, dtype=torch.float32
        )
        eligible_blocks = target_probability_mass >= (
            target_probability_total * minimum_block_target_probability_ratio
        )
        block_priority = target_probability_mass
    if (
        block_probability_top_p is None
        and block_target_probability_top_p is None
        and block_f_beta is None
    ):
        eligible_blocks &= causal_key_blocks.view(
            1, 1, num_query_blocks, num_key_blocks
        )
        eligible_blocks |= protected_blocks
        block_priority.masked_fill_(~eligible_blocks, float("-inf"))
        block_priority.masked_fill_(protected_blocks, float("inf"))
        if final_token_budget is None and maximum_selected_blocks is None:
            selected_key_blocks = eligible_blocks
        else:
            top_blocks = num_key_blocks
            if final_token_budget is not None:
                top_blocks = min(
                    top_blocks,
                    max(1, final_token_budget // key_block_size),
                )
            if maximum_selected_blocks is not None:
                top_blocks = min(top_blocks, maximum_selected_blocks)
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
    if target_probabilities is None:
        covered_target_probability_mass = covered_target_tokens.to(torch.float32)
    else:
        covered_target_probability_mass = (
            torch.where(
                selected_mask,
                target_probabilities,
                torch.zeros_like(target_probabilities),
            )
        ).sum(-1, dtype=torch.float32)
    selected_block_counts = selected_key_blocks.sum(-1, dtype=torch.int64)
    causal_block_counts = causal_key_blocks.sum(-1, dtype=torch.int64).view(
        1, 1, num_query_blocks
    ).expand(batch_size, num_groups, -1)
    return (
        selected_mask,
        selected_block_counts,
        causal_block_counts,
        covered_target_tokens,
        covered_target_probability_mass,
        block_selection_score,
    )


def _top_p_target_mask(
    probabilities: torch.Tensor,
    causal_tokens: torch.Tensor,
    *,
    top_p: float,
    minimum_tokens: int = 0,
    row_chunk_size: int = 16,
) -> torch.Tensor:
    """Select the minimal descending-probability prefix reaching ``top_p``."""

    if minimum_tokens < 0:
        raise ValueError("minimum_tokens must be non-negative")
    rows = probabilities.reshape(-1, probabilities.shape[-1])
    causal_rows = causal_tokens.expand_as(probabilities).reshape_as(rows)
    target_rows = torch.zeros_like(rows, dtype=torch.bool)
    for start in range(0, rows.shape[0], row_chunk_size):
        end = min(start + row_chunk_size, rows.shape[0])
        chunk = rows[start:end]
        causal_chunk = causal_rows[start:end]
        _, sorted_indices = torch.sort(
            chunk.masked_fill(~causal_chunk, -1.0), dim=-1, descending=True
        )
        sorted_probabilities = torch.gather(chunk, -1, sorted_indices)
        sorted_causal = torch.gather(causal_chunk, -1, sorted_indices)
        cumulative = sorted_probabilities.to(torch.float32).cumsum(dim=-1)
        # Using the mass before each token includes the first token that crosses
        # top_p, yielding the smallest prefix whose total mass reaches it.
        keep_sorted = (
            cumulative - sorted_probabilities.to(torch.float32)
        ) < top_p
        if minimum_tokens:
            ranks = torch.arange(chunk.shape[-1], device=chunk.device)
            keep_sorted |= ranks.view(1, -1) < minimum_tokens
        keep_sorted &= sorted_causal
        target_rows[start:end].scatter_(-1, sorted_indices, keep_sorted)
    target_mask = target_rows.view_as(probabilities)
    return target_mask & causal_tokens


def _causal_pair_count_per_selector_row(
    selected_mask: torch.Tensor,
    *,
    seq_len: int,
    query_block_size: int,
) -> torch.Tensor:
    """Count exact legal Q-K pairs for every group/query-tile selector row."""

    batch_size, num_groups, num_query_blocks, _ = selected_mask.shape
    prefix = torch.cumsum(selected_mask, dim=-1, dtype=torch.int32)
    query_positions = torch.arange(
        num_query_blocks * query_block_size, device=selected_mask.device
    ).view(num_query_blocks, query_block_size)
    valid_queries = query_positions < seq_len
    gather_positions = query_positions.clamp(max=seq_len - 1).view(
        1, 1, num_query_blocks, query_block_size
    ).expand(batch_size, num_groups, -1, -1)
    selected_per_query = torch.gather(prefix, -1, gather_positions)
    selected_per_query *= valid_queries.view(
        1, 1, num_query_blocks, query_block_size
    ).to(selected_per_query.dtype)
    return selected_per_query.sum(dim=-1, dtype=torch.int64)


class RepresentativeTokenFirstBlockSelector:
    """Create a token mask first, then cover it with fixed-size K blocks."""

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        logical_key_block_size: int | tuple[int, ...],
        target_token_budget: Optional[int],
        target_token_top_p: Optional[float],
        target_token_min_budget: Optional[int],
        target_top_p_start_layer: Optional[int],
        final_token_budget: Optional[int],
        maximum_selected_blocks: Optional[int],
        fill_final_token_budget: bool,
        minimum_block_coverage_ratio: float,
        minimum_block_target_probability_ratio: Optional[float],
        block_probability_top_p: Optional[float],
        block_target_probability_top_p: Optional[float],
        block_target_count_coverage_ratio: Optional[float],
        block_target_cost_aware: bool,
        block_f_beta: Optional[float],
        measure_fixed_target_probability_mass: bool,
        force_sink_block: bool,
        force_diagonal_block: bool,
        force_sink_target_tokens: bool = False,
        force_diagonal_target_tokens: bool = False,
        target_protected_token_span: int = 128,
        query_score_mode: str = "four_probe_weighted",
        probe_weights: tuple[float, float, float, float] = (
            0.2,
            0.3,
            0.4,
            0.1,
        ),
        adaptive_coverage_tolerance: float = 0.0,
        adaptive_kernel_cost_tolerance: float = 0.0,
        watched_key_ranges: tuple[tuple[int, int], ...] = (),
        force_watched_key_range_blocks: bool = False,
        profile_member_mask_fidelity: bool = False,
        project_target_to_blocks: bool = True,
        member_vertical_slash: bool = False,
        member_vertical_slash_gamma: float = 0.95,
        member_vertical_slash_min_tokens: int = 1024,
        member_vertical_slash_max_tokens: int = 2048,
        member_vertical_slash_head_chunk_size: int = 4,
        member_vertical_slash_layers: Optional[Sequence[int]] = None,
        residual_mode: str = "none",
        oracle_residual_tokens: int = 0,
        oracle_member_topk_budget: int = 8192,
        oracle_head_chunk_size: int = 4,
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
        if block_probability_top_p is None:
            if target_top_p_start_layer is None:
                if (target_token_budget is None) == (target_token_top_p is None):
                    raise ValueError(
                        "Exactly one of target_token_budget and "
                        "target_token_top_p is required"
                    )
            elif target_token_budget is None or target_token_top_p is None:
                raise ValueError(
                    "Layerwise target selection requires both a fixed token "
                    "budget and token top-p"
                )
        elif (
            target_token_budget is not None
            or target_token_top_p is not None
            or target_top_p_start_layer is not None
        ):
            raise ValueError(
                "Block probability top-p cannot be combined with a token target budget"
            )
        if target_token_budget is not None and target_token_budget <= 0:
            raise ValueError("target_token_budget must be positive")
        if target_token_top_p is not None and not 0.0 < target_token_top_p <= 1.0:
            raise ValueError("target_token_top_p must be in (0, 1]")
        if target_token_min_budget is not None and target_token_min_budget <= 0:
            raise ValueError("target_token_min_budget must be positive")
        if target_token_min_budget is not None and target_token_top_p is None:
            raise ValueError(
                "target_token_min_budget requires probability Top-p selection"
            )
        if target_top_p_start_layer is not None and target_top_p_start_layer < 0:
            raise ValueError("target_top_p_start_layer must be non-negative")
        if final_token_budget is not None and final_token_budget <= 0:
            raise ValueError("final_token_budget must be positive")
        if target_protected_token_span <= 0:
            raise ValueError("target_protected_token_span must be positive")
        if (
            force_sink_target_tokens or force_diagonal_target_tokens
        ) and target_token_budget is None:
            raise ValueError(
                "Target-token protection requires a fixed target_token_budget"
            )
        if maximum_selected_blocks is not None and maximum_selected_blocks <= 0:
            raise ValueError("maximum_selected_blocks must be positive")
        if fill_final_token_budget and final_token_budget is None:
            raise ValueError(
                "fill_final_token_budget requires a finite final_token_budget"
            )
        if block_probability_top_p is not None and not (
            0.0 < block_probability_top_p <= 1.0
        ):
            raise ValueError("block_probability_top_p must be in (0, 1]")
        if block_target_probability_top_p is not None and not (
            0.0 < block_target_probability_top_p <= 1.0
        ):
            raise ValueError(
                "block_target_probability_top_p must be in (0, 1]"
            )
        if block_target_count_coverage_ratio is not None and not (
            0.0 < block_target_count_coverage_ratio <= 1.0
        ):
            raise ValueError(
                "block_target_count_coverage_ratio must be in (0, 1]"
            )
        if (
            block_target_count_coverage_ratio is not None
            and block_target_probability_top_p is None
        ):
            raise ValueError(
                "Target-count coverage requires target-probability block selection"
            )
        if block_f_beta is not None and block_f_beta <= 0.0:
            raise ValueError("block_f_beta must be positive")
        block_objectives = sum(
            option is not None
            for option in (
                block_probability_top_p,
                block_target_probability_top_p,
                block_f_beta,
            )
        )
        if block_objectives > 1:
            raise ValueError(
                "Block probability top-p, target-mass top-p, and F-beta are "
                "mutually exclusive"
            )
        if fill_final_token_budget and block_objectives:
            raise ValueError(
                "Matched-budget filling cannot be combined with another block objective"
            )
        if not 0.0 < minimum_block_coverage_ratio <= 1.0:
            raise ValueError("minimum_block_coverage_ratio must be in (0, 1]")
        if minimum_block_target_probability_ratio is not None and not (
            0.0 < minimum_block_target_probability_ratio <= 1.0
        ):
            raise ValueError(
                "minimum_block_target_probability_ratio must be in (0, 1]"
            )
        if not 0.0 <= adaptive_coverage_tolerance <= 1.0:
            raise ValueError("adaptive_coverage_tolerance must be in [0, 1]")
        if not 0.0 <= adaptive_kernel_cost_tolerance <= 1.0:
            raise ValueError(
                "adaptive_kernel_cost_tolerance must be in [0, 1]"
            )
        if query_score_mode not in {"four_probe_weighted", "full_query_mean"}:
            raise ValueError(
                "query_score_mode must be 'four_probe_weighted' or "
                "'full_query_mean'"
            )
        if len(probe_weights) != 4 or any(weight < 0 for weight in probe_weights):
            raise ValueError("probe_weights must contain four non-negative values")
        if not math.isclose(sum(probe_weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probe_weights must sum to 1")
        if not 0.0 < member_vertical_slash_gamma <= 1.0:
            raise ValueError("member_vertical_slash_gamma must be in (0, 1]")
        if member_vertical_slash_min_tokens <= 0:
            raise ValueError("member_vertical_slash_min_tokens must be positive")
        if member_vertical_slash_max_tokens < member_vertical_slash_min_tokens:
            raise ValueError(
                "member_vertical_slash_max_tokens must be at least the minimum"
            )
        if member_vertical_slash_head_chunk_size <= 0:
            raise ValueError(
                "member_vertical_slash_head_chunk_size must be positive"
            )
        if residual_mode not in {"none", "oracle"}:
            raise ValueError("residual_mode must be 'none' or 'oracle'")
        if oracle_residual_tokens < 0:
            raise ValueError("oracle_residual_tokens must be non-negative")
        if residual_mode == "none" and oracle_residual_tokens:
            raise ValueError("oracle_residual_tokens requires residual_mode='oracle'")
        if oracle_member_topk_budget <= 0:
            raise ValueError("oracle_member_topk_budget must be positive")
        if oracle_head_chunk_size <= 0:
            raise ValueError("oracle_head_chunk_size must be positive")
        self.layers = group_config["layers"]
        self.logical_key_block_sizes = tuple(sorted(logical_key_block_sizes))
        self.target_token_budget = (
            int(target_token_budget) if target_token_budget is not None else None
        )
        self.target_token_top_p = (
            float(target_token_top_p) if target_token_top_p is not None else None
        )
        self.target_token_min_budget = (
            int(target_token_min_budget)
            if target_token_min_budget is not None
            else None
        )
        self.target_top_p_start_layer = (
            int(target_top_p_start_layer)
            if target_top_p_start_layer is not None
            else None
        )
        self.final_token_budget = (
            int(final_token_budget) if final_token_budget is not None else None
        )
        self.maximum_selected_blocks = (
            int(maximum_selected_blocks)
            if maximum_selected_blocks is not None
            else None
        )
        self.fill_final_token_budget = bool(fill_final_token_budget)
        self.minimum_block_coverage_ratio = float(minimum_block_coverage_ratio)
        self.minimum_block_target_probability_ratio = (
            float(minimum_block_target_probability_ratio)
            if minimum_block_target_probability_ratio is not None
            else None
        )
        self.block_probability_top_p = block_probability_top_p
        self.block_target_probability_top_p = block_target_probability_top_p
        self.block_target_count_coverage_ratio = (
            block_target_count_coverage_ratio
        )
        self.block_target_cost_aware = bool(block_target_cost_aware)
        self.block_f_beta = block_f_beta
        self.measure_fixed_target_probability_mass = bool(
            measure_fixed_target_probability_mass
        )
        self.force_sink_block = bool(force_sink_block)
        self.force_diagonal_block = bool(force_diagonal_block)
        self.force_sink_target_tokens = bool(force_sink_target_tokens)
        self.force_diagonal_target_tokens = bool(
            force_diagonal_target_tokens
        )
        self.target_protected_token_span = int(target_protected_token_span)
        self.query_score_mode = str(query_score_mode)
        self.probe_weights = tuple(float(weight) for weight in probe_weights)
        self.adaptive_coverage_tolerance = float(adaptive_coverage_tolerance)
        self.adaptive_kernel_cost_tolerance = float(
            adaptive_kernel_cost_tolerance
        )
        normalized_ranges = []
        for start, end in watched_key_ranges:
            if start < 0 or end <= start:
                raise ValueError(
                    "Watched key ranges must be non-negative half-open intervals"
                )
            normalized_ranges.append((int(start), int(end)))
        self.watched_key_ranges = tuple(normalized_ranges)
        self.force_watched_key_range_blocks = bool(
            force_watched_key_range_blocks
        )
        self.profile_member_mask_fidelity = bool(profile_member_mask_fidelity)
        self.project_target_to_blocks = bool(project_target_to_blocks)
        self.member_vertical_slash = bool(member_vertical_slash)
        self.member_vertical_slash_gamma = float(member_vertical_slash_gamma)
        self.member_vertical_slash_min_tokens = int(
            member_vertical_slash_min_tokens
        )
        self.member_vertical_slash_max_tokens = int(
            member_vertical_slash_max_tokens
        )
        self.member_vertical_slash_head_chunk_size = int(
            member_vertical_slash_head_chunk_size
        )
        self.member_vertical_slash_layers = (
            frozenset(int(layer) for layer in member_vertical_slash_layers)
            if member_vertical_slash_layers is not None
            else None
        )
        self.residual_mode = str(residual_mode)
        self.oracle_residual_tokens = int(oracle_residual_tokens)
        self.oracle_member_topk_budget = int(oracle_member_topk_budget)
        self.oracle_head_chunk_size = int(oracle_head_chunk_size)
        self.oracle_residual_dump_dir: Optional[Path] = None
        self.oracle_residual_dump_sample_limit = 0
        self.vertical_slash_get_active_blocks = None
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("token_first_block_sparse_layer", default=None)
        )
        self.stats = TokenCompactedSelectorStats()
        self.ranked_probability_dump_dir: Optional[Path] = None
        self._ranked_probability_dumped_layers: set[int] = set()

    def _score_representative_keys(
        self,
        representative_q: torch.Tensor,
        representative_k: torch.Tensor,
        *,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score every key for each query tile with the configured Q summary."""

        batch_size, seq_len, num_groups, head_dim = representative_q.shape
        pooled_q = _block_mean(representative_q, block_size)
        num_query_blocks = pooled_q.shape[1]
        query_starts = torch.arange(
            num_query_blocks, device=representative_q.device
        ) * block_size
        query_ends = (query_starts + block_size).clamp(max=seq_len)
        query_lengths = (query_ends - query_starts).clamp_min(1)

        token_scores = torch.einsum(
            "bqhd,bthd->bhqt", pooled_q, representative_k
        ) / math.sqrt(head_dim)
        if self.query_score_mode == "four_probe_weighted":
            probe_positions = query_starts[:, None] + torch.stack(
                (
                    (query_lengths - 1) // 3,
                    2 * (query_lengths - 1) // 3,
                    query_lengths - 1,
                ),
                dim=-1,
            )
            token_scores.mul_(self.probe_weights[3])
            for probe_idx in range(3):
                probe_q = representative_q.index_select(
                    1, probe_positions[:, probe_idx]
                )
                probe_scores = torch.einsum(
                    "bqhd,bthd->bhqt", probe_q, representative_k
                ) / math.sqrt(head_dim)
                token_scores.add_(
                    probe_scores, alpha=self.probe_weights[probe_idx]
                )
            return token_scores, query_starts, query_ends, query_lengths

        # For keys preceding a tile, mean(Q) @ K exactly equals the arithmetic
        # mean of every query-token QK logit. Inside the diagonal tile, a key is
        # legal only for its suffix of query rows, so replace those entries with
        # suffix-mean QK scores. This is mathematically identical to materializing
        # all 128 x sequence_length logits, without the prohibitive 5-D tensor.
        padded_seq_len = num_query_blocks * block_size
        pad_tokens = padded_seq_len - seq_len
        if pad_tokens:
            representative_q_padded = torch.nn.functional.pad(
                representative_q, (0, 0, 0, 0, 0, pad_tokens)
            )
            representative_k_padded = torch.nn.functional.pad(
                representative_k, (0, 0, 0, 0, 0, pad_tokens)
            )
        else:
            representative_q_padded = representative_q
            representative_k_padded = representative_k
        query_blocks = representative_q_padded.view(
            batch_size, num_query_blocks, block_size, num_groups, head_dim
        )
        key_blocks = representative_k_padded.view(
            batch_size, num_query_blocks, block_size, num_groups, head_dim
        )
        positions = torch.arange(
            padded_seq_len, device=representative_q.device
        ).view(num_query_blocks, block_size)
        valid_positions = positions < seq_len
        reversed_queries = torch.flip(query_blocks.to(torch.float32), dims=(2,))
        suffix_sums = torch.flip(
            torch.cumsum(reversed_queries, dim=2), dims=(2,)
        )
        suffix_counts = torch.flip(
            torch.cumsum(
                torch.flip(valid_positions.to(torch.float32), dims=(1,)), dim=1
            ),
            dims=(1,),
        ).clamp_min(1.0)
        suffix_means = suffix_sums / suffix_counts.view(
            1, num_query_blocks, block_size, 1, 1
        )
        diagonal_scores = torch.einsum(
            "bqphd,bqphd->bhqp",
            suffix_means.to(representative_k.dtype),
            key_blocks,
        ) / math.sqrt(head_dim)
        diagonal_scores.masked_fill_(
            ~valid_positions.view(1, 1, num_query_blocks, block_size),
            float("-inf"),
        )
        if pad_tokens:
            token_scores = torch.nn.functional.pad(
                token_scores, (0, pad_tokens), value=float("-inf")
            )
        diagonal_indices = positions.view(
            1, 1, num_query_blocks, block_size
        ).expand(batch_size, num_groups, -1, -1)
        token_scores.scatter_(-1, diagonal_indices, diagonal_scores)
        return (
            token_scores[..., :seq_len],
            query_starts,
            query_ends,
            query_lengths,
        )

    def configure_oracle_residual_dump(
        self, dump_dir: str | Path, *, sample_limit: int = 1
    ) -> None:
        if sample_limit <= 0:
            raise ValueError("Oracle residual dump sample_limit must be positive")
        self.oracle_residual_dump_dir = Path(dump_dir).resolve()
        self.oracle_residual_dump_dir.mkdir(parents=True, exist_ok=True)
        self.oracle_residual_dump_sample_limit = int(sample_limit)

    def configure_ranked_probability_dump(self, dump_dir: str | Path) -> None:
        self.ranked_probability_dump_dir = Path(dump_dir).resolve()
        self.ranked_probability_dump_dir.mkdir(parents=True, exist_ok=True)
        self._ranked_probability_dumped_layers.clear()

    @torch.no_grad()
    def dump_ranked_probability_distribution(
        self,
        q_tile: torch.Tensor,
        k: torch.Tensor,
        *,
        layer_idx: int,
        gqa_interleave: bool = False,
        source: str,
    ) -> None:
        dump_dir = self.ranked_probability_dump_dir
        if dump_dir is None or layer_idx in self._ranked_probability_dumped_layers:
            return
        if q_tile.shape[0] != 1:
            raise ValueError("Ranked probability profiling requires batch size one")

        groups = self.layers[str(layer_idx)]
        _, tile_length, num_q_heads, head_dim = q_tile.shape
        seq_len = k.shape[1]
        num_kv_heads = k.shape[2]
        if num_q_heads % num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        num_share_q_heads = num_q_heads // num_kv_heads
        representatives = torch.tensor(
            [int(group["representative"]) for group in groups],
            dtype=torch.long,
            device=q_tile.device,
        )
        if gqa_interleave:
            kv_indices = representatives % num_kv_heads
        else:
            kv_indices = representatives // num_share_q_heads
        representative_q = q_tile.index_select(2, representatives)
        representative_k = k.index_select(2, kv_indices)

        pooled_q = representative_q.mean(dim=1)
        probe_positions = torch.tensor(
            [
                (tile_length - 1) // 3,
                2 * (tile_length - 1) // 3,
                tile_length - 1,
            ],
            dtype=torch.long,
            device=q_tile.device,
        )
        scores = torch.einsum(
            "bhd,bthd->bht", pooled_q, representative_k
        ) / math.sqrt(head_dim)
        scores.mul_(self.probe_weights[3])
        for probe_idx in range(3):
            probe_q = representative_q[:, probe_positions[probe_idx]]
            probe_scores = torch.einsum(
                "bhd,bthd->bht", probe_q, representative_k
            ) / math.sqrt(head_dim)
            scores.add_(probe_scores, alpha=self.probe_weights[probe_idx])

        probabilities = torch.softmax(scores.to(torch.float32), dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, dim=-1, descending=True
        )
        payload = {
            "layer": int(layer_idx),
            "source": source,
            "sequence_length": int(seq_len),
            "query_tile_start": int(seq_len - tile_length),
            "query_tile_end": int(seq_len),
            "query_tile_length": int(tile_length),
            "probe_offsets_in_tile": probe_positions.cpu(),
            "probe_weights": torch.tensor(self.probe_weights, dtype=torch.float32),
            "representatives": representatives.cpu(),
            "groups": [
                {
                    "representative": int(group["representative"]),
                    "members": [int(member) for member in group["members"]],
                }
                for group in groups
            ],
            "watched_key_ranges": list(self.watched_key_ranges),
            "sorted_probabilities": sorted_probabilities[0].cpu(),
            "sorted_key_indices": sorted_indices[0].to(torch.int32).cpu(),
        }
        output_path = dump_dir / f"layer_{layer_idx:02d}.pt"
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        self._ranked_probability_dumped_layers.add(layer_idx)

    def _build_member_vertical_slash_index(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_mask: torch.Tensor,
        target_mask: torch.Tensor,
        token_probabilities: Optional[torch.Tensor],
        causal_tokens: torch.Tensor,
        chosen_block_sizes: torch.Tensor,
        head_to_group: torch.Tensor,
        query_ends: torch.Tensor,
        gqa_interleave: bool,
    ) -> tuple[TokenCompactedIndex, Dict[str, Any]]:
        """Union per-Q-head FlexPrefill Vertical/Slash blocks with AutoBlock."""

        if self.vertical_slash_get_active_blocks is None:
            raise RuntimeError("Original FlexPrefill selector was not captured")
        batch_size, seq_len, num_q_heads, _ = q.shape
        num_query_blocks = selected_mask.shape[2]
        num_structural_blocks = math.ceil(seq_len / 128)
        vertical_slash_blocks = self.vertical_slash_get_active_blocks(
            q,
            k,
            v,
            128,
            self.member_vertical_slash_gamma,
            math.ceil(self.member_vertical_slash_min_tokens / 128),
            math.ceil(self.member_vertical_slash_max_tokens / 128),
            -1.0,
            gqa_interleave,
        )

        base_group_for_head = head_to_group.to(torch.long)
        row_starts_chunks: List[torch.Tensor] = []
        row_ends_chunks: List[torch.Tensor] = []
        token_index_chunks: List[torch.Tensor] = []
        token_offset = 0
        selected_blocks = 0
        selected_token_pairs = 0
        compacted_key_tokens = 0
        target_mask_tokens = 0
        covered_target_tokens = 0
        target_probability_mass = 0.0
        covered_target_probability_mass = 0.0
        vs_candidate_blocks = 0
        vs_added_blocks = 0
        vs_added_key_tokens = 0
        watched_key_range_stats: Dict[str, Dict[str, float]] = {}

        pad_tokens = num_structural_blocks * 128 - seq_len
        for head_start in range(
            0, num_q_heads, self.member_vertical_slash_head_chunk_size
        ):
            head_end = min(
                head_start + self.member_vertical_slash_head_chunk_size,
                num_q_heads,
            )
            group_indices = base_group_for_head[head_start:head_end]
            base_chunk = selected_mask.index_select(1, group_indices)
            final_chunk = base_chunk.clone()
            rescue_blocks = torch.zeros(
                (
                    batch_size,
                    head_end - head_start,
                    num_query_blocks,
                    num_structural_blocks,
                ),
                dtype=torch.bool,
                device=q.device,
            )
            for batch_idx in range(batch_size):
                for local_head, global_head in enumerate(
                    range(head_start, head_end)
                ):
                    indices = vertical_slash_blocks[batch_idx][global_head]
                    indices = indices[
                        (indices >= 0)
                        & (indices < num_query_blocks * num_structural_blocks)
                    ]
                    rescue_blocks[batch_idx, local_head].view(-1)[indices] = True

            if pad_tokens:
                padded_base = torch.nn.functional.pad(
                    base_chunk, (0, pad_tokens), value=False
                )
            else:
                padded_base = base_chunk
            base_blocks = padded_base.view(
                batch_size,
                head_end - head_start,
                num_query_blocks,
                num_structural_blocks,
                128,
            ).any(dim=-1)
            rescue_tokens = rescue_blocks.repeat_interleave(128, dim=-1)[
                ..., :seq_len
            ]
            final_chunk |= rescue_tokens
            final_chunk &= causal_tokens

            vs_candidate_blocks += int(rescue_blocks.sum(dtype=torch.int64).item())
            vs_added_blocks += int(
                (rescue_blocks & ~base_blocks).sum(dtype=torch.int64).item()
            )
            vs_added_key_tokens += int(
                (final_chunk & ~base_chunk).sum(dtype=torch.int64).item()
            )
            final_padded = (
                torch.nn.functional.pad(final_chunk, (0, pad_tokens), value=False)
                if pad_tokens
                else final_chunk
            )
            selected_blocks += int(
                final_padded.view(
                    batch_size,
                    head_end - head_start,
                    num_query_blocks,
                    num_structural_blocks,
                    128,
                )
                .any(dim=-1)
                .sum(dtype=torch.int64)
                .item()
            )
            selected_token_pairs += int(
                _causal_pair_count_per_selector_row(
                    final_chunk,
                    seq_len=seq_len,
                    query_block_size=128,
                )
                .sum(dtype=torch.int64)
                .item()
            )
            compacted_key_tokens += int(
                final_chunk.sum(dtype=torch.int64).item()
            )

            target_chunk = target_mask.index_select(1, group_indices)
            target_mask_tokens += int(
                target_chunk.sum(dtype=torch.int64).item()
            )
            covered_target_tokens += int(
                (target_chunk & final_chunk).sum(dtype=torch.int64).item()
            )
            probability_chunk = (
                token_probabilities.index_select(1, group_indices)
                if token_probabilities is not None
                else None
            )
            if probability_chunk is not None:
                target_probability_mass += float(
                    torch.where(
                        target_chunk,
                        probability_chunk,
                        torch.zeros_like(probability_chunk),
                    )
                    .sum(dtype=torch.float64)
                    .item()
                )
                covered_target_probability_mass += float(
                    torch.where(
                        target_chunk & final_chunk,
                        probability_chunk,
                        torch.zeros_like(probability_chunk),
                    )
                    .sum(dtype=torch.float64)
                    .item()
                )

            for watched_start, watched_end in self.watched_key_ranges:
                clipped_start = min(watched_start, seq_len)
                clipped_end = min(watched_end, seq_len)
                if clipped_end <= clipped_start:
                    continue
                range_name = f"{watched_start}:{watched_end}"
                values = watched_key_range_stats.setdefault(
                    range_name,
                    {
                        "probability_mass": 0.0,
                        "selector_rows": 0.0,
                        "valid_token_slots": 0.0,
                        "target_token_slots": 0.0,
                        "selected_token_slots": 0.0,
                        "sink_token_slots": 0.0,
                    },
                )
                watched_valid = causal_tokens[
                    ..., clipped_start:clipped_end
                ].expand(
                    batch_size,
                    head_end - head_start,
                    -1,
                    -1,
                )
                watched_target = target_chunk[..., clipped_start:clipped_end]
                watched_selected = final_chunk[..., clipped_start:clipped_end]
                if probability_chunk is not None:
                    values["probability_mass"] += float(
                        (
                            probability_chunk[..., clipped_start:clipped_end]
                            * watched_valid
                        )
                        .sum(dtype=torch.float64)
                        .item()
                    )
                values["selector_rows"] += float(
                    watched_valid.any(dim=-1).sum(dtype=torch.int64).item()
                )
                values["valid_token_slots"] += float(
                    watched_valid.sum(dtype=torch.int64).item()
                )
                values["target_token_slots"] += float(
                    (watched_target & watched_valid)
                    .sum(dtype=torch.int64)
                    .item()
                )
                values["selected_token_slots"] += float(
                    (watched_selected & watched_valid)
                    .sum(dtype=torch.int64)
                    .item()
                )
                watched_positions = torch.arange(
                    clipped_start, clipped_end, device=q.device
                )
                watched_sink = watched_positions.view(1, 1, 1, -1) < (
                    chosen_block_sizes.index_select(1, group_indices).unsqueeze(-1)
                )
                values["sink_token_slots"] += float(
                    (watched_sink & watched_valid)
                    .sum(dtype=torch.int64)
                    .item()
                )

            rows = final_chunk.reshape(-1, seq_len)
            row_counts = rows.sum(dim=-1, dtype=torch.int32)
            row_ends_flat = (
                torch.cumsum(row_counts, dim=0, dtype=torch.int32) + token_offset
            )
            row_starts_flat = row_ends_flat - row_counts
            chunk_token_indices = torch.nonzero(rows, as_tuple=False)[:, 1].to(
                torch.int32
            )
            row_starts_chunks.append(
                row_starts_flat.view(
                    batch_size, head_end - head_start, num_query_blocks
                )
            )
            row_ends_chunks.append(
                row_ends_flat.view(
                    batch_size, head_end - head_start, num_query_blocks
                )
            )
            token_index_chunks.append(chunk_token_indices)
            token_offset += int(chunk_token_indices.numel())

        total_causal_blocks = int(
            batch_size
            * num_q_heads
            * num_structural_blocks
            * (num_structural_blocks + 1)
            // 2
        )
        total_causal_token_pairs = int(
            batch_size * num_q_heads * seq_len * (seq_len + 1) // 2
        )
        candidate_key_tokens = int(
            batch_size * num_q_heads * query_ends.sum(dtype=torch.int64).item()
        )
        index = TokenCompactedIndex(
            row_starts=torch.cat(row_starts_chunks, dim=1).contiguous(),
            row_ends=torch.cat(row_ends_chunks, dim=1).contiguous(),
            token_indices=torch.cat(token_index_chunks, dim=0).contiguous(),
            head_to_group=torch.arange(
                num_q_heads, dtype=torch.int32, device=q.device
            ),
            num_groups=num_q_heads,
            num_query_blocks=num_query_blocks,
            query_block_size=128,
        )
        return index, {
            "selected_blocks": selected_blocks,
            "causal_blocks": total_causal_blocks,
            "selected_token_pairs": selected_token_pairs,
            "causal_token_pairs": total_causal_token_pairs,
            "compacted_key_tokens": compacted_key_tokens,
            "candidate_key_tokens": candidate_key_tokens,
            "target_mask_tokens": target_mask_tokens,
            "covered_target_tokens": covered_target_tokens,
            "target_probability_mass": target_probability_mass,
            "covered_target_probability_mass": covered_target_probability_mass,
            "watched_key_range_stats": watched_key_range_stats,
            "vertical_slash_candidate_blocks": vs_candidate_blocks,
            "vertical_slash_added_blocks": vs_added_blocks,
            "vertical_slash_added_key_tokens": vs_added_key_tokens,
        }

    def _build_member_oracle_residual_index(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        selected_mask: torch.Tensor,
        target_mask: torch.Tensor,
        representative_scores: torch.Tensor,
        causal_tokens: torch.Tensor,
        chosen_block_sizes: torch.Tensor,
        head_to_group: torch.Tensor,
        query_starts: torch.Tensor,
        query_ends: torch.Tensor,
        probe_positions: torch.Tensor,
        gqa_interleave: bool,
        layer_idx: int,
    ) -> tuple[TokenCompactedIndex, Dict[str, Any]]:
        """Add exact member-head oracle residual tokens to the shared mask.

        Dense member QK scores are used only to rank residual targets. The
        returned CSR index contains the shared projected mask plus exact
        head-specific residual tokens, so the sparse kernel remains unchanged.
        """

        groups = self.layers[str(layer_idx)]
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        num_query_blocks = selected_mask.shape[2]
        num_share_q_heads = num_q_heads // num_kv_heads
        num_structural_blocks = math.ceil(seq_len / 128)
        residual_k = min(self.oracle_residual_tokens, seq_len)
        member_top_k = min(self.oracle_member_topk_budget, seq_len)
        base_group_for_head = head_to_group.to(torch.long)
        representative_for_head = torch.tensor(
            [int(groups[int(group_idx)]["representative"]) for group_idx in base_group_for_head.tolist()],
            dtype=torch.long,
            device=q.device,
        )

        row_starts_chunks: List[torch.Tensor] = []
        row_ends_chunks: List[torch.Tensor] = []
        token_index_chunks: List[torch.Tensor] = []
        token_offset = 0
        selected_blocks = 0
        selected_token_pairs = 0
        compacted_key_tokens = 0
        target_mask_tokens = 0
        covered_target_tokens = 0
        oracle_stats: Dict[int, Dict[str, Any]] = {}
        watched_key_range_stats: Dict[str, Dict[str, float]] = {}
        dump_rows: List[Dict[str, Any]] = []
        sample_index = self.stats.calls // max(len(self.layers), 1)
        dump_this_layer = (
            self.oracle_residual_dump_dir is not None
            and sample_index < self.oracle_residual_dump_sample_limit
            and layer_idx in {0, 8, 16, 24, len(self.layers) - 1}
        )
        sampled_heads = set()
        if dump_this_layer:
            for group in groups:
                representative = int(group["representative"])
                sampled_heads.add(representative)
                sampled_heads.update(
                    [
                        int(member)
                        for member in group["members"]
                        if int(member) != representative
                    ][:1]
                )
        sampled_query_tiles = {
            max(num_query_blocks // 4, 0),
            max(num_query_blocks // 2, 0),
            max(num_query_blocks - 1, 0),
        }

        pad_tokens = num_structural_blocks * 128 - seq_len
        for head_start in range(0, num_q_heads, self.oracle_head_chunk_size):
            head_end = min(head_start + self.oracle_head_chunk_size, num_q_heads)
            head_ids = torch.arange(head_start, head_end, device=q.device)
            group_indices = base_group_for_head[head_start:head_end]
            shared_target = target_mask.index_select(1, group_indices)
            base_chunk = selected_mask.index_select(1, group_indices)
            member_q = q[:, :, head_start:head_end]
            if gqa_interleave:
                member_kv_indices = head_ids % num_kv_heads
            else:
                member_kv_indices = head_ids // num_share_q_heads
            member_k = k.index_select(2, member_kv_indices)

            # Oracle-only dense selector scores: every member Q head ranks all
            # causally valid keys using the same four AutoBlock probes.
            member_pooled_q = _block_mean(member_q, 128)
            member_scores = torch.einsum(
                "bqhd,bthd->bhqt", member_pooled_q, member_k
            ) / math.sqrt(head_dim)
            member_scores.mul_(self.probe_weights[3])
            for probe_idx in range(3):
                member_probe_q = member_q.index_select(
                    1, probe_positions[:, probe_idx]
                )
                member_probe_scores = torch.einsum(
                    "bqhd,bthd->bhqt", member_probe_q, member_k
                ) / math.sqrt(head_dim)
                member_scores.add_(
                    member_probe_scores, alpha=self.probe_weights[probe_idx]
                )
            causal_chunk = causal_tokens.expand(
                batch_size, head_end - head_start, -1, -1
            )
            member_scores.masked_fill_(~causal_chunk, float("-inf"))

            _, member_top_indices = torch.topk(
                member_scores, k=member_top_k, dim=-1, sorted=False
            )
            member_top_counts = torch.minimum(
                query_ends,
                query_ends.new_full((), member_top_k),
            )
            member_top_ranks = torch.arange(
                member_top_k, device=q.device
            ).view(1, 1, 1, member_top_k)
            member_top_keep = member_top_ranks < member_top_counts.view(
                1, 1, num_query_blocks, 1
            )
            member_top_mask = torch.zeros_like(member_scores, dtype=torch.bool)
            member_top_mask.scatter_(
                -1,
                member_top_indices,
                member_top_keep.expand_as(member_top_indices),
            )
            member_top_mask &= causal_chunk

            residual_mask = torch.zeros_like(member_top_mask)
            if residual_k:
                residual_scores = member_scores.clone()
                residual_scores.masked_fill_(
                    shared_target | base_chunk | ~causal_chunk,
                    float("-inf"),
                )
                _, residual_indices = torch.topk(
                    residual_scores, k=residual_k, dim=-1, sorted=False
                )
                available_counts = (
                    causal_chunk & ~shared_target & ~base_chunk
                ).sum(-1, dtype=torch.int64)
                residual_keep_counts = torch.minimum(
                    available_counts,
                    available_counts.new_full((), residual_k),
                )
                residual_ranks = torch.arange(
                    residual_k, device=q.device
                ).view(1, 1, 1, residual_k)
                residual_keep = residual_ranks < residual_keep_counts.unsqueeze(-1)
                residual_mask.scatter_(
                    -1,
                    residual_indices,
                    residual_keep.expand_as(residual_indices),
                )
                is_representative = head_ids == representative_for_head[
                    head_start:head_end
                ]
                residual_mask &= ~is_representative.view(1, -1, 1, 1)
                residual_mask &= causal_chunk

            final_target = shared_target | residual_mask
            final_chunk = (base_chunk | residual_mask) & causal_chunk
            target_mask_tokens += int(final_target.sum(dtype=torch.int64).item())
            covered_target_tokens += int(
                (final_target & final_chunk).sum(dtype=torch.int64).item()
            )
            compacted_key_tokens += int(final_chunk.sum(dtype=torch.int64).item())
            selected_token_pairs += int(
                _causal_pair_count_per_selector_row(
                    final_chunk, seq_len=seq_len, query_block_size=128
                ).sum(dtype=torch.int64).item()
            )
            final_padded = (
                torch.nn.functional.pad(final_chunk, (0, pad_tokens), value=False)
                if pad_tokens
                else final_chunk
            )
            selected_blocks += int(
                final_padded.view(
                    batch_size,
                    head_end - head_start,
                    num_query_blocks,
                    num_structural_blocks,
                    128,
                ).any(dim=-1).sum(dtype=torch.int64).item()
            )

            representative_last = representative_scores.index_select(
                1, group_indices
            )[:, :, -1]
            member_last = member_scores[:, :, -1]
            score_valid = causal_chunk[:, :, -1]
            score_difference = torch.where(
                score_valid,
                (member_last - representative_last).abs(),
                torch.zeros_like(member_last),
            )
            for local_head, global_head in enumerate(range(head_start, head_end)):
                group_idx = int(group_indices[local_head].item())
                representative_head = int(groups[group_idx]["representative"])
                member_top = member_top_mask[:, local_head : local_head + 1]
                shared = shared_target[:, local_head : local_head + 1]
                shared_kernel = base_chunk[:, local_head : local_head + 1]
                residual = residual_mask[:, local_head : local_head + 1]
                target_final = final_target[:, local_head : local_head + 1]
                kernel_final = final_chunk[:, local_head : local_head + 1]
                oracle_stats[global_head] = {
                    "group_index": group_idx,
                    "representative_head": representative_head,
                    "kv_head": int(member_kv_indices[local_head].item()),
                    "is_representative": int(global_head == representative_head),
                    "selector_rows": batch_size * num_query_blocks,
                    "member_top8192_tokens": int(
                        member_top.sum(dtype=torch.int64).item()
                    ),
                    "shared_target_tokens": int(
                        shared.sum(dtype=torch.int64).item()
                    ),
                    "shared_kernel_tokens": int(
                        shared_kernel.sum(dtype=torch.int64).item()
                    ),
                    "shared_overlap_tokens": int(
                        (member_top & shared).sum(dtype=torch.int64).item()
                    ),
                    "shared_kernel_overlap_tokens": int(
                        (member_top & shared_kernel).sum(dtype=torch.int64).item()
                    ),
                    "residual_tokens": int(
                        residual.sum(dtype=torch.int64).item()
                    ),
                    "residual_new_hits": int(
                        (member_top & residual).sum(dtype=torch.int64).item()
                    ),
                    "final_target_tokens": int(
                        target_final.sum(dtype=torch.int64).item()
                    ),
                    "final_kernel_tokens": int(
                        kernel_final.sum(dtype=torch.int64).item()
                    ),
                    "final_overlap_tokens": int(
                        (member_top & target_final).sum(dtype=torch.int64).item()
                    ),
                    "kernel_overlap_tokens": int(
                        (member_top & kernel_final).sum(dtype=torch.int64).item()
                    ),
                    "residual_shared_overlap_tokens": int(
                        (residual & shared).sum(dtype=torch.int64).item()
                    ),
                    "residual_kernel_missing_tokens": int(
                        (residual & ~kernel_final).sum(dtype=torch.int64).item()
                    ),
                    "residual_causal_violation_tokens": int(
                        (residual & ~causal_chunk[:, local_head : local_head + 1])
                        .sum(dtype=torch.int64)
                        .item()
                    ),
                    "score_abs_difference_sum": float(
                        score_difference[:, local_head].sum(dtype=torch.float64).item()
                    ),
                    "score_abs_difference_count": int(
                        score_valid[:, local_head].sum(dtype=torch.int64).item()
                    ),
                }

                if dump_this_layer and global_head in sampled_heads:
                    for query_tile in sorted(sampled_query_tiles):
                        if query_tile >= num_query_blocks:
                            continue
                        dump_rows.append(
                            {
                                "sample_index": int(sample_index),
                                "layer": int(layer_idx),
                                "group": group_idx,
                                "head": int(global_head),
                                "representative_head": representative_head,
                                "is_representative": int(
                                    global_head == representative_head
                                ),
                                "query_tile": int(query_tile),
                                "query_start": int(query_starts[query_tile].item()),
                                "query_end": int(query_ends[query_tile].item()),
                                "shared_indices": torch.nonzero(
                                    shared[0, 0, query_tile], as_tuple=False
                                )[:, 0].to(torch.int32).cpu(),
                                "shared_kernel_indices": torch.nonzero(
                                    base_chunk[0, local_head, query_tile],
                                    as_tuple=False,
                                )[:, 0].to(torch.int32).cpu(),
                                "oracle_residual_indices": torch.nonzero(
                                    residual[0, 0, query_tile], as_tuple=False
                                )[:, 0].to(torch.int32).cpu(),
                                "member_top8192_indices": torch.nonzero(
                                    member_top[0, 0, query_tile], as_tuple=False
                                )[:, 0].to(torch.int32).cpu(),
                                "final_kernel_indices": torch.nonzero(
                                    kernel_final[0, 0, query_tile], as_tuple=False
                                )[:, 0].to(torch.int32).cpu(),
                            }
                        )

            for watched_start, watched_end in self.watched_key_ranges:
                clipped_start = min(watched_start, seq_len)
                clipped_end = min(watched_end, seq_len)
                if clipped_end <= clipped_start:
                    continue
                range_name = f"{watched_start}:{watched_end}"
                values = watched_key_range_stats.setdefault(
                    range_name,
                    {
                        "probability_mass": 0.0,
                        "selector_rows": 0.0,
                        "valid_token_slots": 0.0,
                        "target_token_slots": 0.0,
                        "selected_token_slots": 0.0,
                        "sink_token_slots": 0.0,
                    },
                )
                watched_valid = causal_chunk[..., clipped_start:clipped_end]
                watched_target = final_target[..., clipped_start:clipped_end]
                watched_selected = final_chunk[..., clipped_start:clipped_end]
                values["selector_rows"] += float(
                    watched_valid.any(dim=-1).sum(dtype=torch.int64).item()
                )
                values["valid_token_slots"] += float(
                    watched_valid.sum(dtype=torch.int64).item()
                )
                values["target_token_slots"] += float(
                    (watched_target & watched_valid).sum(dtype=torch.int64).item()
                )
                values["selected_token_slots"] += float(
                    (watched_selected & watched_valid).sum(dtype=torch.int64).item()
                )
                watched_positions = torch.arange(
                    clipped_start, clipped_end, device=q.device
                )
                watched_sink = watched_positions.view(1, 1, 1, -1) < (
                    chosen_block_sizes.index_select(1, group_indices).unsqueeze(-1)
                )
                values["sink_token_slots"] += float(
                    (watched_sink & watched_valid).sum(dtype=torch.int64).item()
                )

            rows = final_chunk.reshape(-1, seq_len)
            row_counts = rows.sum(dim=-1, dtype=torch.int32)
            row_ends_flat = (
                torch.cumsum(row_counts, dim=0, dtype=torch.int32) + token_offset
            )
            row_starts_flat = row_ends_flat - row_counts
            chunk_token_indices = torch.nonzero(rows, as_tuple=False)[:, 1].to(
                torch.int32
            )
            row_starts_chunks.append(
                row_starts_flat.view(
                    batch_size, head_end - head_start, num_query_blocks
                )
            )
            row_ends_chunks.append(
                row_ends_flat.view(
                    batch_size, head_end - head_start, num_query_blocks
                )
            )
            token_index_chunks.append(chunk_token_indices)
            token_offset += int(chunk_token_indices.numel())

        if dump_rows and self.oracle_residual_dump_dir is not None:
            output_path = self.oracle_residual_dump_dir / (
                f"sample_{sample_index:03d}_layer_{layer_idx:02d}.pt"
            )
            temporary_path = output_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "sample_index": int(sample_index),
                    "layer": int(layer_idx),
                    "shared_topk": int(self.target_token_budget or 0),
                    "oracle_residual_topk": int(self.oracle_residual_tokens),
                    "member_topk_budget": int(self.oracle_member_topk_budget),
                    "rows": dump_rows,
                },
                temporary_path,
            )
            os.replace(temporary_path, output_path)

        total_causal_blocks = int(
            batch_size
            * num_q_heads
            * num_structural_blocks
            * (num_structural_blocks + 1)
            // 2
        )
        total_causal_token_pairs = int(
            batch_size * num_q_heads * seq_len * (seq_len + 1) // 2
        )
        candidate_key_tokens = int(
            batch_size * num_q_heads * query_ends.sum(dtype=torch.int64).item()
        )
        index = TokenCompactedIndex(
            row_starts=torch.cat(row_starts_chunks, dim=1).contiguous(),
            row_ends=torch.cat(row_ends_chunks, dim=1).contiguous(),
            token_indices=torch.cat(token_index_chunks, dim=0).contiguous(),
            head_to_group=torch.arange(
                num_q_heads, dtype=torch.int32, device=q.device
            ),
            num_groups=num_q_heads,
            num_query_blocks=num_query_blocks,
            query_block_size=128,
        )
        return index, {
            "selected_blocks": selected_blocks,
            "causal_blocks": total_causal_blocks,
            "selected_token_pairs": selected_token_pairs,
            "causal_token_pairs": total_causal_token_pairs,
            "compacted_key_tokens": compacted_key_tokens,
            "candidate_key_tokens": candidate_key_tokens,
            "target_mask_tokens": target_mask_tokens,
            "covered_target_tokens": covered_target_tokens,
            "target_probability_mass": 0.0,
            "covered_target_probability_mass": 0.0,
            "watched_key_range_stats": watched_key_range_stats,
            "member_mask_fidelity_stats": oracle_stats,
        }

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
        del gamma, min_budget, max_budget, tau
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

        self.dump_ranked_probability_distribution(
            q[:, -min(block_size, seq_len) :],
            k,
            layer_idx=layer_idx,
            gqa_interleave=gqa_interleave,
            source="sparse_selector",
        )

        token_scores, query_starts, query_ends, query_lengths = (
            self._score_representative_keys(
                representative_q,
                representative_k,
                block_size=block_size,
            )
        )
        num_query_blocks = token_scores.shape[2]
        probe_positions = query_starts[:, None] + torch.stack(
            (
                (query_lengths - 1) // 3,
                2 * (query_lengths - 1) // 3,
                query_lengths - 1,
            ),
            dim=-1,
        )

        key_positions = torch.arange(seq_len, device=q.device)
        causal_tokens = key_positions.view(1, 1, 1, seq_len) < query_ends.view(
            1, 1, num_query_blocks, 1
        )
        token_scores.masked_fill_(~causal_tokens, float("-inf"))
        token_probabilities = None
        use_probability_target = self.target_token_top_p is not None and (
            self.target_top_p_start_layer is None
            or layer_idx >= self.target_top_p_start_layer
        )
        if self.block_probability_top_p is not None:
            probability_rows = token_scores.reshape(-1, seq_len)
            for start in range(0, probability_rows.shape[0], 16):
                end = min(start + 16, probability_rows.shape[0])
                normalized = torch.softmax(
                    probability_rows[start:end].to(torch.float32), dim=-1
                ).to(probability_rows.dtype)
                probability_rows[start:end].copy_(normalized)
            token_probabilities = token_scores
            target_mask = causal_tokens.expand_as(token_scores)
        elif not use_probability_target:
            if self.target_token_budget is None:
                raise RuntimeError("Fixed token target budget is missing")
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
            target_mask.scatter_(
                -1, target_indices, keep_target.expand_as(target_indices)
            )
            target_mask &= causal_tokens
            if self.measure_fixed_target_probability_mass:
                probability_rows = token_scores.reshape(-1, seq_len)
                for start in range(0, probability_rows.shape[0], 16):
                    end = min(start + 16, probability_rows.shape[0])
                    normalized = torch.softmax(
                        probability_rows[start:end].to(torch.float32), dim=-1
                    ).to(probability_rows.dtype)
                    probability_rows[start:end].copy_(normalized)
                token_probabilities = token_scores
        else:
            # Store normalized probabilities in the score tensor itself to
            # avoid another full [batch, group, query_tile, key] allocation.
            probability_rows = token_scores.reshape(-1, seq_len)
            for start in range(0, probability_rows.shape[0], 16):
                end = min(start + 16, probability_rows.shape[0])
                normalized = torch.softmax(
                    probability_rows[start:end].to(torch.float32), dim=-1
                ).to(probability_rows.dtype)
                probability_rows[start:end].copy_(normalized)
            token_probabilities = token_scores
            target_mask = _top_p_target_mask(
                token_probabilities,
                causal_tokens,
                top_p=float(self.target_token_top_p),
                minimum_tokens=self.target_token_min_budget or 0,
            )

        if self.force_sink_target_tokens or self.force_diagonal_target_tokens:
            protected_target_mask = torch.zeros_like(target_mask)
            if self.force_sink_target_tokens:
                protected_target_mask[..., : self.target_protected_token_span] = True
            if self.force_diagonal_target_tokens:
                protected_local_tokens = (
                    key_positions.view(1, seq_len)
                    >= query_starts.view(num_query_blocks, 1)
                ) & (
                    key_positions.view(1, seq_len)
                    < query_ends.view(num_query_blocks, 1)
                )
                protected_target_mask |= protected_local_tokens.view(
                    1, 1, num_query_blocks, seq_len
                )
            protected_target_mask &= causal_tokens

            target_keep_counts = torch.minimum(
                query_ends,
                query_ends.new_full((), self.target_token_budget),
            )
            protected_target_counts = protected_target_mask.sum(
                -1, dtype=torch.int64
            )
            ordinary_target_counts = (
                target_keep_counts.view(1, 1, num_query_blocks)
                - protected_target_counts
            ).clamp_min(0)
            ordinary_target_scores = token_scores.masked_fill(
                ~target_mask | protected_target_mask, float("-inf")
            )
            ordinary_target_top_k = min(self.target_token_budget, seq_len)
            ordinary_target_values, ordinary_target_indices = torch.topk(
                ordinary_target_scores,
                k=ordinary_target_top_k,
                dim=-1,
                sorted=False,
            )
            ordinary_target_ranks = torch.arange(
                ordinary_target_top_k, device=q.device
            ).view(1, 1, 1, ordinary_target_top_k)
            keep_ordinary_targets = (
                ordinary_target_ranks < ordinary_target_counts.unsqueeze(-1)
            )
            keep_ordinary_targets &= torch.isfinite(ordinary_target_values)
            ordinary_target_mask = torch.zeros_like(target_mask)
            ordinary_target_mask.scatter_(
                -1, ordinary_target_indices, keep_ordinary_targets
            )
            target_mask = protected_target_mask | ordinary_target_mask

        target_tokens_per_row = target_mask.sum(-1, dtype=torch.int64)
        target_probability_mass_per_row = (
            torch.where(
                target_mask, token_probabilities, torch.zeros_like(token_probabilities)
            ).sum(-1, dtype=torch.float32)
            if token_probabilities is not None
            else target_tokens_per_row.to(torch.float32)
        )
        selected_mask = None
        chosen_block_counts = None
        chosen_causal_block_counts = None
        chosen_block_sizes = None
        best_coverage = None
        best_count_coverage = None
        best_feasible = None
        best_kernel_keys = None
        minimum_kernel_pairs = None
        candidate_kernel_pairs_by_size: Dict[int, int] = {}
        group_sizes = torch.tensor(
            [len(group["members"]) for group in groups],
            dtype=torch.int64,
            device=q.device,
        )
        if not self.project_target_to_blocks:
            if self.final_token_budget is None:
                selected_mask = target_mask
            else:
                # Token-only matched-budget control: reserve sink and local
                # tokens first, then fill the remaining slots from the same
                # representative-head TopK targets. Ordinary targets are never
                # expanded to whole blocks.
                protected_mask = torch.zeros_like(target_mask)
                protection_span = max(self.logical_key_block_sizes)
                if self.force_sink_block:
                    protected_mask[..., :protection_span] = True
                if self.force_diagonal_block:
                    local_tokens = (
                        key_positions.view(1, seq_len)
                        >= query_starts.view(num_query_blocks, 1)
                    ) & (
                        key_positions.view(1, seq_len)
                        < query_ends.view(num_query_blocks, 1)
                    )
                    protected_mask |= local_tokens.view(
                        1, 1, num_query_blocks, seq_len
                    )
                protected_mask &= causal_tokens

                final_keep_counts = torch.minimum(
                    query_ends,
                    query_ends.new_full((), self.final_token_budget),
                )
                protected_counts = protected_mask.sum(-1, dtype=torch.int64)
                ordinary_keep_counts = (
                    final_keep_counts.view(1, 1, num_query_blocks)
                    - protected_counts
                ).clamp_min(0)
                ordinary_scores = token_scores.masked_fill(
                    ~target_mask | protected_mask, float("-inf")
                )
                ordinary_top_k = min(self.final_token_budget, seq_len)
                ordinary_values, ordinary_indices = torch.topk(
                    ordinary_scores,
                    k=ordinary_top_k,
                    dim=-1,
                    sorted=False,
                )
                ordinary_ranks = torch.arange(
                    ordinary_top_k, device=q.device
                ).view(1, 1, 1, ordinary_top_k)
                keep_ordinary = ordinary_ranks < ordinary_keep_counts.unsqueeze(-1)
                keep_ordinary &= torch.isfinite(ordinary_values)
                ordinary_mask = torch.zeros_like(target_mask)
                ordinary_mask.scatter_(-1, ordinary_indices, keep_ordinary)
                selected_mask = protected_mask | ordinary_mask
            chosen_block_counts = torch.zeros(
                target_mask.shape[:-1], dtype=torch.int32, device=q.device
            )
            chosen_causal_block_counts = torch.zeros_like(chosen_block_counts)
            chosen_block_sizes = torch.zeros_like(chosen_block_counts)

        for key_block_size in (
            self.logical_key_block_sizes if self.project_target_to_blocks else ()
        ):
            (
                candidate_mask,
                candidate_block_counts,
                candidate_causal_block_counts,
                candidate_covered_targets,
                candidate_covered_probability_mass,
                candidate_selection_score,
            ) = _cover_target_mask_with_key_blocks(
                target_mask,
                causal_tokens,
                query_starts,
                query_ends,
                key_block_size=key_block_size,
                final_token_budget=self.final_token_budget,
                maximum_selected_blocks=self.maximum_selected_blocks,
                fill_final_token_budget=self.fill_final_token_budget,
                minimum_block_coverage_ratio=(
                    self.minimum_block_coverage_ratio
                ),
                target_probabilities=token_probabilities,
                minimum_block_target_probability_ratio=(
                    self.minimum_block_target_probability_ratio
                ),
                block_probability_top_p=self.block_probability_top_p,
                block_target_probability_top_p=(
                    self.block_target_probability_top_p
                ),
                block_target_count_coverage_ratio=(
                    self.block_target_count_coverage_ratio
                ),
                block_target_cost_aware=self.block_target_cost_aware,
                block_f_beta=self.block_f_beta,
                forced_key_ranges=(
                    self.watched_key_ranges
                    if self.force_watched_key_range_blocks
                    else ()
                ),
                force_sink_block=self.force_sink_block,
                force_diagonal_block=self.force_diagonal_block,
            )
            if self.block_f_beta is not None:
                candidate_coverage = candidate_selection_score
            elif self.block_target_probability_top_p is not None:
                candidate_coverage = candidate_selection_score
            elif self.block_probability_top_p is not None:
                candidate_coverage = (
                    torch.where(
                        candidate_mask,
                        token_probabilities,
                        torch.zeros_like(token_probabilities),
                    )
                ).sum(-1, dtype=torch.float32)
            elif self.minimum_block_target_probability_ratio is None:
                candidate_coverage = (
                    candidate_covered_targets.to(torch.float32)
                    / target_tokens_per_row.clamp_min(1).to(torch.float32)
                )
            else:
                candidate_coverage = (
                    candidate_covered_probability_mass
                    / target_probability_mass_per_row.clamp_min(1e-12)
                )
            candidate_count_coverage = None
            candidate_feasible = None
            if self.block_target_probability_top_p is not None:
                candidate_count_coverage = (
                    candidate_covered_targets.to(torch.float32)
                    / target_tokens_per_row.clamp_min(1).to(torch.float32)
                )
                candidate_feasible = candidate_coverage >= (
                    self.block_target_probability_top_p - 1e-6
                )
                if self.block_target_count_coverage_ratio is not None:
                    candidate_feasible &= candidate_count_coverage >= (
                        self.block_target_count_coverage_ratio - 1e-6
                    )
            candidate_kernel_pairs = None
            if self.block_target_probability_top_p is not None:
                candidate_kernel_pairs = _causal_pair_count_per_selector_row(
                    candidate_mask,
                    seq_len=seq_len,
                    query_block_size=block_size,
                )
                candidate_kernel_pairs_by_size[key_block_size] = int(
                    (
                        candidate_kernel_pairs
                        * group_sizes.view(1, len(groups), 1)
                    ).sum().item()
                )
            if selected_mask is None:
                selected_mask = candidate_mask
                chosen_block_counts = candidate_block_counts
                chosen_causal_block_counts = candidate_causal_block_counts
                chosen_block_sizes = torch.full_like(
                    candidate_block_counts, key_block_size, dtype=torch.int32
                )
                best_coverage = candidate_coverage
                best_count_coverage = candidate_count_coverage
                best_feasible = candidate_feasible
                best_kernel_keys = candidate_mask.sum(-1, dtype=torch.int64)
                minimum_kernel_pairs = candidate_kernel_pairs
                continue

            if best_coverage is None:
                raise RuntimeError("Adaptive coverage state was not initialized")
            candidate_kernel_keys = candidate_mask.sum(-1, dtype=torch.int64)
            if self.block_f_beta is not None:
                if best_kernel_keys is None:
                    raise RuntimeError("Adaptive kernel cost was not initialized")
                score_tolerance = 0.005
                materially_better = candidate_coverage > (
                    best_coverage + score_tolerance
                )
                near_best = candidate_coverage >= (
                    best_coverage - score_tolerance
                )
                choose_larger = materially_better | (
                    near_best & (candidate_kernel_keys <= best_kernel_keys)
                )
            elif self.block_target_probability_top_p is not None:
                if (
                    candidate_kernel_pairs is None
                    or minimum_kernel_pairs is None
                    or candidate_count_coverage is None
                    or best_count_coverage is None
                    or candidate_feasible is None
                    or best_feasible is None
                ):
                    raise RuntimeError(
                        "Adaptive target-coverage state is missing"
                    )
                lower_cost = candidate_kernel_pairs.to(torch.float64) <= (
                    minimum_kernel_pairs.to(torch.float64)
                    * (1.0 + self.adaptive_kernel_cost_tolerance)
                )
                candidate_quality = candidate_coverage
                best_quality = best_coverage
                if self.block_target_count_coverage_ratio is not None:
                    candidate_quality = torch.minimum(
                        candidate_coverage
                        / self.block_target_probability_top_p,
                        candidate_count_coverage
                        / self.block_target_count_coverage_ratio,
                    )
                    best_quality = torch.minimum(
                        best_coverage / self.block_target_probability_top_p,
                        best_count_coverage
                        / self.block_target_count_coverage_ratio,
                    )
                choose_larger = (
                    candidate_feasible & ~best_feasible
                ) | (
                    candidate_feasible & best_feasible & lower_cost
                ) | (
                    ~candidate_feasible
                    & ~best_feasible
                    & (
                        (candidate_quality > best_quality + 1e-6)
                        | (
                            torch.isclose(
                                candidate_quality,
                                best_quality,
                                rtol=0.0,
                                atol=1e-6,
                            )
                            & lower_cost
                        )
                    )
                )
            elif self.block_probability_top_p is not None:
                if best_kernel_keys is None:
                    raise RuntimeError("Adaptive kernel cost was not initialized")
                choose_larger = candidate_kernel_keys <= best_kernel_keys
            else:
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
            best_coverage = torch.where(
                choose_larger, candidate_coverage, best_coverage
            )
            if candidate_count_coverage is not None:
                if best_count_coverage is None:
                    best_count_coverage = candidate_count_coverage
                else:
                    best_count_coverage = torch.where(
                        choose_larger,
                        candidate_count_coverage,
                        best_count_coverage,
                    )
            if candidate_feasible is not None:
                if best_feasible is None:
                    best_feasible = candidate_feasible
                else:
                    best_feasible = torch.where(
                        choose_larger, candidate_feasible, best_feasible
                    )
            if best_kernel_keys is not None:
                best_kernel_keys = torch.where(
                    choose_larger, candidate_kernel_keys, best_kernel_keys
                )
            if candidate_kernel_pairs is not None:
                if minimum_kernel_pairs is None:
                    minimum_kernel_pairs = candidate_kernel_pairs
                else:
                    minimum_kernel_pairs = torch.where(
                        choose_larger,
                        candidate_kernel_pairs,
                        minimum_kernel_pairs,
                    )

        if (
            selected_mask is None
            or chosen_block_counts is None
            or chosen_causal_block_counts is None
            or chosen_block_sizes is None
        ):
            raise RuntimeError("No logical K block candidate was evaluated")

        if self.residual_mode == "oracle":
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
            index, member_metrics = self._build_member_oracle_residual_index(
                q=q,
                k=k,
                selected_mask=selected_mask,
                target_mask=target_mask,
                representative_scores=token_scores,
                causal_tokens=causal_tokens,
                chosen_block_sizes=chosen_block_sizes,
                head_to_group=head_to_group,
                query_starts=query_starts,
                query_ends=query_ends,
                probe_positions=probe_positions,
                gqa_interleave=gqa_interleave,
                layer_idx=layer_idx,
            )
            chosen_block_size_rows = {
                size: int(
                    (
                        (chosen_block_sizes == size).sum(
                            dim=-1, dtype=torch.int64
                        )
                        * group_sizes[None, :]
                    ).sum().item()
                )
                for size in self.logical_key_block_sizes
            }
            self.stats.record(
                selected_blocks=member_metrics["selected_blocks"],
                causal_blocks=member_metrics["causal_blocks"],
                selected_token_pairs=member_metrics["selected_token_pairs"],
                causal_token_pairs=member_metrics["causal_token_pairs"],
                compacted_key_tokens=member_metrics["compacted_key_tokens"],
                candidate_key_tokens=member_metrics["candidate_key_tokens"],
                target_mask_tokens=member_metrics["target_mask_tokens"],
                covered_target_tokens=member_metrics["covered_target_tokens"],
                target_probability_mass=member_metrics[
                    "target_probability_mass"
                ],
                covered_target_probability_mass=member_metrics[
                    "covered_target_probability_mass"
                ],
                chosen_block_size_rows=chosen_block_size_rows,
                candidate_kernel_pairs_by_size=candidate_kernel_pairs_by_size,
                layer_idx=layer_idx,
                selector_rows=batch_size * num_q_heads * num_query_blocks,
                target_selection_mode="fixed_top_k_plus_oracle_member_residual",
                watched_key_range_stats=member_metrics[
                    "watched_key_range_stats"
                ],
                member_mask_fidelity_stats=member_metrics[
                    "member_mask_fidelity_stats"
                ],
                timing_events=(timing_start, timing_end),
            )
            timing_end.record()
            return index

        member_mask_fidelity_stats: Dict[int, Dict[str, Any]] = {}
        if self.profile_member_mask_fidelity:
            if self.target_token_budget is None:
                raise RuntimeError(
                    "Member-mask profiling requires fixed TopK target selection"
                )
            for group_idx, group in enumerate(groups):
                representative_head = int(group["representative"])
                representative_target = target_mask[:, group_idx : group_idx + 1]
                representative_final = selected_mask[:, group_idx : group_idx + 1]
                representative_target_tokens = int(
                    representative_target.sum(dtype=torch.int64).item()
                )
                for member_value in group["members"]:
                    member_head = int(member_value)
                    if gqa_interleave:
                        member_kv_head = member_head % num_kv_heads
                    else:
                        member_kv_head = member_head // num_share_q_heads
                    if member_head == representative_head:
                        member_target = representative_target
                    else:
                        member_q = q[:, :, member_head : member_head + 1]
                        member_k = k[:, :, member_kv_head : member_kv_head + 1]
                        member_pooled_q = _block_mean(member_q, block_size)
                        member_scores = torch.einsum(
                            "bqhd,bthd->bhqt", member_pooled_q, member_k
                        ) / math.sqrt(head_dim)
                        member_scores.mul_(self.probe_weights[3])
                        for probe_idx in range(3):
                            member_probe_q = member_q.index_select(
                                1, probe_positions[:, probe_idx]
                            )
                            member_probe_scores = torch.einsum(
                                "bqhd,bthd->bhqt", member_probe_q, member_k
                            ) / math.sqrt(head_dim)
                            member_scores.add_(
                                member_probe_scores,
                                alpha=self.probe_weights[probe_idx],
                            )
                        member_scores.masked_fill_(
                            ~causal_tokens, float("-inf")
                        )
                        member_target_k = min(self.target_token_budget, seq_len)
                        _, member_indices = torch.topk(
                            member_scores,
                            k=member_target_k,
                            dim=-1,
                            sorted=False,
                        )
                        member_keep_counts = torch.minimum(
                            query_ends,
                            query_ends.new_full((), member_target_k),
                        )
                        member_ranks = torch.arange(
                            member_target_k, device=q.device
                        ).view(1, 1, 1, member_target_k)
                        member_keep = member_ranks < member_keep_counts.view(
                            1, 1, num_query_blocks, 1
                        )
                        member_target = torch.zeros_like(
                            member_scores, dtype=torch.bool
                        )
                        member_target.scatter_(
                            -1,
                            member_indices,
                            member_keep.expand_as(member_indices),
                        )
                        member_target &= causal_tokens

                    member_target_tokens = int(
                        member_target.sum(dtype=torch.int64).item()
                    )
                    representative_overlap_tokens = int(
                        (member_target & representative_target)
                        .sum(dtype=torch.int64)
                        .item()
                    )
                    final_overlap_tokens = int(
                        (member_target & representative_final)
                        .sum(dtype=torch.int64)
                        .item()
                    )
                    watched_stats: Dict[str, Dict[str, float]] = {}
                    for watched_start, watched_end in self.watched_key_ranges:
                        clipped_start = min(watched_start, seq_len)
                        clipped_end = min(watched_end, seq_len)
                        if clipped_end <= clipped_start:
                            continue
                        range_name = f"{watched_start}:{watched_end}"
                        watched_valid = causal_tokens[
                            ..., clipped_start:clipped_end
                        ].expand(batch_size, 1, -1, -1)
                        watched_member = member_target[
                            ..., clipped_start:clipped_end
                        ]
                        watched_representative = representative_target[
                            ..., clipped_start:clipped_end
                        ]
                        watched_final = representative_final[
                            ..., clipped_start:clipped_end
                        ]
                        watched_stats[range_name] = {
                            "valid_slots": float(
                                watched_valid.sum(dtype=torch.int64).item()
                            ),
                            "member_target_slots": float(
                                (watched_member & watched_valid)
                                .sum(dtype=torch.int64)
                                .item()
                            ),
                            "representative_target_slots": float(
                                (watched_representative & watched_valid)
                                .sum(dtype=torch.int64)
                                .item()
                            ),
                            "final_kernel_slots": float(
                                (watched_final & watched_valid)
                                .sum(dtype=torch.int64)
                                .item()
                            ),
                        }
                    member_mask_fidelity_stats[member_head] = {
                        "group_index": group_idx,
                        "representative_head": representative_head,
                        "kv_head": member_kv_head,
                        "is_representative": int(
                            member_head == representative_head
                        ),
                        "selector_rows": batch_size * num_query_blocks,
                        "member_target_tokens": member_target_tokens,
                        "representative_target_tokens": (
                            representative_target_tokens
                        ),
                        "representative_overlap_tokens": (
                            representative_overlap_tokens
                        ),
                        "final_overlap_tokens": final_overlap_tokens,
                        "watched_key_ranges": watched_stats,
                    }

        if self.member_vertical_slash and (
            self.member_vertical_slash_layers is None
            or layer_idx in self.member_vertical_slash_layers
        ):
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
            index, member_metrics = self._build_member_vertical_slash_index(
                q=q,
                k=k,
                v=v,
                selected_mask=selected_mask,
                target_mask=target_mask,
                token_probabilities=token_probabilities,
                causal_tokens=causal_tokens,
                chosen_block_sizes=chosen_block_sizes,
                head_to_group=head_to_group,
                query_ends=query_ends,
                gqa_interleave=gqa_interleave,
            )
            chosen_block_size_rows = {
                size: int(
                    (
                        (chosen_block_sizes == size).sum(
                            dim=-1, dtype=torch.int64
                        )
                        * group_sizes[None, :]
                    ).sum().item()
                )
                for size in self.logical_key_block_sizes
            }
            self.stats.record(
                selected_blocks=member_metrics["selected_blocks"],
                causal_blocks=member_metrics["causal_blocks"],
                selected_token_pairs=member_metrics["selected_token_pairs"],
                causal_token_pairs=member_metrics["causal_token_pairs"],
                compacted_key_tokens=member_metrics["compacted_key_tokens"],
                candidate_key_tokens=member_metrics["candidate_key_tokens"],
                target_mask_tokens=member_metrics["target_mask_tokens"],
                covered_target_tokens=member_metrics["covered_target_tokens"],
                target_probability_mass=member_metrics[
                    "target_probability_mass"
                ],
                covered_target_probability_mass=member_metrics[
                    "covered_target_probability_mass"
                ],
                chosen_block_size_rows=chosen_block_size_rows,
                candidate_kernel_pairs_by_size=candidate_kernel_pairs_by_size,
                layer_idx=layer_idx,
                selector_rows=batch_size * num_q_heads * num_query_blocks,
                target_selection_mode=(
                    "probability_top_p_plus_member_vertical_slash"
                    if self.target_token_top_p is not None
                    else "fixed_top_k_plus_member_vertical_slash"
                ),
                watched_key_range_stats=member_metrics[
                    "watched_key_range_stats"
                ],
                member_mask_fidelity_stats=member_mask_fidelity_stats,
                vertical_slash_candidate_blocks=member_metrics[
                    "vertical_slash_candidate_blocks"
                ],
                vertical_slash_added_blocks=member_metrics[
                    "vertical_slash_added_blocks"
                ],
                vertical_slash_added_key_tokens=member_metrics[
                    "vertical_slash_added_key_tokens"
                ],
                timing_events=(timing_start, timing_end),
            )
            timing_end.record()
            return index

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
        selected_pairs_per_row = _causal_pair_count_per_selector_row(
            selected_mask,
            seq_len=seq_len,
            query_block_size=block_size,
        )
        selected_pairs_per_group = selected_pairs_per_row.sum(
            dim=-1, dtype=torch.int64
        )
        selected_token_pairs = int(
            (selected_pairs_per_group * group_sizes[None, :]).sum().item()
        )
        selected_q_heads = int(group_sizes.sum().item())
        total_causal_token_pairs = (
            batch_size * selected_q_heads * seq_len * (seq_len + 1) // 2
        )
        selected_keys_per_group = selected_mask.sum(
            dim=(-1, -2), dtype=torch.int64
        )
        causal_keys_per_row = query_ends.sum(dtype=torch.int64)
        compacted_key_tokens = int(
            (selected_keys_per_group * group_sizes[None, :]).sum().item()
        )
        candidate_key_tokens = int(
            batch_size * selected_q_heads * causal_keys_per_row.item()
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
        if token_probabilities is None:
            target_probability_mass = 0.0
            covered_target_probability_mass = 0.0
        else:
            target_mass_per_group = (
                torch.where(
                    target_mask,
                    token_probabilities,
                    torch.zeros_like(token_probabilities),
                )
            ).sum(dim=(-1, -2), dtype=torch.float64)
            covered_mass_per_group = (
                torch.where(
                    target_mask & selected_mask,
                    token_probabilities,
                    torch.zeros_like(token_probabilities),
                )
            ).sum(dim=(-1, -2), dtype=torch.float64)
            target_probability_mass = float(
                (target_mass_per_group * group_sizes[None, :]).sum().item()
            )
            covered_target_probability_mass = float(
                (covered_mass_per_group * group_sizes[None, :]).sum().item()
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
        watched_key_range_stats: Dict[str, Dict[str, float]] = {}
        if token_probabilities is not None and self.watched_key_ranges:
            group_weights = group_sizes.view(1, len(groups), 1, 1)
            row_group_weights = group_sizes.view(1, len(groups), 1)
            for watched_start, watched_end in self.watched_key_ranges:
                clipped_start = min(watched_start, seq_len)
                clipped_end = min(watched_end, seq_len)
                if clipped_end <= clipped_start:
                    continue
                range_name = f"{watched_start}:{watched_end}"
                watched_positions = torch.arange(
                    clipped_start, clipped_end, device=q.device
                )
                watched_valid = causal_tokens[
                    ..., clipped_start:clipped_end
                ].expand(batch_size, len(groups), -1, -1)
                watched_target = target_mask[..., clipped_start:clipped_end]
                watched_selected = selected_mask[..., clipped_start:clipped_end]
                watched_probabilities = token_probabilities[
                    ..., clipped_start:clipped_end
                ]
                watched_sink = watched_positions.view(1, 1, 1, -1) < (
                    chosen_block_sizes.unsqueeze(-1)
                )
                watched_sink &= watched_valid
                watched_key_range_stats[range_name] = {
                    "probability_mass": float(
                        (
                            watched_probabilities.to(torch.float64)
                            * watched_valid.to(torch.float64)
                            * group_weights
                        ).sum().item()
                    ),
                    "selector_rows": float(
                        (
                            watched_valid.any(dim=-1).to(torch.int64)
                            * row_group_weights
                        ).sum().item()
                    ),
                    "valid_token_slots": float(
                        (
                            watched_valid.to(torch.int64) * group_weights
                        ).sum().item()
                    ),
                    "target_token_slots": float(
                        (
                            (watched_target & watched_valid).to(torch.int64)
                            * group_weights
                        ).sum().item()
                    ),
                    "selected_token_slots": float(
                        (
                            (watched_selected & watched_valid).to(torch.int64)
                            * group_weights
                        ).sum().item()
                    ),
                    "sink_token_slots": float(
                        (
                            watched_sink.to(torch.int64) * group_weights
                        ).sum().item()
                    ),
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
            target_probability_mass=target_probability_mass,
            covered_target_probability_mass=covered_target_probability_mass,
            chosen_block_size_rows=chosen_block_size_rows,
            candidate_kernel_pairs_by_size=candidate_kernel_pairs_by_size,
            layer_idx=layer_idx,
            selector_rows=batch_size * selected_q_heads * num_query_blocks,
            target_selection_mode=(
                "block_probability_top_p"
                if self.block_probability_top_p is not None
                else "probability_top_p"
                if use_probability_target
                else "fixed_top_k"
            ),
            watched_key_range_stats=watched_key_range_stats,
            member_mask_fidelity_stats=member_mask_fidelity_stats,
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


class ChunkedPerHeadTokenFirstSelector:
    """Run independent Q-head selectors in bounded-memory head chunks."""

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        num_q_heads: int,
        head_chunk_size: int = 4,
        **selector_kwargs: Any,
    ) -> None:
        if num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if head_chunk_size <= 0:
            raise ValueError("head_chunk_size must be positive")
        layer_keys = tuple(group_config["layers"].keys())
        self.num_q_heads = int(num_q_heads)
        self.head_chunk_size = int(head_chunk_size)
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("chunked_per_head_sparse_layer", default=None)
        )
        self._stats = TokenCompactedSelectorStats()
        self.children: List[RepresentativeTokenFirstBlockSelector] = []
        for start in range(0, self.num_q_heads, self.head_chunk_size):
            end = min(start + self.head_chunk_size, self.num_q_heads)
            chunk_config = {
                "layers": {
                    layer: [
                        {"representative": head, "members": [head]}
                        for head in range(start, end)
                    ]
                    for layer in layer_keys
                }
            }
            child = RepresentativeTokenFirstBlockSelector(
                chunk_config,
                **selector_kwargs,
            )
            child.stats = self._stats
            self.children.append(child)

    @property
    def stats(self) -> TokenCompactedSelectorStats:
        return self._stats

    @stats.setter
    def stats(self, value: TokenCompactedSelectorStats) -> None:
        self._stats = value
        for child in getattr(self, "children", ()):
            child.stats = value

    def __call__(self, *args: Any, **kwargs: Any) -> TokenCompactedIndex:
        layer_idx = self.current_layer.get()
        if layer_idx is None:
            raise RuntimeError(
                "Per-head selector was called without layer context"
            )
        q = args[0] if args else kwargs["q"]
        if int(q.shape[2]) != self.num_q_heads:
            raise ValueError(
                f"Expected {self.num_q_heads} Q heads, got {q.shape[2]}"
            )

        row_starts = []
        row_ends = []
        token_indices = []
        offset = 0
        num_query_blocks = None
        query_block_size = None
        for child in self.children:
            token = child.current_layer.set(layer_idx)
            try:
                index = child(*args, **kwargs)
            finally:
                child.current_layer.reset(token)
            row_starts.append(index.row_starts + offset)
            row_ends.append(index.row_ends + offset)
            token_indices.append(index.token_indices)
            offset += int(index.token_indices.numel())
            num_query_blocks = index.num_query_blocks
            query_block_size = index.query_block_size

        if num_query_blocks is None or query_block_size is None:
            raise RuntimeError("No per-head selector chunk was evaluated")
        return TokenCompactedIndex(
            row_starts=torch.cat(row_starts, dim=1).contiguous(),
            row_ends=torch.cat(row_ends, dim=1).contiguous(),
            token_indices=torch.cat(token_indices, dim=0).contiguous(),
            head_to_group=torch.arange(
                self.num_q_heads, dtype=torch.int32, device=q.device
            ),
            num_groups=self.num_q_heads,
            num_query_blocks=num_query_blocks,
            query_block_size=query_block_size,
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
    dense_layers: tuple[int, ...] = ()

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
    force_sink_target_tokens: bool = False,
    force_diagonal_target_tokens: bool = False,
    target_protected_token_span: int = 128,
    selection_mode: str = "per_block_top_p",
    candidate_block_count: int = 256,
    final_token_budget: Optional[int] = 8192,
    maximum_selected_blocks: Optional[int] = None,
    fill_final_token_budget: bool = False,
    query_score_mode: str = "four_probe_weighted",
    probe_weights: tuple[float, float, float, float] = (0.2, 0.3, 0.4, 0.1),
    logical_key_block_size: int = 128,
    minimum_block_coverage_ratio: float = 0.125,
    adaptive_coverage_tolerance: float = 0.0,
    target_token_budget: Optional[int] = None,
    target_token_top_p: Optional[float] = None,
    target_token_min_budget: Optional[int] = None,
    target_top_p_start_layer: Optional[int] = None,
    minimum_block_target_probability_ratio: Optional[float] = None,
    block_probability_top_p: Optional[float] = None,
    block_target_probability_top_p: Optional[float] = None,
    block_target_count_coverage_ratio: Optional[float] = None,
    block_target_cost_aware: bool = False,
    block_f_beta: Optional[float] = None,
    measure_fixed_target_probability_mass: bool = False,
    adaptive_kernel_cost_tolerance: float = 0.0,
    watched_key_ranges: tuple[tuple[int, int], ...] = (),
    force_watched_key_range_blocks: bool = False,
    profile_member_mask_fidelity: bool = False,
    dense_layers: Optional[Sequence[int]] = None,
    per_head_selection: bool = False,
    project_target_to_blocks: bool = True,
    member_vertical_slash: bool = False,
    member_vertical_slash_gamma: float = 0.95,
    member_vertical_slash_min_tokens: int = 1024,
    member_vertical_slash_max_tokens: int = 2048,
    member_vertical_slash_head_chunk_size: int = 4,
    member_vertical_slash_layers: Optional[Sequence[int]] = None,
    residual_mode: str = "none",
    oracle_residual_tokens: int = 0,
    oracle_member_topk_budget: int = 8192,
    oracle_head_chunk_size: int = 4,
) -> TokenCompactedFlexPrefillPatch:
    """Patch long prefill with block-selected, token-compacted attention."""

    from flex_prefill import patch_model
    import flex_prefill.ops.flex_prefill_attention as ops_module

    num_layers = int(model.config.num_hidden_layers)
    validate_group_config(
        group_config,
        num_layers=num_layers,
        num_heads=int(model.config.num_attention_heads),
    )
    dense_layer_tuple = tuple(sorted(set(dense_layers or ())))
    if len(dense_layer_tuple) != len(dense_layers or ()):
        raise ValueError("dense_layers must not contain duplicates")
    invalid_dense_layers = [
        layer_idx
        for layer_idx in dense_layer_tuple
        if layer_idx < 0 or layer_idx >= num_layers
    ]
    if invalid_dense_layers:
        raise ValueError(
            f"dense_layers out of range [0, {num_layers}): "
            f"{invalid_dense_layers}"
        )

    attention_modules = []
    original_forwards = {}
    for module in model.modules():
        layer_idx = getattr(module, "layer_idx", None)
        if (
            layer_idx is None
            or not hasattr(module, "q_proj")
            or not hasattr(module, "k_proj")
        ):
            continue
        attention_modules.append(module)
        original_forwards[id(module)] = module.forward
    if len(attention_modules) != num_layers:
        raise RuntimeError(
            f"Found {len(attention_modules)} attention layers, expected "
            f"{num_layers}"
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
            force_sink_target_tokens=force_sink_target_tokens,
            force_diagonal_target_tokens=force_diagonal_target_tokens,
            target_protected_token_span=target_protected_token_span,
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
        selector_kwargs = dict(
            logical_key_block_size=logical_key_block_size,
            target_token_budget=(
                final_token_budget
                if target_token_budget is None and target_token_top_p is None
                else target_token_budget
            ),
            target_token_top_p=target_token_top_p,
            target_token_min_budget=target_token_min_budget,
            target_top_p_start_layer=target_top_p_start_layer,
            final_token_budget=final_token_budget,
            maximum_selected_blocks=maximum_selected_blocks,
            fill_final_token_budget=fill_final_token_budget,
            minimum_block_coverage_ratio=minimum_block_coverage_ratio,
            minimum_block_target_probability_ratio=(
                minimum_block_target_probability_ratio
            ),
            block_probability_top_p=block_probability_top_p,
            block_target_probability_top_p=block_target_probability_top_p,
            block_target_count_coverage_ratio=(
                block_target_count_coverage_ratio
            ),
            block_target_cost_aware=block_target_cost_aware,
            block_f_beta=block_f_beta,
            measure_fixed_target_probability_mass=(
                measure_fixed_target_probability_mass
            ),
            force_sink_block=force_sink_block,
            force_diagonal_block=force_diagonal_block,
            query_score_mode=query_score_mode,
            probe_weights=probe_weights,
            adaptive_coverage_tolerance=adaptive_coverage_tolerance,
            adaptive_kernel_cost_tolerance=adaptive_kernel_cost_tolerance,
            watched_key_ranges=watched_key_ranges,
            force_watched_key_range_blocks=force_watched_key_range_blocks,
            profile_member_mask_fidelity=profile_member_mask_fidelity,
            project_target_to_blocks=project_target_to_blocks,
            member_vertical_slash=member_vertical_slash,
            member_vertical_slash_gamma=member_vertical_slash_gamma,
            member_vertical_slash_min_tokens=member_vertical_slash_min_tokens,
            member_vertical_slash_max_tokens=member_vertical_slash_max_tokens,
            member_vertical_slash_head_chunk_size=(
                member_vertical_slash_head_chunk_size
            ),
            member_vertical_slash_layers=member_vertical_slash_layers,
            residual_mode=residual_mode,
            oracle_residual_tokens=oracle_residual_tokens,
            oracle_member_topk_budget=oracle_member_topk_budget,
            oracle_head_chunk_size=oracle_head_chunk_size,
        )
        if per_head_selection:
            selector = ChunkedPerHeadTokenFirstSelector(
                group_config,
                num_q_heads=int(model.config.num_attention_heads),
                head_chunk_size=4,
                **selector_kwargs,
            )
        else:
            selector = RepresentativeTokenFirstBlockSelector(
                group_config,
                **selector_kwargs,
            )
    else:
        raise ValueError(f"Unknown token selection mode: {selection_mode}")
    original_get_active_blocks = ops_module.get_active_blocks
    original_block_wise_attention = ops_module.triton_block_wise_attention
    if isinstance(selector, RepresentativeTokenFirstBlockSelector):
        selector.vertical_slash_get_active_blocks = original_get_active_blocks

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
    restored_dense_layers = 0
    dense_layer_set = set(dense_layer_tuple)
    for module in attention_modules:
        layer_idx = int(module.layer_idx)
        if layer_idx in dense_layer_set:
            original_forward = original_forwards[id(module)]

            def dense_layer_forward(
                self,
                *args,
                __forward=original_forward,
                __layer_idx=layer_idx,
                **kwargs,
            ):
                result = __forward(*args, **kwargs)
                if selector.ranked_probability_dump_dir is None:
                    return result
                hidden_states = kwargs.get(
                    "hidden_states", args[0] if args else None
                )
                if hidden_states is None or hidden_states.shape[1] <= 1:
                    return result
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    return result

                from transformers.models.qwen2.modeling_qwen2 import (
                    apply_rotary_pos_emb,
                )

                batch_size, seq_len, _ = hidden_states.shape
                tile_length = min(128, seq_len)
                query_states = self.q_proj(hidden_states[:, -tile_length:])
                key_states = self.k_proj(hidden_states)
                query_states = query_states.view(
                    batch_size,
                    tile_length,
                    self.num_heads,
                    self.head_dim,
                ).transpose(1, 2)
                key_states = key_states.view(
                    batch_size,
                    seq_len,
                    self.num_key_value_heads,
                    self.head_dim,
                ).transpose(1, 2)
                rotary_seq_len = max(
                    seq_len,
                    int(position_ids[:, -1].max().item()) + 1,
                )
                cos, sin = self.rotary_emb(key_states, seq_len=rotary_seq_len)
                query_states, _ = apply_rotary_pos_emb(
                    query_states,
                    query_states,
                    cos,
                    sin,
                    position_ids[:, -tile_length:],
                )
                _, key_states = apply_rotary_pos_emb(
                    key_states,
                    key_states,
                    cos,
                    sin,
                    position_ids,
                )
                selector.dump_ranked_probability_distribution(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    layer_idx=__layer_idx,
                    source="dense_observation",
                )
                return result

            module.forward = types.MethodType(dense_layer_forward, module)
            restored_dense_layers += 1
            continue
        original_forward = module.forward

        def layer_forward(
            self,
            *args,
            __forward=original_forward,
            __layer_idx=layer_idx,
            **kwargs,
        ):
            token = selector.current_layer.set(__layer_idx)
            try:
                return __forward(*args, **kwargs)
            finally:
                selector.current_layer.reset(token)

        module.forward = types.MethodType(layer_forward, module)
        wrapped_layers += 1

    expected_sparse_layers = num_layers - len(dense_layer_tuple)
    if (
        wrapped_layers != expected_sparse_layers
        or restored_dense_layers != len(dense_layer_tuple)
    ):
        ops_module.get_active_blocks = original_get_active_blocks
        ops_module.triton_block_wise_attention = original_block_wise_attention
        raise RuntimeError(
            f"Configured {wrapped_layers} sparse and "
            f"{restored_dense_layers} dense attention layers; expected "
            f"{expected_sparse_layers} sparse and "
            f"{len(dense_layer_tuple)} dense"
        )

    return TokenCompactedFlexPrefillPatch(
        selector=selector,
        ops_module=ops_module,
        original_get_active_blocks=original_get_active_blocks,
        original_block_wise_attention=original_block_wise_attention,
        dense_layers=dense_layer_tuple,
    )
