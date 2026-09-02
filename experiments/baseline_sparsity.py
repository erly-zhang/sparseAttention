#!/usr/bin/env python3
"""Lossless selector instrumentation for official sparse-attention baselines."""

from __future__ import annotations

import base64
import contextvars
import json
import math
import time
import types
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def _encoded_array(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    payload = zlib.compress(array.tobytes(), level=1)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "codec": "zlib-base64",
        "data": base64.b64encode(payload).decode("ascii"),
    }


def decode_array(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode an array stored in a selector dump record."""

    raw = zlib.decompress(base64.b64decode(payload["data"]))
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(
        payload["shape"]
    )


def _csr_from_rows(rows: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    values: list[np.ndarray] = []
    for row in rows:
        item = np.asarray(row, dtype=np.int32).reshape(-1)
        values.append(item)
        offsets.append(offsets[-1] + item.size)
    flat = np.concatenate(values) if values else np.empty(0, dtype=np.int32)
    return np.asarray(offsets, dtype=np.int64), flat


class BaselineSelectorStats:
    def __init__(self) -> None:
        self.calls = 0
        self.selected_blocks = 0
        self.causal_blocks = 0
        self.selected_token_pairs = 0
        self.causal_token_pairs = 0
        self.instrumentation_latency_sec = 0.0
        self.watched_key_range_stats: dict[str, dict[str, float]] = {}
        self.layer_watched_key_range_stats: dict[
            int, dict[str, dict[str, float]]
        ] = {}

    def record(
        self,
        *,
        selected_blocks: int,
        causal_blocks: int,
        selected_token_pairs: int,
        causal_token_pairs: int,
        instrumentation_latency_sec: float,
    ) -> None:
        self.calls += 1
        self.selected_blocks += int(selected_blocks)
        self.causal_blocks += int(causal_blocks)
        self.selected_token_pairs += int(selected_token_pairs)
        self.causal_token_pairs += int(causal_token_pairs)
        self.instrumentation_latency_sec += float(instrumentation_latency_sec)

    def snapshot(self) -> dict[str, Any]:
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
            "instrumentation_latency_sec": self.instrumentation_latency_sec,
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
        }

    def record_watched_ranges(
        self,
        layer: int,
        values: Mapping[str, Mapping[str, float]],
    ) -> None:
        layer_totals = self.layer_watched_key_range_stats.setdefault(
            int(layer), {}
        )
        for range_name, range_values in values.items():
            aggregate = self.watched_key_range_stats.setdefault(
                str(range_name), {}
            )
            layer_aggregate = layer_totals.setdefault(str(range_name), {})
            for name, value in range_values.items():
                aggregate[str(name)] = aggregate.get(str(name), 0.0) + float(
                    value
                )
                layer_aggregate[str(name)] = layer_aggregate.get(
                    str(name), 0.0
                ) + float(value)


class BaselineSparsityInstrumentation:
    """Record final sparse-kernel selectors without materializing dense masks."""

    schema_version = 1

    def __init__(
        self,
        model,
        method: str,
        dump_path: str | Path | None = None,
    ) -> None:
        if method not in {"flexprefill", "minference"}:
            raise ValueError(f"Unsupported instrumented baseline: {method}")
        self.model = model
        self.method = method
        self.dump_path = Path(dump_path) if dump_path is not None else None
        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        self.selector = self
        self.stats = BaselineSelectorStats()
        self.current_layer: contextvars.ContextVar[int | None] = (
            contextvars.ContextVar("baseline_sparse_layer", default=None)
        )
        self.current_head: contextvars.ContextVar[int | None] = (
            contextvars.ContextVar("baseline_sparse_head", default=None)
        )
        self._sample: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._pending_stats: list[tuple[Any, Any, Any, Any]] = []
        self._pending_watched_stats: list[
            tuple[int, Mapping[str, Mapping[str, Any]]]
        ] = []
        self.watched_key_ranges: tuple[tuple[int, int], ...] = ()
        self._selection_events = 0
        self._handles: list[Any] = []
        self._install_layer_context()
        if method == "flexprefill":
            self._install_flexprefill()
        else:
            self._install_minference()

    def reset_stats(self) -> None:
        self.stats = BaselineSelectorStats()
        self._pending_stats = []
        self._pending_watched_stats = []
        self._selection_events = 0
        if self.dump_path is not None:
            self.dump_path.write_text("", encoding="utf-8")

    def begin_sample(
        self,
        *,
        call_index: int,
        input_ids_sha256: str,
        input_tokens: int,
        context: Mapping[str, Any],
    ) -> None:
        self._sample = {
            "schema_version": self.schema_version,
            "call_index": int(call_index),
            "method": self.method,
            **dict(context),
            "input_tokens": int(input_tokens),
            "input_ids_sha256": input_ids_sha256,
        }
        self._records = []
        self._pending_stats = []
        self._pending_watched_stats = []

    def end_sample(self) -> dict[str, Any]:
        if self._sample is None:
            return {}
        metadata: dict[str, Any] = {}
        if self.dump_path is not None:
            row = {**self._sample, "selectors": self._records}
            started = time.perf_counter()
            with self.dump_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=True, separators=(",", ":"))
                )
                stream.write("\n")
            metadata = {
                "selector_dump_path": str(self.dump_path),
                "selector_dump_line": int(self._sample["call_index"]),
                "selector_dump_write_latency_sec": time.perf_counter() - started,
            }
        self._sample = None
        self._records = []
        self._pending_stats = []
        self._pending_watched_stats = []
        return metadata

    def abort_sample(self) -> None:
        self._sample = None
        self._records = []
        self._pending_stats = []
        self._pending_watched_stats = []

    def _append(self, record: dict[str, Any]) -> None:
        if self._sample is not None and self.dump_path is not None:
            self._records.append(record)

    def _queue_stats(
        self,
        selected_blocks: Any,
        causal_blocks: Any,
        selected_token_pairs: Any,
        causal_token_pairs: Any,
    ) -> None:
        if self._sample is None:
            return
        self._selection_events += 1
        self._pending_stats.append(
            (
                selected_blocks,
                causal_blocks,
                selected_token_pairs,
                causal_token_pairs,
            )
        )

    @staticmethod
    def _sum_pending(values: Sequence[Any]) -> int:
        tensors = [value.reshape(()) for value in values if torch.is_tensor(value)]
        scalar = sum(int(value) for value in values if not torch.is_tensor(value))
        if tensors:
            scalar += int(torch.stack(tensors).sum().item())
        return scalar

    def flush_stats(self) -> None:
        if not self._pending_stats and not self._pending_watched_stats:
            return
        started = time.perf_counter()
        if self._pending_stats:
            columns = list(zip(*self._pending_stats))
            self.stats.record(
                selected_blocks=self._sum_pending(columns[0]),
                causal_blocks=self._sum_pending(columns[1]),
                selected_token_pairs=self._sum_pending(columns[2]),
                causal_token_pairs=self._sum_pending(columns[3]),
                instrumentation_latency_sec=time.perf_counter() - started,
            )
        for layer, ranges in self._pending_watched_stats:
            resolved = {
                range_name: {
                    name: float(self._sum_pending([value]))
                    for name, value in values.items()
                }
                for range_name, values in ranges.items()
            }
            self.stats.record_watched_ranges(layer, resolved)
        self._pending_stats = []
        self._pending_watched_stats = []

    def _queue_watched_stats(
        self,
        layer: int,
        values: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if self._sample is not None and values:
            self._pending_watched_stats.append((int(layer), values))

    @staticmethod
    def _num_heads(module) -> int:
        for name in ("num_heads", "num_attention_heads"):
            value = getattr(module, name, None)
            if value is not None:
                return int(value)
        return int(module.config.num_attention_heads)

    def _install_layer_context(self) -> None:
        instrumentation = self
        for module in self.model.modules():
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None or not hasattr(module, "q_proj"):
                continue
            original_forward = module.forward

            def wrapped_forward(
                module_self,
                *args,
                __original=original_forward,
                __layer_idx=int(layer_idx),
                **kwargs,
            ):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                q_len = int(hidden.shape[-2]) if hidden is not None else 1
                batch = int(hidden.shape[0]) if hidden is not None else 1
                before_events = instrumentation._selection_events
                token = instrumentation.current_layer.set(__layer_idx)
                try:
                    return __original(*args, **kwargs)
                finally:
                    if (
                        q_len > 1
                        and instrumentation._sample is not None
                        and instrumentation._selection_events == before_events
                    ):
                        instrumentation._record_dense(
                            __layer_idx,
                            q_len,
                            batch * instrumentation._num_heads(module_self),
                        )
                    instrumentation.current_layer.reset(token)

            module.forward = types.MethodType(wrapped_forward, module)

    def _record_dense(self, layer: int, seq_len: int, heads: int) -> None:
        pairs = heads * seq_len * (seq_len + 1) // 2
        self._queue_stats(0, 0, pairs, pairs)
        self._append(
            {
                "layer": int(layer),
                "head": "all",
                "head_count": int(heads),
                "kind": "dense_causal",
                "sequence_length": int(seq_len),
            }
        )

    @staticmethod
    def _block_pair_count(
        block_ids: torch.Tensor, seq_len: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = block_ids.to(torch.int64).reshape(-1)
        num_blocks = math.ceil(seq_len / block_size)
        q_block = torch.div(ids, num_blocks, rounding_mode="floor")
        k_block = ids.remainder(num_blocks)
        valid = (q_block < num_blocks) & (k_block <= q_block)
        q_block = q_block[valid]
        k_block = k_block[valid]
        q_len = (seq_len - q_block * block_size).clamp(0, block_size)
        k_len = (seq_len - k_block * block_size).clamp(0, block_size)
        counts = q_len * k_len
        diagonal = q_block == k_block
        counts[diagonal] = q_len[diagonal] * (q_len[diagonal] + 1) // 2
        return counts.sum(), valid.sum()

    @staticmethod
    def _range_pair_prefix(
        query_end: torch.Tensor,
        key_start: torch.Tensor,
        key_end: torch.Tensor,
    ) -> torch.Tensor:
        width = (key_end - key_start).clamp_min(0)
        steps = (query_end - key_start).clamp_min(0)
        triangular_steps = torch.minimum(steps, width)
        triangular = triangular_steps * (triangular_steps + 1) // 2
        return triangular + (steps - width).clamp_min(0) * width

    @classmethod
    def _selected_range_pair_count(
        cls,
        block_ids: torch.Tensor,
        *,
        seq_len: int,
        block_size: int,
        watched_start: int,
        watched_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = block_ids.to(torch.int64).reshape(-1)
        num_blocks = math.ceil(seq_len / block_size)
        k_block = ids.remainder(num_blocks)
        first_watched_block = watched_start // block_size
        last_watched_block = (watched_end - 1) // block_size
        overlaps = (k_block >= first_watched_block) & (
            k_block <= last_watched_block
        )
        ids = ids[overlaps]
        k_block = k_block[overlaps]
        q_block = torch.div(ids, num_blocks, rounding_mode="floor")
        q_start = q_block * block_size
        q_end = (q_start + block_size).clamp_max(seq_len)
        key_start = torch.maximum(
            k_block * block_size,
            ids.new_full(ids.shape, int(watched_start)),
        )
        key_end = torch.minimum(
            (k_block + 1) * block_size,
            ids.new_full(ids.shape, int(watched_end)),
        ).clamp_max(seq_len)
        valid_overlap = key_end > key_start
        q_start = q_start[valid_overlap]
        q_end = q_end[valid_overlap]
        key_start = key_start[valid_overlap]
        key_end = key_end[valid_overlap]
        selected_slots = (key_end - key_start).sum()
        selected_pairs = (
            cls._range_pair_prefix(q_end, key_start, key_end)
            - cls._range_pair_prefix(q_start, key_start, key_end)
        ).sum()
        return selected_slots, selected_pairs

    def _install_flexprefill(self) -> None:
        import flex_prefill.ops.flex_prefill_attention as flex_ops

        original = flex_ops.get_active_blocks
        original_transform = flex_ops.transform_veritcal_slash_idx
        instrumentation = self
        instrumentation._flex_primitives = None

        def wrapped_transform(v_idx, s_idx, num_blocks):
            result = original_transform(v_idx, s_idx, num_blocks)
            if instrumentation.current_layer.get() is not None:
                instrumentation._flex_primitives = {
                    "vertical": v_idx.detach().cpu().to(torch.int32),
                    "slash": s_idx.detach().cpu().to(torch.int32),
                    "base_blocks": [
                        [head_blocks.clone() for head_blocks in batch]
                        for batch in result
                    ],
                }
            return result

        flex_ops.transform_veritcal_slash_idx = wrapped_transform

        def wrapped_get_active_blocks(q, k, v, block_size, *args, **kwargs):
            instrumentation._flex_primitives = None
            block_idx = original(q, k, v, block_size, *args, **kwargs)
            layer = instrumentation.current_layer.get()
            if layer is None:
                return block_idx
            started = time.perf_counter()
            seq_len = int(q.shape[1])
            primitives = instrumentation._flex_primitives
            if primitives is None:
                raise RuntimeError("FlexPrefill selector primitives were not captured")
            base_blocks = primitives["base_blocks"]
            extra_rows: list[np.ndarray] = []
            selected_pair_tensors: list[torch.Tensor] = []
            selected_block_tensors: list[torch.Tensor] = []
            watched_selected_slots: dict[str, list[torch.Tensor]] = {}
            watched_selected_pairs: dict[str, list[torch.Tensor]] = {}
            for batch_index, batch in enumerate(block_idx):
                for head_index, head_indices in enumerate(batch):
                    indices = torch.unique(head_indices.to(torch.int64), sorted=True)
                    pair_count, block_count = instrumentation._block_pair_count(
                        indices, seq_len, int(block_size)
                    )
                    selected_pair_tensors.append(pair_count)
                    selected_block_tensors.append(block_count)
                    for watched_start, watched_end in (
                        instrumentation.watched_key_ranges
                    ):
                        clipped_start = min(int(watched_start), seq_len)
                        clipped_end = min(int(watched_end), seq_len)
                        if clipped_end <= clipped_start:
                            continue
                        range_name = f"{watched_start}:{watched_end}"
                        range_slots, range_pairs = (
                            instrumentation._selected_range_pair_count(
                                indices,
                                seq_len=seq_len,
                                block_size=int(block_size),
                                watched_start=clipped_start,
                                watched_end=clipped_end,
                            )
                        )
                        watched_selected_slots.setdefault(
                            range_name, []
                        ).append(range_slots)
                        watched_selected_pairs.setdefault(
                            range_name, []
                        ).append(range_pairs)
                    if instrumentation.dump_path is not None:
                        base = torch.unique(
                            base_blocks[batch_index][head_index].to(torch.int64),
                            sorted=True,
                        )
                        extras = indices[~torch.isin(indices, base)]
                        extra_rows.append(
                            extras.detach().cpu().numpy().astype(np.int32)
                        )
            heads = sum(len(batch) for batch in block_idx)
            num_blocks = math.ceil(seq_len / int(block_size))
            causal_pairs = heads * seq_len * (seq_len + 1) // 2
            causal_blocks = heads * num_blocks * (num_blocks + 1) // 2
            instrumentation._queue_stats(
                torch.stack(selected_block_tensors).sum(),
                causal_blocks,
                torch.stack(selected_pair_tensors).sum(),
                causal_pairs,
            )
            watched_stats: dict[str, dict[str, Any]] = {}
            if instrumentation.watched_key_ranges:
                query_ends = (
                    torch.arange(
                        1,
                        num_blocks + 1,
                        device=q.device,
                        dtype=torch.int64,
                    )
                    * int(block_size)
                ).clamp_max(seq_len)
                for watched_start, watched_end in (
                    instrumentation.watched_key_ranges
                ):
                    clipped_start = min(int(watched_start), seq_len)
                    clipped_end = min(int(watched_end), seq_len)
                    if clipped_end <= clipped_start:
                        continue
                    range_name = f"{watched_start}:{watched_end}"
                    valid_slots_per_head = (
                        torch.minimum(
                            query_ends,
                            query_ends.new_full(
                                query_ends.shape, clipped_end
                            ),
                        )
                        - clipped_start
                    ).clamp(0, clipped_end - clipped_start).sum()
                    selector_rows_per_head = (query_ends > clipped_start).sum()
                    valid_pairs_per_head = instrumentation._valid_range_pairs(
                        0, seq_len, clipped_start, clipped_end
                    )
                    watched_stats[range_name] = {
                        "selector_rows": selector_rows_per_head * heads,
                        "valid_token_slots": valid_slots_per_head * heads,
                        "selected_token_slots": torch.stack(
                            watched_selected_slots.get(range_name, [])
                        ).sum(),
                        "valid_causal_pairs": valid_pairs_per_head * heads,
                        "selected_causal_pairs": torch.stack(
                            watched_selected_pairs.get(range_name, [])
                        ).sum(),
                    }
            instrumentation._queue_watched_stats(int(layer), watched_stats)
            if instrumentation.dump_path is None:
                return block_idx
            extra_offsets, extra_ids = _csr_from_rows(extra_rows)
            vertical = primitives["vertical"].numpy().astype(np.int32)
            slash = primitives["slash"].numpy().astype(np.int32)
            instrumentation._append(
                {
                    "layer": int(layer),
                    "head": "csr_rows",
                    "head_count": heads,
                    "kind": "flexprefill_final_block_ids",
                    "sequence_length": seq_len,
                    "block_size": int(block_size),
                    "num_blocks": num_blocks,
                    "reconstruction": (
                        "official transform(vertical_indices, slash_indices) "
                        "union extra_block_ids"
                    ),
                    "vertical_indices": _encoded_array(vertical),
                    "slash_indices": _encoded_array(slash),
                    "extra_head_offsets": _encoded_array(extra_offsets),
                    "extra_block_ids": _encoded_array(extra_ids),
                }
            )
            return block_idx

        flex_ops.get_active_blocks = wrapped_get_active_blocks

    @staticmethod
    def _valid_range_pairs(
        q_start: int, q_end: int, k_start: int, k_end: int
    ) -> int:
        total = 0
        for query in range(q_start, q_end):
            total += max(0, min(k_end, query + 1) - k_start)
        return total

    def _record_minference_mixed(
        self,
        outputs: Sequence[torch.Tensor],
        *,
        seq_len: int,
        block_size_m: int,
        block_size_n: int,
    ) -> None:
        layer = self.current_layer.get()
        head = self.current_head.get()
        if layer is None or head is None:
            return
        started = time.perf_counter()
        block_count, block_offset, column_count, column_index = outputs
        row_count = block_count.shape[-1]
        device = block_count.device
        q_start = (
            torch.arange(row_count, device=device, dtype=torch.int64)
            * block_size_m
        ).view(1, 1, row_count, 1)
        q_end = (q_start + block_size_m).clamp(max=seq_len)
        q_len = q_end - q_start

        block_slots = torch.arange(
            block_offset.shape[-1], device=device, dtype=torch.int64
        ).view(1, 1, 1, -1)
        valid_blocks = block_slots < block_count.to(torch.int64).unsqueeze(-1)
        key_start = block_offset.to(torch.int64)
        key_len = (seq_len - key_start).clamp(0, block_size_n)
        block_pairs = q_len * key_len
        diagonal = key_start == q_start
        block_pairs = torch.where(
            diagonal,
            q_len * (q_len + 1) // 2,
            block_pairs,
        )
        block_pairs = torch.where(valid_blocks, block_pairs, 0).sum()

        column_slots = torch.arange(
            column_index.shape[-1], device=device, dtype=torch.int64
        ).view(1, 1, 1, -1)
        valid_columns = column_slots < column_count.to(torch.int64).unsqueeze(-1)
        columns = column_index.to(torch.int64)
        column_pairs = (
            q_end - torch.maximum(q_start, columns)
        ).clamp(min=0)
        column_pairs = torch.where(valid_columns, column_pairs, 0).sum()

        batch_heads = int(block_count.shape[0] * block_count.shape[1])
        causal_pairs = batch_heads * seq_len * (seq_len + 1) // 2
        num_q_blocks = math.ceil(seq_len / block_size_m)
        num_k_blocks = math.ceil(seq_len / block_size_n)
        if block_size_m == block_size_n:
            causal_blocks = (
                batch_heads * num_q_blocks * (num_q_blocks + 1) // 2
            )
        else:
            causal_blocks = batch_heads * num_q_blocks * num_k_blocks
        self._queue_stats(
            valid_blocks.sum(),
            causal_blocks,
            block_pairs + column_pairs,
            causal_pairs,
        )
        if self.dump_path is None:
            return
        block_count_np = block_count.detach().cpu().numpy()
        block_offset_np = block_offset.detach().cpu().numpy()
        column_count_np = column_count.detach().cpu().numpy()
        column_index_np = column_index.detach().cpu().numpy()
        block_rows: list[np.ndarray] = []
        column_rows: list[np.ndarray] = []
        selected_pairs = 0
        selected_blocks = 0
        for batch in range(block_count_np.shape[0]):
            for local_head in range(block_count_np.shape[1]):
                for row in range(block_count_np.shape[2]):
                    q_start = row * block_size_m
                    q_end = min(q_start + block_size_m, seq_len)
                    n_blocks = int(block_count_np[batch, local_head, row])
                    starts = block_offset_np[batch, local_head, row, :n_blocks]
                    starts = np.asarray(starts, dtype=np.int32)
                    block_rows.append(starts)
                    selected_blocks += starts.size
                    for key_start in starts.tolist():
                        selected_pairs += self._valid_range_pairs(
                            q_start,
                            q_end,
                            int(key_start),
                            min(int(key_start) + block_size_n, seq_len),
                        )
                    n_columns = int(column_count_np[batch, local_head, row])
                    columns = column_index_np[
                        batch, local_head, row, :n_columns
                    ]
                    columns = np.asarray(columns, dtype=np.int32)
                    column_rows.append(columns)
                    for column in columns.tolist():
                        selected_pairs += max(
                            0, q_end - max(q_start, int(column))
                        )
        block_row_offsets, block_starts = _csr_from_rows(block_rows)
        column_row_offsets, columns = _csr_from_rows(column_rows)
        causal_pairs = batch_heads * seq_len * (seq_len + 1) // 2
        num_rows = math.ceil(seq_len / block_size_m)
        causal_blocks = batch_heads * num_rows * (num_rows + 1) // 2
        self._append(
            {
                "layer": int(layer),
                "head": int(head),
                "kind": "minference_vertical_slash_final_indices",
                "sequence_length": seq_len,
                "query_block_size": block_size_m,
                "key_block_size": block_size_n,
                "block_row_offsets": _encoded_array(block_row_offsets),
                "block_starts": _encoded_array(block_starts),
                "column_row_offsets": _encoded_array(column_row_offsets),
                "columns": _encoded_array(columns),
            }
        )

    def _record_minference_block_sparse(
        self, block_index: torch.Tensor, block_size: int
    ) -> None:
        layer = self.current_layer.get()
        head = self.current_head.get()
        if layer is None or head is None:
            return
        started = time.perf_counter()
        seq_len = int(self._sample["input_tokens"]) if self._sample else int(
            block_index.shape[-2] * block_size
        )
        indices = block_index.to(torch.int64)
        num_blocks = math.ceil(seq_len / block_size)
        q_block = torch.arange(
            indices.shape[-2], device=indices.device, dtype=torch.int64
        ).view(1, 1, -1, 1)
        valid = (q_block < num_blocks) & (indices <= q_block)
        q_start = q_block * block_size
        q_len = (seq_len - q_start).clamp(0, block_size)
        key_start = indices * block_size
        key_len = (seq_len - key_start).clamp(0, block_size)
        pairs = q_len * key_len
        pairs = torch.where(
            indices == q_block,
            q_len * (q_len + 1) // 2,
            pairs,
        )
        batch_heads = int(indices.shape[0] * indices.shape[1])
        causal_pairs = batch_heads * seq_len * (seq_len + 1) // 2
        causal_blocks = batch_heads * num_blocks * (num_blocks + 1) // 2
        self._queue_stats(
            valid.sum(),
            causal_blocks,
            torch.where(valid, pairs, 0).sum(),
            causal_pairs,
        )
        if self.dump_path is None:
            return
        index_np = block_index.detach().cpu().numpy().astype(np.int32)
        rows = [
            np.unique(index_np[b, h, row])
            for b in range(index_np.shape[0])
            for h in range(index_np.shape[1])
            for row in range(index_np.shape[2])
        ]
        row_offsets, flat = _csr_from_rows(rows)
        selected_pairs = 0
        selected_blocks = 0
        cursor = 0
        for q_block, row in enumerate(rows[:num_blocks]):
            q_start = q_block * block_size
            q_end = min(q_start + block_size, seq_len)
            for key_block in row.tolist():
                if int(key_block) > q_block:
                    continue
                selected_blocks += 1
                selected_pairs += self._valid_range_pairs(
                    q_start,
                    q_end,
                    int(key_block) * block_size,
                    min((int(key_block) + 1) * block_size, seq_len),
                )
            cursor += row.size
        self._append(
            {
                "layer": int(layer),
                "head": int(head),
                "kind": "minference_block_sparse_final_indices",
                "sequence_length": seq_len,
                "block_size": int(block_size),
                "row_offsets": _encoded_array(row_offsets),
                "block_indices": _encoded_array(flat),
            }
        )

    def _record_minference_streaming(
        self, q: torch.Tensor, n_init: int, n_local: int
    ) -> None:
        layer = self.current_layer.get()
        head = self.current_head.get()
        if layer is None or head is None:
            return
        seq_len = int(q.shape[2])
        query = torch.arange(seq_len, device=q.device, dtype=torch.int64)
        prefix = torch.minimum(
            torch.full_like(query, int(n_init)), query + 1
        )
        local_start = (query + 1 - int(n_local)).clamp(min=0)
        overlap = (torch.minimum(prefix, query + 1) - local_start).clamp(min=0)
        selected_pairs = (prefix + query + 1 - local_start - overlap).sum()
        causal_pairs = seq_len * (seq_len + 1) // 2
        self._queue_stats(0, 0, selected_pairs, causal_pairs)
        self._append(
            {
                "layer": int(layer),
                "head": int(head),
                "kind": "minference_streaming",
                "sequence_length": seq_len,
                "n_init": int(n_init),
                "n_local": int(n_local),
            }
        )

    def _install_minference(self) -> None:
        import minference.modules.minference_forward as mf_forward
        import minference.ops.block_sparse_flash_attention as block_ops
        import minference.ops.pit_sparse_flash_attention_v2 as mixed_ops

        instrumentation = self
        original_kernel = mf_forward.minference_prefill_kernel

        def wrapped_kernel(q, k, v, head_id, layer_idx, config):
            layer_token = instrumentation.current_layer.set(int(layer_idx))
            head_token = instrumentation.current_head.set(int(head_id))
            try:
                return original_kernel(q, k, v, head_id, layer_idx, config)
            finally:
                instrumentation.current_head.reset(head_token)
                instrumentation.current_layer.reset(layer_token)

        mf_forward.minference_prefill_kernel = wrapped_kernel

        def wrap_conversion(name: str, *, optimized: bool) -> None:
            original = getattr(mixed_ops, name, None)
            if original is None:
                return

            def wrapped(*args, **kwargs):
                outputs = original(*args, **kwargs)
                if optimized:
                    seq_len = int(args[4])
                    block_m = int(args[5])
                    block_n = int(args[6])
                else:
                    seq_len = int(args[3])
                    block_m = int(args[4])
                    block_n = int(args[5])
                instrumentation._record_minference_mixed(
                    outputs,
                    seq_len=seq_len,
                    block_size_m=block_m,
                    block_size_n=block_n,
                )
                return outputs

            setattr(mixed_ops, name, wrapped)

        wrap_conversion("convert_vertical_slash_indexes_opt", optimized=True)
        wrap_conversion("convert_vertical_slash_indexes", optimized=False)

        original_build = block_ops._build_block_index

        def wrapped_build(query, key, top_k, block_size_m=64, block_size_n=64):
            result = original_build(
                query, key, top_k, block_size_m, block_size_n
            )
            instrumentation._record_minference_block_sparse(
                result, int(block_size_n)
            )
            return result

        block_ops._build_block_index = wrapped_build

        original_streaming = mf_forward.streaming_forward

        def wrapped_streaming(q, k, v, n_init, n_local):
            instrumentation._record_minference_streaming(q, n_init, n_local)
            return original_streaming(q, k, v, n_init, n_local)

        mf_forward.streaming_forward = wrapped_streaming


def install_baseline_sparsity_instrumentation(
    model, method: str, dump_path: str | Path | None = None
) -> BaselineSparsityInstrumentation:
    return BaselineSparsityInstrumentation(model, method, dump_path)
