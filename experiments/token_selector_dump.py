#!/usr/bin/env python3
"""Diagnostic dump for final token-compacted kernel selectors."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.baseline_sparsity import _encoded_array
from experiments.token_compacted_sparse import TokenCompactedIndex


class TokenCompactedSelectorDump:
    """Write exact final key-token rows using run-length encoded CSR."""

    schema_version = 1

    def __init__(self, patch: Any, dump_path: str | Path) -> None:
        self.patch = patch
        self.dump_path = Path(dump_path)
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        self._sample: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []

    def reset_stats(self) -> None:
        stats = self.patch.selector.stats
        self.patch.selector.stats = type(stats)()
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
            **dict(context),
            "input_tokens": int(input_tokens),
            "input_ids_sha256": str(input_ids_sha256),
        }
        self._records = []

    @torch.no_grad()
    def capture(self, index: Any, *, layer: int | None) -> None:
        if self._sample is None or not isinstance(index, TokenCompactedIndex):
            return
        if layer is None:
            raise RuntimeError("Token selector dump requires layer context")

        tokens = index.token_indices.to(torch.int64)
        row_starts = index.row_starts.reshape(-1).to(torch.int64)
        row_ends = index.row_ends.reshape(-1).to(torch.int64)
        token_count = int(tokens.numel())

        if token_count:
            run_start = torch.ones(token_count, dtype=torch.bool, device=tokens.device)
            run_start[1:] = tokens[1:] != tokens[:-1] + 1
            valid_row_starts = row_starts[row_starts < token_count]
            run_start[valid_row_starts] = True
            run_positions = torch.nonzero(run_start, as_tuple=False).flatten()
            next_positions = torch.cat(
                (
                    run_positions[1:],
                    run_positions.new_tensor([token_count]),
                )
            )
            run_starts = tokens.index_select(0, run_positions).to(torch.int32)
            run_lengths = (next_positions - run_positions).to(torch.int32)
            run_positions_np = run_positions.detach().cpu().numpy()
        else:
            run_starts = torch.empty(0, dtype=torch.int32, device=tokens.device)
            run_lengths = torch.empty(0, dtype=torch.int32, device=tokens.device)
            run_positions_np = np.empty(0, dtype=np.int64)

        row_starts_np = row_starts.detach().cpu().numpy()
        row_ends_np = row_ends.detach().cpu().numpy()
        run_row_starts = np.searchsorted(
            run_positions_np, row_starts_np, side="left"
        ).astype(np.int64)
        run_row_ends = np.searchsorted(
            run_positions_np, row_ends_np, side="left"
        ).astype(np.int64)
        run_row_offsets = np.empty(row_starts_np.size + 1, dtype=np.int64)
        run_row_offsets[:-1] = run_row_starts
        run_row_offsets[-1] = int(run_positions_np.size)
        if not np.array_equal(run_row_offsets[1:], run_row_ends):
            raise RuntimeError("Run-length rows do not match selector CSR rows")

        self._records.append(
            {
                "layer": int(layer),
                "kind": "token_compacted_final_key_runs",
                "sequence_length": int(self._sample["input_tokens"]),
                "query_block_size": int(index.query_block_size),
                "num_query_blocks": int(index.num_query_blocks),
                "num_groups": int(index.num_groups),
                "row_shape": list(index.row_starts.shape),
                "run_row_offsets": _encoded_array(run_row_offsets),
                "run_starts": _encoded_array(
                    run_starts.detach().cpu().numpy()
                ),
                "run_lengths": _encoded_array(
                    run_lengths.detach().cpu().numpy()
                ),
                "head_to_group": _encoded_array(
                    index.head_to_group.detach().cpu().numpy().astype(np.int32)
                ),
                "selected_key_slots": int(token_count),
            }
        )

    def end_sample(self) -> dict[str, Any]:
        if self._sample is None:
            return {}
        started = time.perf_counter()
        row = {**self._sample, "selectors": self._records}
        with self.dump_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            stream.write("\n")
        metadata = {
            "selector_dump_path": str(self.dump_path),
            "selector_dump_line": int(self._sample["call_index"]),
            "selector_dump_write_latency_sec": time.perf_counter() - started,
        }
        self._sample = None
        self._records = []
        return metadata

    def abort_sample(self) -> None:
        self._sample = None
        self._records = []


def install_token_compacted_selector_dump(
    patch: Any, dump_path: str | Path
) -> TokenCompactedSelectorDump:
    """Attach final-selector dumping without changing selection or execution."""

    instrumentation = TokenCompactedSelectorDump(patch, dump_path)
    original_selector = patch.ops_module.get_active_blocks

    def wrapped_selector(*args: Any, **kwargs: Any) -> Any:
        index = original_selector(*args, **kwargs)
        instrumentation.capture(
            index, layer=patch.selector.current_layer.get()
        )
        return index

    patch.ops_module.get_active_blocks = wrapped_selector
    patch.begin_sample = instrumentation.begin_sample
    patch.end_sample = instrumentation.end_sample
    patch.abort_sample = instrumentation.abort_sample
    patch.reset_stats = instrumentation.reset_stats
    patch.selector_dump_instrumentation = instrumentation
    return instrumentation
