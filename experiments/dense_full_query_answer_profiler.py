"""Side-channel answer-token scoring for an otherwise dense Llama prefill."""

from __future__ import annotations

from collections import defaultdict
import json
import math
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _projection_rows(heads: Sequence[int], head_dim: int, device) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                int(head) * head_dim,
                (int(head) + 1) * head_dim,
                device=device,
            )
            for head in heads
        ]
    )


class DenseFullQueryAnswerProfiler:
    """Measure selector-style answer scores without changing dense attention."""

    def __init__(
        self,
        model,
        group_config_path: str | Path,
        selector,
        output_dir: str | Path,
        *,
        query_block_size: int = 128,
        query_chunk_size: int = 8,
        topk_budget: int = 8192,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.selector = selector
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.output_dir / "answer_score_rows.jsonl"
        self.rows_path.write_text("", encoding="utf-8")
        config = json.loads(Path(group_config_path).read_text(encoding="utf-8"))
        self.layers = config["layers"]
        self.query_block_size = int(query_block_size)
        self.query_chunk_size = int(query_chunk_size)
        self.topk_budget = int(topk_budget)
        self.top_p = float(top_p)
        self.rows: list[dict[str, Any]] = []
        self.original_forwards: dict[int, Any] = {}
        self.sample_index = -1
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        modules = []
        for module in self.model.modules():
            layer_idx = getattr(module, "layer_idx", None)
            if (
                layer_idx is None
                or not hasattr(module, "q_proj")
                or not hasattr(module, "k_proj")
            ):
                continue
            modules.append(module)
        expected = int(self.model.config.num_hidden_layers)
        if len(modules) != expected:
            raise RuntimeError(
                f"Found {len(modules)} attention modules, expected {expected}"
            )
        for module in modules:
            layer_idx = int(module.layer_idx)
            original_forward = module.forward
            self.original_forwards[id(module)] = original_forward
            profiler = self

            def wrapped_forward(
                module_self,
                *args,
                __forward=original_forward,
                __layer_idx=layer_idx,
                **kwargs,
            ):
                hidden_states = kwargs.get(
                    "hidden_states", args[0] if args else None
                )
                ranges = tuple(
                    getattr(profiler.selector, "watched_key_ranges", ())
                )
                if (
                    hidden_states is not None
                    and hidden_states.ndim == 3
                    and hidden_states.shape[0] == 1
                    and hidden_states.shape[1] > 1
                    and ranges
                ):
                    if __layer_idx == 0:
                        profiler.sample_index += 1
                    profiler._profile_layer(
                        module_self,
                        hidden_states,
                        kwargs.get("position_embeddings"),
                        layer_idx=__layer_idx,
                        answer_ranges=ranges,
                    )
                return __forward(*args, **kwargs)

            module.forward = types.MethodType(wrapped_forward, module)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for module in self.model.modules():
            original = self.original_forwards.get(id(module))
            if original is not None:
                module.forward = original
        self._installed = False

    @torch.no_grad()
    def _profile_layer(
        self,
        module,
        hidden_states: torch.Tensor,
        position_embeddings,
        *,
        layer_idx: int,
        answer_ranges: Sequence[tuple[int, int]],
    ) -> None:
        if position_embeddings is None:
            raise RuntimeError("Dense profiler requires external RoPE embeddings")
        if len(answer_ranges) != 1:
            raise RuntimeError(
                f"Expected one answer-value range, found {answer_ranges}"
            )

        groups = self.layers[str(layer_idx)]
        representatives = [int(group["representative"]) for group in groups]
        num_q_heads = int(module.num_heads)
        num_kv_heads = int(module.num_key_value_heads)
        if num_q_heads % num_kv_heads:
            raise ValueError("Q heads must be divisible by KV heads")
        q_per_kv = num_q_heads // num_kv_heads
        kv_heads = [head // q_per_kv for head in representatives]
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = int(module.head_dim)

        q_rows = _projection_rows(representatives, head_dim, hidden_states.device)
        k_rows = _projection_rows(kv_heads, head_dim, hidden_states.device)
        q_bias = (
            module.q_proj.bias.index_select(0, q_rows)
            if module.q_proj.bias is not None
            else None
        )
        k_bias = (
            module.k_proj.bias.index_select(0, k_rows)
            if module.k_proj.bias is not None
            else None
        )
        q = F.linear(
            hidden_states,
            module.q_proj.weight.index_select(0, q_rows),
            q_bias,
        ).view(batch_size, seq_len, len(representatives), head_dim)
        k = F.linear(
            hidden_states,
            module.k_proj.weight.index_select(0, k_rows),
            k_bias,
        ).view(batch_size, seq_len, len(representatives), head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q[0].to(torch.float32)
        k = k[0].to(torch.float32)

        block = self.query_block_size
        num_tiles = (seq_len + block - 1) // block
        pooled = []
        starts = []
        ends = []
        for tile_idx in range(num_tiles):
            start = tile_idx * block
            end = min(start + block, seq_len)
            starts.append(start)
            ends.append(end)
            pooled.append(q[:, start:end].mean(dim=1))
        pooled_q = torch.stack(pooled, dim=1)
        answer_start, answer_end = answer_ranges[0]
        answer_indices = torch.arange(
            answer_start,
            min(answer_end, seq_len),
            device=hidden_states.device,
        )
        if answer_indices.numel() == 0:
            raise RuntimeError(
                f"Answer range {answer_ranges[0]} is outside length {seq_len}"
            )

        accumulators = [
            {
                "slot_count": 0,
                "tile_count": 0,
                "score_sum": 0.0,
                "score_sq_sum": 0.0,
                "score_min": float("inf"),
                "score_max": float("-inf"),
                "rank_sum": 0.0,
                "rank_fraction_sum": 0.0,
                "percentile_sum": 0.0,
                "probability_sum": 0.0,
                "answer_mass_sum": 0.0,
                "topk_hits": 0,
                "topp_hits": 0,
            }
            for _ in representatives
        ]
        scale = 1.0 / math.sqrt(head_dim)
        key_positions = torch.arange(seq_len, device=hidden_states.device)

        for chunk_start in range(0, num_tiles, self.query_chunk_size):
            chunk_end = min(chunk_start + self.query_chunk_size, num_tiles)
            scores = torch.einsum(
                "grd,gtd->grt", pooled_q[:, chunk_start:chunk_end], k
            )
            scores.mul_(scale)
            for local_idx, tile_idx in enumerate(range(chunk_start, chunk_end)):
                start, end = starts[tile_idx], ends[tile_idx]
                tile_q = q[:, start:end]
                suffix_sum = torch.flip(
                    torch.cumsum(torch.flip(tile_q, dims=(1,)), dim=1),
                    dims=(1,),
                )
                suffix_count = torch.arange(
                    end - start,
                    0,
                    -1,
                    device=hidden_states.device,
                    dtype=torch.float32,
                ).view(1, -1, 1)
                suffix_mean = suffix_sum / suffix_count
                scores[:, local_idx, start:end] = (
                    suffix_mean * k[:, start:end]
                ).sum(dim=-1) * scale
                scores[:, local_idx].masked_fill_(key_positions >= end, float("-inf"))

                valid_answer = answer_indices[answer_indices < end]
                if valid_answer.numel() == 0:
                    continue
                row_scores = scores[:, local_idx, :end]
                answer_scores = row_scores.index_select(1, valid_answer)
                log_normalizer = torch.logsumexp(row_scores, dim=-1)
                all_probabilities = torch.exp(
                    row_scores - log_normalizer.unsqueeze(1)
                )
                answer_probabilities = all_probabilities.index_select(
                    1, valid_answer
                )
                comparisons = row_scores.unsqueeze(1) > answer_scores.unsqueeze(2)
                ranks = comparisons.sum(dim=-1) + 1
                higher_mass = (
                    comparisons.to(all_probabilities.dtype)
                    * all_probabilities.unsqueeze(1)
                ).sum(dim=-1)
                rank_fractions = ranks.to(torch.float32) / float(end)
                percentiles = 1.0 - (
                    (ranks.to(torch.float32) - 1.0) / max(end - 1, 1)
                )
                topk_limit = min(self.topk_budget, end)

                for group_idx, accumulator in enumerate(accumulators):
                    group_scores = answer_scores[group_idx]
                    group_probabilities = answer_probabilities[group_idx]
                    slots = int(group_scores.numel())
                    accumulator["slot_count"] += slots
                    accumulator["tile_count"] += 1
                    accumulator["score_sum"] += float(group_scores.sum().item())
                    accumulator["score_sq_sum"] += float(
                        group_scores.square().sum().item()
                    )
                    accumulator["score_min"] = min(
                        accumulator["score_min"], float(group_scores.min().item())
                    )
                    accumulator["score_max"] = max(
                        accumulator["score_max"], float(group_scores.max().item())
                    )
                    accumulator["rank_sum"] += float(ranks[group_idx].sum().item())
                    accumulator["rank_fraction_sum"] += float(
                        rank_fractions[group_idx].sum().item()
                    )
                    accumulator["percentile_sum"] += float(
                        percentiles[group_idx].sum().item()
                    )
                    accumulator["probability_sum"] += float(
                        group_probabilities.sum().item()
                    )
                    accumulator["answer_mass_sum"] += float(
                        group_probabilities.sum().item()
                    )
                    accumulator["topk_hits"] += int(
                        (ranks[group_idx] <= topk_limit).sum().item()
                    )
                    accumulator["topp_hits"] += int(
                        (higher_mass[group_idx] < self.top_p).sum().item()
                    )

        layer_rows = []
        for group_idx, (group, kv_head, accumulator) in enumerate(
            zip(groups, kv_heads, accumulators)
        ):
            slots = int(accumulator["slot_count"])
            tiles = int(accumulator["tile_count"])
            if slots == 0 or tiles == 0:
                raise RuntimeError(
                    f"No legal answer slots at layer {layer_idx}, group {group_idx}"
                )
            mean_score = accumulator["score_sum"] / slots
            variance = max(
                accumulator["score_sq_sum"] / slots - mean_score * mean_score,
                0.0,
            )
            row = {
                "sample_index": int(self.sample_index),
                "layer": int(layer_idx),
                "group_index": int(group_idx),
                "representative_head": int(group["representative"]),
                "kv_head": int(kv_head),
                "sequence_length": int(seq_len),
                "answer_value_token_start": int(answer_start),
                "answer_value_token_end": int(answer_end),
                "answer_value_token_count": int(answer_end - answer_start),
                "query_tiles_with_legal_answer": tiles,
                "answer_token_query_slots": slots,
                "mean_logit_score": mean_score,
                "std_logit_score": math.sqrt(variance),
                "min_logit_score": accumulator["score_min"],
                "max_logit_score": accumulator["score_max"],
                "mean_rank": accumulator["rank_sum"] / slots,
                "mean_rank_fraction": accumulator["rank_fraction_sum"] / slots,
                "mean_score_percentile": accumulator["percentile_sum"] / slots,
                "mean_softmax_probability_per_answer_token": (
                    accumulator["probability_sum"] / slots
                ),
                "mean_answer_probability_mass_per_query_tile": (
                    accumulator["answer_mass_sum"] / tiles
                ),
                "topk_8192_inclusion_ratio": accumulator["topk_hits"] / slots,
                "top_p_0_95_inclusion_ratio": accumulator["topp_hits"] / slots,
                "score_definition": "causal_full_query_tile_mean_qk_logit",
                "probability_denominator": "all_causally_legal_key_tokens_in_tile",
            }
            layer_rows.append(row)
            self.rows.append(row)
        with self.rows_path.open("a", encoding="utf-8") as stream:
            for row in layer_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finalize(self, metric_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        expected_samples = self.sample_index + 1
        if expected_samples != len(metric_rows):
            raise RuntimeError(
                f"Profiled {expected_samples} samples, metrics contain {len(metric_rows)}"
            )
        for row in self.rows:
            row["input_ids_sha256"] = metric_rows[row["sample_index"]][
                "input_ids_sha256"
            ]
        with self.rows_path.open("w", encoding="utf-8") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
            weights = [float(row["answer_token_query_slots"]) for row in rows]
            total = sum(weights)
            tile_total = sum(float(row["query_tiles_with_legal_answer"]) for row in rows)

            def weighted(name: str) -> float:
                return sum(
                    float(row[name]) * weight for row, weight in zip(rows, weights)
                ) / total

            return {
                "answer_token_query_slots": int(total),
                "mean_logit_score": weighted("mean_logit_score"),
                "mean_rank": weighted("mean_rank"),
                "mean_rank_fraction": weighted("mean_rank_fraction"),
                "mean_score_percentile": weighted("mean_score_percentile"),
                "mean_softmax_probability_per_answer_token": weighted(
                    "mean_softmax_probability_per_answer_token"
                ),
                "mean_answer_probability_mass_per_query_tile": sum(
                    float(row["mean_answer_probability_mass_per_query_tile"])
                    * float(row["query_tiles_with_legal_answer"])
                    for row in rows
                )
                / tile_total,
                "topk_8192_inclusion_ratio": weighted(
                    "topk_8192_inclusion_ratio"
                ),
                "top_p_0_95_inclusion_ratio": weighted(
                    "top_p_0_95_inclusion_ratio"
                ),
            }

        by_layer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_layer[str(row["layer"])].append(row)
        summary = {
            "schema_version": 1,
            "sample_count": expected_samples,
            "layer_count": len(by_layer),
            "representatives_per_layer": len(self.layers["0"]),
            "query_block_size": self.query_block_size,
            "topk_budget": self.topk_budget,
            "top_p": self.top_p,
            "score_formula": (
                "s[r,j] = mean_{i in query_tile_r, j <= i} "
                "dot(q[i], k[j]) / sqrt(head_dim)"
            ),
            "probability_formula": (
                "p[r,j] = exp(s[r,j]) / sum_{k causal for tile r} exp(s[r,k])"
            ),
            "scope": "AE3 representative Q heads observed during full dense attention",
            "overall": aggregate(self.rows),
            "layers": {
                layer: aggregate(rows) for layer, rows in sorted(by_layer.items(), key=lambda x: int(x[0]))
            },
        }
        (self.output_dir / "answer_score_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary
