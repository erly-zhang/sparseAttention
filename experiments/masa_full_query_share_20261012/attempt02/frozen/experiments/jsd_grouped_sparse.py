"""JSD-based offline head grouping and representative block-top-p execution.

The offline path compares token-level attention distributions. The online path
uses a project-owned block-top-p selector and FlexPrefill's Triton attention
kernel. Only offline representative heads estimate blocks; group members reuse
their representative's indices.
"""

from __future__ import annotations

import contextvars
import math
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


def _normalized_probabilities(
    attention: torch.Tensor,
    eps: float = 1e-12,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    probabilities = attention.to(
        dtype=torch.float32,
        device=device if device is not None else attention.device,
    ).clamp_min(0)
    probabilities = probabilities + eps
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)


def token_sqrt_jsd_distance(
    layer_attention: torch.Tensor,
    eps: float = 1e-12,
    *,
    compute_device: Optional[torch.device] = None,
    token_chunk_size: int = 512,
) -> torch.Tensor:
    """Return pairwise token-level sqrt(JSD / ln(2)) for one layer.

    Args:
        layer_attention: ``[num_heads, last_q, seq_len]`` attention
            probabilities. The caller is responsible for using the intended
            query rows (32 for the formal offline classification).
    """

    if compute_device is None:
        compute_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    probabilities = _normalized_probabilities(
        layer_attention, eps=eps, device=compute_device
    )
    num_heads = probabilities.shape[0]
    last_q = probabilities.shape[1]
    seq_len = probabilities.shape[2]
    pair_jsd = torch.zeros(
        (num_heads, num_heads, last_q),
        dtype=torch.float32,
        device=compute_device,
    )

    for start in range(0, seq_len, token_chunk_size):
        chunk = probabilities[..., start : start + token_chunk_size]
        p = chunk[:, None, :, :]
        q = chunk[None, :, :, :]
        midpoint = 0.5 * (p + q)
        pair_jsd.add_(
            0.5
            * (
                (p * (torch.log(p) - torch.log(midpoint))).sum(dim=-1)
                + (q * (torch.log(q) - torch.log(midpoint))).sum(dim=-1)
            )
        )

    distance = torch.sqrt(
        torch.clamp(pair_jsd / math.log(2.0), min=0.0, max=1.0)
    ).mean(dim=-1)
    distance.fill_diagonal_(0.0)
    return distance.to(device="cpu")


@dataclass
class JSDDistanceAccumulator:
    """Online accumulator so calibration attention tensors can be released."""

    sums: Optional[List[torch.Tensor]] = None
    count: int = 0

    def add(self, attentions: torch.Tensor) -> None:
        """Add ``[num_layers, num_heads, last_q, seq_len]`` attentions."""

        if attentions.ndim != 4:
            raise ValueError(f"Expected 4-D attentions, got {attentions.shape}")
        sample_distances = [
            token_sqrt_jsd_distance(attentions[layer_idx])
            for layer_idx in range(attentions.shape[0])
        ]
        if self.sums is None:
            self.sums = [matrix.clone() for matrix in sample_distances]
        else:
            if len(self.sums) != len(sample_distances):
                raise ValueError("Calibration samples disagree on layer count")
            for layer_idx, matrix in enumerate(sample_distances):
                if self.sums[layer_idx].shape != matrix.shape:
                    raise ValueError(
                        "Calibration samples disagree on attention head count"
                    )
                self.sums[layer_idx].add_(matrix)
        self.count += 1

    def mean(self) -> List[torch.Tensor]:
        if self.sums is None or self.count <= 0:
            raise ValueError("No calibration samples were accumulated")
        return [matrix / self.count for matrix in self.sums]


def kmedoids(distance: torch.Tensor, num_groups: int) -> List[List[int]]:
    """Deterministic farthest-first initialized k-medoids."""

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    num_heads = distance.shape[0]
    if not 1 <= num_groups <= num_heads:
        raise ValueError(
            f"num_groups must be in [1, {num_heads}], got {num_groups}"
        )

    first = int(distance.sum(dim=1).argmin().item())
    medoids = [first]
    while len(medoids) < num_groups:
        nearest = distance[:, medoids].min(dim=1).values
        nearest[torch.tensor(medoids, dtype=torch.long)] = -1
        medoids.append(int(nearest.argmax().item()))

    for _ in range(100):
        assignments = distance[:, medoids].argmin(dim=1)
        new_medoids: List[int] = []
        for group_idx, old_medoid in enumerate(medoids):
            members = torch.where(assignments == group_idx)[0]
            if members.numel() == 0:
                new_medoids.append(old_medoid)
                continue
            within = distance[members][:, members]
            new_medoids.append(int(members[within.sum(dim=1).argmin()].item()))
        if new_medoids == medoids:
            break
        medoids = new_medoids

    assignments = distance[:, medoids].argmin(dim=1)
    groups: List[List[int]] = []
    for group_idx in range(num_groups):
        members = torch.where(assignments == group_idx)[0].tolist()
        groups.append(sorted(int(member) for member in members))
    return groups


def build_group_config(
    mean_distances: Sequence[torch.Tensor],
    *,
    num_groups: int,
    calibration_sample_ids: Sequence[str],
    classification_last_q: int,
) -> Dict[str, Any]:
    layers: Dict[str, List[Dict[str, Any]]] = {}
    for layer_idx, distance in enumerate(mean_distances):
        layer_groups: List[Dict[str, Any]] = []
        for members in kmedoids(distance, num_groups):
            member_tensor = torch.tensor(members, dtype=torch.long)
            within = distance[member_tensor][:, member_tensor]
            representative = members[int(within.sum(dim=1).argmin().item())]
            representative_distances = distance[representative, member_tensor]
            layer_groups.append(
                {
                    "representative": representative,
                    "members": members,
                    "mean_jsd_distance": float(
                        representative_distances.mean().item()
                    ),
                    "max_jsd_distance": float(
                        representative_distances.max().item()
                    ),
                }
            )
        layers[str(layer_idx)] = layer_groups

    return {
        "schema_version": 1,
        "classification_metric": "token_sqrt_jsd",
        "classification_last_q": classification_last_q,
        "num_groups_per_layer": num_groups,
        "num_calibration_samples": len(calibration_sample_ids),
        "calibration_sample_ids": list(calibration_sample_ids),
        "layers": layers,
    }


def validate_group_config(
    group_config: Mapping[str, Any], *, num_layers: int, num_heads: int
) -> None:
    layers = group_config.get("layers", {})
    if len(layers) != num_layers:
        raise ValueError(
            f"Group config has {len(layers)} layers, model has {num_layers}"
        )
    expected = set(range(num_heads))
    for layer_idx in range(num_layers):
        groups = layers.get(str(layer_idx))
        if not groups:
            raise ValueError(f"Layer {layer_idx} has no groups")
        members = [
            int(head)
            for group in groups
            for head in group.get("members", [])
        ]
        if set(members) != expected or len(members) != num_heads:
            raise ValueError(
                f"Layer {layer_idx} groups must cover every head exactly once"
            )
        for group in groups:
            representative = int(group["representative"])
            if representative not in group["members"]:
                raise ValueError(
                    f"Layer {layer_idx} representative {representative} "
                    "is not a group member"
                )


@dataclass
class GroupedSelectorStats:
    calls: int = 0
    selected_blocks: int = 0
    causal_blocks: int = 0
    covered_attention_mass: float = 0.0
    attention_rows: int = 0
    per_layer: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def record(
        self,
        layer_idx: int,
        selected_blocks: int,
        causal_blocks: int,
        covered_attention_mass: float,
        attention_rows: int,
    ) -> None:
        self.calls += 1
        self.selected_blocks += selected_blocks
        self.causal_blocks += causal_blocks
        self.covered_attention_mass += covered_attention_mass
        self.attention_rows += attention_rows
        layer = self.per_layer.setdefault(
            layer_idx,
            {
                "calls": 0,
                "selected_blocks": 0,
                "causal_blocks": 0,
                "covered_attention_mass": 0.0,
                "attention_rows": 0,
            },
        )
        layer["calls"] += 1
        layer["selected_blocks"] += selected_blocks
        layer["causal_blocks"] += causal_blocks
        layer["covered_attention_mass"] += covered_attention_mass
        layer["attention_rows"] += attention_rows

    def snapshot(self) -> Dict[str, Any]:
        ratio = (
            self.selected_blocks / self.causal_blocks
            if self.causal_blocks
            else float("nan")
        )
        return {
            "calls": self.calls,
            "selected_blocks": self.selected_blocks,
            "causal_blocks": self.causal_blocks,
            "mean_block_keep_ratio": ratio,
            "mean_block_sparsity": 1.0 - ratio if math.isfinite(ratio) else ratio,
            "mean_selected_attention_mass": (
                self.covered_attention_mass / self.attention_rows
                if self.attention_rows
                else float("nan")
            ),
            "per_layer": {str(k): v for k, v in self.per_layer.items()},
        }


def _block_mean(sequence: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean pool ``[batch, seq, heads, dim]`` without padding bias."""

    batch_size, seq_len, num_heads, head_dim = sequence.shape
    num_blocks = math.ceil(seq_len / block_size)
    padded_len = num_blocks * block_size
    if padded_len != seq_len:
        sequence = torch.nn.functional.pad(
            sequence, (0, 0, 0, 0, 0, padded_len - seq_len)
        )
    pooled = sequence.view(
        batch_size, num_blocks, block_size, num_heads, head_dim
    ).sum(dim=2)
    valid_counts = torch.full(
        (num_blocks,),
        block_size,
        dtype=pooled.dtype,
        device=pooled.device,
    )
    valid_counts[-1] = seq_len - (num_blocks - 1) * block_size
    return pooled / valid_counts[None, :, None, None]


class RepresentativeBlockTopPSelector:
    """Project-owned representative-head block selector.

    Representative Q/K states are mean-pooled into contiguous blocks. For each
    causal query block, the smallest highest-scoring key-block set whose
    normalized mass reaches ``gamma`` is retained. The resulting flattened block
    indices are shared with every member of the representative's offline group.
    """

    def __init__(
        self,
        group_config: Mapping[str, Any],
        *,
        force_sink_block: bool = False,
        force_diagonal_block: bool = False,
    ) -> None:
        self.layers = group_config["layers"]
        self.force_sink_block = force_sink_block
        self.force_diagonal_block = force_diagonal_block
        self.current_layer: contextvars.ContextVar[Optional[int]] = (
            contextvars.ContextVar("grouped_flexprefill_layer", default=None)
        )
        self.stats = GroupedSelectorStats()

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
    ):
        layer_idx = self.current_layer.get()
        if layer_idx is None:
            raise RuntimeError("Grouped selector was called without a layer context")
        groups = self.layers[str(layer_idx)]
        num_q_heads = q.shape[2]
        num_kv_heads = k.shape[2]
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("Query head count must be divisible by KV head count")
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
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", pooled_q, pooled_k
        ) / math.sqrt(q.shape[-1])
        num_blocks = scores.shape[-1]
        causal = torch.tril(
            torch.ones(
                (num_blocks, num_blocks),
                dtype=torch.bool,
                device=scores.device,
            )
        )
        scores = scores.masked_fill(
            ~causal[None, None, :, :], float("-inf")
        )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)

        # FlexPrefill converts token budgets to block counts before invoking
        # get_active_blocks; this replacement follows that exact contract.
        min_blocks = min(num_blocks, max(1, int(min_budget)))
        max_blocks = min(num_blocks, max(1, int(max_budget)))
        representative_blocks: List[List[torch.Tensor]] = [
            [] for _ in range(q.shape[0])
        ]
        for batch_idx in range(q.shape[0]):
            for group_idx in range(len(groups)):
                flattened_indices: List[torch.Tensor] = []
                for query_block in range(num_blocks):
                    row = probabilities[
                        batch_idx, group_idx, query_block, : query_block + 1
                    ]
                    sorted_values, sorted_indices = torch.sort(
                        row, descending=True
                    )
                    cumulative = torch.cumsum(sorted_values, dim=0)
                    reached = torch.where(cumulative >= gamma)[0]
                    top_p_count = (
                        int(reached[0].item()) + 1
                        if reached.numel()
                        else int(row.numel())
                    )
                    keep_count = min(
                        max(top_p_count, min_blocks),
                        max_blocks,
                        int(row.numel()),
                    )
                    selected_keys = sorted_indices[:keep_count]
                    forced_keys: List[int] = []
                    if self.force_sink_block:
                        forced_keys.append(0)
                    if self.force_diagonal_block:
                        forced_keys.append(query_block)
                    if forced_keys:
                        selected_keys = torch.unique(
                            torch.cat(
                                [
                                    selected_keys,
                                    torch.tensor(
                                        forced_keys,
                                        device=row.device,
                                        dtype=torch.long,
                                    ),
                                ]
                            )
                        )
                    selected_keys = selected_keys[selected_keys <= query_block]
                    flattened_indices.append(
                        query_block * num_blocks + selected_keys
                    )
                representative_blocks[batch_idx].append(
                    torch.unique(torch.cat(flattened_indices)).to(torch.long)
                )

        batch_size = q.shape[0]
        full_blocks: List[List[Optional[torch.Tensor]]] = [
            [None for _ in range(num_q_heads)] for _ in range(batch_size)
        ]
        for batch_idx in range(batch_size):
            for group_idx, group in enumerate(groups):
                selected = representative_blocks[batch_idx][group_idx]
                for head_idx in group["members"]:
                    full_blocks[batch_idx][int(head_idx)] = selected

        if any(
            block is None for batch_blocks in full_blocks for block in batch_blocks
        ):
            raise RuntimeError(f"Layer {layer_idx} group config missed a head")

        selected_count = sum(
            int(block.numel())
            for batch_blocks in full_blocks
            for block in batch_blocks
            if block is not None
        )
        causal_count = (
            batch_size
            * num_q_heads
            * num_blocks
            * (num_blocks + 1)
            // 2
        )
        group_sizes = [len(group["members"]) for group in groups]
        expanded_mass = 0.0
        expanded_rows = 0
        for group_idx, group_size in enumerate(group_sizes):
            group_mass = 0.0
            group_rows = 0
            for batch_idx in range(batch_size):
                for query_block in range(num_blocks):
                    selected_flat = representative_blocks[batch_idx][group_idx]
                    row_mask = (
                        selected_flat // num_blocks == query_block
                    )
                    selected_keys = (
                        selected_flat[row_mask] % num_blocks
                    )
                    row = probabilities[
                        batch_idx, group_idx, query_block, : query_block + 1
                    ]
                    group_mass += float(row[selected_keys].sum().item())
                    group_rows += 1
            expanded_mass += group_mass * group_size
            expanded_rows += group_rows * group_size
        self.stats.record(
            layer_idx,
            selected_count,
            causal_count,
            expanded_mass,
            expanded_rows,
        )
        return full_blocks


@dataclass
class GroupedFlexPrefillPatch:
    selector: RepresentativeBlockTopPSelector
    ops_module: Any
    original_get_active_blocks: Any
    dense_layers: tuple[int, ...] = ()

    def restore_selector(self) -> None:
        self.ops_module.get_active_blocks = self.original_get_active_blocks


def install_grouped_flexprefill(
    model,
    group_config: Mapping[str, Any],
    *,
    block_size: int = 128,
    gamma: float = 0.85,
    tau: float = 0.1,
    min_budget: int = 1024,
    max_budget: Optional[int] = None,
    force_sink_block: bool = False,
    force_diagonal_block: bool = False,
    dense_layers: Optional[Sequence[int]] = None,
) -> GroupedFlexPrefillPatch:
    """Patch Qwen2 with grouped sparse prefill and optional dense layers."""

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

    original_get_active_blocks = ops_module.get_active_blocks
    selector = RepresentativeBlockTopPSelector(
        group_config,
        force_sink_block=force_sink_block,
        force_diagonal_block=force_diagonal_block,
    )
    ops_module.get_active_blocks = selector

    wrapped_layers = 0
    restored_dense_layers = 0
    dense_layer_set = set(dense_layer_tuple)
    for module in attention_modules:
        layer_idx = int(module.layer_idx)
        if layer_idx in dense_layer_set:
            # The model was loaded with FlashAttention-2. Restoring this
            # pre-patch bound method gives selected layers a true dense path.
            module.forward = original_forwards[id(module)]
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
        raise RuntimeError(
            f"Configured {wrapped_layers} sparse and "
            f"{restored_dense_layers} dense attention layers; expected "
            f"{expected_sparse_layers} sparse and "
            f"{len(dense_layer_tuple)} dense"
        )

    return GroupedFlexPrefillPatch(
        selector=selector,
        ops_module=ops_module,
        original_get_active_blocks=original_get_active_blocks,
        dense_layers=dense_layer_tuple,
    )
