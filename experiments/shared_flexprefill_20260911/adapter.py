"""Share official FlexPrefill block indices, never representative Q/K/V outputs."""
from __future__ import annotations

import contextvars
import importlib
import inspect
import types

import torch


METHOD = "flexprefill_ae3_shared_pattern"


def validate_groups(config, num_layers, num_heads, expected_groups=3):
    layers = config["layers"]
    result = {}
    if set(layers) != {str(i) for i in range(num_layers)}:
        raise ValueError("Group config must contain exactly all model layers")
    for layer in range(num_layers):
        groups = layers[str(layer)]
        if len(groups) != expected_groups:
            raise ValueError("Unexpected number of groups")
        members = [h for g in groups for h in g["members"]]
        if sorted(members) != list(range(num_heads)):
            raise ValueError("Groups must partition Q heads exactly once")
        for group in groups:
            if group["representative"] not in group["members"]:
                raise ValueError("Representative must belong to its group")
        result[layer] = groups
    return result


def representative_indices(original, q, k, v, groups, selection_kwargs):
    """Run the unmodified selector on representatives and their true KV heads."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected Q/K/V in B,N,H,D layout")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Only equal-length self-attention prefill is supported")
    heads, kv_heads = q.shape[2], k.shape[2]
    if heads % kv_heads:
        raise ValueError("Invalid GQA head counts")
    representatives = [g["representative"] for g in groups]
    interleave = selection_kwargs.get("gqa_interleave", False)
    kv = [h % kv_heads if interleave else h // (heads // kv_heads)
          for h in representatives]
    rep_ids = torch.tensor(representatives, device=q.device, dtype=torch.long)
    kv_ids = torch.tensor(kv, device=k.device, dtype=torch.long)
    kwargs = dict(selection_kwargs, gqa_interleave=False)
    # Gathered tensors have a one-to-one Q-to-KV mapping, including repeated KV.
    masks = original(q.index_select(2, rep_ids), k.index_select(2, kv_ids),
                     v.index_select(2, kv_ids), **kwargs)
    owner = [None] * heads
    for group_id, group in enumerate(groups):
        for h in group["members"]:
            if not 0 <= h < heads or owner[h] is not None:
                raise ValueError("Overlapping or invalid member head")
            owner[h] = group_id
    if any(g is None for g in owner):
        raise ValueError("Missing member head")
    if len(masks) != q.shape[0] or any(len(row) != len(groups) for row in masks):
        raise RuntimeError("Unexpected official selector result shape")
    # Indices are read-only. The original kernel still receives the full Q/K/V.
    return [[row[owner[h]] for h in range(heads)] for row in masks]


class SharedFlexPrefill:
    def __init__(self, model, config, ops=None):
        self.ops = ops or importlib.import_module("flex_prefill.ops.flex_prefill_attention")
        self.groups = validate_groups(config, model.config.num_hidden_layers,
                                      model.config.num_attention_heads)
        self.layer = contextvars.ContextVar("shared_flexprefill_layer", default=None)
        self.original = self.ops.get_active_blocks
        self.signature = inspect.signature(self.original)
        self.forwards = []
        self.selection_calls = 0
        modules = [m for m in model.modules()
                   if hasattr(m, "q_proj") and getattr(m, "layer_idx", None) is not None]
        if sorted(int(m.layer_idx) for m in modules) != sorted(self.groups):
            raise ValueError("Cannot identify exactly one attention module per layer")
        for module in modules:
            original_forward = module.forward
            layer_idx = int(module.layer_idx)

            def forward(_self, *args, _forward=original_forward,
                        _layer=layer_idx, **kwargs):
                token = self.layer.set(_layer)
                try:
                    return _forward(*args, **kwargs)
                finally:
                    self.layer.reset(token)

            self.forwards.append((module, original_forward))
            module.forward = types.MethodType(forward, module)
        self.ops.get_active_blocks = self.select

    def select(self, *args, **kwargs):
        layer = self.layer.get()
        if layer is None:
            raise RuntimeError("Shared selector called outside a model layer")
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        q, k, v = (params.pop(name) for name in ("q", "k", "v"))
        masks = representative_indices(self.original, q, k, v, self.groups[layer], params)
        self.selection_calls += 1
        return masks

    def close(self):
        self.ops.get_active_blocks = self.original
        for module, forward in self.forwards:
            module.forward = forward
