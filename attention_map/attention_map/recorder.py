"""Capture and save attention maps indexed by (layer_i, head_j)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


def _attn_to_numpy(attn: torch.Tensor) -> np.ndarray:
    """[batch, heads, q_len, k_len] -> float16 numpy on CPU."""
    if attn is None:
        raise ValueError("Attention weights are None; use attn_implementation='eager'.")
    return attn.detach().float().cpu().numpy().astype(np.float16)


def save_layer_head_maps(
    attn: np.ndarray,
    out_dir: Path,
    layer_idx: int,
    query_slice: str = "all",
    query_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Save attention for each head at layer `layer_idx`.

    attn shape: [batch, num_heads, q_len, k_len]
    Files: layer_{i:02d}/head_{j:02d}.npy

    query_slice:
      - "all": full [q_len, k_len] per head
      - "last": only last query row -> [k_len]
      - "index": row query_index -> [k_len]
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected 4D attention, got shape {attn.shape}")

    batch, num_heads, q_len, k_len = attn.shape
    if batch != 1:
        attn = attn[:1]

    layer_dir = out_dir / f"layer_{layer_idx:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    for head_idx in range(num_heads):
        head_attn = attn[0, head_idx]
        if query_slice == "last":
            head_attn = head_attn[-1:]
        elif query_slice == "index":
            if query_index is None:
                raise ValueError("query_index required when query_slice='index'")
            head_idx_q = max(0, min(int(query_index), q_len - 1))
            head_attn = head_attn[head_idx_q : head_idx_q + 1]

        rel_path = f"layer_{layer_idx:02d}/head_{head_idx:02d}.npy"
        np.save(layer_dir / f"head_{head_idx:02d}.npy", head_attn)
        saved.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "path": rel_path,
                "shape": list(head_attn.shape),
            }
        )

    return {
        "layer": layer_idx,
        "num_heads": num_heads,
        "q_len": q_len,
        "k_len": k_len,
        "query_slice": query_slice,
        "query_index": query_index,
        "heads": saved,
    }


class AttentionHookRecorder:
    """Register hooks on each decoder layer; flush attention to disk per layer."""

    def __init__(
        self,
        model: nn.Module,
        output_dir: Path,
        query_slice: str = "last",
        query_index: Optional[int] = None,
        dtype_save: str = "float16",
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.query_slice = query_slice
        self.query_index = query_index
        self.dtype_save = dtype_save
        self._handles: List[Any] = []
        self.layer_manifests: List[Dict[str, Any]] = []

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError("Unsupported model architecture: cannot find .model.layers")

    def register(self) -> None:
        layers = self._get_layers()

        def make_hook(layer_idx: int):
            def hook(_module, _args, output):
                # Qwen2Attention: (attn_output, attn_weights, past_key_value)
                if not isinstance(output, tuple) or len(output) < 2:
                    return
                attn_weights = output[1]
                if attn_weights is None:
                    return
                manifest = save_layer_head_maps(
                    _attn_to_numpy(attn_weights),
                    self.output_dir,
                    layer_idx,
                    query_slice=self.query_slice,
                    query_index=self.query_index,
                )
                self.layer_manifests.append(manifest)
                del attn_weights

            return hook

        for i, layer in enumerate(layers):
            handle = layer.self_attn.register_forward_hook(make_hook(i))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def write_manifest(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        manifest = {
            "indexing": "(layer_i, head_j) -> layer_{i:02d}/head_{j:02d}.npy",
            "query_slice": self.query_slice,
            "layers": self.layer_manifests,
        }
        if extra:
            manifest.update(extra)
        path = self.output_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return path


def record_via_output_attentions(
    attentions: Tuple[torch.Tensor, ...],
    output_dir: Path,
    query_slice: str = "last",
    query_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fallback: save from model forward output_attentions tuple."""
    manifests = []
    for layer_idx, attn in enumerate(attentions):
        manifest = save_layer_head_maps(
            _attn_to_numpy(attn),
            output_dir,
            layer_idx,
            query_slice=query_slice,
            query_index=query_index,
        )
        manifests.append(manifest)
    return manifests
