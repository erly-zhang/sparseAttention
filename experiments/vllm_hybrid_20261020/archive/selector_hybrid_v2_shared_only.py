"""Hybrid W1 pool-compaction selector for TriAttention's vLLM runtime.

Runs when the runtime triggers a compaction (pool > budget + divide_length, i.e. with
KV_BUDGET=K+4096 and DIVIDE_LENGTH=1024 the pool peaks at K+5120 tokens). Keeps
  latest W1 core (the K positions selected at the previous decode step for this
  request and layer) U the full prompt U uniform-random fill up to budget_total,
in chronological order, shared by all KV heads (mode="shared"; per-head layouts would
break the shared position space the per-step selection relies on).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import torch

from . import hybrid_attention as ha
from .config import TriAttentionRuntimeConfig

COUNTS = {"events": 0, "core_missing": 0, "length_mismatch": 0, "in_flight": 0}


def _stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def build_hybrid_selector(
    config: TriAttentionRuntimeConfig,
    base_runner: Any | None = None,
) -> tuple[Callable[..., dict[str, Any] | None], None, str]:
    base_seed = int(os.environ.get("TRIATTN_RUNTIME_RANDOM_SEED", "1234"))
    generators: dict[torch.device, torch.Generator] = {}
    if base_runner is not None and ha.RUNNER is None:
        ha.RUNNER = base_runner

    def _active_req_id() -> str:
        return str(getattr(base_runner, "_triattention_active_req_id", "?"))

    def _select_keep_indices(
        *,
        keys_dense: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None,
        block_ids: "list[int] | torch.Tensor | None" = None,
        block_size: int | None = None,
        total_tokens: int,
        prefill_len: int,
        protect_prefill: bool,
        layer_idx: int,
        round_start: int,
        budget_total: int,
    ) -> dict[str, Any] | None:
        if total_tokens <= budget_total:
            return {"mode": "shared", "indices": list(range(total_tokens))}
        if prefill_len > budget_total:
            return None
        device = kv_cache.device if kv_cache is not None else keys_dense.device
        req = _active_req_id()
        COUNTS["events"] += 1
        keep = torch.zeros(total_tokens, device=device, dtype=torch.bool)
        entry = ha.REGISTRY.get(layer_idx)
        row = entry[0].get(req) if entry is not None else None
        if row is None:
            # No recorded selection (cannot happen once n > K); keep recent 64 instead.
            COUNTS["core_missing"] += 1
            keep[max(0, total_tokens-64):] = True
        else:
            _, idx, n_cpu, entry_step = entry
            n_prev = int(n_cpu[row])
            if n_prev != total_tokens:
                # Worker self-triggered events (async scheduling) count the in-flight token:
                # total_tokens == n_prev + 1, and that slot is overwritten by this step's
                # token after compaction, so it must stay in the kept set (the runtime's own
                # selectors keep it through their recency window).
                key = "in_flight" if total_tokens == n_prev+1 else "length_mismatch"
                COUNTS[key] = COUNTS.get(key, 0)+1
                if key == "length_mismatch" and COUNTS[key] <= 5:
                    print(f"[HYBRID-SELECTOR] length mismatch req={req} layer={layer_idx} "
                          f"recorded={n_prev} total_tokens={total_tokens} entry_step={entry_step} "
                          f"step={ha.STEP[0]} round_start={round_start}", flush=True)
                if n_prev > total_tokens:
                    raise RuntimeError(f"hybrid registry newer than the cache: {n_prev} > {total_tokens}")
            core = idx[row, :min(n_prev, ha.hybrid_k())].long()
            keep[core] = True
            keep[n_prev:] = True
        keep[:prefill_len] = True
        if int(keep.sum()) > budget_total:
            raise RuntimeError("hybrid core + prompt exceed the pool budget")
        gen = generators.get(device)
        if gen is None:
            gen = generators[device] = torch.Generator(device=device)
        gen.manual_seed(_stable_seed(base_seed, ha.stable_req_id(req), round_start, layer_idx))
        scores = torch.rand(total_tokens, device=device, generator=gen)
        scores = torch.where(keep, 2.0, scores)
        chosen = torch.topk(scores, k=int(budget_total), largest=True, sorted=False).indices
        if COUNTS["events"] % 3600 == 0:
            print(f"[HYBRID-SELECTOR] counts {COUNTS}", flush=True)
        return {"mode": "shared", "indices": torch.sort(chosen).values.contiguous()}

    setattr(_select_keep_indices, "_supports_paged", True)
    return _select_keep_indices, None, "hybrid_w1"
