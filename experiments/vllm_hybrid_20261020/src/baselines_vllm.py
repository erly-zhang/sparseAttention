"""SnapKV / R-KV / VaSE inside TriAttention's vLLM 0.19 runtime (our port, not an official release).

Selection = the upstream Random Attention engine code (baseline_scoring.py, verified identical to
EvictLayer on the same tensors). vLLM's fused attention never materialises queries or attention
weights, so this module
  1. wraps FlashAttentionImpl.forward to record, per request and layer, the post-RoPE queries of the
     last W=64 positions (the upstream circular query buffer; attention itself is untouched);
  2. provides the runtime selector: at a compaction event it gathers the request's candidate K/V
     from the paged cache, scores them with the recorded queries and returns, per KV head, the
     selected K_sel = budget_total - W candidates plus the W most recent positions (upstream: K-64
     selected + residual 64), sorted.
Upstream semantics kept: no prompt protection in the selection (the runtime runs with PROTECT_PREFILL=1 only for
event-handling parity with the other arms; the selector ignores the flag), smooth=True, temperature
1, R-KV lambda 0.5, VaSE n_large = KV_BUDGET/4 with 4 sinks, query window 64. Cadence comes from the
runtime (DIVIDE_LENGTH=64 = upstream residual 64). Enabled only with
TRIATTN_RUNTIME_SELECTOR in {snapkv, rkv, vase}.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

import numpy as np
import torch

from . import baseline_scoring as bs
from .kv_compaction import _resolve_token_slots_contiguous_range, _split_kv_axes

logger = logging.getLogger("triattention.vllm.runtime.baselines_vllm")
SELECTORS = ("snapkv", "rkv", "vase")
WINDOW = int(os.environ.get("BASELINE_QUERY_WINDOW", "64"))
RUNNER = None
_ORIG_FORWARD = None
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
RINGS: dict[int, torch.Tensor] = {}   # layer -> [slots, W, Hq, D] post-RoPE queries
SLOT_OF: dict[str, int] = {}          # request id -> ring slot
STATS = {"steps": 0, "events": 0, "slot_resets": 0}


def stable_req_id(req_id: str) -> str:
    head, _, tail = str(req_id).rpartition("-")
    return head if head and len(tail) == 8 else str(req_id)


def _num_slots() -> int:
    base = getattr(getattr(RUNNER, "scheduler_config", None), "max_num_seqs", 256)
    return int(os.environ.get("BASELINE_QUERY_SLOTS", base+8))


def _assign_slots(req_ids: list[str]) -> list[int]:
    # A slot is reclaimed only when its request has finished (left the runner's request states);
    # a live request that is merely unscheduled this step keeps its recorded queries.
    alive = getattr(RUNNER, "requests", None)
    live = set(alive) if isinstance(alive, dict) else set(req_ids)
    live.update(req_ids)
    slots = []
    for rid in req_ids:
        s = SLOT_OF.get(rid)
        if s is None:
            used = set(SLOT_OF.values())
            free = [i for i in range(_num_slots()) if i not in used]
            if not free:
                for old in [r for r in SLOT_OF if r not in live]:
                    del SLOT_OF[old]
                used = set(SLOT_OF.values())
                free = [i for i in range(_num_slots()) if i not in used]
            if not free:
                raise RuntimeError("query-ring slots exhausted")
            s = SLOT_OF[rid] = free[0]
            for ring in RINGS.values():   # upstream buffer starts at zero
                ring[s].zero_()
            STATS["slot_resets"] += 1
        slots.append(s)
    return slots


class StepPlan:
    __slots__ = ("tok", "slot", "ring")


def _plan(md, device):
    """Tokens whose queries enter the rings this step: the last <= W scheduled tokens of each row."""
    batch = RUNNER.input_batch
    B = batch.num_reqs
    qsl = RUNNER.query_start_loc.np[:B+1].astype(np.int64)
    slots = _assign_slots(list(batch.req_ids[:B]))
    prompt = np.asarray(batch.num_prompt_tokens[:B], dtype=np.int64)
    tok, slot, off = [], [], []
    for r in range(B):
        lo = max(int(qsl[r]), int(qsl[r+1])-WINDOW)
        tok.append(np.arange(lo, int(qsl[r+1])))
        slot.append(np.full(int(qsl[r+1])-lo, slots[r]))
        # upstream circular layout: the last min(P, W) prompt queries fill slots 0.., decode continues
        # round-robin, i.e. position t sits in slot (t - P + min(P, W)) mod W
        off.append(np.full(int(qsl[r+1])-lo, int(prompt[r])-min(int(prompt[r]), WINDOW)))
    plan = StepPlan()
    plan.tok = torch.from_numpy(np.concatenate(tok)).to(device, non_blocking=True)
    plan.slot = torch.from_numpy(np.concatenate(slot)).to(device, non_blocking=True)
    offs = torch.from_numpy(np.concatenate(off)).to(device, non_blocking=True)
    plan.ring = torch.remainder(RUNNER.positions[:md.num_actual_tokens].index_select(0, plan.tok)-offs, WINDOW)
    STATS["steps"] += 1
    return plan


def _layer_idx(layer):
    li = getattr(layer, "_baseline_layer_idx", None)
    if li is None:
        li = layer._baseline_layer_idx = int(_LAYER_RE.search(layer.layer_name).group(1))
    return li


def _patched_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                     output_scale=None, output_block_scale=None):
    md = attn_metadata
    if md is not None and RUNNER is not None:
        plan = getattr(md, "_baseline_plan", None)
        if plan is None:
            plan = md._baseline_plan = _plan(md, query.device)
        li = _layer_idx(layer)
        ring = RINGS.get(li)
        if ring is None:
            ring = RINGS[li] = torch.zeros((_num_slots(), WINDOW, self.num_heads, self.head_size),
                                           dtype=query.dtype, device=query.device)
        ring[plan.slot, plan.ring] = query.index_select(0, plan.tok)
    return _ORIG_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata, output,
                         output_scale, output_block_scale)


def _stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def build_baseline_selector(config, base_runner=None):
    kind = os.environ["TRIATTN_RUNTIME_SELECTOR"].strip().lower()
    mode = bs.MODES[kind]
    seed0 = int(os.environ.get("TRIATTN_RUNTIME_RANDOM_SEED", "1234"))
    n_large = int(os.environ.get("BASELINE_N_LARGE", config.kv_budget//4))
    rkv_lambda = float(os.environ.get("BASELINE_RKV_LAMBDA", "0.5"))
    generators: dict = {}
    if int(config.window_size) != WINDOW:
        raise RuntimeError(f"baseline selectors need WINDOW_SIZE={WINDOW} (upstream residual), got {config.window_size}")

    def _select_keep_indices(*, keys_dense=None, kv_cache=None, block_ids=None, block_size=None, total_tokens,
                             prefill_len, protect_prefill, layer_idx, round_start, budget_total):
        # Upstream compresses only once the cache exceeds the budget K; anything at or below K is kept.
        if total_tokens <= budget_total or total_tokens <= int(config.kv_budget):
            return {"mode": "shared", "indices": list(range(total_tokens))}
        # protect_prefill is deliberately ignored: the runtime flag is set (PROTECT_PREFILL=1) only so that event
        # triggering/skipping is identical to the random_pp and TriAttention arms (e.g. the scheduler's prefill-time
        # events are skipped as under budget); the selection itself stays upstream (no prompt protection).
        req = str(getattr(base_runner, "_triattention_active_req_id", "?"))
        slot = SLOT_OF.get(req)
        ring = RINGS.get(layer_idx)
        if slot is None or ring is None:
            if total_tokens == prefill_len:
                # Event scheduled before the request's prefill has run (the scheduler already counts the
                # prompt as cached; its KV is not written yet). random_pp/TriAttention keep everything here
                # (their prompt protection lifts the budget to the prompt length); do the same. Upstream
                # never compresses at prefill.
                STATS["prefill_noops"] = STATS.get("prefill_noops", 0)+1
                return {"mode": "shared", "indices": list(range(total_tokens))}
            raise RuntimeError(f"no recorded queries for req={req} layer={layer_idx} total={total_tokens} "
                               f"prefill={prefill_len} budget_total={budget_total}")
        n_cand = total_tokens-WINDOW
        k_sel = budget_total-WINDOW
        key_cache, value_cache = _split_kv_axes(kv_cache)
        blk, off = _resolve_token_slots_contiguous_range(block_ids=block_ids, block_size=block_size, start_token=0,
                                                         num_tokens=n_cand, device=key_cache.device)
        k_cand = key_cache[blk, off].transpose(0, 1).unsqueeze(0)   # [1, Hkv, n, D]
        v_cand = value_cache[blk, off].transpose(0, 1).unsqueeze(0)
        queries = ring[slot].permute(1, 0, 2).contiguous().unsqueeze(0)  # [1, Hq, W, D], upstream layout
        gen = None
        if mode == "range_sink_sample_attn":
            gen = generators.get(k_cand.device)
            if gen is None:
                gen = generators[k_cand.device] = torch.Generator(device=k_cand.device)
            gen.manual_seed(_stable_seed(seed0, stable_req_id(req), round_start, layer_idx))
        ids = bs.select(mode, k_cand, v_cand, queries, k_sel, rkv_lambda=rkv_lambda, smooth=True, n_large=n_large,
                        temperature=1.0, generator=gen)[0]          # [Hkv, k_sel]
        recent = torch.arange(n_cand, total_tokens, device=ids.device).expand(ids.shape[0], -1)
        keep = torch.sort(torch.cat([ids, recent], dim=-1), dim=-1).values.contiguous()
        STATS["events"] += 1
        if STATS["events"] % 36000 == 0:
            logger.info("[Baseline-%s] stats %s", kind, STATS)
        return {"mode": "per_head", "indices": keep}

    setattr(_select_keep_indices, "_supports_paged", True)
    return _select_keep_indices, None, f"baseline_{kind}"


def install():
    global _ORIG_FORWARD
    if _ORIG_FORWARD is not None:
        return
    import vllm.v1.attention.backends.flash_attn as fa
    import vllm.v1.worker.gpu_worker as worker_mod
    _ORIG_FORWARD = fa.FlashAttentionImpl.forward
    fa.FlashAttentionImpl.forward = _patched_forward
    orig_init = worker_mod.Worker.init_device

    def init_device(self):
        global RUNNER
        orig_init(self)
        RUNNER = self.model_runner

    worker_mod.Worker.init_device = init_device
    logger.info("[Baseline] query capture installed for selector=%s window=%d",
                os.environ.get("TRIATTN_RUNTIME_SELECTOR"), WINDOW)
