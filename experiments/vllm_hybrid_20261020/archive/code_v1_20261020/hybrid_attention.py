"""Hybrid W1 decode attention for vLLM 0.19 (FlashAttention backend, paged KV).

Per decode step and layer, for every decode row whose effective KV length n exceeds K:
  1. representative Q head r (hash of request, logical position, layer), raw BF16 QK of
     the current query head r against KV head r//G over the whole paged pool [0, n);
  2. exact chronological selection of K positions: protected prompt prefix pl,
     recent 64, and the top-(K - pl - 64) scores of the middle (ties -> earlier slot);
  3. attention of all Q heads over the selected K positions (each head on its own KV head).
Rows with n <= K attend to all n positions (identical to dense).
The selected positions of the latest step are kept per (layer, request) for the
pool-compaction selector (selector_hybrid.py); compaction itself runs through the
TriAttention runtime (budget K+4096, divide length 1024).
Enabled only with TRIATTN_RUNTIME_SELECTOR=hybrid; every other arm is untouched.
"""
from __future__ import annotations

import logging
import os
import re
import zlib

import numpy as np
import torch
import triton
import triton.language as tl

logger = logging.getLogger("triattention.vllm.runtime.hybrid_attention")

SELECT_BLOCK = 8192
SCORE_BN = 128
ATTN_BN = 64
MAX_LAYERS = 128
RUNNER = None          # base GPUModelRunner of this worker (set in Worker.init_device)
REGISTRY: dict = {}    # layer_idx -> (row_of_req, idx[int32 B,K], n_cpu) of the latest sparse step
STEP = [0]           # forward steps seen by the patch (plans built)
STATS = {"dense_steps": 0, "sparse_steps": 0, "mixed_steps": 0, "sparse_rows": 0, "layer_calls": 0}
_ORIG_FORWARD = None
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def hybrid_k() -> int:
    return int(os.environ["HYBRID_K"])


@triton.jit
def _hyb_score(Q, KC, BT, SEQ, ROWS, QROW, REP, SCORES,
               s_qt, s_qh, s_blk, s_tok, s_head, s_bt, s_sc,
               GRP: tl.constexpr, BS: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    i = tl.program_id(0)
    chunk = tl.program_id(1)
    row = tl.load(ROWS+i)
    n = tl.load(SEQ+row)
    j = chunk*BN+tl.arange(0, BN)
    m = j < n
    if chunk*BN < n:
        rep = tl.load(REP+i)
        kvh = rep//GRP
        d = tl.arange(0, D)
        q = tl.load(Q+tl.load(QROW+i).to(tl.int64)*s_qt+rep*s_qh+d).to(tl.float32)
        blk = tl.load(BT+row.to(tl.int64)*s_bt+j//BS, m, other=0).to(tl.int64)
        base = blk*s_blk+(j % BS)*s_tok+kvh*s_head
        k = tl.load(KC+base[:, None]+d[None, :], m[:, None], other=0).to(tl.float32)
        s = tl.sum(k*q[None, :], axis=1)
        tl.store(SCORES+i.to(tl.int64)*s_sc+j, s.to(tl.bfloat16), m)


@triton.jit
def _hyb_select(SCORES, SEQ, ROWS, PL, IDX, CNT, s_sc, KB: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    row = tl.load(ROWS+i)
    n = tl.load(SEQ+row)
    j = tl.arange(0, BLOCK)
    out = IDX+i.to(tl.int64)*KB
    if n <= KB:
        tl.store(out+j, j, j < n)
        tl.store(CNT+i, n)
    else:
        start = tl.load(PL+i)
        end = tl.maximum(start, n-64)
        extra = KB-start-(n-end)
        tl.store(out+j, j, j < start)
        tl.store(out+start+extra+j, end+j, j < n-end)
        tl.store(CNT+i, KB)
        if extra > 0:
            # Same packed keys and radix select as the HF v4/v6 kernels.
            valid = j < end-start
            score = tl.load(SCORES+i.to(tl.int64)*s_sc+start+j, valid, other=0)
            bits = score.to(tl.uint16, bitcast=True).to(tl.int64)
            order = tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)
            order = tl.where((bits & 0x7FFF) == 0, 0x8000, order)
            key = tl.where(valid, (order << 16) | (0xFFFF-j.to(tl.int64)), -1)
            threshold = tl.sum(tl.zeros([BLOCK], dtype=tl.int64), axis=0)
            for bit in tl.static_range(31, -1, -1):
                candidate = threshold | (1 << bit)
                hits = tl.sum((key >= candidate).to(tl.int32), axis=0)
                threshold = tl.where(hits >= extra, candidate, threshold)
            chosen = key >= threshold
            rank = tl.cumsum(chosen.to(tl.int32), axis=0)-1
            tl.store(out+start+rank, start+j, chosen)


@triton.jit
def _hyb_attn(Q, KC, VC, BT, ROWS, QROW, IDX, CNT, ML, ACC, scale,
              s_qt, s_qh, s_blk, s_tok, s_head, s_bt,
              HQ: tl.constexpr, GRP: tl.constexpr, GP: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
              KB: tl.constexpr, NSPLIT: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr):
    i = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    row = tl.load(ROWS+i)
    cnt = tl.load(CNT+i)
    c0 = sp*CH
    c1 = tl.minimum(c0+CH, cnt)
    g = tl.arange(0, GP)
    gm = g < GRP
    d = tl.arange(0, D)
    head = kvh*GRP+g
    q = tl.load(Q+tl.load(QROW+i).to(tl.int64)*s_qt+head[:, None]*s_qh+d[None, :], gm[:, None], other=0)
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    qk_scale = scale*1.4426950408889634
    for t in range(c0, c1, BN):
        jj = t+tl.arange(0, BN)
        mk = jj < c1
        pos = tl.load(IDX+i.to(tl.int64)*KB+jj, mk, other=0)
        blk = tl.load(BT+row.to(tl.int64)*s_bt+pos//BS, mk, other=0).to(tl.int64)
        base = blk*s_blk+(pos % BS)*s_tok+kvh*s_head
        k = tl.load(KC+base[:, None]+d[None, :], mk[:, None], other=0)
        s = tl.dot(q, tl.trans(k))*qk_scale
        s = tl.where(mk[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s-m_new[:, None])
        alpha = tl.math.exp2(m_i-m_new)
        l_i = l_i*alpha+tl.sum(p, 1)
        v = tl.load(VC+base[:, None]+d[None, :], mk[:, None], other=0)
        acc = acc*alpha[:, None]+tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = (i.to(tl.int64)*HQ+head)*NSPLIT+sp
    tl.store(ML+o*2, m_i, gm)
    tl.store(ML+o*2+1, l_i, gm)
    tl.store(ACC+o[:, None]*D+d[None, :], acc, gm[:, None])


@triton.jit
def _hyb_reduce(ML, ACC, QROW, OUT, s_ot, s_oh, HQ: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.arange(0, NSPLIT)
    o = (i.to(tl.int64)*HQ+h)*NSPLIT+sp
    m = tl.load(ML+o*2)
    l = tl.load(ML+o*2+1)
    mx = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.math.exp2(m-mx))
    d = tl.arange(0, D)
    acc = tl.load(ACC+o[:, None]*D+d[None, :])
    out = tl.sum(acc*w[:, None], 0)/tl.sum(l*w, 0)
    tl.store(OUT+tl.load(QROW+i).to(tl.int64)*s_ot+h*s_oh+d, out.to(OUT.dtype.element_ty))


def _fmix32(x):
    x = x & 0xFFFFFFFF
    x ^= x >> 16
    x = (x*0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x*0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def representative_heads(req_keys, positions, hq, layers=MAX_LAYERS, seed=0):
    """int32 [layers, B]: deterministic per (request, logical position, layer)."""
    base = (np.asarray(req_keys, dtype=np.uint64) ^ (np.asarray(positions, dtype=np.uint64)*np.uint64(2654435761))
            ^ np.uint64(seed*0x27D4EB2F)) & np.uint64(0xFFFFFFFF)
    lay = (np.arange(layers, dtype=np.uint64)+np.uint64(1))*np.uint64(0x9E3779B9)
    with np.errstate(over="ignore"):
        x = (base[None, :] ^ lay[:, None]) & np.uint64(0xFFFFFFFF)
        x = _fmix32(x)
    return (x % np.uint64(hq)).astype(np.int32)


_REQ_KEYS: dict[str, int] = {}


def _req_key(req_id):
    k = _REQ_KEYS.get(req_id)
    if k is None:
        if len(_REQ_KEYS) > 100000:
            _REQ_KEYS.clear()
        k = _REQ_KEYS[req_id] = zlib.crc32(str(req_id).encode())
    return k


class StepPlan:
    __slots__ = ("dense", "pure", "rows", "qrow", "pl", "rep", "row_of_req", "n_cpu", "nb", "max_n", "scores",
                 "ml", "acc", "dev_rows")


def _plan(md, hq, device):
    runner = RUNNER
    plan = StepPlan()
    if runner is None:
        plan.dense = True
        return plan
    K = hybrid_k()
    batch = runner.input_batch
    B = batch.num_reqs
    qsl = runner.query_start_loc.np[:B+1]
    qlen = np.diff(qsl)
    n = runner.optimistic_seq_lens_cpu[:B].numpy().astype(np.int64)
    decode = qlen == 1
    sparse = decode & (n > K)
    # Rows longer than the hybrid pool can be (a request the runtime has already retired
    # but async scheduling still runs for one discarded step) keep the original path.
    overflow = sparse & (n > SELECT_BLOCK)
    if overflow.any():
        STATS["overflow_rows"] = STATS.get("overflow_rows", 0)+int(overflow.sum())
        if STATS["overflow_rows"] <= 10:
            r = int(np.nonzero(overflow)[0][0])
            logger.warning("[Hybrid] overflow row=%d req=%s n=%d nct=%d num_reqs=%d -> original attention",
                           r, batch.req_ids[r], int(n[r]), int(batch.num_computed_tokens_cpu[r]), B)
        sparse &= ~overflow
    if not sparse.any():
        plan.dense = True
        return plan
    plan.dense = False
    plan.pure = bool(decode.all()) and md.max_query_len == 1 and not overflow.any()
    rows = np.arange(B) if plan.pure else np.nonzero(sparse)[0]
    req_ids = batch.req_ids[:B]
    prompt = batch.num_prompt_tokens[:B].astype(np.int64)
    pl = np.where(prompt+64 <= K, prompt, K-65)[rows]
    keys = [_req_key(req_ids[r]) for r in rows]
    positions = batch.num_computed_tokens_cpu[:B][rows]
    rep = representative_heads(keys, positions, hq, seed=int(os.environ.get("HYBRID_REP_SEED", "0")))
    nb = len(rows)
    packed = np.empty((3+MAX_LAYERS, nb), dtype=np.int32)
    packed[0] = rows
    packed[1] = qsl[rows]
    packed[2] = pl
    packed[3:] = rep
    dev = torch.from_numpy(packed).pin_memory().to(device, non_blocking=True)
    plan.dev_rows = dev
    plan.rows, plan.qrow, plan.pl, plan.rep = dev[0], dev[1], dev[2], dev[3:]
    plan.row_of_req = {req_ids[r]: i for i, r in enumerate(rows)}
    plan.n_cpu = n[rows]
    plan.nb = nb
    plan.max_n = int(n[rows].max())
    if plan.max_n > SELECT_BLOCK:  # unreachable: overflow rows are excluded above
        raise RuntimeError(f"hybrid pool {plan.max_n} exceeds SELECT_BLOCK={SELECT_BLOCK}")
    plan.scores = torch.empty((nb, SELECT_BLOCK), device=device, dtype=torch.bfloat16)
    plan.ml = None
    plan.acc = None
    if plan.pure:
        STATS["sparse_steps"] += 1
    else:
        STATS["mixed_steps"] += 1
    STATS["sparse_rows"] += int(sparse.sum())
    if os.environ.get("HYBRID_CHECK") == "1":
        gpu_n = md.seq_lens[:B].cpu().numpy()
        if not np.array_equal(gpu_n, n):
            raise RuntimeError(f"hybrid seq_lens mismatch cpu={n.tolist()} gpu={gpu_n.tolist()}")
    return plan


def sparse_attention(query, key_cache, value_cache, block_table, seq_lens, plan, layer_idx, scale, output,
                     hq, hkv, d, K):
    nb = plan.nb
    grp = hq//hkv
    bs = key_cache.shape[1]
    s_blk, s_tok, s_head, s_d = key_cache.stride()
    if s_d != 1 or value_cache.stride() != key_cache.stride() or query.stride(2) != 1 or output.stride(2) != 1:
        raise RuntimeError("unexpected KV/query layout for hybrid attention")
    rep = plan.rep[layer_idx]
    _hyb_score[(nb, triton.cdiv(plan.max_n, SCORE_BN))](
        query, key_cache, block_table, seq_lens, plan.rows, plan.qrow, rep, plan.scores,
        query.stride(0), query.stride(1), s_blk, s_tok, s_head, block_table.stride(0), plan.scores.stride(0),
        GRP=grp, BS=bs, D=d, BN=SCORE_BN, num_warps=4)
    idx = torch.empty((nb, K), device=query.device, dtype=torch.int32)
    cnt = torch.empty(nb, device=query.device, dtype=torch.int32)
    _hyb_select[(nb,)](plan.scores, seq_lens, plan.rows, plan.pl, idx, cnt, plan.scores.stride(0),
                       KB=K, BLOCK=SELECT_BLOCK, num_warps=8)
    nsplit = 8 if K <= 2048 else 16
    ch = triton.cdiv(K, nsplit)
    ch = triton.cdiv(ch, ATTN_BN)*ATTN_BN
    if plan.ml is None:
        plan.ml = torch.empty((nb*hq*nsplit*2,), device=query.device, dtype=torch.float32)
        plan.acc = torch.empty((nb*hq*nsplit, d), device=query.device, dtype=torch.float32)
    _hyb_attn[(nb, hkv, nsplit)](
        query, key_cache, value_cache, block_table, plan.rows, plan.qrow, idx, cnt, plan.ml, plan.acc, scale,
        query.stride(0), query.stride(1), s_blk, s_tok, s_head, block_table.stride(0),
        HQ=hq, GRP=grp, GP=max(16, triton.next_power_of_2(grp)), BS=bs, D=d, KB=K, NSPLIT=nsplit, CH=ch,
        BN=ATTN_BN, num_warps=4)
    _hyb_reduce[(nb, hq)](plan.ml, plan.acc, plan.qrow, output, output.stride(0), output.stride(1),
                          HQ=hq, D=d, NSPLIT=nsplit, num_warps=4)
    REGISTRY[layer_idx] = (plan.row_of_req, idx, plan.n_cpu, STEP[0])
    return idx, cnt


def _layer_idx(layer):
    li = getattr(layer, "_hybrid_layer_idx", None)
    if li is None:
        m = _LAYER_RE.search(getattr(layer, "layer_name", ""))
        if m is None:
            raise RuntimeError(f"cannot parse layer index from {getattr(layer, 'layer_name', None)!r}")
        li = layer._hybrid_layer_idx = int(m.group(1))
    return li


def _patched_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                     output_scale=None, output_block_scale=None):
    md = attn_metadata
    if (md is None or output is None or output_scale is not None or output_block_scale is not None
            or getattr(md, "use_cascade", False) or self.kv_cache_dtype.startswith("fp8")
            or self.sliding_window not in (None, (-1, -1))):
        return _ORIG_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata, output,
                             output_scale, output_block_scale)
    plan = getattr(md, "_hybrid_plan", None)
    if plan is None:
        STEP[0] += 1
        plan = _plan(md, self.num_heads, query.device)
        md._hybrid_plan = plan
        if plan.dense:
            STATS["dense_steps"] += 1
        total = STATS["dense_steps"]+STATS["sparse_steps"]+STATS["mixed_steps"]
        if total % 2048 == 0:
            logger.info("[Hybrid] stats %s", STATS)
    if plan.dense:
        return _ORIG_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata, output,
                             output_scale, output_block_scale)
    if not plan.pure:
        _ORIG_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata, output,
                      output_scale, output_block_scale)
    key_cache, value_cache = kv_cache.unbind(0)
    STATS["layer_calls"] += 1
    sparse_attention(query, key_cache, value_cache, md.block_table, md.seq_lens, plan, _layer_idx(layer),
                     self.scale, output, self.num_heads, self.num_kv_heads, self.head_size, hybrid_k())
    return output


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
        logger.info("[Hybrid] runner captured: K=%d", hybrid_k())

    worker_mod.Worker.init_device = init_device
    logger.info("[Hybrid] FlashAttentionImpl.forward patched (K=%s)", os.environ.get("HYBRID_K"))
