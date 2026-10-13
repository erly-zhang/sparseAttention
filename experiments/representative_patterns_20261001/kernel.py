"""Adapted AutoBlock kernel with exact per-query slash membership.

Original: experiments/token_compacted_sparse.py at frozen source-lock revision.
All heads retain their own Q/K/V and contiguous GQA mapping.
"""
import math
from typing import Optional
import torch
import triton
import triton.language as tl
from experiments.token_compacted_sparse import TokenCompactedIndex

@triton.jit
def _pattern_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    row_starts_ptr,
    row_ends_ptr,
    token_indices_ptr,
    head_to_group_ptr,
    base_flags_ptr,
    slash_lookup_ptr,
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

    m_i = tl.full((BLOCK_SIZE_Q,), -1.0e20, dtype=tl.float32)
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
        base_flags = tl.load(base_flags_ptr + token_ptrs, mask=valid_lane, other=0)
        distance = query_offsets[:, None] - token_ids[None, :]
        slash = tl.load(slash_lookup_ptr + distance,
            mask=causal & (distance < k_len), other=0)
        causal = causal & (base_flags[None, :] | slash)
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


def pattern_prefill_attention(
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
    _pattern_prefill_kernel[(batch_size * num_q_heads, index.num_query_blocks)](
        q,
        k,
        v,
        output,
        index.row_starts,
        index.row_ends,
        index.token_indices,
        index.head_to_group,
        index.base_flags,
        index.slash_lookup,
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

