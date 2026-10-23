"""SnapKV / R-KV / VaSE selection, copied from the Random Attention engine (upstream 64db9688).

Source: kvcompress/engine/cache_utils.py, EvictLayer._run_eviction, modes
  'attn'                    (SnapKV, smooth=True)
  'attn_rkv'                (R-KV, rkv_lambda=0.5, smooth=True)
  'range_sink_sample_attn'  (VaSE, n_large=K/4, smooth=True)
The nested helpers compute_attention_scores / cal_redundancy / multinomial_sample and the
selection branches are reproduced line for line; only `self.*` became arguments and the
multinomial draw takes an explicit generator (upstream: the global CUDA stream). Inputs use the
upstream layout: k/v candidates [B, Hkv, n, D], queries [B, Hq, W, D] (post-RoPE, W = 64).
Returns ids_to_keep [B, Hkv, K_sel] in upstream (unsorted) order.
"""
import math

import torch
import torch.nn.functional as F

MODES = {'snapkv': 'attn', 'rkv': 'attn_rkv', 'vase': 'range_sink_sample_attn'}


def compute_attention_scores(query_cache, keys, temperature, smooth):
    # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L8
    batch_size, q_heads, q_len, head_dim = query_cache.shape
    kv_heads = keys.shape[1]
    num_kv_groups = q_heads // kv_heads
    assert num_kv_groups > 1, "not GQA"

    query_cache = query_cache.view(batch_size, kv_heads, num_kv_groups, q_len, head_dim)
    keys = keys.unsqueeze(2)

    # shape: [batch_size, kv_heads, num_kv_groups, q_len, kv_len]
    attn_weights = torch.matmul(
        query_cache, keys.transpose(3, 4)
    ) / (math.sqrt(head_dim) * temperature)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_cache.dtype)
    scores = attn_weights.mean(-2)  # avg over q_len (window_size)
    if smooth:
        # https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py#L96
        orig_shape = scores.shape
        scores = scores.view(batch_size, q_heads, -1)
        kernel_size = 5
        scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        scores = scores.view(orig_shape)
    scores = scores.mean(2)  # avg heads in the same q_group
    return scores


def cal_redundancy(keys):
    # https://github.com/Zefan-Cai/R-KV/blob/main/HuggingFace/rkv/utils.py#L42
    k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    cos_similarity = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    # zero diagonal (self-similarity)
    diag = torch.eye(keys.size(2), device=keys.device, dtype=torch.bool)
    cos_similarity.masked_fill_(diag.unsqueeze(0).unsqueeze(0), 0.0)
    return cos_similarity.mean(dim=2).softmax(dim=-1)


def multinomial_sample(scores, K, generator=None):
    orig_shape = scores.shape
    flat_scores = scores.view(-1, orig_shape[-1])
    ids_to_keep = torch.multinomial(flat_scores, K, replacement=False, generator=generator)
    return ids_to_keep.view(*orig_shape[:-1], K)


def select(mode, k_candidates, v_candidates, queries, K, *, rkv_lambda=0.5, smooth=True, n_large=256,
           temperature=1.0, n_sink=4, generator=None):
    """Upstream ids_to_keep for MODE in {'attn', 'attn_rkv', 'range_sink_sample_attn'}."""
    v_magnitude = None
    if 'range' in mode:
        v_magnitude = (v_candidates.amax(-1) - v_candidates.amin(-1))
    if v_magnitude is not None and 'sink' in mode:
        v_magnitude[..., :n_sink] = float('inf')
    attn_scores = compute_attention_scores(queries, k_candidates, temperature, smooth)
    if 'rkv' in mode:
        similarities = cal_redundancy(k_candidates)
        attn_scores = attn_scores * rkv_lambda - similarities * (1 - rkv_lambda)
    if mode in ('attn', 'attn_rkv'):
        return attn_scores.topk(K, dim=-1, largest=True).indices
    if mode == 'range_sink_sample_attn':
        ids_large = v_magnitude.topk(n_large, dim=-1, largest=True).indices
        attn_scores.scatter_(dim=-1, index=ids_large, value=0.0)
        ids_attn = multinomial_sample(attn_scores, K - n_large, generator)
        return torch.cat([ids_large, ids_attn], dim=-1)
    raise ValueError(mode)
