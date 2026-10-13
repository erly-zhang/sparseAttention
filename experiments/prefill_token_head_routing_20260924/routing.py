"""Inference-only Llama/Qwen3 routing; explicit repeat + full original o_proj.
No global patches. Full projections and full KV cache are retained.
"""
from contextlib import contextmanager
from types import MethodType
import inspect
import importlib
import torch
from torch.nn import functional as F


def routed_attention(q, k, v, positions, valid, mode, scaling, backend="eager", query_chunk=128):
    """B,H,N,d / B,Hkv,N,d -> B,N,d. Padding excluded, positions uncompressed.
    eligible_pairs counts causal pairs; matmul_pairs includes masked future
    entries in the rectangular domain, not measured kernel FLOPs.
    For FlashAttention this is an upper bound because masked tiles may be skipped.
    """
    if backend not in ("eager", "sdpa", "flash_attention_2") or query_chunk < 1:
        raise ValueError("invalid kernel configuration")
    if mode not in ("q_only", "q_kv"):
        raise ValueError(mode)
    bsz, heads, length, dim = q.shape
    kvheads = k.shape[1]
    if heads % kvheads or k.shape != v.shape or k.shape[2:] != (length, dim):
        raise ValueError("incompatible GQA shapes")
    if positions.shape != (bsz, length) or valid.shape != positions.shape:
        raise ValueError("positions/valid must be B,N")
    merged = q.new_zeros(bsz, length, dim)
    eligible = torch.zeros((), dtype=torch.int64, device=q.device)
    matmul = 0
    for b in range(bsz):
        all_idx = valid[b].nonzero(as_tuple=True)[0]
        p = positions[b, all_idx]
        if p.numel() > 1 and not bool((p[1:] > p[:-1]).all()):
            raise ValueError("valid original positions must increase strictly")
        assignments = p.remainder(heads)
        for h in range(heads):
            qi = all_idx[assignments == h]
            if qi.numel() == 0:
                continue
            ki = qi if mode == "q_kv" else all_idx
            kh = h // (heads // kvheads)
            # Keys/values are identical for every query chunk of this head.
            # Gather once, rather than copying long K/V on every iteration.
            ks, vs = k[b, kh, ki], v[b, kh, ki]
            if mode == "q_kv":
                eligible += qi.numel() * (qi.numel() + 1) // 2
            else:
                eligible += torch.searchsorted(p, positions[b, qi], right=True).sum()
            matmul += qi.numel() * ki.numel()
            if backend == "flash_attention_2" and mode == "q_kv":
                # Equal sorted Q/K indices make triangular masking exact.
                # Do not chunk Q: unequal lengths change FA2 causal alignment.
                from flash_attn import flash_attn_func
                merged[b, qi] = flash_attn_func(
                    q[b, h, qi][None, :, None, :].contiguous(),
                    ks[None, :, None, :].contiguous(),
                    vs[None, :, None, :].contiguous(),
                    dropout_p=0.0, softmax_scale=scaling, causal=True)[0, :, 0]
                continue
            # Irregular Q-only positions require the explicit SDPA mask.
            for chunk in qi.split(query_chunk):
                allowed = positions[b, ki][None, :] <= positions[b, chunk][:, None]
                qs = q[b, h, chunk]
                if backend == "eager":
                    scores = (qs @ ks.T) * scaling
                    probs = scores.float().masked_fill(~allowed, float("-inf")).softmax(-1).to(q.dtype)
                    merged[b, chunk] = probs @ vs
                else:
                    merged[b, chunk] = F.scaled_dot_product_attention(
                        qs[None, None], ks[None, None], vs[None, None],
                        attn_mask=allowed[None, None], dropout_p=0.0,
                        is_causal=False, scale=scaling)[0, 0]
    return merged, {"eligible_pairs": int(eligible), "matmul_pairs": matmul}


def repeat_project(merged, projection, heads, summed=False):
    """PyTorch W is [D_out,H*d]. Bias is applied exactly once."""
    dim = merged.shape[-1]
    if projection.in_features != heads * dim:
        raise ValueError("o_proj width must equal H * head_dim")
    if summed:
        weight = projection.weight.reshape(projection.out_features, heads, dim).sum(1)
        return F.linear(merged, weight, projection.bias)
    return projection(merged.repeat(1, 1, heads))


class RoutingController:
    """Explicit phase prevents mistaking a one-token prompt for decode.
    Use phase('prefill', valid, positions) once with an empty DynamicCache;
    then phase('decode') for each one-token step. Chunked prefill is rejected.
    Only inference, full attention, single-device Llama or Qwen3, eager, SDPA, or FlashAttention-2 backend.
    """
    def __init__(self, model, mode="q_kv", summed=False):
        if mode not in ("dense", "q_only", "q_kv"):
            raise ValueError(mode)
        if model.config.model_type not in ("llama", "qwen3"):
            raise ValueError("only Llama and Qwen3 interfaces are audited")
        if model.config._attn_implementation not in ("eager", "sdpa", "flash_attention_2"):
            raise ValueError("unsupported attention backend")
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1 or any(device.type == "meta" for device in devices):
            raise ValueError("only a fully materialized single-device model is supported")
        self.backend = model.config._attn_implementation
        self.model, self.mode, self.summed = model, mode, summed
        self.state = None
        self.counts = []
        self.originals = []
        for layer in model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "sliding_window", None) is not None:
                raise ValueError("sliding-window attention unsupported")
            if "past_key_values" not in inspect.signature(attn.forward).parameters:
                raise ValueError("unverified Transformers API: requires plural past_key_values")
        for layer in model.model.layers:
            attn = layer.self_attn
            original = attn.forward
            self.originals.append((attn, original))
            attn.forward = MethodType(self._make_forward(original), attn)
        self.guard = model.register_forward_pre_hook(self._guard_inputs, with_kwargs=True)

    def _guard_inputs(self, model, args, kwargs):
        if self.state is None:
            raise RuntimeError("explicit phase required")
        phase, valid, positions = self.state
        if phase == "prefill":
            mask, pos = kwargs.get("attention_mask"), kwargs.get("position_ids")
            if mask is None or pos is None or mask.ndim != 2:
                raise ValueError("explicit 2-D attention_mask and position_ids required")
            if pos.shape != mask.shape or pos.dtype not in (torch.int32, torch.int64):
                raise ValueError("position_ids must be integer B,N matching the mask")
            if not torch.equal(mask.bool(), valid.bool()) or not torch.equal(pos, positions):
                raise ValueError("routing metadata differs from model inputs")
            if not bool(((mask == 0) | (mask == 1)).all()) or not bool(valid.any(-1).all()):
                raise ValueError("binary padding mask with at least one valid token required")
            # Validate before the first layer mutates its cache, in A/B/C alike.
            for row, keep in zip(pos, mask.bool()):
                actual = row[keep]
                if bool((actual < 0).any()) or (actual.numel() > 1 and not bool((actual[1:] > actual[:-1]).all())):
                    raise ValueError("valid original positions must be nonnegative and strictly increasing")

    def close(self):
        self.guard.remove()
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()

    @contextmanager
    def phase(self, name, valid=None, positions=None):
        if self.state is not None or name not in ("prefill", "decode"):
            raise ValueError("invalid/nested phase")
        if name == "prefill" and (valid is None or positions is None):
            raise ValueError("prefill requires explicit valid mask and original positions")
        self.state = (name, valid, positions)
        try:
            yield self
        finally:
            self.state = None

    def _make_forward(self, original):
        ctl = self
        def forward(module, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, **kwargs):
            from transformers.cache_utils import DynamicCache
            apply_rotary_pos_emb = importlib.import_module(type(module).__module__).apply_rotary_pos_emb
            if ctl.state is None:
                raise RuntimeError("explicit prefill/decode phase required")
            if module.training or torch.is_grad_enabled():
                raise RuntimeError("inference only: eval + no_grad required")
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError("full DynamicCache required")
            phase, valid, positions = ctl.state
            cached = past_key_values.get_seq_length(module.layer_idx)
            bsz, length, _ = hidden_states.shape
            if phase == "decode":
                if length != 1 or cached == 0:
                    raise ValueError("decode requires nonempty cache and exactly one token")
                return original(hidden_states, position_embeddings, attention_mask,
                                past_key_values=past_key_values, **kwargs)
            if cached:
                raise ValueError("chunked prefill unsupported: cache must be empty")
            if valid.shape != (bsz, length) or positions.shape != valid.shape:
                raise ValueError("prefill metadata shape mismatch")
            if ctl.mode == "dense":
                lengths = valid.sum(-1)
                heads = module.config.num_attention_heads
                ctl.counts.append({"layer": module.layer_idx,
                    "eligible_pairs": int((lengths * (lengths + 1) // 2).sum()) * heads,
                    "matmul_pairs": bsz * heads * length * length})
                return original(hidden_states, position_embeddings, attention_mask,
                                past_key_values=past_key_values, **kwargs)
            shape = (bsz, length, -1, module.head_dim)
            q = module.q_proj(hidden_states).view(shape)
            k = module.k_proj(hidden_states).view(shape)
            if hasattr(module, "q_norm"):
                q = module.q_norm(q)
            if hasattr(module, "k_norm"):
                k = module.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            # All real KV heads, constructed from this method's actual states.
            past_key_values.update(k, v, module.layer_idx)
            merged, counts = routed_attention(q, k, v, positions, valid.bool(), ctl.mode, module.scaling, backend=ctl.backend)
            ctl.counts.append({"layer": module.layer_idx, **counts})
            return repeat_project(merged, module.o_proj, q.shape[1], ctl.summed), None
        return forward
