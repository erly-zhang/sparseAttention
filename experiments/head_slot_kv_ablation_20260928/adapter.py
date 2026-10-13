"""Isolated Transformers 4.44 Llama adapter using the frozen routing kernel."""
import types
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from assembly import project, head_pairs

DENSE_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31)


def install(model, mode, routed_attention, dense_layers=DENSE_LAYERS, backend="flash_attention_2", kv_mode="q_kv"):
    if kv_mode not in ("q_kv", "q_only"):
        raise ValueError(kv_mode)
    if mode not in ("repeat", "pair", "native"):
        raise ValueError(mode)
    head_pairs(model.config.num_attention_heads)
    dense_layers = set(dense_layers)
    if not dense_layers.issubset(range(len(model.model.layers))):
        raise ValueError("dense layer index out of range")
    state = {"mode": mode, "kv_mode": kv_mode, "pairs": {}, "dense_layers": sorted(dense_layers),
             "head_pairs": head_pairs(model.config.num_attention_heads), "scale": 1.0}
    originals = []
    for i, layer in enumerate(model.model.layers):
        if i in dense_layers:
            continue
        attn = layer.self_attn
        original = attn.forward
        originals.append((attn, original))

        def make_forward(original):
            def forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None, **kwargs):
                cached = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
                if cached:
                    if hidden_states.shape[1] != 1:
                        raise ValueError("chunked prefill unsupported")
                    return original(hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_value,
                        output_attentions=output_attentions, use_cache=use_cache,
                        cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
                b, n, _ = hidden_states.shape
                if b != 1 or n < 1 or self.training or torch.is_grad_enabled():
                    raise ValueError("single-sample eval/inference mode required")
                if output_attentions or past_key_value is None or not use_cache:
                    raise ValueError("full KV cache required; attention maps are not returned")
                if position_ids is None or position_ids.shape != (b, n):
                    raise ValueError("explicit original position_ids required")
                if position_ids.dtype not in (torch.int32, torch.int64):
                    raise ValueError("integer original positions required")
                if bool((position_ids < 0).any()) or (n > 1 and not bool((position_ids[:, 1:] > position_ids[:, :-1]).all())):
                    raise ValueError("original positions must increase")
                # This experiment accepts only contiguous, unpadded prompts.
                expected_positions = torch.arange(n, device=position_ids.device)[None]
                if not torch.equal(position_ids, expected_positions):
                    raise ValueError("production adapter requires contiguous unpadded input")
                if attention_mask is not None:
                    if attention_mask.ndim == 2:
                        if attention_mask.shape != (b, n) or not bool((attention_mask == 1).all()):
                            raise ValueError("padding not supported in production adapter")
                    elif attention_mask.ndim == 4:
                        # Eager/SDPA tiny-model tests: accept only a pure causal mask.
                        allowed = torch.ones(n, n, dtype=torch.bool, device=hidden_states.device).tril()
                        if attention_mask.shape != (1, 1, n, n) or not torch.equal(attention_mask[0, 0] == 0, allowed):
                            raise ValueError("custom attention masks unsupported")
                    else:
                        raise ValueError("unrecognized attention mask")
                h, hkv, d = self.num_heads, self.num_key_value_heads, self.head_dim
                q = self.q_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
                k = self.k_proj(hidden_states).view(b, n, hkv, d).transpose(1, 2)
                v = self.v_proj(hidden_states).view(b, n, hkv, d).transpose(1, 2)
                cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(v, position_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k, v = past_key_value.update(k, v, self.layer_idx,
                    {"cos": cos, "sin": sin, "cache_position": cache_position})
                valid = torch.ones_like(position_ids, dtype=torch.bool)
                merged, counts = routed_attention(q, k, v, position_ids, valid,
                    kv_mode, d**-0.5, backend=backend)
                state["pairs"][self.layer_idx] = counts["eligible_pairs"]
                return project(merged, position_ids, valid, self.o_proj, h, mode), None, past_key_value
            return forward
        attn.forward = types.MethodType(make_forward(original), attn)

    def close():
        for attn, original in originals:
            attn.forward = original
    state["close"] = close
    return state
