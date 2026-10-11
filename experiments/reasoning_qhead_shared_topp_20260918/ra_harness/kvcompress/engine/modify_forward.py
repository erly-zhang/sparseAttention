# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
# Derived from VaSE's eval/reasoning_tasks/modified/transformers/modify_forward.py (https://github.com/terarachang/VaSE, MIT,
# commit bfa2692): the Qwen3 evict-attention forward is upstream's; Random Attention adds the phi3
# (Phi-4-reasoning) and llama (DeepSeek-R1-Distill-Llama) forward patches (+108 lines). See THIRD_PARTY_NOTICES.md.
import torch
from typing import Optional
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from flash_attn import flash_attn_with_kvcache


def cache_attention(layer, query, key, value, scale):
    """Route full-history shared masks before attention; legacy eviction stays separate."""
    if getattr(layer, 'uses_temporary_mask', False) or getattr(layer, 'records_attention_metrics', False):
        return layer.attention(query, key, value, scale)
    kc, vc, lengths = layer.prepare(key, value)
    output = flash_attn_with_kvcache(query, kc, vc, k=key, v=value,
                                   cache_seqlens=lengths, causal=True,
                                   softmax_scale=scale, window_size=(-1, -1))
    layer.post_attention_update(key.shape[1])
    return output


def wrap_evict_attn_forward(model_name_or_path):
    def qwen3_evict_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]            # (B, L)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (B, L, H, D)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        # flash-attn expects [B, L, H, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = cache_attention(layer, query_states, key_states, value_states, self.scaling)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def phi3_evict_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]            # (B, L)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (B, L, H, D)

        # phi3: single combined qkv_proj -> split; NO q_norm/k_norm (unlike Qwen3)
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        # flash-attn expects [B, L, H, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = cache_attention(layer, query_states, key_states, value_states, self.scaling)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def llama_evict_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]            # (B, L)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (B, L, H, D)

        # llama: separate q/k/v_proj; NO q_norm/k_norm (unlike Qwen3); standard RoPE/GQA.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        # flash-attn expects [B, L, H, D]
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        attn_output = cache_attention(layer, query_states, key_states, value_states, self.scaling)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if "Qwen3" in model_name_or_path:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb
        Qwen3Attention.forward = qwen3_evict_attn_forward
    elif "phi-4" in model_name_or_path.lower() or "phi3" in model_name_or_path.lower():
        from transformers.models.phi3.modeling_phi3 import Phi3Attention, apply_rotary_pos_emb
        Phi3Attention.forward = phi3_evict_attn_forward
    elif "llama" in model_name_or_path.lower():
        from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
        LlamaAttention.forward = llama_evict_attn_forward
    else:
        raise ValueError(f"Unknown evict attn implementation: {model_name_or_path}")
