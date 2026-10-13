"""Adapted from audited Llama 4.44 adapter; dispatch changes only prefill attention."""
import types
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from attention import full_attention

LAYERS = tuple(range(32))


def install(model, mode='dense_full', layers=LAYERS, heads=None, options=None, calibration_callback=None):
    if mode not in ('dense_full', 'representative_autoblock', 'representative_token', 'representative_token_vertical', 'representative_token_vertical_slash', 'flexprefill', 'minference'):
        raise ValueError(mode)
    layers = tuple(layers)
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.model.layers) for i in layers):
        raise ValueError('invalid layer selection')
    if getattr(model, '_shared_map_installed', False):
        raise ValueError('adapter already installed')
    if mode.startswith('representative_') and (heads is None or options is None or any(str(i) not in heads for i in layers)):
        raise ValueError('fixed representative heads and options required')
    state = {'mode': mode, 'prefill': {}, 'decode_calls': {}, 'layers': list(layers)}
    originals = []
    for i in layers:
        attn = model.model.layers[i].self_attn
        original = attn.forward
        originals.append((attn, original))

        def make_forward(original):
            def forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None, **kwargs):
                cached = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
                if cached:
                    if hidden_states.shape[1] != 1:
                        raise ValueError('chunked prefill unsupported')
                    state['decode_calls'][self.layer_idx] = state['decode_calls'].get(self.layer_idx, 0)+1
                    return original(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs)
                b, n, _ = hidden_states.shape
                if b != 1 or n < 1 or self.training or torch.is_grad_enabled():
                    raise ValueError('single-sample eval/inference required')
                if output_attentions or not use_cache or past_key_value is None:
                    raise ValueError('full KV cache required, no returned per-head maps')
                expected = torch.arange(n, device=hidden_states.device)[None]
                if position_ids is None or position_ids.dtype not in (torch.int32, torch.int64) or not torch.equal(position_ids, expected):
                    raise ValueError('contiguous original positions required')
                if attention_mask is not None:
                    if attention_mask.ndim == 2:
                        if attention_mask.shape != (1,n) or not bool((attention_mask == 1).all()):
                            raise ValueError('padding unsupported')
                    elif attention_mask.ndim == 4:
                        if n > 4096:
                            raise ValueError('long inputs require FA2 mask handling, not a quadratic 4D mask')
                        allowed = torch.ones(n,n,dtype=torch.bool,device=hidden_states.device).tril()
                        # Transformers may append an unused, fully masked cache column.
                        if attention_mask.shape[:3] != (1,1,n) or attention_mask.shape[-1] < n:
                            raise ValueError('invalid causal mask dimensions')
                        mask = attention_mask[0,0,:,:n]
                        if not torch.equal(mask == 0, allowed):
                            raise ValueError('only pure causal masks supported')
                        if bool((mask[~allowed] > -1e4).any()) or bool((attention_mask[...,n:] > -1e4).any()):
                            raise ValueError('causal mask must fully suppress future positions')
                    else:
                        raise ValueError('unsupported attention mask')
                h, hk, d = self.num_heads, self.num_key_value_heads, self.head_dim
                q = self.q_proj(hidden_states).view(b,n,h,d).transpose(1,2)
                k = self.k_proj(hidden_states).view(b,n,hk,d).transpose(1,2)
                v = self.v_proj(hidden_states).view(b,n,hk,d).transpose(1,2)
                cos, sin = position_embeddings if position_embeddings is not None else self.rotary_emb(v,position_ids)
                q, k = apply_rotary_pos_emb(q,k,cos,sin)
                k, v = past_key_value.update(k,v,self.layer_idx,
                    {'cos':cos,'sin':sin,'cache_position':cache_position})
                output, shared, stats = full_attention(q,k,v,position_ids,mode,head=heads[str(self.layer_idx)] if heads else None,layer=self.layer_idx,options=options,calibration_callback=calibration_callback)
                state['prefill'][self.layer_idx] = stats
                del shared
                return self.o_proj(output), None, past_key_value
            return forward
        attn.forward = types.MethodType(make_forward(original),attn)
    model._shared_map_installed = True

    def close():
        for attn, original in originals:
            attn.forward = original
        model._shared_map_installed = False
    state['close'] = close
    return state
