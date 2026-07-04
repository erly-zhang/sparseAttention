"""Load Qwen2.5 and run prefill with attention extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "eager",
):
    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if device not in ("auto",) and device != "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def build_input_text(sample_input: str, include_answer_prefix: bool = False) -> str:
    return sample_input


def tokenize(
    tokenizer,
    text: str,
    max_length: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    if device != "cpu":
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


@torch.inference_mode()
def forward_prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = True,
) -> Any:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        use_cache=False,
    )


@torch.inference_mode()
def extract_last_token_attention(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Memory-efficient extraction of the LAST query row of attention.

    Returns, for every (layer, head), the last token's attention distribution
    over all keys -- i.e. the last row of each head's [seq_len, seq_len] map --
    WITHOUT ever materializing the full attention tensor.

    Two-stage strategy:

      Stage 1 (prefill): forward ``input_ids[:, :-1]`` with ``use_cache=True`` and
        ``output_attentions=False``. We only keep ``past_key_values``; no attention
        map is produced/returned. With an eager attention implementation the
        scores are still computed internally, so when ``chunk_size`` is given the
        prefix is fed in chunks to bound each forward's attention to
        ``[num_heads, chunk_size, kv_len]`` instead of ``[num_heads, seq, seq]``.

      Stage 2 (last-token decode): forward only ``input_ids[:, -1:]`` with the
        cached ``past_key_values``, ``use_cache=False`` and
        ``output_attentions=True``. The query length is 1, so each layer returns
        ``[batch, num_heads, 1, seq_len]`` -- exactly the last row.

    Args:
        model: a CausalLM loaded with ``attn_implementation="eager"`` (otherwise
            attentions are not returned).
        input_ids: ``[1, seq_len]`` token ids.
        attention_mask: optional ``[1, seq_len]`` mask (defaults to all ones).
        chunk_size: if set, prefill is done in chunks of this many tokens.

    Returns:
        ``last_token_attn``: CPU float32 tensor of shape
        ``[num_layers, num_heads, seq_len]``.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    batch, seq_len = input_ids.shape
    if batch != 1:
        raise ValueError(f"extract_last_token_attention expects batch=1, got {batch}")
    if seq_len < 1:
        raise ValueError("Empty input_ids")

    prefix_ids = input_ids[:, :-1]
    last_id = input_ids[:, -1:]
    prefix_len = prefix_ids.shape[1]

    past_key_values = None

    # ---- Stage 1: prefill the prefix; keep only past_key_values (no attentions) ----
    if prefix_len > 0:
        if chunk_size is None or chunk_size >= prefix_len:
            out = model(
                input_ids=prefix_ids,
                attention_mask=attention_mask[:, :prefix_len],
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values = out.past_key_values
            del out
        else:
            start = 0
            while start < prefix_len:
                end = min(start + chunk_size, prefix_len)
                out = model(
                    input_ids=prefix_ids[:, start:end],
                    # cumulative mask: covers every position processed so far
                    attention_mask=attention_mask[:, :end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                past_key_values = out.past_key_values
                del out
                start = end
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Stage 2: last token only; query length == 1 -> attn is [B, H, 1, seq_len] ----
    out = model(
        input_ids=last_id,
        attention_mask=attention_mask,  # full length = cached prefix + last token
        past_key_values=past_key_values,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    if out.attentions is None:
        raise RuntimeError(
            "attentions is None. Load the model with attn_implementation='eager'."
        )

    # Move each layer to CPU immediately so all layers don't pile up on the GPU.
    # layer_attn: [batch, num_heads, 1, seq_len] -> [num_heads, seq_len]
    last_token_attn = torch.stack(
        [
            layer_attn[0, :, 0, :].detach().to("cpu", dtype=torch.float32)
            for layer_attn in out.attentions
        ]
    )  # [num_layers, num_heads, seq_len]
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    assert last_token_attn.ndim == 3, f"expected 3D, got {tuple(last_token_attn.shape)}"
    assert last_token_attn.shape[-1] == seq_len, (
        f"last dim {last_token_attn.shape[-1]} != seq_len {seq_len}"
    )
    print("last_token_attn shape:", tuple(last_token_attn.shape))
    return last_token_attn
