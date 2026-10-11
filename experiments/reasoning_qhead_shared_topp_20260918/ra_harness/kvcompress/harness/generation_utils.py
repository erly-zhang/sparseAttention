# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
# Copied from VaSE (https://github.com/terarachang/VaSE, MIT License, Copyright (c) 2026 Ting-Yun Chang):
# eval/reasoning_tasks/generation_utils.py at commit bfa2692 -- only the engine import path changed (kvcompress.engine).
from typing import Optional, List
import torch
from transformers import DynamicCache
from kvcompress.engine.cache_utils import EvictCache

from transformers.generation.logits_process import TopPLogitsWarper
import time
def batch_exist_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_length: int = 100,
    do_sample: bool = False,
    verbose: bool = False,
    metrics: Optional[dict] = None,
    **model_kwargs,
):
    """
    Modified generation loop that dynamically adjusts batch size by filtering finished sequences
    and reorganizing the KV cache.
    """
    # Initialize variables
    if model_kwargs.get('cache_implementation', None) == 'evict':
        cache_config = model_kwargs['cache_config'].copy()
        if cache_config.get('eviction_mode') in ('qhead_shared_mask_random_fixed_recent_gpu', 'qhead_shared_mask_bernoulli_recent_gpu'):
            # Storage bound only: selection still uses complete valid history.
            cache_config['max_cache_length'] = max_length
        if cache_config.get('eviction_mode', '').startswith('grid_'):
            cache_config['max_cache_length'] = max_length
        current_cache = EvictCache(config=model.config, **cache_config)
    else:
        current_cache = DynamicCache()
    if hasattr(current_cache, 'mask_plan'):
        current_cache.mask_plan.max_step = max(0, max_length-input_ids.shape[1]-1)
    mask_precompute_seconds = 0.0
    if hasattr(current_cache, 'compact_mask_plan'):
        torch.cuda.synchronize(input_ids.device)
        mask_precompute_start = time.perf_counter()
        current_cache.compact_mask_plan.precompute_all(input_ids.shape[1],
            max(0,max_length-input_ids.shape[1]-1),input_ids.shape[0],input_ids.device)
        torch.cuda.synchronize(input_ids.device)
        mask_precompute_seconds = time.perf_counter()-mask_precompute_start
    if verbose: print("KV Cache:", current_cache)

    generation_config, model_kwargs = model._prepare_generation_config(None)
    generated = input_ids

    if isinstance(generation_config.eos_token_id, list):
        eos_token_id = generation_config.eos_token_id[0]
        eos_token_ids = torch.tensor(generation_config.eos_token_id, device=input_ids.device)
    else:
        eos_token_id = generation_config.eos_token_id
    initial_batch_size = input_ids.shape[0]

    device = input_ids.device
    finished = torch.zeros(initial_batch_size, dtype=torch.bool, device=device)
    
    cur_input = generated
    cur_to_orig = torch.arange(initial_batch_size, device=device)

    if do_sample:
        top_p_warper = TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)

    measure_phases = metrics is not None and isinstance(current_cache, EvictCache) and (getattr(current_cache.layers[0], 'uses_temporary_mask', False) or getattr(current_cache.layers[0], 'records_attention_metrics', False))
    prefill_seconds = 0.0
    decode_start = None
    decode_calls = 0
    decode_query_tokens = 0
    for step in range(max_length - generated.shape[1]):
        if measure_phases and step == 0:
            torch.cuda.synchronize(device)
            prefill_start = time.perf_counter()
        if measure_phases and step == 1:
            torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
        if measure_phases and step > 0:
            decode_calls += 1
            decode_query_tokens += cur_input.shape[0]
        # Forward pass: get next token logits and updated past_key_values
        with torch.no_grad():
            outputs = model(
                cur_input, 
                attention_mask=attention_mask,
                past_key_values=current_cache, 
                use_cache=True,
                logits_to_keep=1,
        )
            
        if measure_phases and step == 0:
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - prefill_start
        logits = outputs.logits[:, -1, :].clone().float()
        logits = logits.to(input_ids.device)

        if do_sample:
            logits /= generation_config.temperature
            processed_logits = top_p_warper(cur_input, logits)
            probs = torch.softmax(processed_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

        # Update the kv cache with the new keys and values.
        current_cache = outputs.past_key_values

        new_tokens_full = torch.full((initial_batch_size, 1), eos_token_id,
                                     dtype=next_tokens.dtype, device=device)
        new_tokens_full[cur_to_orig] = next_tokens

        # Append the token to each sequence.
        generated = torch.cat([generated, new_tokens_full], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)


        # Update finished flags for the active sequences.
        if isinstance(generation_config.eos_token_id, list):
            finished[cur_to_orig] |= torch.isin(next_tokens.squeeze(1), eos_token_ids)
        else:
            finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)

        # If all sequences are finished, break.
        if finished.all():
            break

        # Determine which sequences in the current batch (cache) are still active.
        current_finished = finished[cur_to_orig]
        active_local = ~current_finished
        if active_local.sum().item() < cur_to_orig.shape[0]:
            active_indices_local = torch.nonzero(active_local, as_tuple=False).squeeze(-1)
            # Update the kv cache using indices relative to the current cache.
            print("active batch index", active_indices_local, "len:", generated.size(-1), flush=True)
            current_cache.batch_select_indices(active_indices_local)
        
            if attention_mask is not None:
                attention_mask = attention_mask[active_indices_local]

            cur_to_orig = cur_to_orig[active_indices_local]

        # Prepare the next input tokens using the updated mapping.
        cur_input = generated[cur_to_orig, -1:].clone()

    if measure_phases:
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_start if decode_start is not None else 0.0
        prefill_pairs = sum(layer.prefill_pairs for layer in current_cache.layers)
        decode_selected = sum(layer.selected_pairs for layer in current_cache.layers) - prefill_pairs
        decode_causal = sum(layer.causal_pairs for layer in current_cache.layers) - prefill_pairs
        metrics.update(prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                       decode_forward_calls=decode_calls, decode_query_tokens=decode_query_tokens,
                       decode_tokens_per_second=decode_query_tokens/decode_seconds if decode_seconds else None,
                       prefill_selected_pairs=prefill_pairs, prefill_causal_pairs=prefill_pairs,
                       decode_selected_pairs=decode_selected, decode_causal_pairs=decode_causal,
                       decode_pair_sparsity=1-decode_selected/decode_causal if decode_causal else 0.0,
                       timing_scope='GPU-synchronized wall: prefill forward; decode from first single-token forward through final sampling/bookkeeping; excludes loading, warmup and metric serialization')
    if metrics is not None and isinstance(current_cache, EvictCache) and all(getattr(layer, 'uses_temporary_mask', False) or getattr(layer, 'records_attention_metrics', False) or layer.MODE == 'qhead_shared_topp_eviction' for layer in current_cache.layers):
        layers = current_cache.layers
        suffix = generated[:, input_ids.shape[1]:]
        eos_values = torch.as_tensor(generation_config.eos_token_id, device=device).reshape(-1)
        eos_matches = torch.isin(suffix, eos_values)
        generated_counts = torch.where(eos_matches.any(dim=1), eos_matches.int().argmax(dim=1)+1, suffix.shape[1]).tolist()
        selected = sum(layer.selected_pairs for layer in layers)
        causal = sum(layer.causal_pairs for layer in layers)
        metrics.update(generated_tokens_per_rollout=generated_counts, query_window=layers[0].query_window, cache_semantics='full_history_temporary_mask' if getattr(layers[0], 'uses_temporary_mask', False) else 'persistent_eviction',
                       max_selected_tokens=max(getattr(layer, 'max_selected_tokens', 0) for layer in layers),
                       final_cache_seqlens=[layer.cache_seqlens.tolist() for layer in layers],
                       final_cumulative_length=[layer.cumulative_length for layer in layers],
                       eos_reached=finished.tolist(), selected_pairs=selected, causal_pairs=causal,
                       global_pair_sparsity=1-selected/causal if causal else 0.0,
                       max_visible_tokens=max(layer.max_visible_tokens for layer in layers),
                       max_capacity=max(layer.k_cache.shape[1] for layer in layers),
                       selection_events=sum(layer.selection_events for layer in layers),
                       scope='all layers and active rollouts, prefill plus every decode attention call')
    if metrics is not None and isinstance(current_cache, EvictCache) and current_cache.layers[0].MODE.startswith('grid_'):
        layers=current_cache.layers;first=layers[0]
        metrics.update(grid_method=first.MODE, grid_storage=first.storage, grid_scorer=first.scorer,
            grid_selector=first.selector, grid_protection=first.protection,
            prompt_outside_budget=False, recent_inside_budget=True,
            attention_token_sum=sum(l.attention_token_sum for l in layers),
            attention_token_observations=sum(l.attention_token_observations for l in layers),
            selection_token_sum=sum(l.selection_token_sum for l in layers),
            selection_token_observations=sum(l.selection_token_observations for l in layers),
            per_layer_selected_pairs=[l.selected_pairs for l in layers],
            per_layer_causal_pairs=[l.causal_pairs for l in layers])
        if hasattr(first, 'implementation'):
            metrics.update(w1_implementation=first.implementation,
                           w1_rank_backend=first.rank_backend,
                           w1_score_mode=first.score_mode)
            if first.storage == 'full':
                metrics.update(full_w1_implementation=first.implementation,
                               full_w1_rank_backend=first.rank_backend,
                               full_w1_score_mode=first.score_mode)
    if metrics is not None and isinstance(current_cache, EvictCache) and all(hasattr(l, 'recent_added_pairs') for l in current_cache.layers):
        metrics.update(decode_topp_core_selected_pairs=sum(l.topp_core_selected_pairs for l in current_cache.layers),
                       decode_recent_added_pairs=sum(l.recent_added_pairs for l in current_cache.layers),
                       gather_allocations=sum(l.gather_allocations for l in current_cache.layers))
    if metrics is not None and isinstance(current_cache, EvictCache) and all(hasattr(l, 'decode_expected_selected_tokens') for l in current_cache.layers):
        metrics.update(decode_selected_tokens_across_layers=sum(l.decode_selected_tokens for l in current_cache.layers),
            decode_expected_selected_tokens_across_layers=sum(l.decode_expected_selected_tokens for l in current_cache.layers),
            decode_selection_count_variance=sum(l.decode_selection_count_variance for l in current_cache.layers))
    if metrics is not None and hasattr(current_cache, 'mask_plan'):
        plan=current_cache.mask_plan
        metrics.update(mask_precompute_window=plan.window, mask_precompute_refills=plan.refills,
            mask_prepared_steps=plan.prepared_steps, mask_plan_peak_bytes=plan.max_plan_bytes,
            mask_prepared_layer_rollout_masks=plan.prepared_layer_rollout_masks,
            mask_consumed_layer_rollout_masks=plan.consumed_layer_rollout_masks,
            mask_precompute_timing='included in synchronized decode and batch wall time')
    if metrics is not None and hasattr(current_cache, 'compact_mask_plan'):
        plan=current_cache.compact_mask_plan
        metrics.update(mask_precompute_seconds=mask_precompute_seconds,
            mask_precompute_phase='before first model forward; included in batch_wall_seconds',
            mask_precompute_chunk_events=plan.chunk_events,
            mask_prepared_events=len(plan.steps),mask_prepared_layer_rollout_masks=plan.prepared_masks,
            mask_consumed_layer_rollout_masks=plan.consumed_masks,
            mask_plan_bytes=plan.ids.numel()*plan.ids.element_size(),
            mask_index_coordinates='relative slots in chronological surviving cache')
    return generated
