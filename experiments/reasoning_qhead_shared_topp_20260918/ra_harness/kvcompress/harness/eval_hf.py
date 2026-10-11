# Modifications Copyright (c) 2026, Salesforce, Inc.  SPDX-License-Identifier: Apache-2.0
# (the original portions remain under their own license -- see the provenance note below and THIRD_PARTY_NOTICES.md)
# Adapted from VaSE (https://github.com/terarachang/VaSE, MIT License, Copyright (c) 2026 Ting-Yun Chang):
# eval/reasoning_tasks/eval_hf.py at commit bfa2692. Random Attention changes: engine import path
# (kvcompress.engine), prompt/data paths resolved relative to this file / $RA_DATA_DIR. Everything
# else (prompt construction, sampling, batch_exist resume) is upstream's, so results are comparable.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root -> `kvcompress` importable
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # harness dir -> Utils/, my_utils, generation_utils
import json
import hashlib
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import re
import importlib.util
import os
import argparse
import random
import time
from datetime import datetime
from tqdm import tqdm
from Utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from Utils.parser import *
from Utils.data_loader import load_data
from Utils.math_normalization import *
from Utils.grader import *
import pickle
from math import comb
from my_utils import *
from generation_utils import batch_exist_generate
from typing import Optional, Tuple


def _stable_hash(*vals):
    x = 0x9E3779B97F4A7C15
    M = 0xFFFFFFFFFFFFFFFF
    for v in vals:
        x = (x ^ (int(v) + 0x9E3779B97F4A7C15 + (x << 6) + (x >> 2))) & M
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & M
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & M
    x = x ^ (x >> 31)
    return x & 0x7FFFFFFFFFFFFFFF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument("--data_dir", default=os.environ.get("RA_DATA_DIR", "./data"), type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--warmup_new_tokens", default=0, type=int, help="untimed warmup on first input; sample seeds reset before scored generation")
    parser.add_argument("--max_tokens", default=32768, type=int)
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--sparsity_method", default='dense', type=str)
    parser.add_argument("--token_budget", default=2048, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--attention_implementation", default="eager", type=str)
    parser.add_argument("--use_batch_exist", action="store_true")
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose information or not")
    parser.add_argument("--ex_start_i", default=0, type=int, help="Start example index for data parallelism")
    parser.add_argument("--ex_end_i", default=None, type=int, help="End example index for data parallelism")
    parser.add_argument("--seed", default=None, type=int, help="generation seed")
    parser.add_argument("--num_shards", default=1, type=int, help="dataset shard count (round-robin)")
    parser.add_argument("--shard_id", default=0, type=int, help="this worker shard id (0..num_shards-1)")
    args, _ = parser.parse_known_args()
    parser = expand_parser_for_methods(parser, args.sparsity_method)
    args = parser.parse_args()
    
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    return tokenized

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", prompt_type, f"prompt.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
        
    print(args)
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    device = f"cuda:{args.rank}"

    examples = load_data(args.data_name, args.split, args.data_dir)
    print(f"{args.data_name}, # Examples: {len(examples)}")
    print('-'*100)

    examples = examples[args.ex_start_i:args.ex_end_i]
    if args.num_shards > 1:
        examples = [e for e in examples if int(e["idx"]) % args.num_shards == args.shard_id]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt_batch = []
    prompt_idxs = []
    for example in tqdm(examples, total=len(examples)):
        # parse question and answer
        question = parse_question(example, args.data_name)
        _, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            if args.data_name in ["aime24", "aime25", "aime26", "hmmt", "math", "olympiadbench"]:
                messages = [
                    {
                        "role": "user",
                        "content": cur_prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                    }
                ]
            else:
                # for gpqa, livecodebench
                messages = [
                    {"role": "user", "content": cur_prompt}
                ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)
        prompt_idxs.append(int(example["idx"]))

    
    is_data_parallel = args.ex_end_i is not None
    completion_filename = f"completions_shard{args.ex_start_i}.jsonl" if is_data_parallel else "completions.jsonl"
    other_info_filename = f"other_info_shard{args.ex_start_i}.json" if is_data_parallel else "other_info.json"

    start_i, generate_lens, batch_times = get_ckpt_start_i(
        args.output_dir, args.run_id, args.batch_size,
        completion_filename, other_info_filename,
    )
    if start_i >= len(examples):
        print(f"Already completed. Exit.")
        exit()

    forward_kwargs = {}
    if args.sparsity_method == "eviction":
        forward_kwargs = init_evict_configs(args)
        from kvcompress.engine.modify_forward import wrap_evict_attn_forward
        wrap_evict_attn_forward(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                        torch_dtype=torch.bfloat16,
                                        device_map=device,
                                        attn_implementation=args.attention_implementation,
                                        )
    
    model.eval()
    eos_id_from_config = getattr(model.generation_config, "eos_token_id", None)
    eos_token_id = eos_id_from_config[0] if isinstance(eos_id_from_config, list) else eos_id_from_config
    batch_size = args.batch_size

    if args.warmup_new_tokens:
        if args.warmup_new_tokens < 2 or not args.use_batch_exist:
            raise ValueError('warmup requires at least 2 tokens and use_batch_exist')
        if os.environ.get('QHEAD_LOG'):
            raise ValueError('disable per-step QHEAD_LOG for untimed warmup')
        warm = prompt_batch[start_i].to(device)
        if args.sparsity_method == 'eviction':
            forward_kwargs['cache_config']['sample_id'] = prompt_idxs[start_i]
        batch_exist_generate(model, input_ids=warm['input_ids'].expand(batch_size,-1),
                             attention_mask=warm['attention_mask'].expand(batch_size,-1),
                             max_length=warm['input_ids'].shape[1]+args.warmup_new_tokens,
                             do_sample=True, **forward_kwargs)
        torch.cuda.synchronize()
        print('Untimed warmup completed; resetting sample RNG for actual generation.', flush=True)
    grid_reference = None
    if os.environ.get('RA_GRID_REFERENCE'):
        with open(os.environ['RA_GRID_REFERENCE']) as stream:
            grid_reference = {r['idx']: r for r in map(json.loads, stream) if r}
    for i in range(start_i, len(prompt_batch)):
        inputs = prompt_batch[i].to(device)
        batch_input_ids = inputs['input_ids'].expand(batch_size, -1)
        attention_mask = inputs['attention_mask'].expand(batch_size, -1)
        sample_idx = prompt_idxs[i]
        if grid_reference is not None:
            reference = grid_reference[sample_idx]
            actual_sha = hashlib.sha256(batch_input_ids[0].detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()
            if batch_input_ids.shape[1] != reference['input_tokens'] or actual_sha != reference['input_ids_sha256']:
                raise RuntimeError(f'input alignment failed: {sample_idx}')
        sample_seed = _stable_hash(args.seed if args.seed is not None else 0, sample_idx)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        if args.sparsity_method == "eviction":
            forward_kwargs['cache_config']['sample_id'] = sample_idx
        if i == start_i: visualize_token(tokenizer, batch_input_ids[0])

        print("start batch: ", i, flush=True)
        generation_metrics = {}
        if args.sparsity_method == 'eviction':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        memory_before_generation = torch.cuda.memory_allocated()
        begin = time.time()
        if args.use_batch_exist:
            outputs = batch_exist_generate(
                model,
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_length = args.max_tokens,
                do_sample=True,
                verbose=args.verbose,
                metrics=generation_metrics,
                **forward_kwargs,
            )
        else:
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                max_length = args.max_tokens,
                do_sample=True,
                num_return_sequences=1,
                **forward_kwargs,
            )

        if args.sparsity_method == 'eviction':
            torch.cuda.synchronize()
        end = time.time()
        batch_time = end - begin
        batch_times.append(batch_time)
        if generation_metrics:
            generation_metrics.update(
                idx=sample_idx, input_tokens=batch_input_ids.shape[1],
                input_ids_sha256=hashlib.sha256(batch_input_ids[0].detach().cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                batch_wall_seconds=batch_time, peak_memory_allocated=torch.cuda.max_memory_allocated(),
                memory_allocated_before_generation=memory_before_generation,
                memory_allocated_after_generation=torch.cuda.memory_allocated(),
                selection_p=args.top_p, query_window=args.query_window, token_budget=args.token_budget, eviction_interval=args.eviction_interval)
            if args.eviction_mode in ('qhead_shared_mask_random','qhead_shared_mask_random_recent_gpu','qhead_shared_mask_random_precomputed_recent_gpu','qhead_shared_mask_random_fixed_recent_gpu'):
                generation_metrics.update(selection_p=None, query_window=None,
                    selection_strategy='uniform_without_replacement', random_keep_ratio=args.random_keep_ratio,
                    mask_seed=args.mask_seed, representative_policy='sample_step_shared_across_layers',
                    representative_seed=args.rep_seed, representative_role='nominal_only_no_query_scoring')
            if args.eviction_mode in ('qhead_shared_mask_random_recent_gpu','qhead_shared_mask_random_precomputed_recent_gpu'):
                generation_metrics.update(selection_strategy='budgeted_recent_plus_gpu_uniform_topk',
                    recent_window=args.recent_window, recent_inside_budget=True, gather_backend='triton_fused_reusable')
            if args.eviction_mode == 'qhead_shared_mask_random_precomputed_recent_gpu':
                generation_metrics.update(mask_implementation='all_layer_batched_exact_legacy_rng',
                    mask_precompute_window=args.mask_precompute_window)
            if args.eviction_mode == 'qhead_shared_mask_bernoulli_recent_gpu':
                generation_metrics.update(selection_p=None, query_window=None,
                    selection_strategy='budgeted_recent_plus_independent_bernoulli',
                    random_keep_ratio=args.random_keep_ratio, mask_seed=args.mask_seed,
                    representative_policy=None, representative_seed=None, representative_role='none',
                    recent_window=args.recent_window, recent_inside_budget=True, prompt_outside_budget=True,
                    budget_semantics='expected_count_not_fixed', gather_backend='triton_fused_reusable_ragged',
                    mask_implementation='all_layer_window_bernoulli_no_topk',
                    mask_precompute_window=args.mask_precompute_window)
            if args.eviction_mode == 'qhead_shared_mask_random_fixed_recent_gpu':
                generation_metrics.update(selection_strategy='fixed_reasoning_budget_recent_plus_gpu_uniform_topk',
                    reasoning_budget=args.reasoning_budget, prompt_outside_budget=True, random_keep_ratio=None,
                    recent_window=args.recent_window, recent_inside_budget=True, gather_backend='triton_fused_reusable')
            if args.eviction_mode in ('qhead_shared_random_compact_recent_gpu','qhead_shared_random_compact_precomputed_gpu'):
                generation_metrics.update(selection_p=None, query_window=None,
                    selection_strategy='persistent_shared_uniform_topk_recent',
                    reasoning_budget=args.reasoning_budget, prompt_outside_budget=True,
                    random_keep_ratio=None, recent_window=args.recent_window, recent_inside_budget=True,
                    mask_seed=args.mask_seed, representative_seed=args.rep_seed,
                    compaction_interval=args.eviction_interval, compaction_phase='after_attention',
                    cache_semantics='persistent_eviction', random_scope='shared_all_heads_per_layer',
                    gather_backend='triton_fused_bounded_scratch')
            if args.eviction_mode == 'qhead_shared_random_compact_precomputed_gpu':
                generation_metrics.update(mask_implementation='all_events_precomputed_exact_legacy_rng')
            if args.eviction_mode in ('qhead_shared_representative_compact_recent_gpu','qhead_shared_representative_compact_optimized_gpu','qhead_shared_representative_window_compact_optimized_gpu'):
                generation_metrics.update(selection_p=None, query_window=args.query_window,
                    selection_strategy='persistent_shared_representative_topk_recent',
                    implementation=args.eviction_mode,
                    reasoning_budget=args.reasoning_budget, prompt_outside_budget=True,
                    random_keep_ratio=None, recent_window=args.recent_window, recent_inside_budget=True,
                    mask_seed=args.mask_seed, representative_seed=args.rep_seed,
                    representative_policy='private_sample_layer_stream_draw_at_compaction',
                    representative_role=('current_query_probability_topk' if args.query_window==1 else 'recent_queries_probability_mean_topk'),
                    selection_ties='earlier_original_position',
                    compaction_interval=args.eviction_interval, compaction_phase='after_attention',
                    cache_semantics='persistent_eviction', shared_scope='all_heads_per_layer',
                    gather_backend='triton_fused_bounded_scratch')
            if args.eviction_mode.startswith('grid_'):
                from kvcompress.engine.selector_grid_cache import decode_mode
                storage,scorer,selector,protection=decode_mode(args.eviction_mode)
                generation_metrics.update(selection_strategy='controlled_shared_'+scorer+'_'+selector,
                    selection_p=args.top_p if selector=='topp' else None,
                    query_window=args.query_window if scorer!='random' else None,
                    mask_seed=args.mask_seed, representative_seed=args.rep_seed,
                    shared_scope='all_kv_heads_share_positions_not_values',
                    protection=protection, compaction_phase='after_attention' if storage=='compact' else None,
                    compaction_interval=args.eviction_interval if storage=='compact' else None,
                    historical_query_causal_mask=False, actual_attention_causal_mask=True,
                    topp_semantics='conditional_unprotected_probability_mass_then_union_protection',
                    budget_semantics='total_K_including_protection' if selector=='topk' else 'variable_uncapped')
            if args.eviction_mode == 'official_random_pp':
                generation_metrics.update(selection_p=None, query_window=None, selection_strategy='official_random_pp',
                    residual_length=args.residual_length, mask_seed=args.mask_seed,
                    cache_semantics='persistent_eviction', random_scope='independent_per_kv_head',
                    upstream_commit='64db9688a12f0926db45fec039370f6ceb1ab4fe',
                    rng_adapter='private_per_sample_layer_stream_with_upstream_sampler_unchanged')
            if args.eviction_mode == 'qhead_shared_mask_w1_recent_gpu':
                generation_metrics.update(selection_strategy='w1_conditional_topp_union_recent',
                    recent_window=args.recent_window, recent_inside_budget=False, random_keep_ratio=None,
                    representative_policy='per_layer_per_step_private_rng', representative_seed=args.rep_seed,
                    gather_backend='triton_fused_reusable_variable_lengths')
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'cache_metrics.jsonl'), 'a') as stream:
                stream.write(json.dumps(generation_metrics) + '\n')
        print("get output in batch: ", i, "time (min):", round(batch_time / 60), "output:", outputs.shape, outputs.device, flush=True)
        

        for j in range(len(outputs)):
            output_seq = outputs[j]
            output_tokens = (output_seq != eos_token_id).sum().item()
            prompt_tokens = (batch_input_ids[j] != eos_token_id).sum().item()
            generate_lens[j].append(output_tokens - prompt_tokens)

        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("finish batch: ", i, flush=True)

        # Write after each batch
        for j in range(batch_size):
            output_runnum_subdir = os.path.join(args.output_dir, f"run_{args.run_id+j}")
            os.makedirs(output_runnum_subdir, exist_ok=True)
            completion_filepath = os.path.join(output_runnum_subdir, completion_filename)
            other_info_filepath = os.path.join(output_runnum_subdir, other_info_filename)

            other_info = {
                "generate_lens": generate_lens[j],
                "batch_times": batch_times,
            }
            with open(other_info_filepath, 'w') as f:
                json.dump(other_info, f)
            with open(completion_filepath, 'a') as f:
                f.write(json.dumps({"idx": sample_idx, "completion": completions[j]}) + '\n')


    print("llm generate done")
    print(f"Successfully saved run{args.run_id}!")

    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    if args.seed is not None:
        import torch as _t
        _t.manual_seed(args.seed)
        _t.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    infer(args)
