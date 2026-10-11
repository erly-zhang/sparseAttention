# Copied from VaSE (https://github.com/terarachang/VaSE, MIT License, Copyright (c) 2026 Ting-Yun Chang):
# eval/reasoning_tasks/my_utils.py at commit bfa2692, unmodified.
import os
import json
from datasets import load_dataset


def get_evict_args(parser):
    #parser.add_argument("--token_budget", default=4096, type=int)
    parser.add_argument("--residual_length", default=64, type=int, help="buffer cache size for recent tokens")
    parser.add_argument("--rkv_lambda", type=float, help="mix_lambda used in the RKV method")
    parser.add_argument("--mode", "--eviction_mode", dest="eviction_mode", type=str, default="None",)
    parser.add_argument("--smooth", action="store_true", help="smooth the attn scores in evict_attn methods")
    parser.add_argument("--n_large", type=int, default=-1, help="number of tokens to keep from the high-importance end; -1 means not used")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for softmax in eviction scoring")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p threshold for qhead_shared_topp")
    parser.add_argument("--eviction_interval", type=int, default=1, help="decode steps between shared Top-p selections when token_budget=0 (uncapped)")
    parser.add_argument("--query_window", type=int, default=64, help="number of recent real queries averaged for full-history shared Top-p masks")
    parser.add_argument("--rep_seed", type=int, default=0, help="seed for representative Q-head RNG")
    parser.add_argument("--random_keep_ratio", type=float, default=None, help="fraction of non-prompt tokens retained by random shared masks")
    parser.add_argument("--reasoning_budget", type=int, default=None, help="non-prompt attention budget for fixed random masks; prompt is additional")
    parser.add_argument("--recent_window", type=int, default=64, help="recent non-prompt positions protected; inside random budget or union with W1 Top-p")
    parser.add_argument("--mask_precompute_window", type=int, default=16, help="future decode steps whose exact random masks are batched across layers")
    parser.add_argument("--mask_seed", type=int, default=42, help="private random-mask seed; does not change sampling RNG")
    return parser

def get_quant_args(parser):
    parser.add_argument("--kbits", default=4, type=int)
    parser.add_argument("--vbits", default=8, type=int)
    parser.add_argument("--axis_key", type=int, default=-1, choices=[0, -1])
    parser.add_argument("--axis_value", type=int, default=0, choices=[0, -1])
    parser.add_argument("--residual_length", default=64, type=int, help="16-bit cache size")
    parser.add_argument("--q_group_size", default=64, type=int, help="group quantization size")
    return parser

def expand_parser_for_methods(parser, method):
    if 'quant' in method.lower():
        parser = get_quant_args(parser)
    elif 'evict' in method.lower():
        parser = get_evict_args(parser)
    return parser


def init_quant_configs(args):
    q_config = {
        "backend": "fake",
        "q_group_size": args.q_group_size,
        "residual_length": args.residual_length,
        "kbits": args.kbits,
        "vbits": args.vbits,
        "axis_key": args.axis_key,
        "axis_value": args.axis_value,
        "verbose": args.verbose,
        }
    print('-'*100)
    print(q_config)
    print('-'*100)
    q_kwargs = {"cache_implementation": "quantized", "cache_config": q_config}
    return q_kwargs


def init_evict_configs(args):
    if args.query_window < 1:
        raise ValueError("query_window must be positive")
    if args.eviction_mode.startswith('grid_'):
        if args.residual_length != 0 or args.token_budget < 0:
            raise ValueError('grid budget/residual mismatch')
    elif args.eviction_mode in ('qhead_shared_topp', 'qhead_shared_mask_topp', 'qhead_shared_mask_w1_recent_gpu', 'qhead_shared_mask_random', 'qhead_shared_mask_random_recent_gpu', 'qhead_shared_mask_random_precomputed_recent_gpu', 'qhead_shared_mask_bernoulli_recent_gpu', 'qhead_shared_mask_random_fixed_recent_gpu'):
        if args.token_budget != 0 or args.residual_length != 0 or args.eviction_interval != 1 or not 0 < args.top_p <= 1:
            raise ValueError('full-history shared masks require token_budget=0, residual=0, interval=1, 0<p<=1')
    elif args.eviction_mode in ('qhead_shared_random_compact_recent_gpu','qhead_shared_random_compact_precomputed_gpu','qhead_shared_representative_compact_recent_gpu','qhead_shared_representative_compact_optimized_gpu','qhead_shared_representative_window_compact_optimized_gpu'):
        if args.token_budget != 0 or args.residual_length != 0 or args.eviction_interval < 1:
            raise ValueError('compact random requires legacy budget/residual=0 and interval>=1')
    elif args.eviction_mode == 'qhead_shared_topp_eviction':
        if args.token_budget < 0 or args.residual_length != 0 or not 0 < args.top_p <= 1 or args.eviction_interval < 1:
            raise ValueError('shared Top-p requires budget>=0 (0=uncapped), residual=0, 0<p<=1, interval>=1')
    else:
        assert args.token_budget > args.residual_length
        assert args.residual_length == 0 or args.token_budget % args.residual_length == 0
    config = {
        "token_budget": args.token_budget,
        "residual_length": args.residual_length,
        "rkv_lambda": args.rkv_lambda,
        "eviction_mode": args.eviction_mode,
        "smooth": args.smooth,
        "n_large": args.n_large,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "query_window": args.query_window,
        "eviction_interval": args.eviction_interval,
        "rep_seed": args.rep_seed,
        "verbose": args.verbose,
        }
    if args.eviction_mode.startswith('grid_'):
        config.update(mask_seed=args.mask_seed)
    if args.eviction_mode == 'official_random_pp':
        allowed = (1024, 2048) if getattr(args, 'data_name', None) == 'hmmt' else (1024,)
        if args.token_budget not in allowed or args.residual_length != 64:
            raise ValueError('official random_pp requires residual64 and an approved task-specific total budget')
        config.update(mask_seed=args.mask_seed)
    if args.eviction_mode == 'qhead_shared_mask_w1_recent_gpu':
        if args.query_window != 1 or args.random_keep_ratio is not None:
            raise ValueError('W1 recent requires query_window=1 and no fixed random ratio')
        config.update(recent_window=args.recent_window)
    if args.eviction_mode in ('qhead_shared_mask_random','qhead_shared_mask_random_recent_gpu','qhead_shared_mask_random_precomputed_recent_gpu', 'qhead_shared_mask_bernoulli_recent_gpu'):
        if args.random_keep_ratio is None or not 0 < args.random_keep_ratio <= 1:
            raise ValueError('random shared masks require 0 < random_keep_ratio <= 1')
        config.pop("query_window")  # unused: random masks never store/query past Q vectors
        config.pop("top_p")
        config.update(random_keep_ratio=args.random_keep_ratio, mask_seed=args.mask_seed)
        if args.eviction_mode in ("qhead_shared_mask_random_recent_gpu", "qhead_shared_mask_random_precomputed_recent_gpu", "qhead_shared_mask_bernoulli_recent_gpu"):
            config.update(recent_window=args.recent_window)
    if args.eviction_mode in ('qhead_shared_mask_random_precomputed_recent_gpu', 'qhead_shared_mask_bernoulli_recent_gpu','qhead_shared_random_compact_precomputed_gpu'):
        config.update(mask_precompute_window=args.mask_precompute_window)
    if args.eviction_mode in ('qhead_shared_mask_random_fixed_recent_gpu','qhead_shared_random_compact_recent_gpu','qhead_shared_random_compact_precomputed_gpu'):
        if args.reasoning_budget is None or args.reasoning_budget < 1 or args.random_keep_ratio is not None:
            raise ValueError('fixed random masks require positive reasoning_budget and no random_keep_ratio')
        if not 0 <= args.recent_window <= args.reasoning_budget:
            raise ValueError('recent_window must fit inside reasoning_budget')
        config.pop('query_window')
        config.pop('top_p')
        config.update(reasoning_budget=args.reasoning_budget, recent_window=args.recent_window, mask_seed=args.mask_seed)
    if args.eviction_mode in ('qhead_shared_representative_compact_recent_gpu','qhead_shared_representative_compact_optimized_gpu','qhead_shared_representative_window_compact_optimized_gpu'):
        window_mode=args.eviction_mode=='qhead_shared_representative_window_compact_optimized_gpu'
        valid_window=(1 <= args.query_window <= args.eviction_interval) if window_mode else args.query_window==1
        if not valid_window or args.reasoning_budget is None or args.reasoning_budget < 1 or args.random_keep_ratio is not None:
            raise ValueError('representative compact requires a valid window, positive reasoning budget, no random ratio')
        if not 0 <= args.recent_window <= args.reasoning_budget:
            raise ValueError('recent_window must fit reasoning budget')
        config.pop('top_p')
        config.update(reasoning_budget=args.reasoning_budget,recent_window=args.recent_window,mask_seed=args.mask_seed)
    print('-'*100)
    print(config)
    print('-'*100)
    kwargs = {"cache_implementation": "evict", "cache_config": config}
    return kwargs


def load_gsm8k(model_name, tokenizer, n_samples, split, n_return_sequences=1):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    extra_info = " Put your final answer within \\boxed{}."

    batch_inputs, all_answers = [], []
    for i in range(n_samples):
        question = ds['question'][i].strip()
        ans = ds['answer'][i].rsplit('#### ', 1)[-1]

        messages = [
            {
                "role": "user",
                "content": question + extra_info
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        inputs['input_ids'] = inputs['input_ids'].expand(n_return_sequences, -1)
        inputs['attention_mask'] = inputs['attention_mask'].expand(n_return_sequences, -1)
        batch_inputs.append(inputs)
        all_answers.append(ans)

    return batch_inputs, all_answers


def load_raw_gsm8k(tokenizer, n_samples, split, model_name): # for vllm
    ds = load_dataset("openai/gsm8k", "main", split=split)
    extra_info = " Put your final answer within \\boxed{}."

    all_inputs, all_answers = [], []
    for i in range(n_samples):
        question = ds['question'][i].strip()
        ans = ds['answer'][i].rsplit('#### ', 1)[-1]

        messages = [
            {"role": "user", "content": question + extra_info}
        ]
        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_inputs.append(templated_text)
        all_answers.append(ans)

    return all_inputs, all_answers



def get_ckpt_start_i(output_dir, run_id, batch_size,
                     completion_filename="completions.jsonl",
                     other_info_filename="other_info.json"):
    """Return (start_i, generate_lens, batch_times) for resuming a run."""
    completion_filepath = os.path.join(output_dir, f"run_{run_id}", completion_filename)
    lines = []
    if os.path.exists(completion_filepath):
        print(f"Loading checkpoint from {completion_filepath}")
        with open(completion_filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                lines.append(item["completion"])
    start_i = len(lines)
    print(f"Resuming from ex-{start_i}...")

    generate_lens = [[] for _ in range(batch_size)]
    batch_times = []
    if start_i > 0:
        for j in range(batch_size):
            info_path = os.path.join(output_dir, f"run_{run_id + j}", other_info_filename)
            with open(info_path) as f:
                info = json.load(f)
            generate_lens[j] = list(info['generate_lens'])
            if j == 0:
                batch_times = list(info['batch_times'])

    return start_i, generate_lens, batch_times


def visualize_token(tokenizer, tokens: list[int]):
    from rich.console import Console
    from rich.text import Text

    COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
    i = 0 
    console = Console()
    rich_text = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]                                                                                                   
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)
    console.print(rich_text)
