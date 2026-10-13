"""Bounded validation only. Random tiny Llama by default; no task-accuracy claim.
Optional local checkpoint + audited input manifest; never downloads weights.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation.logits_process import TopPLogitsWarper
from routing import RoutingController


def stable_hash(*vals):
    x, mask = 0x9E3779B97F4A7C15, 0xFFFFFFFFFFFFFFFF
    for value in vals:
        x = (x ^ (int(value) + 0x9E3779B97F4A7C15 + (x << 6) + (x >> 2))) & mask
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & mask
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & mask
    return (x ^ (x >> 31)) & 0x7FFFFFFFFFFFFFFF


def require_allocation():
    job = os.environ.get('SLURM_JOB_ID')
    if not job or not torch.cuda.is_available():
        raise RuntimeError('existing GPU allocation required; will not request one')
    status = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    fields = dict(x.split('=', 1) for x in status.split() if '=' in x)
    hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).split()
    if fields['JobState'] != 'RUNNING' or socket.gethostname().split('.')[0] not in hosts:
        raise RuntimeError('not inside the running allocation')
    return status.strip()


def timed_forward(model, ctl, phase, ids, mask, pos, cache):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), ctl.phase(phase, mask.bool() if phase == 'prefill' else None,
                                   pos if phase == 'prefill' else None):
        out = model(ids, attention_mask=mask, position_ids=pos, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def generate(model, mode, ids, max_new, eos, temperature, top_p, sample_seed, sample=True):
    """Batch two rollout semantics, sample-local RNG, dynamic removal as old runner.
    Returns actual stop events and token lengths; last sampled token is not cached.
    """
    ctl = RoutingController(model, mode)
    mask = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    cache = DynamicCache(config=model.config)
    histories = [[] for _ in ids]
    reasons = ['max_new_tokens_smoke'] * len(ids)
    mapping = list(range(len(ids)))
    eos = eos if isinstance(eos, list) else [eos]
    warp = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)
    torch.manual_seed(sample_seed); torch.cuda.manual_seed_all(sample_seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    decode_seconds = 0.0
    decode_calls = 0
    decode_eligible_pairs = 0
    decode_matmul_pairs = 0
    try:
        out, prefill_seconds = timed_forward(model, ctl, 'prefill', ids, mask, pos, cache)
        prefill_counts = list(ctl.counts)
        prompt_cache_shapes = [list(layer.keys.shape) for layer in cache.layers]
        first_logits = out.logits[:, -1].detach().cpu()
        for step in range(max_new):
            logits = out.logits[:, -1].clone().float()
            if sample:
                token = torch.multinomial(warp(ids, logits / temperature).softmax(-1), 1)
            else:
                token = logits.argmax(-1, keepdim=True)
            keep = []
            for local, original in enumerate(mapping):
                value = int(token[local, 0])
                histories[original].append(value)
                if value in eos:
                    reasons[original] = 'eos'
                else:
                    keep.append(local)
            if not keep or step == max_new - 1:
                break
            active = torch.tensor(keep, dtype=torch.long, device=ids.device)
            if len(keep) != len(mapping):
                cache.batch_select_indices(active)
            mapping = [mapping[i] for i in keep]
            token = token[active]
            mask = torch.cat((mask[active], torch.ones(len(keep), 1, dtype=mask.dtype, device=mask.device)), -1)
            pos = torch.full_like(token, ids.shape[1] + step)
            out, elapsed = timed_forward(model, ctl, 'decode', token, mask, pos, cache)
            decode_seconds += elapsed
            decode_calls += 1
            pairs = len(keep) * model.config.num_attention_heads * mask.shape[1] * model.config.num_hidden_layers
            decode_eligible_pairs += pairs
            decode_matmul_pairs += pairs
        torch.cuda.synchronize()
        total = time.perf_counter() - start
        return dict(mode=mode, generated_token_ids=histories, stop_reasons=reasons,
                    generated_lengths_including_eos=list(map(len, histories)),
                    sample_seed=sample_seed, prefill_seconds=prefill_seconds,
                    decode_forward_seconds=decode_seconds, total_seconds=total,
                    decode_forward_calls=decode_calls, prompt_cache_shapes=prompt_cache_shapes,
                    prefill_pairs_by_layer=prefill_counts,
                    decode_eligible_pairs=decode_eligible_pairs, decode_matmul_pairs=decode_matmul_pairs,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()), first_logits
    finally:
        ctl.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', help='local real checkpoint; no network loading')
    parser.add_argument('--input-manifest', help='audited JSON records with idx/input_ids; max 2 samples')
    parser.add_argument('--max-new', type=int, default=12)
    parser.add_argument('--backend', choices=['eager','sdpa'], default='eager')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    args = parser.parse_args()
    if not 1 <= args.max_new <= 32:
        parser.error('smoke limited to 1..32 new tokens')
    allocation = require_allocation()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=False)
    dtype = getattr(torch, args.dtype)
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('native bfloat16 required for protocol parity; do not silently change dtype')
    torch.set_num_threads(2)
    tokenizer = None
    if args.model:
        if not args.input_manifest:
            parser.error('real checkpoint requires an audited input manifest')
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    dtype=dtype, attn_implementation=args.backend).to('cuda').eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        inputs = json.loads(Path(args.input_manifest).read_text())['records'][:2]
        provenance = {'kind': 'local_checkpoint', 'path': str(Path(args.model).resolve()),
                      'input_manifest_sha256': hashlib.sha256(Path(args.input_manifest).read_bytes()).hexdigest()}
        cfg = model.generation_config
        temperature, top_p, eos = cfg.temperature, cfg.top_p, cfg.eos_token_id
    else:
        os.environ["ROUTING_TEST_BACKEND"] = args.backend
        from test_routing import small_model
        model = small_model().to(device='cuda', dtype=dtype)
        inputs = [{'idx': 0, 'input_ids': [2,3,4,5,6,7,8]}, {'idx': 1, 'input_ids': [5]}]
        provenance = {'kind': 'random_tiny_llama', 'initialization_seed': 192, 'not_pretrained_Llama_8B': True}
        temperature, top_p, eos = 0.6, 0.95, 42
    provenance.update(torch=torch.__version__, transformers=transformers.__version__,
                      gpu=torch.cuda.get_device_name(), allocation=allocation,
                      config=model.config.to_dict(), dtype=args.dtype, batch_size=1, do_sample=False, backend=args.backend,
                      max_new_tokens=args.max_new, temperature=temperature, top_p=top_p,
                      eos_token_id=eos, source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob('*.py')})
    (outdir/'provenance.json').write_text(json.dumps(provenance, indent=2, default=str)+'\n')
    runtime = []
    for layer in model.model.layers:
        attn = layer.self_attn
        runtime.append({'layer':attn.layer_idx,'head_dim':attn.head_dim,'scaling':attn.scaling,
                        'q_norm':hasattr(attn,'q_norm'),'k_norm':hasattr(attn,'k_norm'),
                        'projections':{name:{'weight':list(getattr(attn,name).weight.shape),'bias':getattr(attn,name).bias is not None}
                                       for name in ('q_proj','k_proj','v_proj','o_proj')}})
    (outdir/'runtime_attention.json').write_text(json.dumps(runtime,indent=2)+'\n')
    results = []
    for inp in inputs:
        ids = torch.tensor([inp['input_ids']], device='cuda').expand(1, -1)
        seed = stable_hash(42, inp['idx'])
        dense_logits = None
        for mode in ('dense', 'q_only', 'q_kv'):
            # Warm up each exact condition and shape, then reset sample RNG and peak stats.
            generate(model, mode, ids, 2, eos, temperature, top_p, seed, sample=False)
            record, logits = generate(model, mode, ids, args.max_new, eos, temperature, top_p, seed, sample=False)
            if mode == 'dense':
                dense_logits = logits
            record.update(idx=inp['idx'], input_ids=inp['input_ids'],
                          input_sha256=hashlib.sha256(json.dumps(inp['input_ids']).encode()).hexdigest(),
                          first_logits_max_abs_vs_dense=float((logits-dense_logits).abs().max()),
                          first_argmax=logits.argmax(-1).tolist())
            if tokenizer is not None:
                record['predictions'] = tokenizer.batch_decode(record['generated_token_ids'], skip_special_tokens=True)
            results.append(record)
            print(json.dumps(record), flush=True)
    (outdir/'results.json').write_text(json.dumps(results, indent=2)+'\n')

if __name__ == '__main__':
    main()
