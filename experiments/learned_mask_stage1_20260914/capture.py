"""Capture group-independent teachers without modifying dense attention outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import export_record


def tile_starts(length, count=8):
    candidates = list(range(10240, length, 128))
    if len(candidates) < count:
        raise ValueError("Not enough long-history tiles")
    indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[i] for i in indices]


def tokenize(tokenizer, prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    maximum = 131072 - 128
    if len(ids) > maximum:
        left = maximum // 2
        ids = ids[:left] + ids[-(maximum-left):]
    return torch.tensor([ids], dtype=torch.long)


def verify_restoration(root):
    data = root / "experiments/data/InfiniteBench_full_20260914"
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    original = read(data / "kv_retrieval.jsonl")
    historical = root / "experiments/data/kv_randomscore_20260910"
    formal = read(historical / "filtered_data/kv_retrieval.jsonl")
    raw = read(data / "restored_calibration/raw_kv_first3.jsonl")
    calibration = read(data / "restored_calibration/calibration.jsonl")
    if len(original) != 500 or original[3:] != formal or raw != original[:3] or len(calibration) != 3:
        raise ValueError("Raw calibration restoration does not match original source")
    template = "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}"
    for index, (source, row) in enumerate(zip(raw, calibration)):
        expected = template.format(**source)
        if row["_id"] != f"kv_retrieval-{index}" or row["prompt"] != expected:
            raise ValueError("Restored prompt/identity differs from historical calibration builder")
        if row["prompt_sha256"] != hashlib.sha256(expected.encode()).hexdigest():
            raise ValueError("Restored prompt checksum mismatch")
    if {r["context"] for r in raw} & {r["context"] for r in formal}:
        raise ValueError("Calibration context overlaps formal inputs")
    return calibration, historical


def collect(root, model_path, output, debug=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    calibration, historical = verify_restoration(root)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    reference = historical / "references/original_autoblock/online_metrics.jsonl"
    reserved = {json.loads(l)["input_ids_sha256"] for l in reference.read_text().splitlines() if l.strip()}
    groups = json.loads((historical / "calibration/shareprefill_ae_k3_head_groups.json").read_text())["layers"]
    encoded = []
    for row in calibration:
        ids = tokenize(tokenizer, row["prompt"])
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if digest in reserved:
            raise ValueError("Calibration input unexpectedly matches formal reference")
        encoded.append((row, ids, digest))
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    model.requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.hidden_size) != (32, 32, 8, 4096):
        raise ValueError("Expected Llama-3.1-8B GQA configuration")
    original_rope = modeling_llama.apply_rotary_pos_emb
    context, entries, identities = {}, [], []
    handles = []
    def before(layer):
        def hook(module, args):
            context["layer"] = layer
        return hook
    def rope(*args, **kwargs):
        q, k = original_rope(*args, **kwargs)
        layer = context["layer"]
        if layer in context["seen"]:
            raise RuntimeError("Expected one dense prefill call per layer")
        context["seen"].add(layer)
        if not debug or layer == 0:
            if q.shape != (1, 32, context["n"], 128) or k.shape != (1, 8, context["n"], 128):
                raise ValueError("Unexpected post-RoPE layout")
            for start in context["starts"]:
                path = output / f'{context["family"]}_l{layer:02d}_q{start}.pt'
                entries.append(export_record(path, q_tile=q[0, :, start:start+128].transpose(0, 1),
                    k=k[0].transpose(0, 1), groups=groups, layer=layer, start=start,
                    input_tokens=context["n"], input_ids_sha256=context["identity"],
                    family_id=context["family"]))
            print("CAPTURE", context["family"], layer, flush=True)
        return q, k
    try:
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(before(i)))
        modeling_llama.apply_rotary_pos_emb = rope
        for row, ids, digest in encoded[:1] if debug else encoded:
            n = ids.shape[1]
            starts = tile_starts(n, 1 if debug else 8)
            context.update(family=row["_id"], identity=digest, n=n, starts=starts, seen=set())
            begin = time.monotonic()
            with torch.inference_mode():
                # Backbone only: avoid allocating unused [N,vocab] LM logits.
                model.model(input_ids=ids.cuda(), attention_mask=torch.ones_like(ids).cuda(),
                            use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            if context["seen"] != set(range(32)):
                raise RuntimeError("Post-RoPE hook did not cover all layers")
            identities.append(dict(family_id=row["_id"], input_tokens=n, input_ids_sha256=digest,
                                   tile_starts=starts, capture_seconds=time.monotonic()-begin))
    finally:
        modeling_llama.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()
    (output / "records.json").write_text(json.dumps(entries, indent=2))
    proof = dict(debug=debug, teacher_state="dense_post_rope", identities=identities,
                 records=len(entries), peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                 calibration_prompt_hash_verified=True, formal_input_overlap=0,
                 historical_calibration_token_hash_available=False,
                 tokenization="user chat template; no extra BOS; middle truncation to 130944",
                 historical_groups_used_only_to_serialize_unused_shared_p=True)
    (output / "capture_verified.json").write_text(json.dumps(proof, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps(collect(args.root, args.model, args.output, args.debug), indent=2))


if __name__ == "__main__":
    main()
