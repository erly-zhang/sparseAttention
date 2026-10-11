"""Independent masked-attention checks for masks emitted by all four NNs."""
import argparse
import json
from pathlib import Path
import time

import torch

from .model import Config, Predictor, VARIANTS, exact_probability_mean, final_mask, shared_masks, teacher_loss


def check():
    from experiments.token_compacted_sparse import TokenCompactedIndex, triton_token_compacted_prefill_attention
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    groups = {"0": [{"representative": 6, "members": [0, 2, 4, 6]},
                     {"representative": 1, "members": [1, 3, 5, 7]}]}
    config = Config(budget=320, core_budget=256)
    results = []
    for n in (129, 385):
        q = torch.randn(1, 8, n, 128, dtype=torch.bfloat16, device=device).transpose(1, 2)
        k = torch.randn(1, 2, n, 128, dtype=torch.bfloat16, device=device).transpose(1, 2)
        v = torch.randn_like(k)
        causal = torch.arange(n, device=device)[:, None] >= torch.arange(n, device=device)[None]
        dense = [(q[0, :, h].float() @ k[0, :, h//4].float().T / 128**.5)
                 .masked_fill(~causal, -torch.inf).softmax(-1) for h in range(8)]
        for variant in VARIANTS:
            model = Predictor(groups, variant, config).to(device)
            masks = torch.zeros(8, (n+127)//128, n, dtype=torch.bool, device=device)
            for start in range(0, n, 128):
                end = min(n, start+128)
                for group in groups["0"]:
                    rep = group["representative"]
                    p = exact_probability_mean(q[0, start:end, rep], k[0, :, rep//4], start)
                    torch.testing.assert_close(p, dense[rep][start:end].mean(0), atol=2e-6, rtol=2e-4)
                    protect, core, full = shared_masks(p, start, end, config)
                    for h in group["members"]:
                        if h == rep:
                            prediction = full
                        else:
                            condition = full if variant == "full_mask" else core
                            score = model(q[0, start:end, h], k[0, :, h//4], p, condition, 0, h)
                            fixed = protect if variant == "full_mask" else core
                            loss = teacher_loss(score, dense[h][start:end].mean(0), fixed, end)
                            loss.backward()
                            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                                raise ValueError("Nonfinite predictor gradient")
                            model.zero_grad(set_to_none=True)
                            prediction = final_mask(score.detach(), core, full, protect, end, config, variant)
                        if int(prediction.sum()) != min(320, end) or prediction[end:].any():
                            raise ValueError("Learned mask violates final budget or tile causality")
                        if not prediction[protect].all():
                            raise ValueError("Protection missing")
                        masks[h, start//128] = prediction
            counts = masks.sum(-1, dtype=torch.int32)
            ends = counts.flatten().cumsum(0).to(torch.int32)
            starts = ends - counts.flatten()
            ids = masks.nonzero()[:, -1].to(torch.int32).contiguous()
            index = TokenCompactedIndex(starts.view(1, 8, -1).contiguous(),
                ends.view(1, 8, -1).contiguous(), ids, torch.arange(8, device=device, dtype=torch.int32),
                8, masks.shape[1], 128)
            output = triton_token_compacted_prefill_attention(q, k, v, index, token_chunk_size=32)
            errors = []
            for h in range(8):
                logits = q[0, :, h].float() @ k[0, :, h//4].float().T / 128**.5
                mask = masks[h, torch.arange(n, device=device)//128] & causal
                ref = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v[0, :, h//4].float()
                relative = float((output[0, :, h].float()-ref).norm()/ref.norm())
                if relative >= .005:
                    raise ValueError(f"Compact kernel relative error {relative}")
                errors.append(relative)
            results.append(dict(n=n, variant=variant, max_relative_error=max(errors)))
    torch.cuda.synchronize()
    return dict(status="passed", device=torch.cuda.get_device_name(0), tests=results,
                scope="synthetic NN masks, gradients, GQA, partial tile, exact causal compact attention",
                timestamp=time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
