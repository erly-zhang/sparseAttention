"""Fail-closed pre-submit gate. All computations run inside the developer GPU job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from .data import load_record
from .gpu_debug import check
from .model import Config, Predictor, VARIANTS, teacher_loss, final_mask
from .stage1 import source_hashes, dependency_hashes
from .train import cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Developer Slurm GPU allocation required")
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if busy.strip():
        raise ValueError("GPU busy; will not interfere")
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-m", __package__+".test_model"], check=True)
    numeric = check()
    (args.output/"gpu_numeric.json").write_text(json.dumps(numeric, indent=2))
    # Run in a separate process so dense model allocations are released before NN debug.
    capture = args.output / "capture"
    subprocess.run([sys.executable, "-u", "-m", __package__+".capture", "--root", str(args.root),
                    "--model", args.model, "--output", str(capture), "--debug"], check=True)
    groups = json.loads((args.root/"experiments/data/kv_randomscore_20260910/calibration/shareprefill_ae_k3_head_groups.json").read_text())["layers"]
    entries = json.loads((capture/"records.json").read_text())
    record = load_record(entries[0], "cuda")
    reports = []
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = Predictor({"0": groups["0"]}, variant, Config()).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        begin = time.monotonic()
        losses = []
        for h, q, k, prior, protect, core, full, condition, fixed in cases(model, record):
            scores = model(q, k, prior, condition, 0, h)
            loss = teacher_loss(scores, record["teacher_p"][h], fixed, len(k))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite real-data loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Nonfinite real-data gradient")
            opt.step()
            mask = final_mask(scores.detach(), core, full, protect, len(k), Config(), variant)
            if int(mask.sum()) != 10240 or not mask[protect].all():
                raise ValueError("Real-data budget/protection failed")
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        reports.append(dict(variant=variant, cases=len(losses), mean_loss=sum(losses)/len(losses),
                            seconds=time.monotonic()-begin))
        del model, opt
    # Ensure the historical AE architecture's forward/backward path works in this environment.
    from experiments.run_shareprefill_autoencoder_clustering import SharePrefillAutoencoder
    ae = SharePrefillAutoencoder().cuda()
    x = torch.zeros(1, 1, 1936, 1936, device="cuda")
    begin = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstruction, latent = ae(x)
    loss = (reconstruction.float()-x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if latent.shape != (1, 64) or not torch.isfinite(loss):
        raise ValueError("AE smoke failed")
    result = dict(status="passed", source_sha256=source_hashes(), dependency_sha256=dependency_hashes(args.root),
                  model=args.model, real_record_count=len(entries), real_nn_steps=reports,
                  ae_step_seconds=time.monotonic()-begin, gpu_numeric=numeric,
                  timestamp=time.time(), scope="debug only; no accuracy or CV scores",
                  historical_groups_for_debug_only=True)
    (args.output/"preflight_passed.json").write_text(json.dumps(result, indent=2))
    print("PREFLIGHT_PASSED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
