"""One fresh, foreground three-input developer-GPU smoke; never submits sbatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["flexprefill_ae3_shared_pattern", "flexprefill"],
               default="flexprefill_ae3_shared_pattern")
p.add_argument("--attempt", type=int, default=1)
args = p.parse_args()
root = Path("/project/yl258/jz648/sparseAttention")
assert os.environ.get("SLURM_JOB_ID"), "Must run inside the authorized GPU allocation"
gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
assert not gpu.strip(), "GPU is busy; do not disturb other workloads"
out = root / "experiments/outputs/shared_flexprefill_20260911" / f"smoke_{args.method}_attempt{args.attempt}"
out.mkdir(parents=True, exist_ok=False)
data = root / "experiments/data/kv_randomscore_20260910"
config = root / "experiments/outputs/random_score_residual_20260910/aligned_recovery/data/0_3/task_configs"
python = "/project/yl258/jz648/miniconda3/envs/sparseattention-score26/bin/python"
command = [python, "-u", "-m", "experiments.shared_flexprefill_20260911.run",
           "--model", "/mmfs1/scratch/yl258/jz648/Llama-3.1-8B-Instruct",
           "--method", args.method, "--task", "kv_retrieval", "--limit", "3",
           "--max_length", "131072", "--batch_size", "1", "--seed", "42", "--chat",
           "--group_config", str(data / "calibration/shareprefill_ae_k3_head_groups.json"),
           "--calibration_scope", "benchmark_specific", "--flexprefill_root", str(root / "FlexPrefill"),
           "--task_config_dir", str(config),
           "--reference_metrics", str(data / "references/original_autoblock/online_metrics.jsonl"),
           "--record_sparsity", "--output_dir", str(out)]
env = dict(os.environ, PYTHONPATH=f"{root}:{root / 'FlexPrefill'}", PYTHONNOUSERSITE="1",
           OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
manifest = dict(command=command, job_id=env["SLURM_JOB_ID"], start_time=time.time(),
                source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in Path(__file__).parent.glob("*.py")})
(out / "command.json").write_text(json.dumps(manifest, indent=2))
shutil.copy2("/tmp/shared_flexprefill_correctness_20260911.log", out / "correctness.log")
print("SMOKE_OUTPUT", out, flush=True)
with (out / "runner.log").open("x") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
(out / "exit.json").write_text(json.dumps(dict(returncode=result.returncode, end_time=time.time())))
print("SMOKE_EXIT", result.returncode, flush=True)
raise SystemExit(result.returncode)
