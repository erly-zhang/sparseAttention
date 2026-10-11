"""Single-allocation, fail-closed full KV run with immutable launch evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path("/project/yl258/jz648/sparseAttention")
SOURCE = ROOT / "experiments/shared_flexprefill_20260911"
OUT = ROOT / "experiments/outputs/shared_flexprefill_20260911/formal_shared"
DATA = ROOT / "experiments/data/kv_randomscore_20260910"
METHOD = "flexprefill_ae3_shared_pattern"
REFERENCE = DATA / "references/original_autoblock/online_metrics.jsonl"
SMOKE_CONFIG = ROOT / "experiments/outputs/random_score_residual_20260910/aligned_recovery/data/0_3/task_configs"
OWNS_OUTPUT = False


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def prepare_config():
    destination = OUT / "task_configs"
    destination.mkdir()
    for path in SMOKE_CONFIG.glob("*.py"):
        shutil.copy2(path, destination / path.name)
    node = yaml.compose((SMOKE_CONFIG / "kv_retrieval.yaml").read_text())
    fields = {key.value: value for key, value in node.value}
    values = {key.value: value for key, value in fields["dataset_kwargs"].value}
    values["data_dir"].value = str(DATA / "filtered_data")
    fields["dataset_path"].value = str(DATA / "task_configs/InfiniteBench.py")
    (destination / "kv_retrieval.yaml").write_text(yaml.serialize(node))
    assert len(rows(DATA / "filtered_data/kv_retrieval.jsonl")) == 497
    assert len(rows(REFERENCE)) == 497
    return destination


def verify():
    folder = OUT / METHOD / "kv_retrieval"
    measured, reference = rows(folder / "online_metrics.jsonl"), rows(REFERENCE)
    identity = lambda row: (int(row["input_tokens"]), row["input_ids_sha256"])
    got, expected = list(map(identity, measured)), list(map(identity, reference))
    assert len(got) == len(set(got)) == len(expected) == 497
    assert got == expected, "Generation order or input identity mismatch"
    s = json.loads((folder / "summary.json").read_text())
    a = s["input_alignment"]
    assert a["status"] == "passed" and a["alignment_scope"] == "full"
    assert a["count"] == a["reference_count"] == 497
    m = s["method_metadata"]
    assert m["num_groups_per_layer"] == 3 and m["fixed_token_cap"] is None
    assert m["implementation"] == METHOD
    assert m["online_config"] == dict(block_size=128, flex_prefill_gamma=.95,
                                      flex_prefill_tau=.1, flex_prefill_min_budget=512,
                                      flex_prefill_max_budget=None)
    assert s["run_args"]["max_new_tokens"] == 128
    assert not s["run_args"]["profile_one_token"]
    for r in measured:
        n = r["input_tokens"]
        assert r["causal_token_pairs"] == 1024 * n * (n + 1) // 2
        assert 0 < r["selected_token_pairs"] <= r["causal_token_pairs"]
        assert 0 < r["generated_tokens"] <= 128
        assert 0 < r["prefill_latency_sec"] < r["total_latency_sec"]
    import statistics
    selected = sum(r["selected_token_pairs"] for r in measured)
    causal = sum(r["causal_token_pairs"] for r in measured)
    prefill = sorted(r["prefill_latency_sec"] for r in measured)
    score = s["official_lm_eval_results"]["kv_retrieval"]["get_score_one_kv_retrieval,none"]
    result = dict(status="passed", count=497, input_alignment=a,
                  accuracy=score, correct=round(497 * score),
                  selected_token_pairs=selected, causal_token_pairs=causal,
                  global_pair_sparsity=1-selected/causal,
                  prefill_mean=statistics.mean(prefill), prefill_median=statistics.median(prefill),
                  prefill_p90=statistics.quantiles(prefill, n=100, method="inclusive")[89],
                  decode_mean=statistics.mean(r["decode_latency_sec"] for r in measured),
                  total_mean=statistics.mean(r["total_latency_sec"] for r in measured),
                  peak_memory_bytes=max(r["peak_memory_bytes"] for r in measured),
                  hardware=s["hardware"], timing_scope=s["timing_scope"])
    save(OUT / "verified_complete.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    global OWNS_OUTPUT
    assert os.environ.get("SLURM_JOB_ID"), "Requires independent sbatch allocation"
    assert os.environ.get("SLURM_JOB_NAME") == "ae3-shared-flex"
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
    assert len(gpu.strip().splitlines()) == 1 and "A100" in gpu and "80GB" in gpu
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    assert not processes.strip(), "Allocated GPU already has a compute process"
    OUT.mkdir(exist_ok=False)
    OWNS_OUTPUT = True
    config = prepare_config()
    sources = OUT / "source"
    sources.mkdir()
    for path in SOURCE.iterdir():
        if path.suffix in {".py", ".sbatch"}:
            shutil.copy2(path, sources / path.name)
    command = [sys.executable, "-u", "-m", "experiments.shared_flexprefill_20260911.formal_run",
               "--model", "/mmfs1/scratch/yl258/jz648/Llama-3.1-8B-Instruct",
               "--method", METHOD, "--task", "kv_retrieval", "--limit", "0",
               "--max_length", "131072", "--batch_size", "1", "--seed", "42", "--chat",
               "--group_config", str(DATA / "calibration/shareprefill_ae_k3_head_groups.json"),
               "--calibration_scope", "benchmark_specific", "--flexprefill_root", str(ROOT / "FlexPrefill"),
               "--task_config_dir", str(config), "--reference_metrics", str(REFERENCE),
               "--record_sparsity", "--output_dir", str(OUT)]
    tracked = list(sources.iterdir()) + [REFERENCE, DATA / "filtered_data/kv_retrieval.jsonl",
              DATA / "calibration/shareprefill_ae_k3_head_groups.json", config / "kv_retrieval.yaml",
              ROOT / "experiments/run_dense_multimodel_infinitebench.py"]
    save(OUT / "command.json", dict(command=command, job_id=os.environ["SLURM_JOB_ID"],
         start_time=time.time(), gpu=gpu, sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}))
    with (OUT / "correctness.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "experiments.shared_flexprefill_20260911.test_adapter"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (OUT / "runner.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OWNS_OUTPUT and not (OUT / "exit.json").exists():
            save(OUT / "exit.json", dict(returncode=1, error=repr(error), end_time=time.time()))
        raise
    else:
        save(OUT / "exit.json", dict(returncode=0, end_time=time.time()))
