"""Single-GPU offline job; never invokes sbatch or runs formal benchmark inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .capture import verify_restoration
from .folds import build_folds


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.iterdir() if p.suffix in {".py", ".sbatch"}}


def dependency_hashes(root):
    paths = ("experiments/run_shareprefill_autoencoder_clustering.py",
             "experiments/token_compacted_sparse.py",
             "experiments/data/InfiniteBench_full_20260914/restored_calibration/calibration.jsonl",
             "experiments/data/kv_randomscore_20260910/references/original_autoblock/online_metrics.jsonl")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}


def run(command, log, root):
    print("RUN", command, flush=True)
    with log.open("x") as stream:
        subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    args = p.parse_args()
    proof = json.loads(args.preflight.read_text())
    if proof.get("status") != "passed" or proof.get("source_sha256") != source_hashes():
        raise ValueError("Debug must pass for this exact source snapshot before formal run")
    if proof.get("dependency_sha256") != dependency_hashes(args.root) or proof.get("model") != args.model:
        raise ValueError("Debug dependencies/model changed")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("An authorized GPU Slurm allocation is required")
    calibration, historical = verify_restoration(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(model=args.model, job_id=os.environ["SLURM_JOB_ID"], source_sha256=source_hashes(),
                  protocol="3 folds, 2 fit inputs + 1 checkpoint-selection input; no independent test",
                  nn_epochs=5, nn_lr=.001, nn_weight_decay=.01, seed=42,
                  ae_epochs=1000, ae_patience=30, ae_lr=.001,
                  ae_scope="historical architecture and optimizer; refitted separately per fold",
                  resource_cap="1 A10080GB; 6 hours imposed by sbatch; completion not guaranteed",
                  purpose="offline validation only, no 497-row generation runs")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    start = time.time()
    code, error = 0, None
    try:
        capture = args.output / "teacher"
        run([sys.executable, "-u", "-m", __package__+".capture", "--root", str(args.root),
             "--model", args.model, "--output", str(capture)], args.output/"capture.log", args.root)
        records = json.loads((capture/"records.json").read_text())
        if len(records) != 3*32*8:
            raise ValueError("Dense teacher capture incomplete")
        group_files = []
        for fold in range(3):
            directory = args.output / f"ae_fold{fold}"
            directory.mkdir()
            rows = [r for i, r in enumerate(calibration) if i != fold]
            data = directory / "calibration.jsonl"
            data.write_text("".join(json.dumps(r)+"\n" for r in rows))
            # The old all-three-fitted AE and latents are never loaded here.
            run([sys.executable, "-u", "-m", "experiments.run_shareprefill_autoencoder_clustering", "all",
                 "--model_name_or_path", args.model, "--data_path", str(data),
                 "--calibration_format", "prompt_jsonl", "--head_selection_num_samples", "2",
                 "--artifact_dir", str(directory/"fit"), "--groups", "3", "--epochs", "1000",
                 "--patience", "30", "--seed", "42", "--device", "cuda"], directory/"fit.log", args.root)
            group_files.append(directory/"fit/shareprefill_ae_k3_head_groups.json")
        plans = build_folds(records, group_files,
                            historical/"references/original_autoblock/online_metrics.jsonl", args.output/"folds")
        results = []
        for fold, plan in enumerate(plans):
            directory = args.output / f"nn_fold{fold}"
            run([sys.executable, "-u", "-m", __package__+".train", "--manifest", plan["manifest"],
                 "--groups", plan["groups"], "--output", str(directory), "--device", "cuda",
                 "--epochs", "5", "--learning-rate", ".001", "--seed", "42"],
                args.output/f"nn_fold{fold}.log", args.root)
            for variant in ("per_head", "per_cluster", "per_layer", "full_mask"):
                result = json.loads((directory/variant/"validation.json").read_text())
                result.pop("rows")
                result.update(fold=fold, validation_family=plan["validation_family"])
                results.append(result)
        (args.output/"summary.json").write_text(json.dumps(dict(status="completed", results=results,
             interpretation="validation used for checkpoint selection; not an independent test or KV accuracy"), indent=2))
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        (args.output/"exit.json").write_text(json.dumps(dict(returncode=code, error=error,
             start_time=start, end_time=time.time()), indent=2))


if __name__ == "__main__":
    main()
