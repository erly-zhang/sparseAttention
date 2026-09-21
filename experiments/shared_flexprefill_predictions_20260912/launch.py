"""Independent rerun with the original command and strict prediction audit."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import hashlib

from experiments.shared_flexprefill_20260911 import launch_formal as base
from experiments.shared_flexprefill_predictions_20260912.predictions import export

ROOT = base.ROOT
SOURCE = ROOT / "experiments/shared_flexprefill_predictions_20260912"
OUT = ROOT / "experiments/outputs/shared_flexprefill_predictions_20260912/formal_shared"
OLD = ROOT / "experiments/outputs/shared_flexprefill_20260911/formal_shared"


def scorer():
    path = base.SMOKE_CONFIG / "metrics.py"
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == "get_score_one_kv_retrieval"]
    assert len(nodes) == 1
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_score_one_kv_retrieval"]


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_NAME") == "ae3-flex-preds"
    assert json.loads((OUT.parent / "preflight_identity_passed.json").read_text())["status"] == "passed"
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
    assert len(gpu.strip().splitlines()) == 1 and "A100" in gpu and "80GB" in gpu
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    OUT.mkdir(exist_ok=False)
    base.OUT = OUT
    try:
        config = base.prepare_config()
        old = json.loads((OLD / "command.json").read_text())
        # Refuse algorithm/runner drift relative to the completed historical run.
        for path, digest in old["sha256"].items():
            path = Path(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
        for name in ("adapter.py", "run.py", "formal_run.py"):
            assert (base.SOURCE / name).read_bytes() == (OLD / "source" / name).read_bytes()
        snapshot = OUT / "source"
        shutil.copytree(SOURCE, snapshot, ignore=shutil.ignore_patterns("__pycache__", "._*"))
        command = list(old["command"])
        command[command.index("-m") + 1] = "experiments.shared_flexprefill_predictions_20260912.run"
        command[command.index("--output_dir") + 1] = str(OUT)
        command[command.index("--task_config_dir") + 1] = str(config)
        base.save(OUT / "command.json", dict(command=command, job_id=os.environ["SLURM_JOB_ID"],
                  parent_job="1277421", changed_fields=["logging wrapper", "output_dir", "equivalent task_config_dir"],
                  start_time=time.time(), gpu=gpu,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*") if p.is_file()}))
        with (OUT / "runner.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        export(OUT / base.METHOD / "kv_retrieval", base.DATA / "filtered_data/kv_retrieval.jsonl", scorer(), 497)
        base.verify()
    except BaseException as error:
        base.save(OUT / "exit.json", dict(returncode=1, error=repr(error), end_time=time.time()))
        raise
    base.save(OUT / "exit.json", dict(returncode=0, end_time=time.time()))


if __name__ == "__main__":
    main()
