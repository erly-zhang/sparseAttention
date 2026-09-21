"""Submission only from an authenticated login node, after exact-snapshot debug."""
import json
from pathlib import Path
import socket
import subprocess
import time

from .stage1 import source_hashes, dependency_hashes


def main():
    host = socket.gethostname().split(".")[0]
    if not host.startswith("login"):
        raise RuntimeError("Submit from an authenticated login-node terminal, not a compute node")
    root = Path("/project/yl258/jz648/sparseAttention")
    proof = json.loads((root/"experiments/outputs/learned_mask_20260914_preflight/preflight_passed.json").read_text())
    if (proof.get("status") != "passed" or proof.get("source_sha256") != source_hashes()
            or proof.get("dependency_sha256") != dependency_hashes(root)):
        raise RuntimeError("Preflight missing, failed, or source changed")
    folder = Path(__file__).parent
    marker = folder/"submission.json"
    if marker.exists() or (root/"experiments/outputs/learned_mask_20260914_stage1").exists():
        raise RuntimeError("Existing submission/results: inspect before any retry")
    queue = subprocess.check_output(["squeue", "-u", "jz648", "-h", "-o", "%j"], text=True)
    if "learned-mask-stage1" in queue.splitlines():
        raise RuntimeError("This experiment is already queued")
    job = subprocess.check_output(["sbatch", "--parsable", str(folder/"formal.sbatch")], text=True).strip()
    with marker.open("x") as stream:
        json.dump(dict(job=job, host=host, timestamp=time.time(), preflight=proof["timestamp"]), stream, indent=2)
    print("SUBMITTED", job, flush=True)


if __name__ == "__main__":
    main()
