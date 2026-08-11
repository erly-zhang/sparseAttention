#!/usr/bin/env python3
"""Single CLI for sparse-attention benchmark adapters.

The benchmark subcommand changes only data loading, prompt construction, and
evaluation. Method installation, generation, timing, and input hashing are
shared by all adapters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.benchmark_shareprefill_ae3 import METHODS


ADAPTERS = {
    "infinitebench": "run_shareprefill_ae3_infinitebench.py",
    "ruler": "run_shareprefill_ae3_ruler.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=ADAPTERS)
    parser.add_argument("--method", choices=METHODS, required=True)
    args, adapter_args = parser.parse_known_args()

    script = Path(__file__).with_name(ADAPTERS[args.benchmark])
    command = [
        sys.executable,
        str(script),
        "--method",
        args.method,
        *adapter_args,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
