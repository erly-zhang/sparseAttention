#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python -m attention_map.visualize \
  --input_dir /home/ubuntu/work/attention_map/outputs/4k/niah_single_1
