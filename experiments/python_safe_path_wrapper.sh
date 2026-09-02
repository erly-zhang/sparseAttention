#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/ubuntu/work/experiments/ruler_compat
cd /tmp
exec /home/ubuntu/miniconda3/envs/official_flex/bin/python "$@"
