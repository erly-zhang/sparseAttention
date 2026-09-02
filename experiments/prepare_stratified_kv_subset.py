#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORK = Path("/home/ubuntu/work")
SOURCE = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/"
    "filtered_data/kv_retrieval.jsonl"
)
SOURCE_YAML = (
    WORK
    / "experiments/data/infinitebench_benchmark_specific_calibration/"
    "task_configs/kv_retrieval.yaml"
)
SOURCE_METRICS = SOURCE_YAML.parent / "metrics.py"
SOURCE_ROUGE = SOURCE_YAML.parent / "rouge.py"
SOURCE_UTILS = SOURCE_YAML.parent / "utils.py"
ROOT = WORK / "experiments/data/kv_oracle_stratified12_20260823"
DATA_DIR = ROOT / "filtered_data"
CONFIG_DIR = ROOT / "task_configs"
INDICES = [0, 45, 90, 135, 180, 225, 271, 316, 361, 406, 451, 496]


lines = SOURCE.read_text(encoding="utf-8").splitlines()
if len(lines) != 497:
    raise RuntimeError(f"Expected 497 KV Retrieval rows, found {len(lines)}")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
selected = [lines[index] for index in INDICES]
(DATA_DIR / "kv_retrieval.jsonl").write_text(
    "\n".join(selected) + "\n", encoding="utf-8"
)
yaml_text = SOURCE_YAML.read_text(encoding="utf-8")
old_dir = (
    "/home/ubuntu/work/experiments/data/"
    "infinitebench_benchmark_specific_calibration/filtered_data"
)
yaml_text = yaml_text.replace(old_dir, str(DATA_DIR))
(CONFIG_DIR / "kv_retrieval.yaml").write_text(yaml_text, encoding="utf-8")
(CONFIG_DIR / "metrics.py").write_bytes(SOURCE_METRICS.read_bytes())
(CONFIG_DIR / "rouge.py").write_bytes(SOURCE_ROUGE.read_bytes())
(CONFIG_DIR / "utils.py").write_bytes(SOURCE_UTILS.read_bytes())
manifest = {
    "source": str(SOURCE),
    "source_count": len(lines),
    "selected_indices": INDICES,
    "selected_count": len(selected),
    "metrics_sha256": hashlib.sha256(
        SOURCE_METRICS.read_bytes()
    ).hexdigest(),
    "rouge_sha256": hashlib.sha256(SOURCE_ROUGE.read_bytes()).hexdigest(),
    "utils_sha256": hashlib.sha256(SOURCE_UTILS.read_bytes()).hexdigest(),
    "selected_line_sha256": [
        hashlib.sha256(line.encode("utf-8")).hexdigest()
        for line in selected
    ],
}
(ROOT / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
