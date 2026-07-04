"""Load RULER jsonl samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class RulerSample:
    sample_id: int
    input_text: str
    outputs: List[str]
    token_position_answer: Optional[int]
    source_file: str
    line_index: int


def discover_jsonl_files(data_root: Path) -> List[Path]:
    """Find all validation.jsonl under ruler_data (e.g. 4k/, 8k/)."""
    files = sorted(data_root.glob("**/validation.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No validation.jsonl found under {data_root}. "
            "Expected layout: ruler_data/4k/<task>/validation.jsonl"
        )
    return files


def load_samples(
    data_root: Path,
    task: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    start_line: int = 0,
) -> List[RulerSample]:
    """
    Load samples from ruler_data.

    Args:
        data_root: e.g. /home/ubuntu/work/ruler_data
        task: e.g. niah_single_1 (subdir name). None = all tasks.
        split: e.g. 4k or 8k (top-level folder). None = all splits.
        max_samples: cap total samples across files.
        start_line: skip this many lines at the start of each matched jsonl.
    """
    files = discover_jsonl_files(data_root)
    samples: List[RulerSample] = []

    for path in files:
        parts = path.relative_to(data_root).parts
        file_split = parts[0] if parts else ""
        file_task = parts[1] if len(parts) > 1 else ""

        if split is not None and file_split != split:
            continue
        if task is not None and file_task != task:
            continue

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < start_line:
                    continue
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
                row = json.loads(line)
                samples.append(
                    RulerSample(
                        sample_id=int(row.get("index", line_idx)),
                        input_text=row["input"],
                        outputs=row.get("outputs", []),
                        token_position_answer=row.get("token_position_answer"),
                        source_file=str(path),
                        line_index=line_idx,
                    )
                )
    if not samples:
        raise ValueError(
            f"No samples matched data_root={data_root}, split={split}, task={task}"
        )
    return samples


def iter_samples(*args, **kwargs) -> Iterator[RulerSample]:
    for s in load_samples(*args, **kwargs):
        yield s
