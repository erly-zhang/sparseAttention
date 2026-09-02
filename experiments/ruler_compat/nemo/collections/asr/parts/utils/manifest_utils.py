"""Minimal NeMo manifest helpers required by the official RULER evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(
    path: str | Path, rows: Iterable[dict[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
