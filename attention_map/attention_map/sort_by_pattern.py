#!/usr/bin/env python3
"""
Sort saved attention maps into folders by predicted pattern.

Input:
  - sample_dir: attention_map output sample_* directory that contains layer_XX/head_YY.npy
  - pattern_json: produced by attention_map.classify (default: <sample_dir>/pattern_classification.json)

Output:
  <out_dir>/
    Vertical-Slash/
      layer_00_head_00.npy
      ...
    StreamLLM(A-shape)/
      ...
    Block-wise/
      ...

By default out_dir = <sample_dir>/pattern_sorted
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


PATTERN_TO_DIR = {
    "vertical_and_slash": "Vertical-Slash",
    "stream_llm": "StreamLLM(A-shape)",
    "block_sparse": "Block-wise",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sort attention maps into folders by pattern classification")
    p.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="sample_* directory containing layer_XX/head_YY.npy",
    )
    p.add_argument(
        "--pattern_json",
        type=str,
        default=None,
        help="Default: <sample_dir>/pattern_classification.json",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Default: <sample_dir>/pattern_sorted",
    )
    p.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="copy: duplicate npy files; symlink: create symlinks (saves disk)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in out_dir",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing .npy/.png in pattern subfolders before sorting",
    )
    return p.parse_args()


def load_classification(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def head_file(sample_dir: Path, layer: int, head: int) -> Path:
    return sample_dir / f"layer_{layer:02d}" / f"head_{head:02d}.npy"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    pattern_json = Path(args.pattern_json) if args.pattern_json else (sample_dir / "pattern_classification.json")
    out_dir = Path(args.out_dir) if args.out_dir else (sample_dir / "pattern_sorted")

    cls = load_classification(pattern_json)
    results = cls.get("results", {})

    for d in PATTERN_TO_DIR.values():
        folder = out_dir / d
        if args.clean and folder.exists():
            for f in folder.glob("*.npy"):
                f.unlink()
            for f in folder.glob("aggregated_*.png"):
                f.unlink()
        ensure_dir(folder)

    moved = {d: 0 for d in PATTERN_TO_DIR.values()}
    skipped = 0

    for layer_key, heads in results.items():
        layer = int(layer_key)
        for head_key, info in heads.items():
            head = int(head_key)
            best = info.get("best", {})
            pattern = best.get("pattern")
            folder = PATTERN_TO_DIR.get(pattern, "Unknown")
            ensure_dir(out_dir / folder)

            src = head_file(sample_dir, layer, head)
            if not src.exists():
                skipped += 1
                continue

            dst_name = f"layer_{layer:02d}_head_{head:02d}.npy"
            dst = out_dir / folder / dst_name
            write_link_or_copy(src, dst, mode=args.mode, overwrite=args.overwrite)
            moved[folder] = moved.get(folder, 0) + 1

    summary = {
        "sample_dir": str(sample_dir),
        "pattern_json": str(pattern_json),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "moved": moved,
        "skipped_missing_src": skipped,
        "pattern_to_dir": PATTERN_TO_DIR,
    }
    (out_dir / "sort_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

