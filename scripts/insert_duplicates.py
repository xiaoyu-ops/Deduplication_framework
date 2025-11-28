"""Insert random duplicate copies into a dataset directory.

Usage:
  python scripts\insert_duplicates.py --dataset mix_dataset --ratio 0.1 --seed 42

This will select ~10% of files (by default) and copy them back into the same directory
with new filenames to create duplicates for dedup testing.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
import shutil
from typing import List


def list_files(dataset_dir: Path) -> List[Path]:
    return [p for p in dataset_dir.rglob('*') if p.is_file()]


def make_dup_name(src: Path, idx: int) -> Path:
    parent = src.parent
    stem = src.stem
    suffix = src.suffix
    new_name = f"{stem}_dup{idx}{suffix}"
    dst = parent / new_name
    # ensure not to overwrite existing file — if exists, increment idx until free
    while dst.exists():
        idx += 1
        new_name = f"{stem}_dup{idx}{suffix}"
        dst = parent / new_name
    return dst


def insert_duplicates(dataset_dir: Path, ratio: float = 0.1, seed: int | None = None, count: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)

    files = list_files(dataset_dir)
    n = len(files)
    if n == 0:
        return {"error": "no files found in dataset_dir"}

    if count is None:
        target = int(round(ratio * n))
    else:
        target = int(count)

    # If target > n, allow sampling with replacement
    if target <= n:
        chosen = random.sample(files, target)
    else:
        chosen = [random.choice(files) for _ in range(target)]

    created = []
    idx = 1
    for src in chosen:
        dst = make_dup_name(src, idx)
        try:
            shutil.copy2(src, dst)
            created.append(str(dst))
        except Exception as exc:
            # skip failures but continue
            print(f"Failed to copy {src} -> {dst}: {exc}")
        idx += 1

    return {"dataset_dir": str(dataset_dir), "original_files": n, "duplicates_requested": target, "duplicates_created": len(created), "created_files": created}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Insert duplicate copies into a dataset directory")
    parser.add_argument("--dataset", "-d", default="mix_dataset", help="Path to dataset directory (recursive)")
    parser.add_argument("--ratio", "-r", type=float, default=0.1, help="Fraction of files to duplicate (e.g. 0.1)")
    parser.add_argument("--count", "-c", type=int, default=None, help="Explicit number of duplicates to create (overrides ratio)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Dataset dir not found: {dataset_dir}")
        return 2

    result = insert_duplicates(dataset_dir, ratio=args.ratio, seed=args.seed, count=args.count)
    if "error" in result:
        print("Error:", result["error"])
        return 1

    print("Inserted duplicates summary:")
    print(f"  dataset_dir: {result['dataset_dir']}")
    print(f"  original_files: {result['original_files']}")
    print(f"  duplicates_requested: {result['duplicates_requested']}")
    print(f"  duplicates_created: {result['duplicates_created']}")
    # print a few created files
    for p in result['created_files'][:20]:
        print("   ", p)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
