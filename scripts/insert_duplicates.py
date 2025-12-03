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


def list_files(dataset_dir: Path, exts: List[str] | None = None, exclude_dirs: List[Path] | None = None) -> List[Path]:
    files: List[Path] = []
    exclude_dirs = exclude_dirs or []
    for p in dataset_dir.rglob('*'):
        if not p.is_file():
            continue
        # skip files under any excluded directory
        if any(str(p).startswith(str(ed)) for ed in exclude_dirs):
            continue
        if exts:
            if p.suffix.lower() not in exts:
                continue
        files.append(p)
    return files


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


def insert_duplicates(dataset_dir: Path, ratio: float = 0.1, seed: int | None = None, count: int | None = None,
                      exts: List[str] | None = None, exclude_dirs: List[Path] | None = None, dry_run: bool = False) -> dict:
    if seed is not None:
        random.seed(seed)

    files = list_files(dataset_dir, exts=exts, exclude_dirs=exclude_dirs)
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
            if not dry_run:
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
    parser.add_argument("--exts", type=str, default=None, help="Comma-separated list of extensions to include (e.g. .wav,.mp3). Default: all files")
    parser.add_argument("--exclude", type=str, default=None, help="Relative subdirectory to exclude (e.g. unusable_bins). Can be comma-separated for multiple")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually copy files; just report what would be done")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Dataset dir not found: {dataset_dir}")
        return 2

    exts = None
    if args.exts:
        exts = [e.strip().lower() if e.strip().startswith('.') else f".{e.strip().lower()}" for e in args.exts.split(',')]

    exclude_dirs = None
    if args.exclude:
        exclude_dirs = []
        for part in args.exclude.split(','):
            part = part.strip()
            if not part:
                continue
            exclude_dirs.append((dataset_dir / part).resolve())

    result = insert_duplicates(dataset_dir, ratio=args.ratio, seed=args.seed, count=args.count,
                               exts=exts, exclude_dirs=exclude_dirs, dry_run=args.dry_run)
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
