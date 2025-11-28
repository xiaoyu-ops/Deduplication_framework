"""Generate dummy binary files in `mix_dataset/unusable_bins` for testing.

Creates N files with random binary content and varying sizes to simulate corrupted/unusable files.

Usage:
  python scripts\generate_unusable_bins.py --output mix_dataset/unusable_bins --count 2000 --min-size 512 --max-size 16384
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def generate_bins(out_dir: Path, count: int = 2000, min_size: int = 512, max_size: int = 16384) -> tuple[int,int]:
    ensure_dir(out_dir)
    written = 0
    total_bytes = 0
    for i in range(count):
        fname = out_dir / f"bin_{i:06d}.bin"
        size = random.randint(min_size, max_size)
        # use os.urandom for binary randomness
        data = os.urandom(size)
        with open(fname, "wb") as fh:
            fh.write(data)
        written += 1
        total_bytes += size
    return written, total_bytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="mix_dataset/unusable_bins")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=16384)
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    written, total_bytes = generate_bins(out_dir, count=args.count, min_size=args.min_size, max_size=args.max_size)
    print(f"Generated {written} files totaling {total_bytes} bytes in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
