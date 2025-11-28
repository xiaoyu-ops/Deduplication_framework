"""Download allenai/dolma3_mix-6T-1025 (streaming) and write examples to JSONL in `mix_dataset`.

This script uses `datasets.load_dataset(..., streaming=True)` to avoid full upfront
indexing or native decoder issues. It writes raw example JSON objects (one per line)
so you get a direct dump of the dataset into `mix_dataset`.

Usage:
  python scripts\download_dolma3_jsonl.py --output mix_dataset --split train --max 10000

Dependencies:
  pip install datasets tqdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Any

from datasets import load_dataset
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def open_dataset_iter(dataset_id: str, split: str) -> Iterable[dict]:
    try:
        return load_dataset(dataset_id, split=split, streaming=True)
    except Exception:
        # try loading non-stream as fallback (may block)
        ds = load_dataset(dataset_id)
        if split not in ds.keys():
            raise RuntimeError(f"Split {split} not found in dataset {dataset_id}")
        return ds[split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="allenai/dolma3_mix-6T-1025")
    parser.add_argument("--split", "-s", default="train")
    parser.add_argument("--output", "-o", default="mix_dataset")
    parser.add_argument("--jsonl", default="dolma3_raw.jsonl", help="Output JSONL filename")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000, help="Lines per flush")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    ensure_dir(out_dir)
    out_path = out_dir / args.jsonl

    mode = "w" if args.overwrite else "a"
    f = open(out_path, mode, encoding="utf-8")

    ds_iter = open_dataset_iter(args.dataset, args.split)

    written = 0
    skipped = 0
    i = 0

    try:
        for ex in tqdm(ds_iter, desc="download"):
            if i < args.start:
                i += 1
                skipped += 1
                continue
            if args.max is not None and written >= args.max:
                break
            # write full example as JSON object (preserves all fields)
            try:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            except Exception:
                # fallback: try to convert problematic fields
                f.write(json.dumps({k: str(v) for k, v in ex.items()}, ensure_ascii=False) + "\n")
            written += 1
            i += 1
            if written % args.chunk_size == 0:
                f.flush()
    finally:
        f.flush()
        f.close()

    print(f"Done. Written {written} examples (skipped {skipped}) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
