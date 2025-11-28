"""Convert a JSONL file into individual .json files (one JSON object per file).

Default behavior:
  - Input: `mix_dataset/dolma3_raw.jsonl`
  - Output directory: `mix_dataset/dolma3_raw_files/`
  - Filenames: zero-padded index (e.g. 00000001.json)

Options:
  --use-source-id: if present, use the example's `source_id` or `id` as filename (sanitized);
                    falls back to indexed names on collision or missing id.
  --pad: zero-pad width for index filenames (default 8)
  --overwrite: overwrite existing files instead of skipping

Usage:
  python scripts\jsonl_to_json_files.py --input mix_dataset/dolma3_raw.jsonl --output mix_dataset\dolma3_raw_files --max 1000

"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm


_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str, max_len: int = 200) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = _SANITIZE_RE.sub("_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s or "id"


def choose_filename(obj: dict, idx: int, out_dir: Path, pad: int, use_source_id: bool, overwrite: bool) -> Path:
    if use_source_id:
        sid = obj.get("source_id") or obj.get("id") or obj.get("idx")
        if sid:
            name = sanitize_filename(str(sid))
            candidate = out_dir / (name + ".json")
            if candidate.exists() and not overwrite:
                # fallback to index-based name
                return out_dir / (f"{idx:0{pad}d}.json")
            return candidate
    # default: index-based
    return out_dir / (f"{idx:0{pad}d}.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="mix_dataset/dolma3_raw.jsonl", help="Input JSONL path")
    parser.add_argument("--output", "-o", default="mix_dataset/dolma3_raw_files", help="Output directory for .json files")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--use-source-id", action="store_true", help="Use source_id/id as filename when available")
    parser.add_argument("--pad", type=int, default=8, help="Zero-pad width for index filenames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args(argv)

    inp = Path(args.input)
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    if not inp.exists():
        print(f"Input file not found: {inp}")
        return 2

    written = 0
    skipped = 0
    i = 0

    with inp.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="converting"):
            if i < args.start:
                i += 1
                skipped += 1
                continue
            if args.max is not None and written >= args.max:
                break
            line = line.rstrip("\n\r")
            if not line:
                i += 1
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip malformed line
                i += 1
                skipped += 1
                continue
            out_path = choose_filename(obj, i, out_dir, args.pad, args.use_source_id, args.overwrite)
            if out_path.exists() and not args.overwrite:
                i += 1
                skipped += 1
                continue
            try:
                with out_path.open("w", encoding="utf-8") as of:
                    json.dump(obj, of, ensure_ascii=False, indent=2)
            except Exception:
                i += 1
                skipped += 1
                continue
            written += 1
            i += 1

    print(f"Done. Written {written} files, skipped {skipped} (from {inp}) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
