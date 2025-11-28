"""Extract images from the Hugging Face dataset `yfszzx/inspiration` (train split) into `mix_dataset/`.

Saves images with filename <sequential_number>.<ext> where ext is inferred by PIL when possible.
Writes every sample (no deduplication) and logs each saved file for traceability.

Usage:
  python scripts\extract_inspiration_images.py --output mix_dataset --split train --max 1000

Requirements:
  pip install datasets pillow requests tqdm
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_ext_and_save(b: bytes, out_dir: Path, seq: int, preferred_ext: str | None = None) -> Path:
    """Persist the bytes to disk using a sequential name.
    Optionally honor a preferred extension if provided."""
    fmt = preferred_ext
    if not fmt:
        try:
            img = Image.open(io.BytesIO(b))
            fmt = img.format.lower() if img.format else None
        except Exception:
            fmt = None
    ext = fmt or "bin"
    # Use a sequential filename to avoid overwriting and remove dedup behavior
    name = f"{seq:08d}.{ext}"
    out_path = out_dir / name
    out_path.write_bytes(b)
    print(f"Saved: {out_path}")
    return out_path


def download_url(url: str, timeout: int = 20) -> bytes | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def pil_image_to_bytes(img: Image.Image) -> tuple[bytes, str | None]:
    """Convert a PIL Image into raw bytes plus a best-effort extension."""
    fmt = (img.format or "PNG").upper()
    buf = io.BytesIO()
    try:
        img.save(buf, format=fmt)
    except Exception:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        fmt = "PNG"
    return buf.getvalue(), fmt.lower() if fmt else None


def extract_from_example(example: dict, field: str, out_dir: Path, seq_start: int) -> int:
    """Extract image(s) from the example's field. Always write files.
    Returns number written. `seq_start` is the starting sequence number for filenames;
    the caller is responsible for incrementing it across calls."""
    val = example.get(field)
    if val is None:
        print("No field", field, "in example; skipping")
        return 0

    # Handle lists/tuples of images by processing each element
    if isinstance(val, (list, tuple)):
        total = 0
        seq = seq_start
        for item in val:
            total += extract_from_example({field: item}, field, out_dir, seq)
            seq += 1
        return total

    # Handle already-decoded PIL Image objects
    if isinstance(val, Image.Image):
        b, ext = pil_image_to_bytes(val)
        infer_ext_and_save(b, out_dir, seq_start, preferred_ext=ext)
        return 1

    written = 0
    seq = seq_start
    # If it's bytes-like
    if isinstance(val, (bytes, bytearray)):
        infer_ext_and_save(bytes(val), out_dir, seq)
        written += 1
        return written

    # If it's a dict (datasets Image feature often becomes dict)
    if isinstance(val, dict):
        # local cache path
        p = val.get("path")
        if isinstance(p, str) and os.path.exists(p):
            try:
                b = Path(p).read_bytes()
            except Exception:
                print("Failed reading path", p)
                return 0
            ext = Path(p).suffix.lstrip(".").lower() or None
            infer_ext_and_save(b, out_dir, seq, preferred_ext=ext)
            written += 1
            return written
        # bytes field
        if "bytes" in val and isinstance(val.get("bytes"), (bytes, bytearray)):
            b = bytes(val.get("bytes"))
            infer_ext_and_save(b, out_dir, seq)
            written += 1
            return written
        # url field
        u = val.get("url")
        if isinstance(u, str) and u.startswith("http"):
            b = download_url(u)
            if not b:
                print("Failed to download url:", u)
                return 0
            # Skip suspiciously large binary blobs saved as .bin
            if len(b) > 5 * 1024 * 1024:
                try:
                    tmp_img = Image.open(io.BytesIO(b))
                except Exception:
                    print("Downloaded content appears large and not a recognizable image; skipping url:", u)
                    return 0
            infer_ext_and_save(b, out_dir, seq)
            written += 1
            return written

    # If it's a string URL
    if isinstance(val, str) and val.startswith("http"):
        b = download_url(val)
        if not b:
            print("Failed to download url:", val)
            return 0
        if len(b) > 5 * 1024 * 1024:
            try:
                tmp_img = Image.open(io.BytesIO(b))
            except Exception:
                print("Downloaded content appears large and not a recognizable image; skipping url:", val)
                return 0
        infer_ext_and_save(b, out_dir, seq)
        written += 1
        return written

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="mix_dataset", help="Output directory")
    parser.add_argument("--split", "-s", default="train", help="Dataset split to use")
    parser.add_argument("--field", "-f", default="image", help="Field name that contains image data")
    parser.add_argument("--max", "-m", type=int, default=None, help="Max examples to process")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    print("Loading dataset yfszzx/inspiration...")
    ds = load_dataset("yfszzx/inspiration")
    if args.split not in ds.keys():
        print("Split not found. Available splits:", list(ds.keys()))
        return 2
    dataset = ds[args.split]
    print(f"Using split {args.split} with {len(dataset)} examples. Extracting field '{args.field}'")

    # No deduplication: write every downloaded image with a sequential filename
    # Start sequence after any existing numbered files to avoid overwriting
    seq = 1
    try:
        existing_seqs = []
        for p in out_dir.glob("*"):
            if p.is_file():
                stem = p.stem
                if stem.isdigit():
                    try:
                        existing_seqs.append(int(stem))
                    except Exception:
                        continue
        if existing_seqs:
            seq = max(existing_seqs) + 1
    except Exception:
        seq = 1
    count = 0
    it = range(len(dataset)) if args.max is None else range(min(args.max, len(dataset)))
    for i in tqdm(it, desc="examples"):
        ex = dataset[i]
        try:
            written = extract_from_example(ex, args.field, out_dir, seq)
            seq += written
            count += written
        except Exception:
            continue

    print(f"Done. Written {count} new images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
