"""Extract audio from Hugging Face dataset `danavery/urbansound8K` into `mix_dataset/` as WAV files.

Usage:
  python scripts\extract_urbansound8k.py --output mix_dataset --split train --max 1000

Requirements:
  pip install datasets soundfile numpy requests tqdm

This script handles audio fields provided as dicts with 'array'/'sampling_rate', local cached
paths, or http URLs. It writes files as sequential WAV files: `00000001.wav`, ...
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import requests
import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_url(url: str, timeout: int = 30) -> bytes | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def read_audio_from_path(p: Path) -> tuple[np.ndarray, int] | None:
    if not sf:
        raise RuntimeError("soundfile is required to read audio from path (pip install soundfile)")
    try:
        data, sr = sf.read(str(p))
        return np.array(data), int(sr)
    except Exception:
        return None


def read_audio_from_bytes(b: bytes) -> tuple[np.ndarray, int] | None:
    if not sf:
        raise RuntimeError("soundfile is required to read audio bytes (pip install soundfile)")
    try:
        with sf.SoundFile(io.BytesIO(b)) as f:
            data = f.read(dtype="float32")
            sr = f.samplerate
        return np.array(data), int(sr)
    except Exception:
        return None


def write_wav(out_path: Path, data: np.ndarray, sr: int) -> bool:
    if not sf:
        raise RuntimeError("soundfile is required to write WAV files (pip install soundfile)")
    try:
        # soundfile accepts float32 or integer arrays
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        sf.write(str(out_path), data, sr)
        print(f"Saved: {out_path}")
        return True
    except Exception as e:
        print(f"Failed to write {out_path}: {e}")
        return False


def infer_start_seq(out_dir: Path) -> int:
    seq = 1
    try:
        existing = []
        for p in out_dir.glob("*.wav"):
            stem = p.stem
            if stem.isdigit():
                try:
                    existing.append(int(stem))
                except Exception:
                    continue
        if existing:
            seq = max(existing) + 1
    except Exception:
        seq = 1
    return seq


def extract_from_example(example: dict, field: str, out_dir: Path, seq: int) -> int:
    """Attempt to extract one audio from example and write as WAV. Returns 1 on success else 0."""
    val = example.get(field)
    if val is None:
        return 0

    # If datasets returns a dict with array + sampling_rate
    if isinstance(val, dict):
        # common structures: {'array': [...], 'sampling_rate': 22050} or {'path': '...'}
        if "array" in val and "sampling_rate" in val:
            arr = np.array(val["array"])
            sr = int(val["sampling_rate"])
            out_path = out_dir / f"{seq:08d}.wav"
            if write_wav(out_path, arr, sr):
                return 1
            return 0

        p = val.get("path")
        if isinstance(p, str):
            # sometimes path points into cache or to a packaged archive; try reading if file exists
            if os.path.exists(p):
                res = read_audio_from_path(Path(p))
                if res:
                    arr, sr = res
                    out_path = out_dir / f"{seq:08d}.wav"
                    if write_wav(out_path, arr, sr):
                        return 1
                    return 0
            # try url field inside dict
        if "url" in val and isinstance(val.get("url"), str):
            b = download_url(val.get("url"))
            if not b:
                return 0
            res = read_audio_from_bytes(b)
            if res:
                arr, sr = res
                out_path = out_dir / f"{seq:08d}.wav"
                if write_wav(out_path, arr, sr):
                    return 1
            return 0

    # If val is a path string
    if isinstance(val, str):
        if val.startswith("http"):
            b = download_url(val)
            if not b:
                return 0
            res = read_audio_from_bytes(b)
            if res:
                arr, sr = res
                out_path = out_dir / f"{seq:08d}.wav"
                if write_wav(out_path, arr, sr):
                    return 1
            return 0
        if os.path.exists(val):
            res = read_audio_from_path(Path(val))
            if res:
                arr, sr = res
                out_path = out_dir / f"{seq:08d}.wav"
                if write_wav(out_path, arr, sr):
                    return 1
            return 0

    # If val is raw bytes (should be rare)
    if isinstance(val, (bytes, bytearray)):
        res = read_audio_from_bytes(bytes(val))
        if res:
            arr, sr = res
            out_path = out_dir / f"{seq:08d}.wav"
            if write_wav(out_path, arr, sr):
                return 1
        return 0

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="mix_dataset", help="Output directory")
    parser.add_argument("--split", "-s", default="train", help="Dataset split to use")
    parser.add_argument("--field", "-f", default="audio", help="Field name that contains audio data")
    parser.add_argument("--max", "-m", type=int, default=None, help="Max examples to process")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    print("Loading dataset danavery/urbansound8K...")
    ds = load_dataset("danavery/urbansound8K")
    if args.split not in ds.keys():
        print("Split not found. Available splits:", list(ds.keys()))
        return 2
    dataset = ds[args.split]
    total = len(dataset)
    print(f"Using split {args.split} with {total} examples. Extracting field '{args.field}'")

    seq = infer_start_seq(out_dir)
    count = 0
    start_idx = max(0, args.start)
    end_idx = total if args.max is None else min(total, start_idx + args.max)

    for i in tqdm(range(start_idx, end_idx), desc="examples"):
        ex = dataset[i]
        try:
            written = extract_from_example(ex, args.field, out_dir, seq)
            if written:
                seq += 1
                count += 1
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

    print(f"Done. Written {count} new audio files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
