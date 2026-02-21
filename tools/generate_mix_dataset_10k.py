"""Generate a 10k-scale mixed modality dataset with challenging cases.

This script synthesizes files for image/audio/text categories alongside
tricky variants (missing extensions, wrong extensions, ambiguous
containers, empty files). It also produces a ground-truth JSONL file.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import random
import shutil
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATASET_TOTAL = 10_000

NORMAL_RATIO = 0.70
MISSING_RATIO = 0.10
WRONG_RATIO = 0.10
AMBIGUOUS_RATIO = 0.05
EMPTY_RATIO = 0.05

NORMAL_TOTAL = int(DATASET_TOTAL * NORMAL_RATIO)
MISSING_TOTAL = int(DATASET_TOTAL * MISSING_RATIO)
WRONG_TOTAL = int(DATASET_TOTAL * WRONG_RATIO)
AMBIGUOUS_TOTAL = int(DATASET_TOTAL * AMBIGUOUS_RATIO)
EMPTY_TOTAL = DATASET_TOTAL - (NORMAL_TOTAL + MISSING_TOTAL + WRONG_TOTAL + AMBIGUOUS_TOTAL)

# Distribute normals roughly evenly across three modalities
NORMAL_IMAGE = NORMAL_TOTAL // 3
NORMAL_AUDIO = NORMAL_TOTAL // 3
NORMAL_TEXT = NORMAL_TOTAL - NORMAL_IMAGE - NORMAL_AUDIO

# For tricky sets, choose fixed splits (summing to totals)
MISSING_IMAGE = 400
MISSING_AUDIO = 300
MISSING_TEXT = MISSING_TOTAL - MISSING_IMAGE - MISSING_AUDIO

WRONG_IMAGE = 400
WRONG_AUDIO = 300
WRONG_TEXT = WRONG_TOTAL - WRONG_IMAGE - WRONG_AUDIO

AMBIGUOUS_IMAGE = 200
AMBIGUOUS_AUDIO = 150
AMBIGUOUS_TEXT = AMBIGUOUS_TOTAL - AMBIGUOUS_IMAGE - AMBIGUOUS_AUDIO

EMPTY_UNKNOWN = EMPTY_TOTAL  # label as unknown

PNG_BASE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAB7GkOtAAAAFElEQVR4nGP4////fwYGBgYGABwABQgKDc3nAAAAAElFTkSuQmCC"
)


def generate_audio_frames(duration: float = 0.2, sample_rate: int = 16_000, frequency: float = 440.0) -> bytes:
    total_samples = int(duration * sample_rate)
    amplitude = 0.4
    frames = bytearray()
    for n in range(total_samples):
        sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * n / sample_rate))
        frames.extend(struct.pack("<h", sample))
    return bytes(frames)


AUDIO_FRAMES = generate_audio_frames()


def write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_BASE)


def write_wav(path: Path, duration: float = 0.2, freq: float = 440.0) -> None:
    frames = AUDIO_FRAMES if freq == 440.0 and duration == 0.2 else generate_audio_frames(duration, frequency=freq)
    path.parent.mkdir(parents=True, exist_ok=True)
    import wave

    with wave.open(str(path), "w") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)  # 16-bit PCM
        wave_file.setframerate(16_000)
        wave_file.writeframes(frames)


def write_text_json(path: Path, variant: str = "standard") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, str]] = []

    if variant == "standard":
        records = [{"text": f"sample text content {i}", "title": f"sample_{i}"} for i in range(3)]
    elif variant == "with_urls":
        records = [
            {
                "content": f"image description {i}",
                "url": f"https://example.com/assets/img_{i}.png",
            }
            for i in range(3)
        ]
    elif variant == "ambiguous":
        records = [
            {
                "data": f"payload {i}",
                "meta": {"type": random.choice(["img", "audio", "text"]), "score": random.random()},
            }
            for i in range(3)
        ]
    else:
        records = [{"text": f"sample text content {i}"} for i in range(3)]

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def write_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def allocate_indices(total: int) -> Iterable[int]:
    return range(total)


def register(entries: List[Dict[str, str]], dataset_root: Path, relative_path: Path, label: str) -> None:
    entries.append({"filename": relative_path.as_posix(), "label": label})


def make_normal_samples(dataset_root: Path, entries: List[Dict[str, str]]) -> None:
    for idx in allocate_indices(NORMAL_IMAGE):
        rel = Path("normal/image") / f"image_{idx:05}.png"
        write_png(dataset_root / rel)
        register(entries, dataset_root, rel, "image")

    for idx in allocate_indices(NORMAL_AUDIO):
        rel = Path("normal/audio") / f"audio_{idx:05}.wav"
        write_wav(dataset_root / rel)
        register(entries, dataset_root, rel, "audio")

    for idx in allocate_indices(NORMAL_TEXT):
        rel = Path("normal/text") / f"text_{idx:05}.json"
        write_text_json(dataset_root / rel, variant="standard")
        register(entries, dataset_root, rel, "text")


def make_missing_extension(dataset_root: Path, entries: List[Dict[str, str]]) -> None:
    for idx in allocate_indices(MISSING_IMAGE):
        rel = Path("tricky/missing") / f"noext_image_{idx:05}"
        write_png(dataset_root / rel)
        register(entries, dataset_root, rel, "image")

    for idx in allocate_indices(MISSING_AUDIO):
        rel = Path("tricky/missing") / f"noext_audio_{idx:05}"
        write_wav(dataset_root / rel)
        register(entries, dataset_root, rel, "audio")

    for idx in allocate_indices(MISSING_TEXT):
        rel = Path("tricky/missing") / f"noext_text_{idx:05}"
        write_text_json(dataset_root / rel, variant="standard")
        register(entries, dataset_root, rel, "text")


def make_wrong_extension(dataset_root: Path, entries: List[Dict[str, str]]) -> None:
    for idx in allocate_indices(WRONG_IMAGE):
        rel = Path("tricky/wrong") / f"wrong_image_{idx:05}.txt"
        write_png(dataset_root / rel)
        register(entries, dataset_root, rel, "image")

    for idx in allocate_indices(WRONG_AUDIO):
        rel = Path("tricky/wrong") / f"wrong_audio_{idx:05}.json"
        write_wav(dataset_root / rel)
        register(entries, dataset_root, rel, "audio")

    for idx in allocate_indices(WRONG_TEXT):
        rel = Path("tricky/wrong") / f"wrong_text_{idx:05}.wav"
        write_text_json(dataset_root / rel, variant="standard")
        register(entries, dataset_root, rel, "text")


def make_ambiguous(dataset_root: Path, entries: List[Dict[str, str]]) -> None:
    for idx in allocate_indices(AMBIGUOUS_IMAGE):
        rel = Path("tricky/ambiguous") / f"ambig_image_{idx:05}.bin"
        payload = PNG_BASE[: len(PNG_BASE) // 2] + b"\x00metadata:image"
        write_binary(dataset_root / rel, payload)
        register(entries, dataset_root, rel, "image")

    for idx in allocate_indices(AMBIGUOUS_AUDIO):
        rel = Path("tricky/ambiguous") / f"ambig_audio_{idx:05}.bin"
        payload = AUDIO_FRAMES[: len(AUDIO_FRAMES) // 2] + b"AUDIO"
        write_binary(dataset_root / rel, payload)
        register(entries, dataset_root, rel, "audio")

    for idx in allocate_indices(AMBIGUOUS_TEXT):
        rel = Path("tricky/ambiguous") / f"ambig_text_{idx:05}.json"
        write_text_json(dataset_root / rel, variant="ambiguous")
        register(entries, dataset_root, rel, "text")


def make_empty(dataset_root: Path, entries: List[Dict[str, str]]) -> None:
    for idx in allocate_indices(EMPTY_UNKNOWN):
        choice = random.choice([".txt", ".png", ""])
        name = f"empty_{idx:05}{choice}"
        rel = Path("tricky/empty") / name
        write_empty(dataset_root / rel)
        register(entries, dataset_root, rel, "unknown")


def build_dataset(dataset_root: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    make_normal_samples(dataset_root, entries)
    make_missing_extension(dataset_root, entries)
    make_wrong_extension(dataset_root, entries)
    make_ambiguous(dataset_root, entries)
    make_empty(dataset_root, entries)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10k mixed dataset")
    parser.add_argument("--output-dir", default="test_dataset", help="输出数据集目录")
    parser.add_argument(
        "--ground-truth",
        default="test_dataset_ground_truth.jsonl",
        help="ground truth JSONL 输出路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    random.seed(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = (repo_root / args.output_dir).resolve()
    ground_truth_path = (repo_root / args.ground_truth).resolve()

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    entries = build_dataset(dataset_root)
    elapsed = time.perf_counter() - start

    entries.sort(key=lambda item: item["filename"])

    with ground_truth_path.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_files = len(entries)
    print(f"生成完成: {total_files} 个样本，耗时 {elapsed:.2f} 秒")
    print(f"数据目录: {dataset_root}")
    print(f"ground truth: {ground_truth_path}")


if __name__ == "__main__":
    main()
