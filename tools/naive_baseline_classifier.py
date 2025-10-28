"""Naive baseline classifier using extensions and lightweight heuristics.

This script walks an input directory, predicts a coarse modality label for
each file (audio/image/text/unknown), and saves predictions to a CSV file.
It is intentionally simple to serve as a baseline against the sorter
classifier.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".aac",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
}

TEXT_EXTS = {
    ".txt",
    ".json",
    ".csv",
    ".md",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".log",
    ".tsv",
}

JSON_TEXT_KEYS = {"text", "content", "title", "sentence", "article"}
JSON_AUDIO_KEYS = {"audio", "audio_url", "audio_path", "wav", "mp3"}
JSON_IMAGE_KEYS = {"image", "image_url", "img", "picture", "thumbnail"}

PRINTABLE_THRESHOLD = 0.85
HEADER_BYTES = 4096


def iter_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def is_mostly_printable(data: bytes) -> bool:
    if not data:
        return False
    printable = sum(chr(b).isprintable() or chr(b).isspace() for b in data)
    ratio = printable / len(data)
    return ratio >= PRINTABLE_THRESHOLD


def sniff_magic(header: bytes) -> Optional[str]:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio"
    if header.startswith(b"ID3"):
        return "audio"
    if header.startswith(b"fLaC"):
        return "audio"
    if b"metadata:image" in header or b"<svg" in header.lower():
        return "image"
    if header.lstrip().startswith(b"{") or header.lstrip().startswith(b"["):
        return "text"
    return None


def classify_json(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            content = json.load(f)
    except Exception:
        return "text"

    if isinstance(content, dict):
        keys = set(content.keys())
        if keys & JSON_IMAGE_KEYS:
            return "image"
        if keys & JSON_AUDIO_KEYS:
            return "audio"
        if keys & JSON_TEXT_KEYS:
            return "text"
        if "url" in content and isinstance(content["url"], str):
            url = content["url"].lower()
            if any(url.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
                return "image"
            return "text"
    elif isinstance(content, list) and content:
        votes = {"image": 0, "audio": 0, "text": 0}
        sample = content[: min(10, len(content))]
        for item in sample:
            if not isinstance(item, dict):
                continue
            keys = set(item.keys())
            if keys & JSON_IMAGE_KEYS:
                votes["image"] += 1
            if keys & JSON_AUDIO_KEYS:
                votes["audio"] += 1
            if keys & JSON_TEXT_KEYS:
                votes["text"] += 1
            if "url" in item and isinstance(item["url"], str):
                url = item["url"].lower()
                if any(url.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    votes["image"] += 1
        best = max(votes, key=votes.get)
        if votes[best] > 0:
            return best
    return "text"


def classify_file(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown"

    if size == 0:
        return "unknown"

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    if suffix in AUDIO_EXTS:
        return "audio"
    if suffix in TEXT_EXTS:
        if suffix == ".json":
            return classify_json(path)
        return "text"

    header = b""
    try:
        with path.open("rb") as f:
            header = f.read(HEADER_BYTES)
    except Exception:
        return "unknown"

    sniffed = sniff_magic(header)
    if sniffed:
        if sniffed == "text" and suffix not in TEXT_EXTS:
            try:
                text_sample = header.decode("utf-8", errors="ignore")
                if text_sample.lstrip().startswith("{") or text_sample.lstrip().startswith("["):
                    return classify_json(path)
            except Exception:
                pass
        return sniffed

    if is_mostly_printable(header):
        return "text"

    return "unknown"


def write_predictions(
    predictions: Dict[str, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_label"])
        for filename, label in sorted(predictions.items()):
            writer.writerow([filename.replace("\\", "/"), label])


def predict_directory(
    input_dir: Path,
    *,
    input_root: Optional[Path] = None,
) -> Dict[str, str]:
    if input_root is None:
        input_root = input_dir
    predictions: Dict[str, str] = {}
    for file_path in sorted(iter_files(input_dir)):
        if not file_path.is_file():
            continue
        if file_path.name == "ground_truth.jsonl":
            continue
        try:
            rel_path = file_path.relative_to(input_root)
        except ValueError:
            rel_path = file_path
        label = classify_file(file_path)
        predictions[str(rel_path).replace("\\", "/")] = label
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive baseline modality classifier")
    parser.add_argument("--input", required=True, help="输入目录")
    parser.add_argument("--predictions", required=True, help="预测CSV输出路径")
    parser.add_argument("--input-root", help="相对路径基准，默认为输入目录")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_root = Path(args.input_root).resolve() if args.input_root else input_dir

    predictions = predict_directory(input_dir, input_root=input_root)
    output_path = Path(args.predictions).resolve()
    write_predictions(predictions, output_path)

    total = len(predictions)
    label_counts: Dict[str, int] = {}
    for label in predictions.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"预测完成，文件总数: {total}")
    for label, count in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {count}")
    print(f"预测结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
