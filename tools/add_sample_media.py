"""Download or synthesize sample PNG/WAV files and update ground truth."""
from __future__ import annotations

import base64
import json
import math
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MIX_DATASET_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = MIX_DATASET_DIR / "ground_truth.jsonl"

TARGET_IMAGE_COUNT = 50
TARGET_AUDIO_COUNT = 50

SEED_MEDIA: List[Dict[str, str]] = [
    {
        "filename": "sample_hopper.png",
        "label": "image",
        "url": "https://raw.githubusercontent.com/python-pillow/Pillow/main/Tests/images/hopper.png",
    },
    {
        "filename": "sample_transparency.png",
        "label": "image",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    },
    {
        "filename": "sample_example.png",
        "label": "image",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/79/Example.png",
    },
    {
        "filename": "sample_tone.wav",
        "label": "audio",
        "url": "https://raw.githubusercontent.com/jiaaro/pydub/master/test/data/test1.wav",
    },
    {
        "filename": "sample_speech.wav",
        "label": "audio",
        "url": "https://raw.githubusercontent.com/microsoft/EdgeSpeaker/master/data/samples/hello.wav",
    },
]

IMAGE_COLOR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("0c7bdc", "ffffff"),
    ("ff7043", "ffffff"),
    ("34a853", "ffffff"),
    ("ea4335", "ffffff"),
    ("fbbc05", "000000"),
    ("9b51e0", "ffffff"),
    ("00acc1", "ffffff"),
    ("ef5350", "ffffff"),
)


def build_media_items(
    image_count: int = TARGET_IMAGE_COUNT,
    audio_count: int = TARGET_AUDIO_COUNT,
) -> List[Dict[str, str]]:
    """Create a combined media item list with predictable filenames."""

    items: List[Dict[str, str]] = [dict(seed) for seed in SEED_MEDIA]
    existing_names = {seed["filename"] for seed in items}

    for idx in range(1, image_count + 1):
        filename = f"sample_image_{idx:02}.png"
        if filename in existing_names:
            continue

        bg, fg = IMAGE_COLOR_PAIRS[(idx - 1) % len(IMAGE_COLOR_PAIRS)]
        text = f"img{idx:02}"
        url = f"https://dummyimage.com/256x256/{bg}/{fg}.png&text={text}"
        items.append({
            "filename": filename,
            "label": "image",
            "url": url,
        })
        existing_names.add(filename)

    for idx in range(1, audio_count + 1):
        filename = f"sample_audio_{idx:02}.wav"
        if filename in existing_names:
            continue

        frequency = 220.0 + (idx - 1) * 15.0
        duration = 1.2 + (idx % 4) * 0.2  # introduce small variation
        items.append({
            "filename": filename,
            "label": "audio",
            "frequency": frequency,
            "duration": duration,
            "url": None,
        })
        existing_names.add(filename)

    return items


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_media_item(item: Dict[str, str], timeout: float = 30.0) -> Optional[Path]:
    target_path = MIX_DATASET_DIR / item["filename"]
    if target_path.exists():
        print(f"[skip] {target_path.name} already exists")
        return target_path

    url = item.get("url")
    if not url:
        return None

    print(f"[download] {url} -> {target_path.name}")
    ensure_parent(target_path)

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=timeout) as response, target_path.open("wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        return target_path
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        print(f"[warn] download failed for {target_path.name}: {exc}")
        if target_path.exists():
            target_path.unlink()
        return None


def write_placeholder_image(path: Path) -> Path:
    # 2x2 PNG with simple pattern (base64 encoded)
    placeholder_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAB7GkOtAAAAFElEQVR4nGP4////fwYGBgYGABwABQgKDc3nAAAAAElFTkSuQmCC"
    )
    ensure_parent(path)
    with path.open("wb") as f:
        f.write(placeholder_png)
    return path


def write_placeholder_audio(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> Path:
    ensure_parent(path)
    total_samples = int(duration * sample_rate)
    amplitude = 0.4  # scale to avoid clipping

    with wave.open(str(path), "w") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)  # 16-bit PCM
        wave_file.setframerate(sample_rate)

        frames = bytearray()
        for n in range(total_samples):
            sample = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * n / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wave_file.writeframes(frames)

    return path


def ensure_media_item(item: Dict[str, str]) -> Tuple[Optional[Path], str]:
    target_path = MIX_DATASET_DIR / item["filename"]
    if target_path.exists():
        return target_path, "existing"

    path = download_media_item(item)
    if path is not None:
        return path, "downloaded"

    suffix = target_path.suffix.lower()
    try:
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            write_placeholder_image(target_path)
            return target_path, "placeholder-generated"
        if suffix in {".wav"}:
            write_placeholder_audio(
                target_path,
                duration=float(item.get("duration", 1.0)),
                frequency=float(item.get("frequency", 440.0)),
            )
            return target_path, "placeholder-generated"
    except Exception as exc:
        print(f"[error] failed to generate placeholder for {target_path.name}: {exc}")
        if target_path.exists():
            target_path.unlink()
        return None, "failed"

    print(f"[error] unsupported fallback for {target_path.name}")
    return None, "failed"


def load_ground_truth() -> List[Dict[str, str]]:
    if not GROUND_TRUTH_PATH.exists():
        return []

    entries: List[Dict[str, str]] = []
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in ground truth: {line}") from exc
    return entries


def save_ground_truth(entries: List[Dict[str, str]]) -> None:
    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_ground_truth(entries: List[Dict[str, str]], item: Dict[str, str]) -> List[Dict[str, str]]:
    updated = False
    for entry in entries:
        if entry.get("filename") == item["filename"]:
            if entry.get("label") != item["label"]:
                print(f"[update] {item['filename']} label {entry.get('label')} -> {item['label']}")
                entry["label"] = item["label"]
            updated = True
            break

    if not updated:
        print(f"[append] {item['filename']} -> {item['label']}")
        entries.append({"filename": item["filename"], "label": item["label"]})

    return entries


def main() -> None:
    print("Mix dataset directory:", MIX_DATASET_DIR)
    media_plan = build_media_items()
    entries = load_ground_truth()
    results: List[Tuple[str, str, str]] = []

    for media_item in media_plan:
        path, status = ensure_media_item(media_item)
        if path is None:
            print(f"[skip] could not prepare {media_item['filename']}")
            results.append((media_item["filename"], media_item["label"], status))
            continue

        entries = update_ground_truth(entries, media_item)
        results.append((media_item["filename"], media_item["label"], status))

    # Keep entries sorted for readability
    entries.sort(key=lambda e: e.get("filename", ""))
    save_ground_truth(entries)
    print(f"Ground truth updated: {GROUND_TRUTH_PATH}")

    print("\nSummary:")
    for filename, label, status in results:
        print(f" - {filename} [{label}] -> {status}")

    images_prepared = sum(1 for _, label, status in results if label == "image" and status != "failed")
    audio_prepared = sum(1 for _, label, status in results if label == "audio" and status != "failed")
    print(f"Prepared images: {images_prepared} (target {TARGET_IMAGE_COUNT} + seeds)")
    print(f"Prepared audio: {audio_prepared} (target {TARGET_AUDIO_COUNT} + seeds)")


if __name__ == "__main__":
    main()
