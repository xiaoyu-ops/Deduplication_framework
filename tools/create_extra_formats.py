"""Generate assorted file formats under mix_dataset/extra_formats for sorter testing."""

import base64
import json
import math
import struct
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "mix_dataset" / "extra_formats"


def ensure_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def write_text_files(base: Path) -> None:
    (base / "notes_markdown.md").write_text(
        "# Mixed Data Note\n\n- 包含多种文件格式，用于测试 sorter\n- 覆盖文本/音频/图像等\n",
        encoding="utf-8",
    )
    (base / "sample_page.html").write_text(
        "<!DOCTYPE html>\n<html><head><title>Sorter Demo</title></head><body><p>HTML content for sorter test.</p></body></html>\n",
        encoding="utf-8",
    )
    (base / "vector_icon.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64'>\n  <rect width='64' height='64' fill='#222'/>\n  <circle cx='32' cy='32' r='20' fill='#4CAF50'/>\n</svg>\n",
        encoding="utf-8",
    )
    json_payload = {
        "id": "img-001",
        "title": "Example image metadata",
        "image_url": "https://example.com/demo.png",
        "audio_url": "https://example.com/demo.mp3",
        "text": "Caption text for sorter validation",
    }
    (base / "metadata.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base / "config_sample.yaml").write_text(
        "pipeline:\n  name: sorter-mixed\n  enabled: true\nfiles:\n  - name: gif\n    path: tiny.gif\n",
        encoding="utf-8",
    )
    (base / "ingest.log").write_text(
        "INFO start ingest\nWARN missing frame\nERROR retrying\n",
        encoding="utf-8",
    )


def write_gif(base: Path) -> None:
    gif_bytes = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    (base / "tiny.gif").write_bytes(gif_bytes)


def write_wav(base: Path) -> None:
    wav_path = base / "sine_tone.wav"
    framerate = 16000
    seconds = 1
    tone_hz = 440
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        frames = bytearray()
        for n in range(int(framerate * seconds)):
            value = int(32767 * math.sin(2 * math.pi * tone_hz * n / framerate))
            frames += struct.pack("<h", value)
        wf.writeframes(frames)


def write_pdf(base: Path) -> None:
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        "2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
        "3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>endobj\n"
        "4 0 obj<< /Length 44 >>stream\n"
        "BT /F1 24 Tf 72 120 Td (Hello Sorter) Tj ET\n"
        "endstream endobj\n"
        "xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000060 00000 n \n0000000115 00000 n \n0000000200 00000 n \n"
        "trailer<< /Root 1 0 R >>\nstartxref\n280\n%%EOF\n"
    )
    (base / "report.pdf").write_bytes(pdf_content.encode("utf-8"))


def write_binary_blob(base: Path) -> None:
    (base / "random_blob.bin").write_bytes(bytes(range(256)))


def main() -> None:
    base = ensure_dir()
    write_text_files(base)
    write_gif(base)
    write_wav(base)
    write_pdf(base)
    write_binary_blob(base)
    print(f"Created extra format samples under {base}")


if __name__ == "__main__":
    main()
