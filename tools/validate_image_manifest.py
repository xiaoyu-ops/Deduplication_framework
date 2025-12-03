"""
Validate image manifest entries by attempting to open each path with Pillow.
Usage:
  python tools/validate_image_manifest.py /path/to/manifest.txt
Manifest format: one image path per line (absolute or relative).
"""
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    print("Pillow not installed in this Python environment. Install it or run inside the 'image' env.", file=sys.stderr)
    sys.exit(2)


def check_manifest(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 2

    bad = []
    total = 0
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            p = line.strip()
            if not p:
                continue
            total += 1
            path = Path(p)
            if not path.exists():
                bad.append((p, "missing"))
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:
                bad.append((p, str(exc)))

    print(f"Checked {total} entries; {len(bad)} failures")
    for p, reason in bad:
        print(f"BAD: {p} -> {reason}")

    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_image_manifest.py manifest.txt")
        sys.exit(2)
    manifest = Path(sys.argv[1])
    rc = check_manifest(manifest)
    sys.exit(rc)
