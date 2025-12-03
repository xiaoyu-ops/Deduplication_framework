from __future__ import annotations

import json
import sys
from pathlib import Path


def main(manifest_path: Path, out_dir: Path):
    if not manifest_path.exists():
        print(f"[validate_manifest] manifest not found: {manifest_path}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [l.strip() for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(lines)
    missing = []
    present_count = 0
    for i, line in enumerate(lines):
        p = Path(line)
        if p.exists():
            present_count += 1
        else:
            missing.append(line)
            if len(missing) >= 20:
                # still continue counting total, but stop collecting examples
                pass

    summary = {
        "manifest": str(manifest_path),
        "total_lines": total,
        "present_count": present_count,
        "missing_examples_count": len(missing),
        "missing_examples": missing[:20],
    }

    out_path = out_dir / "manifest_check.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validate_manifest] wrote summary to {out_path}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_text_manifest.py <manifest_path> [out_dir]")
        sys.exit(2)
    manifest = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/local_test/stage2_text")
    code = main(manifest, out)
    sys.exit(code)
