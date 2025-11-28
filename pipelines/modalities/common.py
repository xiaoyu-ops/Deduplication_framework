from __future__ import annotations

import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional


def _read_lines(path: Path) -> List[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        print(f"[modalities] manifest file not found: {path}")
        return []
    except Exception as exc:  # pragma: no cover
        print(f"[modalities] failed to read manifest {path}: {exc}")
        return []


def load_input_paths(env_var: str) -> Tuple[List[Path], Path | None]:
    manifest_env = os.environ.get(env_var)
    if not manifest_env:
        print(f"[modalities] env {env_var} missing; fallback to empty list")
        return [], None
    manifest_path = Path(manifest_env)
    lines = _read_lines(manifest_path)
    paths = [Path(line).resolve() for line in lines]
    resolved: List[Path] = []
    for p in paths:
        if p.exists():
            resolved.append(p)
            continue
        fallback = Path(os.path.normpath(str(p)))
        resolved.append(fallback)
    return resolved, manifest_path


def ensure_output_dir(env_var: str) -> Path | None:
    output_env = os.environ.get(env_var)
    if not output_env:
        print(f"[modalities] env {env_var} missing; no output directory specified")
        return None
    output_path = Path(output_env)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def copy_existing_files(paths: Iterable[Path], output_dir: Path) -> Dict[str, int]:
    copied = 0
    skipped = 0
    missing = 0
    for src in paths:
        if not src.exists():
            missing += 1
            print(f"[modalities] missing file: {src}")
            continue
        try:
            dest = output_dir / src.name
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(src, dest)
            copied += 1
        except Exception as exc:  # pragma: no cover
            skipped += 1
            print(f"[modalities] failed to copy {src} -> {dest}: {exc}")
    return {"copied": copied, "skipped": skipped, "missing": missing}


def _default_summary_payload(modality: str, stats: Dict[str, int], manifest_path: Path | None) -> Dict[str, object]:
    return {
        "modality": modality,
        "stats": stats,
        "manifest": str(manifest_path) if manifest_path else None,
    }


def compute_file_hash(path: Path, chunk_size: int = 65536, algorithm: str = "sha1") -> Optional[str]:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from None

    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
    except (OSError, IOError):
        return None
    return hasher.hexdigest()


def deduplicate_by_hash(
    paths: Iterable[Path],
    *,
    output_dir: Optional[Path],
    hash_algorithm: str = "sha1",
) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    total_paths = list(paths)
    stats = {
        "total_candidates": 0,
        "selected": len(total_paths),
        "missing": 0,
        "unique": 0,
        "duplicates": 0,
        "copied": 0,
    }

    hash_to_entry: Dict[str, Dict[str, object]] = {}
    duplicate_groups: List[Dict[str, object]] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in total_paths:
        stats["total_candidates"] += 1
        if not path.exists():
            stats["missing"] += 1
            continue

        digest = compute_file_hash(path, algorithm=hash_algorithm)
        if not digest:
            stats["missing"] += 1
            continue

        entry = hash_to_entry.get(digest)
        if entry is None:
            # First time we see this content
            copied_target = None
            if output_dir:
                destination = output_dir / path.name
                counter = 1
                while destination.exists():
                    destination = output_dir / f"{path.stem}_{counter}{path.suffix}"
                    counter += 1
                try:
                    shutil.copy2(path, destination)
                    copied_target = destination
                    stats["copied"] += 1
                except Exception as exc:  # pragma: no cover
                    print(f"[modalities] failed to copy unique file {path}: {exc}")
            hash_to_entry[digest] = {
                "original": str(path),
                "output": str(copied_target) if copied_target else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "duplicates": [],
            }
            stats["unique"] += 1
        else:
            entry["duplicates"].append(str(path))
            stats["duplicates"] += 1

    for digest, entry in hash_to_entry.items():
        if entry["duplicates"]:
            duplicate_groups.append({
                "hash": digest,
                "original": entry["original"],
                "duplicates": entry["duplicates"],
                "output": entry.get("output"),
            })

    return stats, duplicate_groups


def write_summary(
    output_dir: Path | None,
    modality: str,
    stats: Dict[str, int],
    manifest_path: Path | None,
    *,
    duplicates: Optional[List[Dict[str, object]]] = None,
) -> None:
    if output_dir is None:
        return
    summary = _default_summary_payload(modality, stats, manifest_path)
    if duplicates:
        duplicates_path = output_dir / f"{modality}_duplicates.json"
        try:
            duplicates_path.write_text(
                json.dumps(duplicates, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary["duplicates_file"] = str(duplicates_path)
        except Exception as exc:  # pragma: no cover
            print(f"[modalities] failed to write duplicates list {duplicates_path}: {exc}")
    summary_path = output_dir / f"{modality}_runner_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        print(f"[modalities] failed to write summary {summary_path}: {exc}")
