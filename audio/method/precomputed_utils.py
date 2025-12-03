"""Utilities to load and map precomputed fingerprint dictionaries to manifest paths.

Provides tolerant key-normalization and matching heuristics so that precomputed
fingerprint maps with keys like "audio_0001.wav" or absolute paths can be
matched against manifest entries like "00000001.wav" or relative names.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _extract_digits(s: str) -> Optional[str]:
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None


def load_fingerprint_map(fp: Path) -> Dict[str, object]:
    data = np.load(fp, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        return data.item()
    raise ValueError("Fingerprint file must contain a pickled dictionary")


def build_index(fingerprint_map: Dict[str, object]) -> Dict[str, List[str]]:
    """Build a mapping from normalized tokens -> list of original keys.

    The tokens include filename, stem, extracted digits, and variants with/without
    common prefixes like 'audio_'. This makes matching flexible.
    """
    index: Dict[str, List[str]] = {}

    for orig_key in fingerprint_map.keys():
        try:
            p = Path(orig_key)
        except Exception:
            p = None

        candidates = set()
        if p is not None:
            candidates.add(str(orig_key))
            candidates.add(p.name)
            candidates.add(p.stem)
        else:
            candidates.add(str(orig_key))

        # Add digits token if present
        stem = p.stem if p is not None else str(orig_key)
        digits = _extract_digits(stem)
        if digits:
            candidates.add(digits)
            candidates.add(str(int(digits)))

        # Common prefix removal/additions
        if stem.startswith("audio_"):
            candidates.add(stem.replace("audio_", ""))
        else:
            candidates.add("audio_" + stem)

        for token in candidates:
            if not token:
                continue
            index.setdefault(token, []).append(orig_key)

    return index


def match_paths_to_map(paths: Iterable[Path], fingerprint_map: Dict[str, object]) -> Tuple[List[object], List[Path], List[Path], int]:
    """Match manifest paths to fingerprint_map entries.

    Returns (vectors, matched_paths, failed_paths, matched_count)
    """
    index = build_index(fingerprint_map)

    vectors: List[object] = []
    matched_paths: List[Path] = []
    failed: List[Path] = []
    matched_count = 0

    for path in paths:
        tokens = set()
        tokens.add(path.name)
        tokens.add(path.stem)
        tokens.add(str(path))
        tokens.add(str(path.resolve()))
        digits = _extract_digits(path.stem)
        if digits:
            tokens.add(digits)
            tokens.add(str(int(digits)))

        found_key = None
        # Prefer exact absolute/relative matches first
        for t in (str(path), str(path.resolve()), path.name):
            if t in fingerprint_map:
                found_key = t
                break

        if found_key is None:
            # Fallback: try index tokens
            for tok in tokens:
                candidates = index.get(tok)
                if candidates:
                    # Prefer candidate whose filename exactly matches path.name
                    pick = None
                    for c in candidates:
                        if Path(c).name == path.name:
                            pick = c
                            break
                    if pick is None:
                        pick = candidates[0]
                    found_key = pick
                    break

        if found_key is None:
            failed.append(path)
            continue

        vectors.append(fingerprint_map[found_key])
        matched_paths.append(path)
        matched_count += 1

    return vectors, matched_paths, failed, matched_count

