"""Audio pipeline helpers for fingerprint-based deduplication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:  # Optional dependency for on-the-fly fingerprint extraction
    from audio.method import spectrum_fingerprint  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    spectrum_fingerprint = None  # type: ignore


@dataclass
class AudioEmbeddingConfig:
    """Configuration for building or loading audio fingerprints."""

    fingerprint_backend: str = "auto"  # auto | precomputed | compute
    precomputed_fingerprints: Optional[str] = None
    save_fingerprints_dir: Optional[str] = None
    threshold: float = 0.85


@dataclass
class AudioDedupConfig:
    method: str = "jaccard"  # jaccard | hash (reserved for future)
    threshold: float = 0.85
    max_candidates: int = 2048


@dataclass
class AudioPipelineConfig:
    embedding: AudioEmbeddingConfig = field(default_factory=AudioEmbeddingConfig)
    dedup: AudioDedupConfig = field(default_factory=AudioDedupConfig)


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    paths: List[Path]
    failed_paths: List[Path]
    backend: Optional[str]


@dataclass
class AudioPipelineResult:
    keepers: List[Path]
    duplicates: List[Dict[str, object]]
    missing: List[Path]
    stats: Dict[str, object]


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(default)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: Optional[str]) -> AudioPipelineConfig:
    """Load configuration from YAML/JSON path or fall back to defaults."""

    defaults: Dict[str, Dict[str, Any]] = {
        "embedding": {
            "fingerprint_backend": "auto",
            "precomputed_fingerprints": None,
            "save_fingerprints_dir": None,
            "threshold": 0.85,
        },
        "dedup": {
            "method": "jaccard",
            "threshold": 0.85,
            "max_candidates": 2048,
        },
    }

    if not config_path:
        config_dict = dict(defaults)
    else:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio pipeline config not found: {path}")
        content = path.read_text(encoding="utf-8")
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(content) or {}
            else:
                loaded = json.loads(content)
        except Exception as exc:  # pragma: no cover - parsing fallback
            raise ValueError(f"Failed to parse audio pipeline config {path}: {exc}") from exc
        config_dict = _merge_dict(defaults, loaded)

    return AudioPipelineConfig(
        embedding=AudioEmbeddingConfig(**config_dict["embedding"]),
        dedup=AudioDedupConfig(**config_dict["dedup"]),
    )


def run_audio_pipeline(paths: Sequence[Path], config: AudioPipelineConfig) -> AudioPipelineResult:
    """Compute fingerprints and run deduplication for audio paths."""

    unique_paths: List[Path] = []
    seen: set[str] = set()
    missing: List[Path] = []

    for path in paths:
        candidate = Path(path)
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            unique_paths.append(candidate)
        else:
            missing.append(candidate)

    if not unique_paths:
        stats = {
            "fingerprint_backend": None,
            "unique": 0,
            "duplicates": 0,
            "missing": len(missing),
            "processed": 0,
        }
        return AudioPipelineResult(keepers=[], duplicates=[], missing=missing, stats=stats)

    embedding_result: Optional[EmbeddingResult] = None

    if config.embedding.precomputed_fingerprints:
        try:
            embedding_result = _load_precomputed_fingerprints(unique_paths, config.embedding)
        except Exception as exc:
            print(f"[audio pipeline] failed to load precomputed fingerprints: {exc}")

    if embedding_result is None:
        try:
            embedding_result = _compute_fingerprints(unique_paths, config.embedding)
        except Exception as exc:
            print(f"[audio pipeline] failed to compute fingerprints for {len(unique_paths)} files: {exc}")
            missing.extend(unique_paths)
            stats = {
                "fingerprint_backend": None,
                "embedding_backend": None,
                "unique": 0,
                "duplicates": 0,
                "missing": len(missing),
                "processed": 0,
                "error": str(exc),
            }
            return AudioPipelineResult(keepers=[], duplicates=[], missing=missing, stats=stats)

    missing.extend(embedding_result.failed_paths)

    dedup_summary = _run_deduplication(
        embedding_result.paths,
        embedding_result.embeddings,
        config.dedup,
    )

    stats = {
        "fingerprint_backend": embedding_result.backend,
        "embedding_backend": embedding_result.backend,
        "unique": len(dedup_summary["keepers"]),
        "duplicates": dedup_summary["duplicate_count"],
        "missing": len(missing),
        "processed": len(embedding_result.paths),
        "skipped_due_to_limit": dedup_summary["skipped"],
    }

    return AudioPipelineResult(
        keepers=dedup_summary["keepers"],
        duplicates=dedup_summary["duplicates"],
        missing=missing,
        stats=stats,
    )


def _load_precomputed_fingerprints(
    paths: Sequence[Path],
    config: AudioEmbeddingConfig,
) -> Optional[EmbeddingResult]:
    if not config.precomputed_fingerprints:
        return None

    fingerprint_path = Path(config.precomputed_fingerprints).expanduser()
    if not fingerprint_path.exists():
        raise FileNotFoundError(f"Precomputed fingerprints not found: {fingerprint_path}")

    data = np.load(fingerprint_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        fingerprint_map = data.item()
    else:
        raise ValueError("Fingerprint file must contain a pickled dictionary")

    vectors: List[np.ndarray] = []
    matched_paths: List[Path] = []
    failed: List[Path] = []

    for path in paths:
        key_candidates = {path.name, str(path), str(path.resolve())}
        vector: Optional[np.ndarray] = None
        for candidate in key_candidates:
            vector = fingerprint_map.get(candidate)
            if vector is not None:
                break
        if vector is None:
            failed.append(path)
            continue
        vectors.append(np.asarray(vector, dtype=np.float32))
        matched_paths.append(path)

    if not vectors:
        raise RuntimeError("None of the provided paths matched precomputed fingerprints")

    stacked = np.stack(vectors, axis=0)
    return EmbeddingResult(
        embeddings=stacked,
        paths=matched_paths,
        failed_paths=failed,
        backend="precomputed",
    )


def _compute_fingerprints(
    paths: Sequence[Path],
    config: AudioEmbeddingConfig,
) -> EmbeddingResult:
    if spectrum_fingerprint is None:
        raise RuntimeError(
            "audio.method.spectrum_fingerprint is required for on-the-fly fingerprint computation"
        )

    vectors: List[np.ndarray] = []
    processed_paths: List[Path] = []
    failed: List[Path] = []

    for path in paths:
        try:
            fingerprint = spectrum_fingerprint.create_binary_spectrogram(str(path))
            vectors.append(np.asarray(fingerprint, dtype=np.float32))
            processed_paths.append(path)
        except Exception as exc:
            print(f"[audio pipeline] failed to compute fingerprint for {path}: {exc}")
            failed.append(path)

    if not vectors:
        raise RuntimeError("No audio fingerprints could be computed")

    stacked = np.stack(vectors, axis=0)
    return EmbeddingResult(
        embeddings=stacked,
        paths=processed_paths,
        failed_paths=failed,
        backend="computed",
    )


def _run_deduplication(
    paths: Sequence[Path],
    embeddings: np.ndarray,
    config: AudioDedupConfig,
) -> Dict[str, Any]:
    if embeddings.size == 0:
        return {
            "keepers": [],
            "duplicates": [],
            "duplicate_count": 0,
            "skipped": 0,
        }

    if embeddings.shape[0] > config.max_candidates:
        print(
            f"[audio pipeline] candidate count {embeddings.shape[0]} exceeds max_candidates={config.max_candidates}; "
            "skipping similarity dedup and treating all files as unique."
        )
        return {
            "keepers": list(paths),
            "duplicates": [],
            "duplicate_count": 0,
            "skipped": embeddings.shape[0],
        }

    method = (config.method or "jaccard").lower()
    if method == "jaccard":
        return _deduplicate_by_jaccard(paths, embeddings, config.threshold)

    raise ValueError(f"Unknown audio deduplication method: {config.method}")


def _deduplicate_by_jaccard(
    paths: Sequence[Path],
    embeddings: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2-dimensional for Jaccard deduplication")

    binary = (embeddings > 0).astype(np.uint8)
    counts = binary.sum(axis=1, keepdims=True)
    # Intersection counts
    intersections = binary @ binary.T
    unions = counts + counts.T - intersections
    unions = np.clip(unions, 1e-12, None)
    similarity = intersections / unions

    np.fill_diagonal(similarity, -np.inf)

    keepers: List[Path] = []
    duplicates: List[Dict[str, object]] = []
    duplicate_count = 0
    seen: set[int] = set()

    n = binary.shape[0]
    for i in range(n):
        if i in seen:
            continue
        keepers.append(paths[i])
        dup_entries: List[Dict[str, object]] = []
        for j in range(i + 1, n):
            if j in seen:
                continue
            sim = similarity[i, j]
            if sim >= threshold:
                dup_entries.append({"path": str(paths[j]), "similarity": float(sim)})
                seen.add(j)
        if dup_entries:
            duplicates.append(
                {
                    "original": str(paths[i]),
                    "duplicates": dup_entries,
                    "similarity_threshold": float(threshold),
                }
            )
            duplicate_count += len(dup_entries)

    return {
        "keepers": keepers,
        "duplicates": duplicates,
        "duplicate_count": duplicate_count,
        "skipped": 0,
    }
