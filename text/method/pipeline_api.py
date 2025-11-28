"""Text pipeline helpers for n-gram based deduplication."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml


@dataclass
class TextEmbeddingConfig:
    """Configuration for building text n-gram signatures."""

    ngram_size: int = 3
    lowercase: bool = True
    strip_non_alnum: bool = True
    collapse_whitespace: bool = True
    encoding: str = "utf-8"
    errors: str = "ignore"


@dataclass
class TextDedupConfig:
    method: str = "jaccard"
    threshold: float = 0.8
    max_candidates: int = 5000


@dataclass
class TextPipelineConfig:
    embedding: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    dedup: TextDedupConfig = field(default_factory=TextDedupConfig)


@dataclass
class TextEmbeddingResult:
    features: List[Set[str]]
    texts: List[str]
    paths: List[Path]
    failed_paths: List[Path]
    backend: str


@dataclass
class TextPipelineResult:
    keepers: List[Path]
    duplicates: List[Dict[str, object]]
    missing: List[Path]
    stats: Dict[str, object]


_NON_ALNUM_RE = re.compile(r"[^\w\s\u4e00-\u9fff]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(default)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: Optional[str]) -> TextPipelineConfig:
    """Load configuration from YAML/JSON path or fall back to defaults."""

    defaults: Dict[str, Dict[str, Any]] = {
        "embedding": {
            "ngram_size": 3,
            "lowercase": True,
            "strip_non_alnum": True,
            "collapse_whitespace": True,
            "encoding": "utf-8",
            "errors": "ignore",
        },
        "dedup": {
            "method": "jaccard",
            "threshold": 0.8,
            "max_candidates": 5000,
        },
    }

    if not config_path:
        config_dict = dict(defaults)
    else:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Text pipeline config not found: {path}")
        content = path.read_text(encoding="utf-8")
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(content) or {}
            else:
                loaded = json.loads(content)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Failed to parse text pipeline config {path}: {exc}") from exc
        config_dict = _merge_dict(defaults, loaded)

    return TextPipelineConfig(
        embedding=TextEmbeddingConfig(**config_dict["embedding"]),
        dedup=TextDedupConfig(**config_dict["dedup"]),
    )


def run_text_pipeline(paths: Sequence[Path], config: TextPipelineConfig) -> TextPipelineResult:
    unique_paths: List[Path] = []
    missing: List[Path] = []
    seen: Set[str] = set()

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
            "embedding_backend": None,
            "unique": 0,
            "duplicates": 0,
            "missing": len(missing),
            "processed": 0,
        }
        return TextPipelineResult(keepers=[], duplicates=[], missing=missing, stats=stats)

    try:
        embedding_result = _compute_text_signatures(unique_paths, config.embedding)
    except Exception as exc:
        print(f"[text pipeline] failed to compute signatures for {len(unique_paths)} files: {exc}")
        missing.extend(unique_paths)
        stats = {
            "embedding_backend": None,
            "unique": 0,
            "duplicates": 0,
            "missing": len(missing),
            "processed": 0,
            "error": str(exc),
        }
        return TextPipelineResult(keepers=[], duplicates=[], missing=missing, stats=stats)

    missing.extend(embedding_result.failed_paths)

    dedup_summary = _run_deduplication(
        embedding_result.paths,
        embedding_result.features,
        config.dedup,
    )

    stats = {
        "embedding_backend": embedding_result.backend,
        "unique": len(dedup_summary["keepers"]),
        "duplicates": dedup_summary["duplicate_count"],
        "missing": len(missing),
        "processed": len(embedding_result.paths),
        "skipped_due_to_limit": dedup_summary["skipped"],
    }

    return TextPipelineResult(
        keepers=dedup_summary["keepers"],
        duplicates=dedup_summary["duplicates"],
        missing=missing,
        stats=stats,
    )


def _normalize_text(content: str, config: TextEmbeddingConfig) -> str:
    text = content
    if config.lowercase:
        text = text.lower()
    if config.strip_non_alnum:
        text = _NON_ALNUM_RE.sub(" ", text)
    if config.collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _compute_ngrams(text: str, n: int) -> Set[str]:
    if not text:
        return set()
    length = len(text)
    if length < n:
        char_ngrams = {text}
    else:
        char_ngrams = {text[i : i + n] for i in range(length - n + 1)}
    words = text.split()
    word_ngrams: Set[str] = set()
    if len(words) >= n:
        for idx in range(len(words) - n + 1):
            word_ngrams.add(" ".join(words[idx : idx + n]))
    return char_ngrams | word_ngrams


def _compute_text_signatures(
    paths: Sequence[Path],
    config: TextEmbeddingConfig,
) -> TextEmbeddingResult:
    features: List[Set[str]] = []
    texts: List[str] = []
    processed_paths: List[Path] = []
    failed: List[Path] = []

    for path in paths:
        try:
            raw_text = path.read_text(encoding=config.encoding, errors=config.errors)
        except Exception as exc:
            print(f"[text pipeline] failed to read {path}: {exc}")
            failed.append(path)
            continue
        normalized = _normalize_text(raw_text, config)
        signature = _compute_ngrams(normalized, max(1, config.ngram_size))
        features.append(signature)
        texts.append(normalized)
        processed_paths.append(path)

    if not features:
        raise RuntimeError("No text signatures could be computed")

    return TextEmbeddingResult(
        features=features,
        texts=texts,
        paths=processed_paths,
        failed_paths=failed,
        backend="computed",
    )


def _run_deduplication(
    paths: Sequence[Path],
    features: Sequence[Set[str]],
    config: TextDedupConfig,
) -> Dict[str, Any]:
    if not features:
        return {
            "keepers": [],
            "duplicates": [],
            "duplicate_count": 0,
            "skipped": 0,
        }

    if len(features) > config.max_candidates:
        print(
            f"[text pipeline] candidate count {len(features)} exceeds max_candidates={config.max_candidates}; "
            "skipping similarity dedup and treating all files as unique."
        )
        return {
            "keepers": list(paths),
            "duplicates": [],
            "duplicate_count": 0,
            "skipped": len(features),
        }

    method = (config.method or "jaccard").lower()
    if method == "jaccard":
        return _deduplicate_by_jaccard(paths, features, config.threshold)

    raise ValueError(f"Unknown text deduplication method: {config.method}")


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    intersection_size = len(a & b)
    return intersection_size / len(union)


def _deduplicate_by_jaccard(
    paths: Sequence[Path],
    features: Sequence[Set[str]],
    threshold: float,
) -> Dict[str, Any]:
    keepers: List[Path] = []
    duplicates: List[Dict[str, object]] = []
    duplicate_count = 0
    seen: Set[int] = set()

    for idx, path in enumerate(paths):
        if idx in seen:
            continue
        keepers.append(path)
        dup_entries: List[Dict[str, object]] = []
        for other_idx in range(idx + 1, len(paths)):
            if other_idx in seen:
                continue
            sim = _jaccard_similarity(features[idx], features[other_idx])
            if sim >= threshold:
                dup_entries.append({"path": str(paths[other_idx]), "similarity": float(sim)})
                seen.add(other_idx)
        if dup_entries:
            duplicates.append(
                {
                    "original": str(path),
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
