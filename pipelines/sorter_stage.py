from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass

from .config import PipelineConfig


@dataclass
class SorterStageResult:
    stats: Dict[str, Any]
    manifest_rows: List[Dict[str, Any]]
    per_modality: Dict[str, List[str]]
    unknown: Dict[str, List[str]]
    details: List[Dict[str, Any]]


def resolve_input_files(input_root: Path) -> List[str]:
    from sorter import read_files_from_directory

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    return read_files_from_directory(str(input_root))


def run_sorter(config: PipelineConfig, manifest_path: Path, logger) -> SorterStageResult:
    sorter_config = config.sorter
    input_root_value = config.general.get("input_root")
    if not input_root_value:
        raise ValueError("general.input_root must be configured for sorter stage")
    input_root = Path(input_root_value)
    move_files = sorter_config.get("move_files", False)
    prediction_path = sorter_config.get("prediction_path")

    files = resolve_input_files(input_root)
    logger.info("Sorter stage scanning %d files", len(files))
    from sorter import sorter as sorter_fn

    start = time.time()
    result = sorter_fn(
        files,
        eval_mode=False,
        prediction_path=prediction_path,
        input_root=str(input_root),
        move_base_dir=sorter_config.get("move_base_dir"),
        collect_only=not move_files,
    )
    elapsed = time.time() - start
    logger.info("Sorter stage completed in %.2f seconds", elapsed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    details = result.get("details", [])
    headers = [
        "source_path",
        "relative_path",
        "category",
        "status",
        "target_path",
        "reason",
    ]
    rows: List[Dict[str, Any]] = []
    for entry in details:
        row = {
            "source_path": entry.get("source_path"),
            "relative_path": entry.get("relative_path"),
            "category": entry.get("category"),
            "status": entry.get("status"),
            "target_path": entry.get("target_path"),
            "reason": entry.get("reason"),
        }
        rows.append(row)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    return SorterStageResult(
        stats={
            "success_count": result["success_count"],
            "fail_count": result["fail_count"],
            "elapsed_seconds": result["elapsed_seconds"],
            "prediction_file": result.get("prediction_file"),
            "move_files": move_files,
        },
        manifest_rows=rows,
        per_modality=result["categorized"],
        unknown=result["unknown"],
        details=details,
    )
