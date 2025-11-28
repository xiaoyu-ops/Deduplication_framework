from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_MODALITIES = ("image", "audio", "text")


@dataclass
class ManifestData:
    rows: List[Dict[str, str]]
    per_modality: Dict[str, List[str]]
    unknown: Dict[str, List[str]]

    @property
    def total(self) -> int:
        return len(self.rows)

    def count_for(self, modality: str) -> int:
        return len(self.per_modality.get(modality, ()))


class ManifestFormatError(ValueError):
    """Raised when the manifest CSV does not meet the expected schema."""


def load_manifest_data(
    manifest_path: Path,
    *,
    expected_modalities: Sequence[str] = DEFAULT_MODALITIES,
    allow_empty: bool = False,
) -> ManifestData:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    expected_modalities_set = set(expected_modalities)
    per_modality: Dict[str, List[str]] = {m: [] for m in expected_modalities_set}
    unknown: Dict[str, List[str]] = {}
    rows: List[Dict[str, str]] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestFormatError("Manifest is missing header row")

        normalized_fields = {
            (name or "").strip().lower(): name for name in reader.fieldnames
        }

        category_field = normalized_fields.get("category")
        if category_field is None:
            raise ManifestFormatError(
                "Manifest must contain a 'category' column, "
                f"found: {reader.fieldnames}"
            )

        path_field = normalized_fields.get("path")
        source_field = normalized_fields.get("source_path")
        relative_field = normalized_fields.get("relative_path")

        if path_field is None and source_field is None and relative_field is None:
            raise ManifestFormatError(
                "Manifest must contain one of 'path', 'source_path', or 'relative_path' columns, "
                f"found: {reader.fieldnames}"
            )

        for row in reader:
            raw_path = None
            if path_field:
                raw_path = row.get(path_field)
            if (not raw_path) and source_field:
                raw_path = row.get(source_field)
            if (not raw_path) and relative_field:
                raw_path = row.get(relative_field)

            category = row.get(category_field)
            if not raw_path or not category:
                raise ManifestFormatError("Manifest row missing required path or category values")

            normalized_category = category.strip()
            normalized_path = raw_path.strip()
            normalized_row: Dict[str, str] = {
                "path": normalized_path,
                "category": normalized_category,
            }
            if source_field:
                source_value = row.get(source_field)
                if source_value:
                    normalized_row["source_path"] = source_value.strip()
            if relative_field:
                relative_value = row.get(relative_field)
                if relative_value:
                    normalized_row["relative_path"] = relative_value.strip()
            rows.append(normalized_row)
            target = per_modality.get(normalized_category)
            if target is not None:
                target.append(normalized_path)
            else:
                unknown.setdefault(normalized_category, []).append(normalized_path)

    if not rows and not allow_empty:
        raise ManifestFormatError("Manifest is empty; rerun sorter stage or provide data")

    return ManifestData(rows=rows, per_modality=per_modality, unknown=unknown)
