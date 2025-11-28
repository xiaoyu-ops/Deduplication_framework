import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any


@dataclass
class StageArtifact:
    stage_name: str
    status: str
    elapsed_seconds: float
    output_paths: Dict[str, str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_artifact(artifact_dir: Path, artifact: StageArtifact) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, ensure_ascii=False, indent=2)
    return output_path
