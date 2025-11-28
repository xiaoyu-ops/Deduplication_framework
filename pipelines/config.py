import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExecutorConfig:
    type: str = "local"
    conda_executable: Optional[str] = None
    envs: Dict[str, str] = None


@dataclass
class PipelineConfig:
    general: Dict[str, Any]
    executor: ExecutorConfig
    sorter: Dict[str, Any]
    image: Dict[str, Any]
    audio: Dict[str, Any]
    text: Dict[str, Any]
    report: Dict[str, Any]

    def modality_enabled(self, name: str) -> bool:
        section = getattr(self, name, {})
        return section.get("enabled", True)

    def env_for(self, stage: str) -> Optional[str]:
        return (self.executor.envs or {}).get(stage)

    def general_option(self, key: str, default: Any = None) -> Any:
        return self.general.get(key, default)


class ConfigLoader:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> PipelineConfig:
        with self.path.open("r", encoding="utf-8") as f:
            if self.path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        executor_data = data.get("executor", {})
        executor_cfg = ExecutorConfig(
            type=executor_data.get("type", "local"),
            conda_executable=executor_data.get("conda_executable"),
            envs=executor_data.get("envs", {}),
        )

        cfg = PipelineConfig(
            general=data.get("general", {}),
            executor=executor_cfg,
            sorter=data.get("sorter", {}),
            image=data.get("image", {}),
            audio=data.get("audio", {}),
            text=data.get("text", {}),
            report=data.get("report", {}),
        )

        base_dir = self.path.parent

        def _abspath(relative_path: Optional[str]) -> Optional[str]:
            if not relative_path:
                return relative_path
            p = Path(relative_path)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            return str(p)

        # Normalize general paths
        for key in ["input_root", "output_root", "temp_root"]:
            if key in cfg.general:
                cfg.general[key] = _abspath(cfg.general[key])

        # Normalize sorter paths
        if "prediction_path" in cfg.sorter:
            cfg.sorter["prediction_path"] = _abspath(cfg.sorter["prediction_path"])
        if "manifest_name" in cfg.sorter:
            cfg.sorter["manifest_name"] = cfg.sorter["manifest_name"]

        # Normalize modality config paths
        for section_name in ["image", "audio", "text", "report"]:
            section = getattr(cfg, section_name)
            for key, value in list(section.items()):
                if (
                    "path" in key
                    or key.endswith("_file")
                    or key.endswith("_dir")
                    or key in {"entrypoint", "workdir"}
                ):
                    if section_name == "report" and key in {"summary_file", "markdown_file"}:
                        continue
                    section[key] = _abspath(value)

        retry_defaults = data.get("general", {}).get("retry", {})
        cfg.general.setdefault("retry", retry_defaults)


        return cfg
