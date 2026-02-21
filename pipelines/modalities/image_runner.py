from __future__ import annotations

import os
import sys
import json
from pathlib import Path

def _ensure_package_root() -> None:
    module_name = "pipelines.modalities.common"
    if module_name in sys.modules:
        return
    pkg_root = Path(__file__).resolve().parents[1]
    project_root = pkg_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_package_root()

from pipelines.modalities.common import (  # type: ignore  # noqa: E402
    copy_existing_files,
    ensure_output_dir,
    load_input_paths,
    write_summary,
)

from image.method.pipeline_api import (  # type: ignore  # noqa: E402
    ImagePipelineConfig,
    ImagePipelineResult,
    load_pipeline_config,
    run_image_pipeline,
)

def main() -> None:
    paths, manifest_path = load_input_paths("PIPELINE_IMAGE_INPUT_LIST")
    total_candidates = int(os.environ.get("PIPELINE_IMAGE_TOTAL", "0") or 0)
    output_dir = ensure_output_dir("PIPELINE_IMAGE_OUTPUT_DIR")
    config_path = os.environ.get("PIPELINE_IMAGE_CONFIG_FILE")
    
    # 新增: 支持导出纯 JSON 列表与跳过物理复制
    export_json_path = os.environ.get("PIPELINE_IMAGE_EXPORT_JSON")
    skip_copy = os.environ.get("PIPELINE_IMAGE_SKIP_COPY", "0") == "1"

    try:
        config: ImagePipelineConfig = load_pipeline_config(config_path)
    except Exception as exc:
        print(f"[image runner] failed to load config {config_path!r}: {exc}")
        raise

    result: ImagePipelineResult = run_image_pipeline(paths, config)

    stats = dict(result.stats)
    stats.setdefault("total_candidates", total_candidates)
    stats.setdefault("selected", len(result.keepers))

    # 如果配置了导出 JSON 路径，则保存纯文件名列表 (适配 benchmark 对比)
    if export_json_path and result.keepers:
        try:
            with open(export_json_path, 'w', encoding='utf-8') as f:
                json.dump([str(p) for p in result.keepers], f, indent=2)
            print(f"[image runner] exported keep list to {export_json_path}")
        except Exception as exc:
            print(f"[image runner] failed to export keep list: {exc}")

    copy_stats = {"copied": 0, "skipped": 0, "missing": 0}
    if not skip_copy and output_dir and result.keepers:
        copy_stats = copy_existing_files(result.keepers, output_dir)
        stats.update(copy_stats)
    else:
        stats.setdefault("copied", 0)
        stats.setdefault("skipped", 0)
        if skip_copy:
            print("[image runner] copy skipped by configuration")
        elif not output_dir:
            print("[image runner] No output directory specified; skipping copy")

    stats["missing"] = len(result.missing) + copy_stats.get("missing", 0)

    # 计算输入集合的字节总量（用于上报吞吐），以 bytes 为单位
    try:
        processed_bytes = 0
        for p in paths:
            try:
                if p.exists():
                    processed_bytes += p.stat().st_size
            except Exception:
                # 忽略单个文件 stat 错误
                continue
        stats["processed_bytes"] = processed_bytes
    except Exception:
        stats.setdefault("processed_bytes", None)

    write_summary(output_dir, "image", stats, manifest_path, duplicates=result.duplicates)

    print(
        f"[image runner] processed {len(paths)} entries (total candidates={total_candidates}); "
        f"unique={stats.get('unique', 0)}, duplicates={stats.get('duplicates', 0)}, "
        f"backend={stats.get('embedding_backend')}"
    )


if __name__ == "__main__":
    main()
