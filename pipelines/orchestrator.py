"""Pipeline orchestrator base implementation."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .artifacts import StageArtifact, save_artifact
from .config import ConfigLoader, PipelineConfig
from .executor import BaseExecutor, ExecutorError, create_executor
from .logger import setup_logger
from .manifest_utils import ManifestData, ManifestFormatError, load_manifest_data
from .sorter_stage import run_sorter
from .stage_utils import (
    StageLockError,
    acquire_stage_lock,
    compute_dict_hash,
    release_stage_lock,
    write_flag,
)


class PipelineOrchestrator:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = ConfigLoader(config_path).load()
        self.executor: BaseExecutor = create_executor(
            self.config.executor.type,
            conda_executable=self.config.executor.conda_executable,
        )
        now = dt.datetime.now(dt.timezone.utc)
        self.run_id = now.strftime("%Y%m%d-%H%M%S")
        general = self.config.general
        self.output_root = Path(general.get("output_root", "outputs"))
        self.artifacts_root = Path(general.get("temp_root", "artifacts")) / self.run_id
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.artifacts_root / "logs" / "pipeline.log"
        self.logger = setup_logger(self.log_path)
        self.logger.info("Pipeline orchestrator initialized")

        self.summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "config": str(self.config_path),
            "stages": [],
        }
        self.summary["modality_plan"] = {}
        self.summary["modality_results"] = {}
        self.summary["timestamps"] = {"started_at": now.isoformat(timespec="seconds")}
        self.run_manifest_path = self.artifacts_root / "run_manifest.json"
        self.summary["run_manifest_path"] = str(self.run_manifest_path)
        self._write_run_manifest()
        self.sorter_manifest: ManifestData | None = None

    def _stage_flags(self, stage_name: str) -> Dict[str, Path]:
        stage_dir = self.stage_artifact_dir(stage_name)
        return {
            "success": stage_dir / "_SUCCESS",
            "failure": stage_dir / "_FAILURE",
            "lock": stage_dir / "_LOCK",
        }

    def _stage_already_completed(self, stage_name: str, config_hash: str) -> bool:
        flags = self._stage_flags(stage_name)
        summary_path = self.stage_artifact_dir(stage_name) / "summary.json"
        if not flags["success"].exists() or not summary_path.exists():
            return False
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary_data = json.load(f)
            return summary_data.get("metadata", {}).get("config_hash") == config_hash
        except Exception:
            return False

    def _stage_should_resume(self, stage_name: str, config_hash: str) -> bool:
        general = self.config.general
        resume = general.get("resume", False)
        if not resume:
            return False
        return self._stage_already_completed(stage_name, config_hash)

    def stage_artifact_dir(self, stage_name: str) -> Path:
        return self.artifacts_root / stage_name

    def save_stage_result(
        self,
        stage_name: str,
        artifact: StageArtifact,
        *,
        success: bool,
    ) -> None:
        artifact_dir = self.stage_artifact_dir(stage_name)
        save_artifact(artifact_dir, artifact)
        flags = self._stage_flags(stage_name)
        if success:
            write_flag(artifact_dir, "_SUCCESS")
            failure_flag = artifact_dir / "_FAILURE"
            if failure_flag.exists():
                failure_flag.unlink()
        else:
            write_flag(artifact_dir, "_FAILURE")
            success_flag = artifact_dir / "_SUCCESS"
            if success_flag.exists():
                success_flag.unlink()
        self.summary["stages"].append(artifact.to_dict())
        self._write_run_manifest()

    def _write_run_manifest(self) -> None:
        try:
            self.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.run_manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(self.summary, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            if hasattr(self, "logger"):
                self.logger.error("Failed to write run manifest %s: %s", self.run_manifest_path, exc)

    def _load_runner_summary(self, modality: str, output_dir: Optional[str]) -> Optional[Dict[str, Any]]:
        if not output_dir:
            return None
        summary_path = Path(output_dir) / f"{modality}_runner_summary.json"
        if not summary_path.exists():
            return None
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                "Failed to load %s runner summary from %s: %s", modality, summary_path, exc
            )
            return None

    def _build_aggregated_results(self) -> Dict[str, Any]:
        def safe_div(numerator: Any, denominator: Any) -> Optional[float]:
            try:
                if numerator is None or denominator is None:
                    return None
                if not isinstance(denominator, (int, float)):
                    return None
                if denominator <= 0:
                    return None
                return float(numerator) / float(denominator)
            except (TypeError, ZeroDivisionError, ValueError):
                return None

        BYTES_IN_MB = 1024 * 1024

        aggregated: Dict[str, Any] = {}

        sorter_info = self.summary.get("sorter_manifest") or {}
        sorter_stage = next(
            (stage for stage in self.summary.get("stages", []) if stage.get("stage_name") == "stage1_sorter"),
            {},
        )
        sorter_metadata = sorter_stage.get("metadata") or {}
        sorter_runtime = self.summary.get("sorter") or {}
        sorter_elapsed = sorter_stage.get("elapsed_seconds")
        if sorter_elapsed is None:
            sorter_elapsed = sorter_metadata.get("elapsed_seconds")
        sorter_total = sorter_info.get("total_rows")
        if sorter_total is None:
            sorter_total = sorter_metadata.get("manifest_rows")
        sorter_success = sorter_metadata.get("success_count")
        sorter_fail = sorter_metadata.get("fail_count")
        sorter_throughput = safe_div(sorter_total, sorter_elapsed)
        sorter_bytes = sorter_metadata.get("total_bytes")
        if sorter_bytes is None:
            sorter_bytes = sorter_runtime.get("total_bytes")
        per_modality_bytes = sorter_metadata.get("per_modality_bytes") or sorter_runtime.get("per_modality_bytes") or {}
        unknown_bytes = sorter_metadata.get("unknown_bytes") or sorter_runtime.get("unknown_bytes") or {}
        sorter_megabytes = safe_div(sorter_bytes, BYTES_IN_MB)
        per_modality_megabytes = {
            modality: safe_div(value, BYTES_IN_MB)
            for modality, value in per_modality_bytes.items()
            if isinstance(value, (int, float))
        }
        unknown_megabytes = {
            category: safe_div(value, BYTES_IN_MB)
            for category, value in unknown_bytes.items()
            if isinstance(value, (int, float))
        }
        sorter_data_throughput = safe_div(sorter_megabytes, sorter_elapsed)

        aggregated["sorter"] = {
            "total_rows": sorter_total,
            "manifest_path": sorter_info.get("manifest_path"),
            "per_modality_counts": sorter_info.get("per_modality_counts", {}),
            "unknown_counts": sorter_info.get("unknown_counts", {}),
            "success_count": sorter_success,
            "fail_count": sorter_fail,
            "elapsed_seconds": sorter_elapsed,
            "files_per_second": sorter_throughput,
            "prediction_file": sorter_metadata.get("prediction_file"),
            "move_files": sorter_metadata.get("move_files"),
            "total_bytes": sorter_bytes,
            "total_megabytes": sorter_megabytes,
            "megabytes_per_second": sorter_data_throughput,
            "per_modality_bytes": per_modality_bytes,
            "per_modality_megabytes": per_modality_megabytes,
            "unknown_bytes": unknown_bytes,
            "unknown_megabytes": unknown_megabytes,
        }

        plan = self.summary.get("modality_plan", {})
        results = self.summary.get("modality_results", {})
        modality_report: Dict[str, Any] = {}

        total_processed = 0
        total_elapsed = 0.0
        total_candidates = 0
        total_unique = 0
        total_duplicates = 0
        total_missing = 0
        total_copied = 0
        enabled_count = 0
        completed_count = 0

        for modality, plan_entry in plan.items():
            result_entry = results.get(modality, {})
            runner_summary = result_entry.get("runner_summary") or {}
            stats = runner_summary.get("stats") or {}

            enabled = plan_entry.get("enabled", True)
            status = plan_entry.get("status")
            files = plan_entry.get("files")
            processed = result_entry.get("file_count")
            elapsed_seconds = result_entry.get("elapsed_seconds")
            if isinstance(processed, (int, float)):
                total_processed += processed
            if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds > 0:
                total_elapsed += elapsed_seconds
            if enabled:
                enabled_count += 1
            if status == "completed":
                completed_count += 1

            candidates = stats.get("total_candidates")
            duplicates = stats.get("duplicates")
            unique = stats.get("unique")
            missing = stats.get("missing")
            copied = stats.get("copied")
            if isinstance(candidates, (int, float)):
                total_candidates += candidates
            if isinstance(unique, (int, float)):
                total_unique += unique
            if isinstance(duplicates, (int, float)):
                total_duplicates += duplicates
            if isinstance(missing, (int, float)):
                total_missing += missing
            if isinstance(copied, (int, float)):
                total_copied += copied

            report_entry = {
                "enabled": enabled,
                "status": status,
                "files": files,
                "processed": processed,
                "output_dir": plan_entry.get("output_dir"),
                "env": plan_entry.get("env"),
                "elapsed_seconds": elapsed_seconds,
                "files_per_second": safe_div(processed, elapsed_seconds),
                "runner_summary": runner_summary,
                "duplicates_file": runner_summary.get("duplicates_file"),
                "manifest": runner_summary.get("manifest"),
                "deduplication_rate": safe_div(duplicates, candidates),
                "unique_ratio": safe_div(unique, candidates),
                "stats": {
                    "total_candidates": candidates,
                    "selected": stats.get("selected"),
                    "unique": unique,
                    "duplicates": duplicates,
                    "missing": missing,
                    "copied": copied,
                },
            }
            modality_report[modality] = report_entry

        aggregated["modalities"] = modality_report

        stats_totals = {
            "total_candidates": total_candidates,
            "unique": total_unique,
            "duplicates": total_duplicates,
            "missing": total_missing,
            "copied": total_copied,
            "duplicates_ratio": safe_div(total_duplicates, total_candidates),
            "unique_ratio": safe_div(total_unique, total_candidates),
        }

        # Drop stats if no modality executed provided any totals.
        if not any(
            isinstance(value, (int, float)) and value > 0
            for key, value in stats_totals.items()
            if key in {"total_candidates", "unique", "duplicates", "missing", "copied"}
        ):
            stats_totals = {}

        overall_stats: Dict[str, Any] = {
            "total_inputs": sorter_total,
            "modalities_enabled": enabled_count,
            "modalities_completed": completed_count,
            "modalities_processed": total_processed,
            "modalities_elapsed_seconds": total_elapsed if total_elapsed > 0 else None,
            "modalities_files_per_second": safe_div(total_processed, total_elapsed) if total_elapsed > 0 else None,
            "stats_totals": stats_totals,
            "total_input_bytes": sorter_bytes,
            "total_input_megabytes": sorter_megabytes,
            "sorter_megabytes_per_second": sorter_data_throughput,
        }
        aggregated["overall"] = overall_stats

        return aggregated

    def finalize(self) -> None:
        report_dir = self.output_root
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_filename = self.config.report.get("summary_file", "summary.json") if self.config.report else "summary.json"
        summary_path = Path(summary_filename)
        if not summary_path.is_absolute():
            summary_path = report_dir / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary["aggregated"] = self._build_aggregated_results()
        self.summary.setdefault("timestamps", {})["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2)
        self.summary.setdefault("report_outputs", {})["summary_json"] = str(summary_path)
        self.logger.info("Pipeline summary saved to %s", summary_path)
        self._write_run_manifest()

    # Placeholder stage runners to be implemented in later phases
    def _append_existing_stage_summary(self, stage_dir: Path) -> None:
        summary_path = stage_dir / "summary.json"
        if not summary_path.exists():
            self.logger.warning("Expected stage summary missing at %s", summary_path)
            return
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.summary["stages"].append(data)
            self._write_run_manifest()
        except Exception as exc:  # pragma: no cover
            self.logger.error("Failed to load existing stage summary: %s", exc)

    def _load_manifest_snapshot(self, manifest_path: Path) -> None:
        try:
            manifest_data = load_manifest_data(manifest_path)
        except (FileNotFoundError, ManifestFormatError) as exc:
            self.logger.error("Unable to load sorter manifest: %s", exc)
            raise
        self.sorter_manifest = manifest_data
        per_modality_counts = {
            modality: len(paths)
            for modality, paths in manifest_data.per_modality.items()
        }
        unknown_total = sum(len(paths) for paths in manifest_data.unknown.values())
        self.summary["sorter_manifest"] = {
            "manifest_path": str(manifest_path),
            "total_rows": manifest_data.total,
            "per_modality_counts": per_modality_counts,
            "unknown_counts": {
                modality: len(paths) for modality, paths in manifest_data.unknown.items()
            },
        }
        self.logger.info(
            "Sorter manifest loaded: total=%d %s unknown=%d",
            manifest_data.total,
            " ".join(
                f"{mod}:{count}" for mod, count in sorted(per_modality_counts.items())
            ),
            unknown_total,
        )
        self._write_run_manifest()

    def _check_manifest_accessibility(self, subset: List[str], stage_dir: Path, modality: str) -> None:
        """
        Assert that manifest subset paths are accessible from the orchestrator's execution environment.
        If the missing fraction or absolute missing count exceeds configured thresholds, write
        a failed StageArtifact and raise RuntimeError to abort the stage early.

        Configuration (optional) under `general.manifest_check`:
        - max_missing_fraction: float (default 0.10)
        - max_missing_count: int or None (default None)
        - sample_limit: int (how many missing paths to include in artifact, default 50)
        """
        input_root = self.config.general.get("input_root")
        max_missing_fraction = 0.10
        max_missing_count = None
        sample_limit = 50
        mcfg = self.config.general.get("manifest_check") or {}
        try:
            if isinstance(mcfg, dict):
                max_missing_fraction = float(mcfg.get("max_missing_fraction", max_missing_fraction))
                if mcfg.get("max_missing_count") is not None:
                    max_missing_count = int(mcfg.get("max_missing_count"))
                sample_limit = int(mcfg.get("sample_limit", sample_limit))
        except Exception:
            # ignore bad config and use defaults
            pass

        missing_files: list[str] = []
        for p in subset:
            # Try as-is first, then relative to input_root if provided
            try_paths = [Path(p)]
            if input_root and not Path(p).is_absolute():
                try_paths.append(Path(input_root) / p)
            exists = False
            for tp in try_paths:
                if tp.exists():
                    exists = True
                    break
            if not exists:
                missing_files.append(p)

        total = len(subset)
        missing_count = len(missing_files)
        missing_fraction = (missing_count / total) if total > 0 else 0.0

        if (missing_fraction > max_missing_fraction) or (
            max_missing_count is not None and missing_count > max_missing_count
        ):
            # Record a failed artifact with details for debugging and fail fast.
            snippet = missing_files[:sample_limit]
            metadata = {
                "config_hash": None,
                "file_count": total,
                "missing_count": missing_count,
                "missing_fraction": missing_fraction,
                "missing_samples": snippet,
                "notes": (
                    f"Manifest accessibility check failed for {modality}: "
                    f"{missing_count}/{total} missing ({missing_fraction:.2%})"
                ),
            }
            artifact = StageArtifact(
                stage_name=f"stage2_{modality}",
                status="failed",
                elapsed_seconds=0.0,
                output_paths={"manifest": str(stage_dir / "input_manifest.txt")},
                metadata=metadata,
            )
            # Save artifact so run manifest captures failure reasons
            self.save_stage_result(f"stage2_{modality}", artifact, success=False)
            # Raise to abort the stage immediately
            raise RuntimeError(metadata["notes"])

    def _prepare_modality_tasks(self) -> list[Dict[str, Any]]:
        if self.sorter_manifest is None:
            raise RuntimeError("Sorter manifest is not loaded; run sorter stage first")

        tasks: list[Dict[str, Any]] = []
        for modality in ("image", "audio", "text"):
            if not self.config.modality_enabled(modality):
                self.logger.info("%s modality stage disabled via configuration", modality)
                self.summary["modality_plan"][modality] = {"enabled": False}
                continue

            files = list(self.sorter_manifest.per_modality.get(modality, []))
            modality_config = getattr(self.config, modality, {}) or {}
            task = {
                "modality": modality,
                "files": files,
                "config": modality_config,
                "stage_name": f"stage2_{modality}",
                "env": self.config.env_for(modality),
            }
            tasks.append(task)
        return tasks

    def _run_single_modality_stage(self, task: Dict[str, Any]) -> None:
        modality = task["modality"]
        stage_name = task["stage_name"]
        stage_dir = self.stage_artifact_dir(stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)

        plan_entry = self.summary["modality_plan"].setdefault(modality, {"enabled": True})
        results_map = self.summary.setdefault("modality_results", {})

        flags = self._stage_flags(stage_name)
        if flags["lock"].exists():
            self.logger.error(
                "%s stage lock detected at %s. Remove the stale lock before retrying.",
                stage_name,
                flags["lock"],
            )
            raise StageLockError(f"Stale modality stage lock: {flags['lock']}")
        if flags["failure"].exists():
            self.logger.warning(
                "Previous %s stage marked as FAILURE at %s; this run will attempt to rerun.",
                stage_name,
                flags["failure"],
            )

        files = task["files"]
        config_snapshot = {
            "config": task["config"],
            "files": files,
        }
        config_hash = compute_dict_hash(config_snapshot)

        if self._stage_should_resume(stage_name, config_hash):
            self.logger.info(
                "%s stage already completed with matching config hash, skipping",
                stage_name,
            )
            self._append_existing_stage_summary(stage_dir)
            plan_entry["resume"] = True
            plan_entry["status"] = "reused"
            try:
                with (stage_dir / "summary.json").open("r", encoding="utf-8") as f:
                    summary_payload = json.load(f)
                results_map[modality] = summary_payload.get("metadata", {})
            except Exception:
                pass
            self._write_run_manifest()
            return

        max_retries = task.get("config", {}).get("max_retries")
        if max_retries is None:
            max_retries = self.config.general.get("retry", {}).get("max_retries", 0)
        retry_delay = self.config.general.get("retry", {}).get("delay_seconds", 0)

        attempts_metadata: list[Dict[str, Any]] = []
        attempt = 0
        while True:
            attempt += 1
            plan_entry["attempt"] = attempt
            try:
                if modality == "image":
                    self._run_image_stage(task, stage_dir, config_hash, plan_entry, results_map)
                elif modality == "audio":
                    self._run_audio_stage(task, stage_dir, config_hash, plan_entry, results_map)
                elif modality == "text":
                    self._run_text_stage(task, stage_dir, config_hash, plan_entry, results_map)
                else:
                    metadata = {
                        "config_hash": config_hash,
                        "file_count": len(files),
                        "env": task.get("env"),
                        "notes": "Pending modality implementation",
                    }
                    artifact = StageArtifact(
                        stage_name=stage_name,
                        status="planned",
                        elapsed_seconds=0.0,
                        output_paths={},
                        metadata=metadata,
                    )
                    self.save_stage_result(stage_name, artifact, success=True)
                    plan_entry["status"] = "planned"
                    results_map[modality] = metadata
                break
            except ExecutorError as exc:
                error_payload = {
                    "attempt": attempt,
                    "error": str(exc),
                }
                attempts_metadata.append(error_payload)
                plan_entry.setdefault("errors", []).append(error_payload)
                if attempt > max_retries:
                    self.logger.error(
                        "%s stage failed after %d attempt(s); no more retries", stage_name, attempt
                    )
                    raise
                self.logger.warning(
                    "%s stage failed on attempt %d/%d; retrying after %.1f seconds",
                    stage_name,
                    attempt,
                    max_retries,
                    retry_delay,
                )
                if retry_delay > 0:
                    time.sleep(retry_delay)
            except Exception:
                raise

        if attempts_metadata:
            results_map.setdefault(modality, {}).setdefault("attempts", attempts_metadata)

    def _run_image_stage(
        self,
        task: Dict[str, Any],
        stage_dir: Path,
        config_hash: str,
        plan_entry: Dict[str, Any],
        results_map: Dict[str, Any],
    ) -> None:
        modality = task["modality"]
        stage_name = task["stage_name"]
        files = task["files"]
        modality_config = task["config"]
        env_name = task.get("env")

        entrypoint = modality_config.get("entrypoint")
        if not entrypoint:
            raise RuntimeError("Image modality requires 'entrypoint' in configuration")

        entrypoint_path = Path(entrypoint)
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"Image entrypoint not found: {entrypoint_path}")

        if not files:
            self.logger.info("Image stage has no files; marking as skipped")
            metadata = {
                "config_hash": config_hash,
                "file_count": 0,
                "env": env_name,
                "reason": "No files available for image stage",
            }
            artifact = StageArtifact(
                stage_name=stage_name,
                status="skipped",
                elapsed_seconds=0.0,
                output_paths={},
                metadata=metadata,
            )
            self.save_stage_result(stage_name, artifact, success=True)
            plan_entry["status"] = "skipped"
            results_map[modality] = metadata
            return

        cwd = modality_config.get("workdir")
        args = ["python", str(entrypoint_path)]
        extra_args = modality_config.get("args", [])
        if extra_args:
            args.extend(extra_args)

        manifest_limit = (
            modality_config.get("manifest_subset_count")
            or modality_config.get("batch_size")
            or self.config.general.get("batch_size")
        )

        # Build chunks like text stage so we process all files instead of only the first subset
        if manifest_limit and manifest_limit > 0:
            chunks = [files[i : i + manifest_limit] for i in range(0, len(files), manifest_limit)]
        else:
            chunks = [files]

        import shutil

        extra_env_base = {
            "PIPELINE_IMAGE_TOTAL": str(len(files)),
        }

        env_overrides = modality_config.get("env")
        if isinstance(env_overrides, dict):
            extra_env_base.update(env_overrides)
        output_dir = modality_config.get("output_dir")
        if output_dir:
            extra_env_base.setdefault("PIPELINE_IMAGE_OUTPUT_DIR", str(output_dir))
        config_file = modality_config.get("config_file")
        if config_file:
            extra_env_base.setdefault("PIPELINE_IMAGE_CONFIG_FILE", str(config_file))

        # Check accessibility for first chunk (fail-fast if many missing)
        first_subset = chunks[0] if chunks else []
        try:
            self._check_manifest_accessibility(first_subset, stage_dir, modality)
        except Exception as exc:
            self.logger.error("Image stage manifest accessibility assertion failed: %s", exc)
            plan_entry["status"] = "failed"
            results_map[modality] = {"error": str(exc)}
            raise

        lock_acquired = False
        chunk_results: List[Dict[str, Any]] = []
        total_elapsed = 0.0
        try:
            acquire_stage_lock(stage_dir)
            lock_acquired = True
            self.logger.info(
                "Running image stage in %d chunk(s) (env=%s), chunk_size=%s",
                len(chunks),
                env_name or "default",
                str(manifest_limit),
            )

            for idx, chunk in enumerate(chunks, start=1):
                chunk_manifest = stage_dir / f"input_manifest_{idx}.txt"
                chunk_manifest.write_text("\n".join(chunk), encoding="utf-8")

                extra_env = dict(extra_env_base)
                extra_env["PIPELINE_IMAGE_INPUT_LIST"] = str(chunk_manifest)

                try:
                    result = self.executor.run(
                        args,
                        env_name=env_name,
                        cwd=cwd,
                        capture_output=True,
                        check=True,
                        extra_env=extra_env,
                    )
                except ExecutorError as exc:
                    self.logger.error("Image chunk %d execution failed: %s", idx, exc)
                    metadata = {
                        "config_hash": config_hash,
                        "env": env_name,
                        "chunk_index": idx,
                        "command": args,
                        "error": str(exc),
                    }
                    artifact = StageArtifact(
                        stage_name=stage_name,
                        status="failed",
                        elapsed_seconds=total_elapsed,
                        output_paths={
                            "manifests": [str(stage_dir / f"input_manifest_{i+1}.txt") for i in range(len(chunks))],
                        },
                        metadata=metadata,
                    )
                    self.save_stage_result(stage_name, artifact, success=False)
                    plan_entry["status"] = "failed"
                    results_map[modality] = metadata
                    raise

                # write per-chunk stdout/stderr
                stdout_path = stage_dir / f"stdout_{idx}.log"
                stderr_path = stage_dir / f"stderr_{idx}.log"
                try:
                    stdout_text = result.stdout if result.stdout is not None else ""
                except Exception:
                    stdout_text = str(result.stdout)
                try:
                    stderr_text = result.stderr if result.stderr is not None else ""
                except Exception:
                    stderr_text = str(result.stderr)
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")

                total_elapsed += result.elapsed_seconds

                runner_summary = self._load_runner_summary(modality, extra_env.get("PIPELINE_IMAGE_OUTPUT_DIR"))
                chunk_dup_path = None
                if runner_summary and runner_summary.get("duplicates_file"):
                    try:
                        dup_src = Path(runner_summary.get("duplicates_file"))
                        if dup_src.exists():
                            chunk_dup_path = stage_dir / f"image_duplicates_{idx}.json"
                            shutil.copy2(dup_src, chunk_dup_path)
                    except Exception:
                        chunk_dup_path = None

                chunk_results.append({
                    "chunk_index": idx,
                    "manifest": str(chunk_manifest),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "runner_summary": runner_summary,
                    "duplicates_copy": str(chunk_dup_path) if chunk_dup_path is not None else None,
                    "elapsed_seconds": result.elapsed_seconds,
                })

        finally:
            if lock_acquired:
                release_stage_lock(stage_dir)

        aggregated_stats: Dict[str, Any] = {
            "config_hash": config_hash,
            "file_count": len(files),
            "total_candidates": len(files),
            "env": env_name,
            "chunks": len(chunk_results),
            "elapsed_seconds": total_elapsed,
        }

        merged_runner_stats: Dict[str, float] = {}
        for cr in chunk_results:
            rs = (cr.get("runner_summary") or {}).get("stats") or {}
            for k, v in rs.items():
                if isinstance(v, (int, float)):
                    merged_runner_stats[k] = merged_runner_stats.get(k, 0) + float(v)

        if merged_runner_stats:
            aggregated_stats["runner_summary"] = {"modality": modality, "stats": merged_runner_stats}

        output_paths = {
            "manifests": [cr["manifest"] for cr in chunk_results],
            "stdout_logs": [cr["stdout"] for cr in chunk_results],
            "stderr_logs": [cr["stderr"] for cr in chunk_results],
            "duplicates": [cr.get("duplicates_copy") for cr in chunk_results if cr.get("duplicates_copy")],
        }

        artifact = StageArtifact(
            stage_name=stage_name,
            status="success",
            elapsed_seconds=total_elapsed,
            output_paths=output_paths,
            metadata=aggregated_stats,
        )
        self.save_stage_result(stage_name, artifact, success=True)
        plan_entry["status"] = "completed"
        plan_entry["processed_files"] = len(files)
        results_map[modality] = aggregated_stats

    def _run_audio_stage(
        self,
        task: Dict[str, Any],
        stage_dir: Path,
        config_hash: str,
        plan_entry: Dict[str, Any],
        results_map: Dict[str, Any],
    ) -> None:
        modality = task["modality"]
        stage_name = task["stage_name"]
        files = task["files"]
        modality_config = task["config"]
        env_name = task.get("env")

        entrypoint = modality_config.get("entrypoint")
        if not entrypoint:
            raise RuntimeError("Audio modality requires 'entrypoint' in configuration")

        entrypoint_path = Path(entrypoint)
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"Audio entrypoint not found: {entrypoint_path}")

        if not files:
            self.logger.info("Audio stage has no files; marking as skipped")
            metadata = {
                "config_hash": config_hash,
                "file_count": 0,
                "env": env_name,
                "reason": "No files available for audio stage",
            }
            artifact = StageArtifact(
                stage_name=stage_name,
                status="skipped",
                elapsed_seconds=0.0,
                output_paths={},
                metadata=metadata,
            )
            self.save_stage_result(stage_name, artifact, success=True)
            plan_entry["status"] = "skipped"
            results_map[modality] = metadata
            return

        cwd = modality_config.get("workdir")
        args = ["python", str(entrypoint_path)]
        extra_args = modality_config.get("args", [])
        if extra_args:
            args.extend(extra_args)

        manifest_limit = (
            modality_config.get("manifest_subset_count")
            or modality_config.get("batch_size")
            or self.config.general.get("batch_size")
        )

        if manifest_limit and manifest_limit > 0:
            chunks = [files[i : i + manifest_limit] for i in range(0, len(files), manifest_limit)]
        else:
            chunks = [files]

        import shutil

        extra_env_base = {
            "PIPELINE_AUDIO_TOTAL": str(len(files)),
        }

        env_overrides = modality_config.get("env")
        if isinstance(env_overrides, dict):
            extra_env_base.update(env_overrides)
        output_dir = modality_config.get("output_dir")
        if output_dir:
            extra_env_base.setdefault("PIPELINE_AUDIO_OUTPUT_DIR", str(output_dir))
        config_file = modality_config.get("config_file")
        if config_file:
            extra_env_base.setdefault("PIPELINE_AUDIO_CONFIG_FILE", str(config_file))

        # Pre-check first chunk accessibility
        first_subset = chunks[0] if chunks else []
        try:
            self._check_manifest_accessibility(first_subset, stage_dir, modality)
        except Exception as exc:
            self.logger.error("Audio stage manifest accessibility assertion failed: %s", exc)
            plan_entry["status"] = "failed"
            results_map[modality] = {"error": str(exc)}
            raise

        lock_acquired = False
        chunk_results: List[Dict[str, Any]] = []
        total_elapsed = 0.0
        try:
            acquire_stage_lock(stage_dir)
            lock_acquired = True
            self.logger.info(
                "Running audio stage in %d chunk(s) (env=%s), chunk_size=%s",
                len(chunks),
                env_name or "default",
                str(manifest_limit),
            )

            for idx, chunk in enumerate(chunks, start=1):
                chunk_manifest = stage_dir / f"input_manifest_{idx}.txt"
                chunk_manifest.write_text("\n".join(chunk), encoding="utf-8")

                extra_env = dict(extra_env_base)
                extra_env["PIPELINE_AUDIO_INPUT_LIST"] = str(chunk_manifest)

                try:
                    result = self.executor.run(
                        args,
                        env_name=env_name,
                        cwd=cwd,
                        capture_output=True,
                        check=True,
                        extra_env=extra_env,
                    )
                except ExecutorError as exc:
                    self.logger.error("Audio chunk %d execution failed: %s", idx, exc)
                    metadata = {
                        "config_hash": config_hash,
                        "env": env_name,
                        "chunk_index": idx,
                        "command": args,
                        "error": str(exc),
                    }
                    artifact = StageArtifact(
                        stage_name=stage_name,
                        status="failed",
                        elapsed_seconds=total_elapsed,
                        output_paths={
                            "manifests": [str(stage_dir / f"input_manifest_{i+1}.txt") for i in range(len(chunks))],
                        },
                        metadata=metadata,
                    )
                    self.save_stage_result(stage_name, artifact, success=False)
                    plan_entry["status"] = "failed"
                    results_map[modality] = metadata
                    raise

                stdout_path = stage_dir / f"stdout_{idx}.log"
                stderr_path = stage_dir / f"stderr_{idx}.log"
                try:
                    stdout_text = result.stdout if result.stdout is not None else ""
                except Exception:
                    stdout_text = str(result.stdout)
                try:
                    stderr_text = result.stderr if result.stderr is not None else ""
                except Exception:
                    stderr_text = str(result.stderr)
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")

                total_elapsed += result.elapsed_seconds

                runner_summary = self._load_runner_summary(modality, extra_env.get("PIPELINE_AUDIO_OUTPUT_DIR"))
                chunk_dup_path = None
                if runner_summary and runner_summary.get("duplicates_file"):
                    try:
                        dup_src = Path(runner_summary.get("duplicates_file"))
                        if dup_src.exists():
                            chunk_dup_path = stage_dir / f"audio_duplicates_{idx}.json"
                            shutil.copy2(dup_src, chunk_dup_path)
                    except Exception:
                        chunk_dup_path = None

                chunk_results.append({
                    "chunk_index": idx,
                    "manifest": str(chunk_manifest),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "runner_summary": runner_summary,
                    "duplicates_copy": str(chunk_dup_path) if chunk_dup_path is not None else None,
                    "elapsed_seconds": result.elapsed_seconds,
                })

        finally:
            if lock_acquired:
                release_stage_lock(stage_dir)

        aggregated_stats: Dict[str, Any] = {
            "config_hash": config_hash,
            "file_count": len(files),
            "total_candidates": len(files),
            "env": env_name,
            "chunks": len(chunk_results),
            "elapsed_seconds": total_elapsed,
        }

        merged_runner_stats: Dict[str, float] = {}
        for cr in chunk_results:
            rs = (cr.get("runner_summary") or {}).get("stats") or {}
            for k, v in rs.items():
                if isinstance(v, (int, float)):
                    merged_runner_stats[k] = merged_runner_stats.get(k, 0) + float(v)

        if merged_runner_stats:
            aggregated_stats["runner_summary"] = {"modality": modality, "stats": merged_runner_stats}

        output_paths = {
            "manifests": [cr["manifest"] for cr in chunk_results],
            "stdout_logs": [cr["stdout"] for cr in chunk_results],
            "stderr_logs": [cr["stderr"] for cr in chunk_results],
            "duplicates": [cr.get("duplicates_copy") for cr in chunk_results if cr.get("duplicates_copy")],
        }

        artifact = StageArtifact(
            stage_name=stage_name,
            status="success",
            elapsed_seconds=total_elapsed,
            output_paths=output_paths,
            metadata=aggregated_stats,
        )
        self.save_stage_result(stage_name, artifact, success=True)
        plan_entry["status"] = "completed"
        plan_entry["processed_files"] = len(files)
        results_map[modality] = aggregated_stats

    def _run_text_stage(
        self,
        task: Dict[str, Any],
        stage_dir: Path,
        config_hash: str,
        plan_entry: Dict[str, Any],
        results_map: Dict[str, Any],
    ) -> None:
        modality = task["modality"]
        stage_name = task["stage_name"]
        files = task["files"]
        modality_config = task["config"]
        env_name = task.get("env")

        entrypoint = modality_config.get("entrypoint")
        if not entrypoint:
            raise RuntimeError("Text modality requires 'entrypoint' in configuration")

        entrypoint_path = Path(entrypoint)
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"Text entrypoint not found: {entrypoint_path}")

        if not files:
            self.logger.info("Text stage has no files; marking as skipped")
            metadata = {
                "config_hash": config_hash,
                "file_count": 0,
                "env": env_name,
                "reason": "No files available for text stage",
            }
            artifact = StageArtifact(
                stage_name=stage_name,
                status="skipped",
                elapsed_seconds=0.0,
                output_paths={},
                metadata=metadata,
            )
            self.save_stage_result(stage_name, artifact, success=True)
            plan_entry["status"] = "skipped"
            results_map[modality] = metadata
            return

        cwd = modality_config.get("workdir")
        args = ["python", str(entrypoint_path)]
        extra_args = modality_config.get("args", [])
        if extra_args:
            args.extend(extra_args)

        manifest_limit = (
            modality_config.get("manifest_subset_count")
            or modality_config.get("batch_size")
            or self.config.general.get("batch_size")
        )
        subset = files[:manifest_limit] if manifest_limit else files

        manifest_path = stage_dir / "input_manifest.txt"
        manifest_path.write_text("\n".join(subset), encoding="utf-8")
        extra_env = {
            "PIPELINE_TEXT_INPUT_LIST": str(manifest_path),
            "PIPELINE_TEXT_TOTAL": str(len(files)),
        }
        # Check manifest accessibility from orchestrator host before launching runner
        try:
            self._check_manifest_accessibility(subset, stage_dir, modality)
        except Exception as exc:
            self.logger.error("Text stage manifest accessibility assertion failed: %s", exc)
            plan_entry["status"] = "failed"
            results_map[modality] = {"error": str(exc)}
            raise
        env_overrides = modality_config.get("env")
        if isinstance(env_overrides, dict):
            extra_env.update(env_overrides)
        output_dir = modality_config.get("output_dir")
        if output_dir:
            extra_env.setdefault("PIPELINE_TEXT_OUTPUT_DIR", str(output_dir))
        config_file = modality_config.get("config_file")
        if config_file:
            extra_env.setdefault("PIPELINE_TEXT_CONFIG_FILE", str(config_file))

        # Implement chunked execution for text stage to avoid handing very
        # large manifests to a single runner invocation. We'll acquire the
        # stage lock once, then iterate over slices of `files` of size
        # `manifest_limit`, invoking the runner for each slice and
        # aggregating per-chunk summaries.
        import shutil

        manifest_limit = manifest_limit
        chunks: List[List[str]]
        if manifest_limit and manifest_limit > 0:
            chunks = [files[i : i + manifest_limit] for i in range(0, len(files), manifest_limit)]
        else:
            chunks = [files]

        lock_acquired = False
        chunk_results: List[Dict[str, Any]] = []
        total_elapsed = 0.0
        try:
            acquire_stage_lock(stage_dir)
            lock_acquired = True
            self.logger.info(
                "Running text stage in %d chunk(s) (env=%s), chunk_size=%s",
                len(chunks),
                env_name or "default",
                str(manifest_limit),
            )

            for idx, chunk in enumerate(chunks, start=1):
                chunk_manifest = stage_dir / f"input_manifest_{idx}.txt"
                chunk_manifest.write_text("\n".join(chunk), encoding="utf-8")

                extra_env_chunk = dict(extra_env)
                extra_env_chunk["PIPELINE_TEXT_INPUT_LIST"] = str(chunk_manifest)
                # Keep PIPELINE_TEXT_TOTAL as total candidates

                try:
                    result = self.executor.run(
                        args,
                        env_name=env_name,
                        cwd=cwd,
                        capture_output=True,
                        check=True,
                        extra_env=extra_env_chunk,
                    )
                except ExecutorError as exc:
                    self.logger.error("Text chunk %d execution failed: %s", idx, exc)
                    metadata = {
                        "config_hash": config_hash,
                        "env": env_name,
                        "chunk_index": idx,
                        "command": args,
                        "error": str(exc),
                    }
                    artifact = StageArtifact(
                        stage_name=stage_name,
                        status="failed",
                        elapsed_seconds=total_elapsed,
                        output_paths={
                            "manifests": [str(stage_dir / f"input_manifest_{i+1}.txt") for i in range(len(chunks))],
                        },
                        metadata=metadata,
                    )
                    self.save_stage_result(stage_name, artifact, success=False)
                    plan_entry["status"] = "failed"
                    results_map[modality] = metadata
                    raise

                # write per-chunk stdout/stderr to distinct files
                stdout_path = stage_dir / f"stdout_{idx}.log"
                stderr_path = stage_dir / f"stderr_{idx}.log"
                try:
                    stdout_text = result.stdout if result.stdout is not None else ""
                except Exception:
                    stdout_text = str(result.stdout)
                try:
                    stderr_text = result.stderr if result.stderr is not None else ""
                except Exception:
                    stderr_text = str(result.stderr)
                stdout_path.write_text(stdout_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")

                total_elapsed += result.elapsed_seconds

                # try to load runner summary from output dir
                runner_summary = self._load_runner_summary(modality, extra_env.get("PIPELINE_TEXT_OUTPUT_DIR"))
                # If runner wrote duplicates file, copy it to stage_dir with chunk suffix
                chunk_dup_path = None
                if runner_summary and runner_summary.get("duplicates_file"):
                    try:
                        dup_src = Path(runner_summary.get("duplicates_file"))
                        if dup_src.exists():
                            chunk_dup_path = stage_dir / f"text_duplicates_{idx}.json"
                            shutil.copy2(dup_src, chunk_dup_path)
                    except Exception:
                        chunk_dup_path = None

                chunk_results.append({
                    "chunk_index": idx,
                    "manifest": str(chunk_manifest),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "runner_summary": runner_summary,
                    "duplicates_copy": str(chunk_dup_path) if chunk_dup_path is not None else None,
                    "elapsed_seconds": result.elapsed_seconds,
                })

        finally:
            if lock_acquired:
                release_stage_lock(stage_dir)

        # Aggregate chunk results into a single artifact metadata
        aggregated_stats: Dict[str, Any] = {
            "config_hash": config_hash,
            "file_count": len(files),
            "total_candidates": len(files),
            "env": env_name,
            "chunks": len(chunk_results),
            "elapsed_seconds": total_elapsed,
        }

        # merge runner_summary.stats across chunks when available
        merged_runner_stats: Dict[str, float] = {}
        for cr in chunk_results:
            rs = (cr.get("runner_summary") or {}).get("stats") or {}
            for k, v in rs.items():
                if isinstance(v, (int, float)):
                    merged_runner_stats[k] = merged_runner_stats.get(k, 0) + float(v)

        if merged_runner_stats:
            aggregated_stats["runner_summary"] = {"modality": modality, "stats": merged_runner_stats}

        output_paths = {
            "manifests": [cr["manifest"] for cr in chunk_results],
            "stdout_logs": [cr["stdout"] for cr in chunk_results],
            "stderr_logs": [cr["stderr"] for cr in chunk_results],
            "duplicates": [cr.get("duplicates_copy") for cr in chunk_results if cr.get("duplicates_copy")],
        }

        artifact = StageArtifact(
            stage_name=stage_name,
            status="success",
            elapsed_seconds=total_elapsed,
            output_paths=output_paths,
            metadata=aggregated_stats,
        )
        self.save_stage_result(stage_name, artifact, success=True)
        plan_entry["status"] = "completed"
        plan_entry["processed_files"] = len(files)
        results_map[modality] = aggregated_stats

    def run_sorter_stage(self) -> None:
        stage_name = "stage1_sorter"
        stage_dir = self.stage_artifact_dir(stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)

        flags = self._stage_flags(stage_name)
        if flags["failure"].exists():
            self.logger.warning(
                "Detected previous sorter stage failure at %s; this run will attempt to rerun. "
                "Inspect prior logs if the upcoming execution also fails.",
                flags["failure"],
            )
        if flags["lock"].exists():
            self.logger.error(
                "Sorter stage lock detected at %s. Remove the stale lock before retrying.",
                flags["lock"],
            )
            raise StageLockError(f"Stale sorter stage lock: {flags['lock']}")

        sorter_config = dict(self.config.sorter)
        sorter_config_snapshot = {
            **sorter_config,
            "input_root": self.config.general.get("input_root"),
        }
        config_hash = compute_dict_hash(sorter_config_snapshot)
        manifest_name = sorter_config.get("manifest_name", "manifest.csv")
        manifest_path = stage_dir / manifest_name

        if self._stage_should_resume(stage_name, config_hash):
            self.logger.info("Sorter stage already completed with matching config hash, skipping")
            self._append_existing_stage_summary(stage_dir)
            self._load_manifest_snapshot(manifest_path)
            return
        if self.config.general.get("resume", False):
            self.logger.info(
                "Resume is enabled but sorter stage artifacts do not match current config hash; rerunning stage.")

        lock_acquired = False
        try:
            acquire_stage_lock(stage_dir)
            lock_acquired = True
            self.logger.info("Running sorter stage")
            result = run_sorter(self.config, manifest_path, self.logger)
        except Exception as exc:  # pragma: no cover
            self.logger.error("Sorter stage failed: %s", exc, exc_info=True)
            artifact = StageArtifact(
                stage_name=stage_name,
                status="failed",
                elapsed_seconds=0.0,
                output_paths={"manifest": str(manifest_path)},
                metadata={"config_hash": config_hash, "error": str(exc)},
            )
            self.save_stage_result(stage_name, artifact, success=False)
            raise
        finally:
            if lock_acquired:
                release_stage_lock(stage_dir)

        artifact = StageArtifact(
            stage_name=stage_name,
            status="success",
            elapsed_seconds=result.stats.get("elapsed_seconds", 0.0),
            output_paths={"manifest": str(manifest_path)},
            metadata={
                "config_hash": config_hash,
                "success_count": result.stats.get("success_count"),
                "fail_count": result.stats.get("fail_count"),
                "manifest_rows": len(result.manifest_rows),
                "manifest_columns": [
                    "source_path",
                    "relative_path",
                    "category",
                    "status",
                    "target_path",
                    "reason",
                ],
                "per_modality_counts": {
                    modality: len(paths)
                    for modality, paths in result.per_modality.items()
                },
                "unknown_counts": {
                    modality: len(paths)
                    for modality, paths in result.unknown.items()
                },
                "per_modality_samples": {
                    modality: paths[:3]
                    for modality, paths in result.per_modality.items()
                    if paths
                },
                "unknown_samples": {
                    modality: paths[:3]
                    for modality, paths in result.unknown.items()
                    if paths
                },
                "prediction_file": result.stats.get("prediction_file"),
                "move_files": result.stats.get("move_files"),
                "total_bytes": result.stats.get("total_bytes"),
                "per_modality_bytes": result.stats.get("per_modality_bytes", {}),
                "unknown_bytes": result.stats.get("unknown_bytes", {}),
            },
        )
        self.save_stage_result(stage_name, artifact, success=True)
        self._load_manifest_snapshot(manifest_path)

        sorter_bytes = {
            "total_bytes": result.stats.get("total_bytes"),
            "per_modality_bytes": result.stats.get("per_modality_bytes", {}),
            "unknown_bytes": result.stats.get("unknown_bytes", {}),
        }
        sorter_summary = self.summary.setdefault("sorter", {})
        sorter_summary.update(
            {
                "success_count": result.stats.get("success_count"),
                "fail_count": result.stats.get("fail_count"),
                "elapsed_seconds": result.stats.get("elapsed_seconds"),
                **sorter_bytes,
            }
        )
        sorter_manifest_summary = self.summary.get("sorter_manifest")
        if sorter_manifest_summary:
            sorter_manifest_summary.update(
                {k: v for k, v in sorter_bytes.items() if v is not None}
            )
        self._write_run_manifest()

    def _submit_modality_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        parallel_enabled = self.config.general.get("parallel_modalities", False)
        max_workers: Optional[int] = self.config.general.get("parallel_workers")
        if isinstance(max_workers, int) and max_workers <= 0:
            max_workers = None

        if not parallel_enabled or len(tasks) == 1:
            for task in tasks:
                self._run_single_modality_stage(task)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: Dict[Future[None], Dict[str, Any]] = {}
            for task in tasks:
                future = executor.submit(self._run_single_modality_stage, task)
                futures[future] = task
            for future in as_completed(futures):
                task = futures[future]
                modality = task.get("modality")
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(
                        "Modality stage %s failed during parallel execution: %s",
                        modality,
                        exc,
                        exc_info=True,
                    )
                    raise

    def run_modality_stages(self) -> None:
        try:
            tasks = self._prepare_modality_tasks()
        except RuntimeError as exc:
            self.logger.error("Cannot run modality stages: %s", exc)
            raise

        if not tasks:
            self.logger.info("No modality stages to run after applying configuration")
            return

        prepared_tasks: List[Dict[str, Any]] = []
        for task in tasks:
            modality = task["modality"]
            files = task["files"]
            env_label = task["env"] or "default"
            output_dir = task["config"].get("output_dir")
            self.summary["modality_plan"][modality] = {
                "enabled": True,
                "files": len(files),
                "env": task["env"],
                "output_dir": output_dir,
                "status": "pending",
            }
            self.logger.info(
                "Dispatch %s stage: %d files, env=%s, output_dir=%s",
                modality,
                len(files),
                env_label,
                output_dir or "<not set>",
            )
            self._write_run_manifest()
            prepared_tasks.append(task)

        self._submit_modality_tasks(prepared_tasks)

    def run_report_stage(self) -> None:
        report_cfg = self.config.report or {}
        markdown_target = report_cfg.get("markdown_file")
        if not markdown_target:
            self.logger.info("Report markdown not configured; skipping report stage")
            return

        markdown_path = Path(markdown_target)
        if not markdown_path.is_absolute():
            markdown_path = self.output_root / markdown_path
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._generate_report_markdown()
        markdown_path.write_text(content, encoding="utf-8")
        self.summary.setdefault("report_outputs", {})["markdown"] = str(markdown_path)
        self.logger.info("Report markdown written to %s", markdown_path)

    def _generate_report_markdown(self) -> str:
        def _fmt_int(value: Any) -> Optional[str]:
            if isinstance(value, (int, float)):
                return f"{int(value):,}"
            return None

        def _fmt_percent(value: Any, precision: int = 2) -> Optional[str]:
            if isinstance(value, (int, float)):
                return f"{float(value) * 100:.{precision}f}%"
            return None

        def _fmt_mb(value: Any, precision: int = 2) -> Optional[str]:
            if isinstance(value, (int, float)):
                return f"{float(value):.{precision}f} MB"
            return None

        def _fmt_bytes_with_mb(bytes_value: Any, mb_value: Any) -> Optional[str]:
            if not isinstance(bytes_value, (int, float)):
                return None
            bytes_part = f"{int(bytes_value):,} bytes"
            mb_part = _fmt_mb(mb_value)
            if mb_part:
                return f"{bytes_part} ({mb_part})"
            return bytes_part

        lines = [
            f"# Pipeline Report - {self.run_id}",
            "",
            "## Overview",
            f"- Config: `{self.summary.get('config')}`",
            f"- Run ID: `{self.run_id}`",
            f"- Stages completed: {len(self.summary.get('stages', []))}",
        ]

        aggregated = self.summary.get("aggregated") or {}
        if not aggregated:
            aggregated = self._build_aggregated_results()
            self.summary["aggregated"] = aggregated

        overall = aggregated.get("overall") or {}
        if overall:
            lines.append("\n## Overall Metrics")
            inputs_str = _fmt_int(overall.get("total_inputs"))
            if inputs_str:
                lines.append(f"- Sorter inputs: {inputs_str}")
            bytes_line = _fmt_bytes_with_mb(
                overall.get("total_input_bytes"),
                overall.get("total_input_megabytes"),
            )
            if bytes_line:
                lines.append(f"- Input volume: {bytes_line}")
            enabled_str = _fmt_int(overall.get("modalities_enabled"))
            if enabled_str:
                lines.append(f"- Modalities enabled: {enabled_str}")
            completed_str = _fmt_int(overall.get("modalities_completed"))
            if completed_str:
                lines.append(f"- Modalities completed: {completed_str}")
            processed_str = _fmt_int(overall.get("modalities_processed"))
            if processed_str:
                lines.append(f"- Files processed across modalities: {processed_str}")
            throughput = overall.get("modalities_files_per_second")
            duration = overall.get("modalities_elapsed_seconds")
            if isinstance(throughput, (int, float)) and isinstance(duration, (int, float)):
                lines.append(
                    f"- Modalities throughput: {throughput:.2f} files/s over {duration:.2f}s"
                )
            sorter_mb_rate = overall.get("sorter_megabytes_per_second")
            if isinstance(sorter_mb_rate, (int, float)):
                lines.append(f"- Sorter data throughput: {sorter_mb_rate:.2f} MB/s")
            stats_totals = overall.get("stats_totals") or {}
            if stats_totals:
                lines.append("- Aggregate dedup stats:")
                candidates_str = _fmt_int(stats_totals.get("total_candidates"))
                if candidates_str:
                    lines.append(f"  - Total candidates: {candidates_str}")
                unique_str = _fmt_int(stats_totals.get("unique"))
                if unique_str:
                    lines.append(f"  - Unique: {unique_str}")
                duplicates_str = _fmt_int(stats_totals.get("duplicates"))
                if duplicates_str:
                    lines.append(f"  - Duplicates: {duplicates_str}")
                missing_str = _fmt_int(stats_totals.get("missing"))
                if missing_str:
                    lines.append(f"  - Missing assets: {missing_str}")
                copied_str = _fmt_int(stats_totals.get("copied"))
                if copied_str:
                    lines.append(f"  - Copied artifacts: {copied_str}")
                dup_ratio = _fmt_percent(stats_totals.get("duplicates_ratio"))
                if dup_ratio:
                    lines.append(f"  - Duplication rate: {dup_ratio}")
                unique_ratio = _fmt_percent(stats_totals.get("unique_ratio"))
                if unique_ratio:
                    lines.append(f"  - Unique ratio: {unique_ratio}")

        sorter_info = aggregated.get("sorter") or {}
        if sorter_info:
            lines.append("\n## Sorter Summary")
            total_rows = sorter_info.get("total_rows")
            if total_rows is not None:
                lines.append(f"- Total inputs: {total_rows}")
            bytes_line = _fmt_bytes_with_mb(
                sorter_info.get("total_bytes"),
                sorter_info.get("total_megabytes"),
            )
            if bytes_line:
                lines.append(f"- Input volume: {bytes_line}")
            manifest_path = sorter_info.get("manifest_path")
            if manifest_path:
                lines.append(f"- Manifest: `{manifest_path}`")
            elapsed = sorter_info.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                lines.append(f"- Elapsed: {elapsed:.2f}s")
            throughput = sorter_info.get("files_per_second")
            if isinstance(throughput, (int, float)):
                lines.append(f"- Throughput: {throughput:.2f} files/s")
            data_throughput = sorter_info.get("megabytes_per_second")
            if isinstance(data_throughput, (int, float)):
                lines.append(f"- Data throughput: {data_throughput:.2f} MB/s")
            success = sorter_info.get("success_count")
            fail = sorter_info.get("fail_count")
            if success is not None or fail is not None:
                pieces = []
                if success is not None:
                    pieces.append(f"success={success}")
                if fail is not None:
                    pieces.append(f"fail={fail}")
                lines.append(f"- Sorter outcomes: {', '.join(pieces)}")
            prediction_file = sorter_info.get("prediction_file")
            if prediction_file:
                lines.append(f"- Prediction file: `{prediction_file}`")
            move_files = sorter_info.get("move_files")
            if move_files is not None:
                lines.append(f"- Move files enabled: {bool(move_files)}")
            per_modality = sorter_info.get("per_modality_counts", {})
            if per_modality:
                lines.append("- Per-modality counts:")
                for modality, count in sorted(per_modality.items()):
                    lines.append(f"  - {modality}: {count}")
            per_modality_bytes = sorter_info.get("per_modality_bytes", {})
            per_modality_mb = sorter_info.get("per_modality_megabytes", {})
            if per_modality_bytes:
                lines.append("- Per-modality volume:")
                for modality, value in sorted(per_modality_bytes.items()):
                    volume_line = _fmt_bytes_with_mb(value, per_modality_mb.get(modality))
                    if volume_line:
                        lines.append(f"  - {modality}: {volume_line}")
            unknown_counts = sorter_info.get("unknown_counts", {})
            if unknown_counts:
                lines.append("- Unknown categories:")
                for category, count in sorted(unknown_counts.items()):
                    lines.append(f"  - {category}: {count}")
            unknown_bytes = sorter_info.get("unknown_bytes", {})
            unknown_megabytes = sorter_info.get("unknown_megabytes", {})
            if unknown_bytes:
                lines.append("- Unknown volume:")
                for category, value in sorted(unknown_bytes.items()):
                    volume_line = _fmt_bytes_with_mb(value, unknown_megabytes.get(category))
                    if volume_line:
                        lines.append(f"  - {category}: {volume_line}")

        stages = self.summary.get("stages", [])
        if stages:
            lines.append("\n## Stage Summary")
            for stage in stages:
                name = stage.get("stage_name")
                status = stage.get("status")
                elapsed = stage.get("elapsed_seconds")
                elapsed_str = f" ({elapsed:.2f}s)" if isinstance(elapsed, (int, float)) else ""
                lines.append(f"- **{name}**: {status}{elapsed_str}")

        modality_report = aggregated.get("modalities") or {}
        if modality_report:
            lines.append("\n## Modality Outcomes")
            for modality, data in sorted(modality_report.items()):
                status = data.get("status") or ("disabled" if not data.get("enabled", True) else "unknown")
                lines.append(f"- **{modality}** (status: {status})")
                stats = data.get("stats") or {}
                env = data.get("env")
                if env:
                    lines.append(f"  - Environment: {env}")
                files = data.get("files")
                if files is not None:
                    lines.append(f"  - Files provided: {files}")
                processed = data.get("processed")
                if processed is not None:
                    lines.append(f"  - Processed subset: {processed}")
                elapsed = data.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)):
                    lines.append(f"  - Elapsed: {elapsed:.2f}s")
                throughput = data.get("files_per_second")
                if isinstance(throughput, (int, float)):
                    lines.append(f"  - Throughput: {throughput:.2f} files/s")
                output_dir = data.get("output_dir")
                if output_dir:
                    lines.append(f"  - Output dir: `{output_dir}`")
                total_candidates = stats.get("total_candidates")
                if total_candidates is not None:
                    lines.append(f"  - Candidates: {total_candidates}")
                unique = stats.get("unique")
                duplicates = stats.get("duplicates")
                missing = stats.get("missing")
                copied = stats.get("copied")
                stats_parts = []
                if unique is not None:
                    stats_parts.append(f"unique={unique}")
                if duplicates is not None:
                    stats_parts.append(f"duplicates={duplicates}")
                if missing is not None:
                    stats_parts.append(f"missing={missing}")
                if copied is not None:
                    stats_parts.append(f"copied={copied}")
                if stats_parts:
                    lines.append("  - Stats: " + ", ".join(stats_parts))
                dup_rate = data.get("deduplication_rate")
                if isinstance(dup_rate, (int, float)):
                    lines.append(f"  - Duplication rate: {dup_rate * 100:.2f}%")
                unique_ratio = data.get("unique_ratio")
                if isinstance(unique_ratio, (int, float)):
                    lines.append(f"  - Unique ratio: {unique_ratio * 100:.2f}%")
                duplicates_file = data.get("duplicates_file")
                if duplicates_file:
                    lines.append(f"  - Duplicates: `{duplicates_file}`")
                manifest = data.get("manifest")
                if manifest:
                    lines.append(f"  - Manifest: `{manifest}`")

        return "\n".join(lines)

    def run(self) -> None:
        self.logger.info("Starting pipeline run %s", self.run_id)
        try:
            self.run_sorter_stage()
            self.run_modality_stages()
            self.run_report_stage()
        finally:
            self.finalize()
