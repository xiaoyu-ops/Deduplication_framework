"""Comparative evaluation runner for sorter and baseline classifiers.

This script prepares an evaluation workspace, runs both the sorter model and
our naive baseline classifier on the same dataset, and computes classification
metrics alongside runtime and memory statistics.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_sorter_evaluation import (  # noqa: E402
    build_confusion,
    collect_misclassifications,
    compute_metrics,
    load_ground_truth,
    load_predictions,
    prepare_evaluation_directory,
)
from tools.naive_baseline_classifier import (  # noqa: E402
    predict_directory,
    write_predictions,
)
import sorter as sorter_module  # noqa: E402

DEFAULT_DATASET_DIR = REPO_ROOT / "test_dataset"
DEFAULT_GROUND_TRUTH = DEFAULT_DATASET_DIR / "ground_truth.jsonl"
DEFAULT_WORK_DIR = REPO_ROOT / "test_dataset_eval"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "comparative"


@dataclass
class ResourceStats:
    elapsed_seconds: float
    cpu_seconds: float
    rss_start_bytes: Optional[int]
    rss_end_bytes: Optional[int]
    rss_peak_bytes: Optional[int]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_seconds": self.cpu_seconds,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "rss_peak_megabytes": (
                self.rss_peak_bytes / (1024 * 1024) if self.rss_peak_bytes else None
            ),
        }


class ResourceMonitor:
    def __init__(self, poll_interval: float = 0.05) -> None:
        self.poll_interval = poll_interval
        self._process = psutil.Process(os.getpid()) if psutil else None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_rss: Optional[int] = None
        self._start_rss: Optional[int] = None
        self._end_rss: Optional[int] = None
        self._start_wall: float = 0.0
        self._end_wall: float = 0.0
        self._start_cpu: float = 0.0
        self._end_cpu: float = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        if self._process is not None:
            try:
                rss = self._process.memory_info().rss
            except psutil.Error:  # pragma: no cover
                rss = None
            self._start_rss = rss
            self._peak_rss = rss
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._process is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            try:
                self._end_rss = self._process.memory_info().rss
            except psutil.Error:  # pragma: no cover
                self._end_rss = None
        self._end_wall = time.perf_counter()
        self._end_cpu = time.process_time()

    def _poll_loop(self) -> None:
        assert self._process is not None
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
            except psutil.Error:
                break
            if rss is not None:
                if self._peak_rss is None or rss > self._peak_rss:
                    self._peak_rss = rss
            time.sleep(self.poll_interval)

    def snapshot(self) -> ResourceStats:
        return ResourceStats(
            elapsed_seconds=self._end_wall - self._start_wall,
            cpu_seconds=self._end_cpu - self._start_cpu,
            rss_start_bytes=self._start_rss,
            rss_end_bytes=self._end_rss,
            rss_peak_bytes=self._peak_rss,
        )


def write_misclassifications_csv(path: Path, rows: Iterable[Tuple[str, str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "actual_label", "predicted_label"])
        writer.writerows(rows)


def dataset_summary(ground_truth: Dict[str, str]) -> Dict[str, int]:
    return dict(Counter(ground_truth.values()))


def run_sorter(
    eval_dir: Path,
    predictions_path: Path,
    input_root: Path,
) -> Tuple[ResourceStats, str, str]:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        files = sorter_module.read_files_from_directory(str(eval_dir))
        with ResourceMonitor() as monitor:
            sorter_module.sorter(
                files,
                eval_mode=True,
                prediction_path=str(predictions_path),
                input_root=str(input_root),
            )
        stats = monitor.snapshot()

    return stats, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def run_baseline(
    eval_dir: Path,
    predictions_path: Path,
    input_root: Path,
) -> Tuple[ResourceStats, str, str]:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        with ResourceMonitor() as monitor:
            predictions = predict_directory(eval_dir, input_root=input_root)
            write_predictions(predictions, predictions_path)
        stats = monitor.snapshot()

    return stats, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def evaluate_predictions(
    ground_truth: Dict[str, str],
    predictions: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]], List[Tuple[str, str, str]]]:
    confusion, actual_labels, predicted_labels = build_confusion(ground_truth, predictions)
    metrics = compute_metrics(confusion, actual_labels, predicted_labels)
    misclassified = collect_misclassifications(ground_truth, predictions)
    return metrics, {label: dict(preds) for label, preds in confusion.items()}, misclassified


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sorter vs baseline evaluation")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="源数据目录")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH), help="ground truth JSONL")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="评估临时工作目录")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="结果输出根目录")
    parser.add_argument("--skip-copy", action="store_true", help="直接在源目录上运行，不复制文件")
    parser.add_argument("--skip-sorter", action="store_true", help="跳过 sorter 评估")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过 baseline 评估")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    ground_truth_path = Path(args.ground_truth).resolve()
    work_dir = Path(args.work_dir).resolve()
    results_root = Path(args.results_dir).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    ground_truth = load_ground_truth(ground_truth_path)
    label_distribution = dataset_summary(ground_truth)

    if args.skip_copy:
        eval_dir = dataset_dir
    else:
        eval_dir = work_dir / "comparative_eval"
        prepare_evaluation_directory(dataset_dir, eval_dir, ground_truth.keys())

    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}
    summary["dataset"] = {
        "dataset_dir": str(dataset_dir),
        "ground_truth": str(ground_truth_path),
        "label_distribution": label_distribution,
        "total_files": len(ground_truth),
        "copied_to": (str(eval_dir) if not args.skip_copy else None),
    }

    input_root = eval_dir

    if not args.skip_sorter:
        sorter_dir = run_dir / "sorter"
        sorter_dir.mkdir(parents=True, exist_ok=True)
        sorter_predictions_path = sorter_dir / "predictions.csv"

        sorter_stats, sorter_stdout, sorter_stderr = run_sorter(eval_dir, sorter_predictions_path, input_root)
        predictions = load_predictions(sorter_predictions_path)
        metrics, confusion, misclassified = evaluate_predictions(ground_truth, predictions)

        throughput = None
        if sorter_stats.elapsed_seconds > 0:
            throughput = metrics["total_files"] / sorter_stats.elapsed_seconds

        model_record = {
            "metrics": metrics,
            "confusion_matrix": confusion,
            "misclassified_count": len(misclassified),
            "performance": {
                "elapsed_seconds": sorter_stats.elapsed_seconds,
                "cpu_seconds": sorter_stats.cpu_seconds,
                "files_per_second": throughput,
            },
            "resources": sorter_stats.to_dict(),
            "stdout": sorter_stdout,
            "stderr": sorter_stderr,
        }
        summary["sorter"] = model_record

        with (sorter_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(model_record, f, ensure_ascii=False, indent=2)

        write_misclassifications_csv(sorter_dir / "misclassified_samples.csv", misclassified)

    if not args.skip_baseline:
        baseline_dir = run_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_predictions_path = baseline_dir / "predictions.csv"

        baseline_stats, baseline_stdout, baseline_stderr = run_baseline(eval_dir, baseline_predictions_path, input_root)
        predictions = load_predictions(baseline_predictions_path)
        metrics, confusion, misclassified = evaluate_predictions(ground_truth, predictions)

        throughput = None
        if baseline_stats.elapsed_seconds > 0:
            throughput = metrics["total_files"] / baseline_stats.elapsed_seconds

        model_record = {
            "metrics": metrics,
            "confusion_matrix": confusion,
            "misclassified_count": len(misclassified),
            "performance": {
                "elapsed_seconds": baseline_stats.elapsed_seconds,
                "cpu_seconds": baseline_stats.cpu_seconds,
                "files_per_second": throughput,
            },
            "resources": baseline_stats.to_dict(),
            "stdout": baseline_stdout,
            "stderr": baseline_stderr,
        }
        summary["baseline"] = model_record

        with (baseline_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(model_record, f, ensure_ascii=False, indent=2)

        write_misclassifications_csv(baseline_dir / "misclassified_samples.csv", misclassified)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    shutil.copy2(ground_truth_path, run_dir / "ground_truth.jsonl")

    print("评估完成。")
    print(f"数据集总数: {summary['dataset']['total_files']}")
    if "sorter" in summary:
        metrics = summary["sorter"]["metrics"]
        perf = summary["sorter"]["performance"]
        print(
            "Sorter -> Accuracy: {0:.4f}, Macro F1: {1:.4f}, Weighted F1: {2:.4f}, Throughput: {3:.2f} 文件/秒".format(
                metrics["accuracy"]["overall"],
                metrics["macro_f1"]["overall"],
                metrics["weighted_f1"]["overall"],
                perf["files_per_second"] or 0.0,
            )
        )
    if "baseline" in summary:
        metrics = summary["baseline"]["metrics"]
        perf = summary["baseline"]["performance"]
        print(
            "Baseline -> Accuracy: {0:.4f}, Macro F1: {1:.4f}, Weighted F1: {2:.4f}, Throughput: {3:.2f} 文件/秒".format(
                metrics["accuracy"]["overall"],
                metrics["macro_f1"]["overall"],
                metrics["weighted_f1"]["overall"],
                perf["files_per_second"] or 0.0,
            )
        )
    print(f"详情见: {summary_path}")


if __name__ == "__main__":
    main()
