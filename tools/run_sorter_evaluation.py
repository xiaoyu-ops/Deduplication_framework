"""Automated evaluation pipeline for sorter classifier.

This script prepares an isolated evaluation workspace, runs sorter.py in
evaluation mode to generate predictions, and computes accuracy, F1 scores,
throughput, and confusion matrix metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "mix_dataset"
DEFAULT_GROUND_TRUTH = DEFAULT_SOURCE_DIR / "ground_truth.jsonl"
DEFAULT_EVAL_DIR = REPO_ROOT / "mix_dataset_eval"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluation" / "sorter"
SORTER_SCRIPT = REPO_ROOT / "sorter.py"


def normalize_key(name: str) -> str:
    """Normalize file identifiers for consistent comparison."""
    return name.replace("\\", "/").lstrip("./")


def load_ground_truth(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            filename = normalize_key(record["filename"])
            label = record["label"].strip()
            mapping[filename] = label
    if not mapping:
        raise ValueError(f"Ground truth file {path} is empty")
    return mapping


def prepare_evaluation_directory(source: Path, dest: Path, filenames: Iterable[str]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel_name in filenames:
        src = source / rel_name
        if not src.exists():
            raise FileNotFoundError(f"Ground truth referenced file not found: {src}")
        dst = dest / rel_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_sorter(eval_dir: Path, predictions_path: Path) -> Tuple[str, str, float]:
    start = time.perf_counter()
    process = subprocess.run(
        [
            sys.executable,
            str(SORTER_SCRIPT),
            "--input",
            str(eval_dir),
            "--eval",
            "--predictions",
            str(predictions_path),
            "--input-root",
            str(eval_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"sorter.py exited with code {process.returncode}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )
    return process.stdout, process.stderr, elapsed


def load_predictions(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "predicted_label" not in reader.fieldnames:
            raise ValueError("Predictions CSV missing required columns")
        for row in reader:
            mapping[normalize_key(row["filename"])] = row["predicted_label"].strip()
    return mapping


def build_confusion(
    ground_truth: Dict[str, str],
    predictions: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, int]], List[str], List[str]]:
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    predicted_labels = set()
    for filename, actual in ground_truth.items():
        predicted = predictions.get(filename, "missing")
        confusion[actual][predicted] += 1
        predicted_labels.add(predicted)
    actual_labels = sorted(set(ground_truth.values()))
    predicted_labels = sorted(predicted_labels)
    return confusion, actual_labels, predicted_labels


def compute_metrics(
    confusion: Dict[str, Dict[str, int]],
    actual_labels: List[str],
    predicted_labels: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    total = sum(sum(row.values()) for row in confusion.values())
    total_float = float(total)
    accuracy_numerator = sum(confusion[label].get(label, 0) for label in actual_labels)
    accuracy = accuracy_numerator / total_float if total_float else 0.0

    label_stats = {}
    total_support = 0
    f1_sum = 0.0
    weighted_f1_sum = 0.0

    for label in actual_labels:
        tp = float(confusion[label].get(label, 0))
        fp = float(sum(confusion[other].get(label, 0) for other in actual_labels if other != label))
        fn = float(sum(count for pred, count in confusion[label].items() if pred != label))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        label_stats[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        total_support += support
        f1_sum += f1
        weighted_f1_sum += f1 * support

    macro_f1 = f1_sum / len(actual_labels) if actual_labels else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support else 0.0

    metrics["accuracy"] = {"overall": accuracy}
    metrics["macro_f1"] = {"overall": macro_f1}
    metrics["weighted_f1"] = {"overall": weighted_f1}
    metrics["per_label"] = label_stats

    pred_counts = {
        label: sum(confusion[actual].get(label, 0) for actual in actual_labels)
        for label in predicted_labels
    }
    metrics["predicted_counts"] = pred_counts
    metrics["total_files"] = int(total)
    return metrics


def collect_misclassifications(
    ground_truth: Dict[str, str],
    predictions: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for filename, actual in ground_truth.items():
        predicted = predictions.get(filename, "missing")
        if predicted != actual:
            rows.append((filename, actual, predicted))
    rows.sort()
    return rows


def save_metrics(
    run_dir: Path,
    metrics: Dict[str, Dict[str, float]],
    confusion: Dict[str, Dict[str, int]],
    stdout: str,
    stderr: str,
    elapsed: float,
) -> None:
    metrics_path = run_dir / "metrics.json"
    total_files = metrics.get("total_files", 0)
    output = {
        "metrics": metrics,
        "confusion_matrix": {actual: dict(preds) for actual, preds in confusion.items()},
        "elapsed_seconds": elapsed,
        "throughput_files_per_sec": (total_files / elapsed if elapsed > 0 else None),
        "sorter_stdout": stdout,
        "sorter_stderr": stderr,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def write_misclassifications(run_dir: Path, rows: List[Tuple[str, str, str]]) -> None:
    if not rows:
        return
    mis_path = run_dir / "misclassified_samples.csv"
    with mis_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "actual_label", "predicted_label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sorter evaluation pipeline")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="源数据目录")
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH), help="ground_truth.jsonl 路径")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="评估临时目录")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_ROOT), help="评估结果输出根目录")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    ground_truth_path = Path(args.ground_truth).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    results_root = Path(args.results_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    ground_truth = load_ground_truth(ground_truth_path)
    prepare_evaluation_directory(source_dir, eval_dir, ground_truth.keys())

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"

    stdout, stderr, elapsed = run_sorter(eval_dir, predictions_path)
    predictions = load_predictions(predictions_path)

    confusion, actual_labels, predicted_labels = build_confusion(ground_truth, predictions)
    metrics = compute_metrics(confusion, actual_labels, predicted_labels)

    misclassified = collect_misclassifications(ground_truth, predictions)
    write_misclassifications(run_dir, misclassified)
    save_metrics(run_dir, metrics, confusion, stdout, stderr, elapsed)

    shutil.copy2(ground_truth_path, run_dir / "ground_truth.jsonl")

    print("评估完成。")
    print(f"评估样本总数: {metrics['total_files']}")
    print(f"Accuracy: {metrics['accuracy']['overall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']['overall']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']['overall']:.4f}")
    if elapsed > 0:
        throughput = metrics['total_files'] / elapsed
        print(f"总耗时: {elapsed:.2f} 秒 (约 {throughput:.2f} 文件/秒)")
    if misclassified:
        print(f"误分类样本数: {len(misclassified)} (详情见 {run_dir / 'misclassified_samples.csv'})")
    else:
        print("未发现误分类样本！")


if __name__ == "__main__":
    main()
