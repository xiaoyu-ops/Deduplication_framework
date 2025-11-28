"""Simple modality sorter with enhanced heuristics for mixed datasets."""

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def read_files_from_directory(directory: str):
    """递归读取目录下的所有文件路径"""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
# 运行流水线时仅允许以下扩展进入 image 阶段，其他即便被 heuristics 识别也强制标记为 unknown。
STRICT_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
STRICT_AUDIO_EXTS = {".wav"}
AUDIO_EXTS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".wma"}
TEXT_EXTS = {".txt", ".json", ".csv", ".md", ".xml", ".yaml", ".yml", ".ini", ".log", ".tsv"}

JSON_TEXT_KEYS = {"text", "content", "title", "sentence", "article"}
JSON_AUDIO_KEYS = {"audio", "audio_url", "audio_path", "wav", "mp3"}
JSON_IMAGE_KEYS = {"image", "image_url", "img", "picture", "thumbnail"}

HEADER_BYTES = 4096
PRINTABLE_THRESHOLD = 0.85


def is_image_url(u: str) -> bool:
    if not isinstance(u, str):
        return False
    u_low = u.lower()
    return any(ext in u_low for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"])


def safe_move_file(src_path: str, dst_path: str, max_retries: int = 3) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)


            original_dst = dst_path
            candidate = dst_path
            counter = 1
            while os.path.exists(candidate):
                name, ext = os.path.splitext(original_dst)
                candidate = f"{name}_{counter}{ext}"
                counter += 1

            shutil.move(src_path, candidate)
            print(f"已将文件 {src_path} 移动到 {candidate}")
            return candidate

        except PermissionError:
            print(f"文件被占用，尝试 {attempt + 1}/{max_retries}: {src_path}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"文件移动失败，可能被其他程序占用: {src_path}")
                print("请关闭可能占用该文件的程序如VSCode、记事本等")
                return None
        except Exception as exc:  # pragma: no cover
            print(f"移动文件时出现未知错误: {exc}")
            return None
    return None


def put_file_in_category(file_path: str, category: str, base_dir: Optional[str] = None) -> Optional[str]:
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent)
    category_dir = os.path.join(base_dir, category, "dataset")
    filename = os.path.basename(file_path)
    dest_path = os.path.join(category_dir, filename)
    return safe_move_file(file_path, dest_path)


def read_header(file_path: str, size: int = HEADER_BYTES) -> bytes:
    try:
        with open(file_path, "rb") as f:
            return f.read(size)
    except Exception:
        return b""


def is_mostly_printable(data: bytes) -> bool:
    if not data:
        return False
    printable = sum((chr(b).isprintable() or chr(b).isspace()) for b in data)
    return printable / len(data) >= PRINTABLE_THRESHOLD


def sniff_magic(header: bytes) -> Optional[str]:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio"
    if header.startswith(b"ID3"):
        return "audio"
    if header.startswith(b"fLaC"):
        return "audio"
    lowered = header.lower()
    if b"metadata:image" in header or b"<svg" in lowered:
        return "image"
    stripped = header.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "text"
    return None


def load_json_payload(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return json.loads(content)
        except Exception:
            return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def classify_json_payload(data: Any) -> str:
    if isinstance(data, dict):
        keys = set(data.keys())
        if keys & JSON_IMAGE_KEYS:
            return "image"
        if keys & JSON_AUDIO_KEYS:
            return "audio"
        if keys & JSON_TEXT_KEYS:
            return "text"
        url = data.get("url") if isinstance(data.get("url"), str) else None
        if url and is_image_url(url):
            return "image"
        if any(isinstance(v, str) and v.strip() for v in data.values()):
            return "text"
        return "text"

    if isinstance(data, list) and data:
        votes: Counter = Counter()
        sample = data[: min(10, len(data))]
        for item in sample:
            if isinstance(item, dict):
                keys = set(item.keys())
                if keys & JSON_IMAGE_KEYS:
                    votes["image"] += 1
                if keys & JSON_AUDIO_KEYS:
                    votes["audio"] += 1
                if keys & JSON_TEXT_KEYS:
                    votes["text"] += 1
                if "url" in item and isinstance(item["url"], str) and is_image_url(item["url"]):
                    votes["image"] += 1
            elif isinstance(item, str) and item.strip():
                votes["text"] += 1
        if votes:
            label, count = votes.most_common(1)[0]
            if count > 0:
                return label
        return "text"

    if isinstance(data, str) and data.strip():
        return "text"

    return "text"


def classify_json_file(file_path: str) -> Optional[str]:
    payload = load_json_payload(file_path)
    if payload is None:
        return None
    return classify_json_payload(payload)


def determine_category(file_path: str) -> str:
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return "error"

    if size == 0:
        return "unknown"

    suffix = Path(file_path).suffix.lower()
    header = read_header(file_path)
    magic = sniff_magic(header)

    if suffix == ".json":
        label = classify_json_file(file_path)
        if label:
            return label
        if magic:
            if magic == "text":
                return "text"
            return magic
        if is_mostly_printable(header):
            return "text"
        return "unknown"

    if suffix in IMAGE_EXTS:
        if magic:
            if magic == "text":
                label = classify_json_file(file_path)
                if label:
                    return label
                return "text"
            return magic
        return "image"

    if suffix in AUDIO_EXTS:
        if magic == "audio":
            return "audio"
        if magic == "image":
            return "image"
        if magic == "text":
            label = classify_json_file(file_path)
            if label:
                return label
            if is_mostly_printable(header):
                return "text"
            return "unknown"
        if not is_mostly_printable(header):
            return "audio"
        label = classify_json_file(file_path)
        if label:
            return label
        return "text"

    if suffix in TEXT_EXTS:
        if magic and magic != "text":
            return magic
        if is_mostly_printable(header):
            return "text"
        label = classify_json_file(file_path)
        if label:
            return label
        return "text"

    if magic:
        if magic == "text":
            label = classify_json_file(file_path)
            if label:
                return label
            return "text"
        return magic

    if is_mostly_printable(header):
        label = classify_json_file(file_path)
        if label:
            return label
        return "text"

    return "unknown"


def sorter(
    files,
    *,
    eval_mode: bool = False,
    prediction_path: Optional[str] = None,
    input_root: Optional[str] = None,
    move_base_dir: Optional[str] = None,
    collect_only: bool = False,
):
    """根据文件内容将其分类为 audio/image/text。

    Args:
        files: 可迭代的文件路径集合。
        eval_mode: 评估模式，仅统计不移动文件。
        prediction_path: 可选，输出预测 CSV。
        input_root: 生成相对路径时使用的根目录。
        move_base_dir: 若提供，则把文件移动到指定目录下的对应模态 dataset/ 中。
        collect_only: 为 True 时不执行移动，只收集分类结果并返回。

    Returns:
        dict: 包含统计信息、分类结果（categorized）与未支持类别（unknown）。
    """

    start_time = time.time()
    success_count = 0
    fail_count = 0
    categorized: Dict[str, List[str]] = {"image": [], "audio": [], "text": []}
    other_categories: Dict[str, List[str]] = {}
    details: List[Dict[str, Any]] = []

    prediction_writer = None
    prediction_file = None

    if prediction_path:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_file = prediction_path.open("w", newline="", encoding="utf-8")
        prediction_writer = csv.writer(prediction_file)
        prediction_writer.writerow(["filename", "predicted_label"])

    def resolve_relative(path: str) -> str:
        if input_root:
            try:
                return os.path.relpath(path, input_root).replace("\\", "/")
            except ValueError:
                pass
        return path.replace("\\", "/")

    def write_prediction(path: str, label: str) -> None:
        if prediction_writer is None:
            return
        prediction_writer.writerow([resolve_relative(path), label])

    iterator = tqdm(files, desc="分类进度") if not eval_mode else files

    for file_path in iterator:
        try:
            category = determine_category(file_path)
            suffix = Path(file_path).suffix.lower()
            if category == "image" and suffix not in STRICT_IMAGE_EXTS:
                category = "unknown"
            if category == "audio" and suffix not in STRICT_AUDIO_EXTS:
                category = "unknown"
        except Exception as exc:
            print(f"处理文件 {file_path} 时出错: {exc}")
            fail_count += 1
            write_prediction(file_path, "error")
            details.append(
                {
                    "source_path": os.path.abspath(file_path),
                    "relative_path": resolve_relative(file_path),
                    "category": "error",
                    "status": "error",
                    "target_path": None,
                    "reason": str(exc),
                }
            )
            continue

        write_prediction(file_path, category)
        is_supported = category in {"audio", "image", "text"}
        abs_path = os.path.abspath(file_path)
        detail_entry: Dict[str, Any] = {
            "source_path": abs_path,
            "relative_path": resolve_relative(file_path),
            "category": category,
            "status": None,
            "target_path": None,
            "reason": None,
        }

        if eval_mode:
            if is_supported:
                success_count += 1
                detail_entry["status"] = "evaluated"
            else:
                fail_count += 1
                detail_entry["status"] = "unsupported"
                detail_entry["reason"] = "unsupported_category"
            details.append(detail_entry)
            continue

        if is_supported:
            categorized[category].append(abs_path)
            if not collect_only:
                moved_path = put_file_in_category(file_path, category, base_dir=move_base_dir)
                if moved_path:
                    success_count += 1
                    detail_entry["status"] = "moved"
                    detail_entry["target_path"] = moved_path
                else:
                    fail_count += 1
                    detail_entry["status"] = "move_failed"
                    detail_entry["reason"] = "move_failed"
            else:
                success_count += 1
                detail_entry["status"] = "collected"
        else:
            other_categories.setdefault(category, []).append(os.path.abspath(file_path))
            print(f"无法归类文件 {file_path}，预测为 {category}")
            fail_count += 1
            detail_entry["status"] = "unknown"
            detail_entry["reason"] = category

        details.append(detail_entry)

    if prediction_file:
        prediction_file.close()

    if not eval_mode and hasattr(iterator, "close"):
        iterator.close()

    elapsed = time.time() - start_time
    print("\n分类完成")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败/跳过: {fail_count} 个文件")
    print(f"总耗时: {elapsed:.2f} 秒")

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "elapsed_seconds": elapsed,
        "categorized": categorized,
        "unknown": other_categories,
        "details": details,
        "prediction_file": str(prediction_path.resolve()) if prediction_path else None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对 mix_dataset 进行分类或评估")
    parser.add_argument("--input", default="./mix_dataset", help="要分类的输入目录")
    parser.add_argument("--eval", action="store_true", help="启用评估模式，不移动文件，只输出预测")
    parser.add_argument("--predictions", help="评估模式下保存预测结果的 CSV 路径")
    parser.add_argument("--input-root", help="预测路径的相对根目录，默认等于 --input")
    parser.add_argument("--move-base", help="实际移动文件时的目标根目录")

    args = parser.parse_args()

    input_directory = args.input
    files = read_files_from_directory(input_directory)
    print(f"找到 {len(files)} 个文件")

    if not files:
        print("没有找到文件")
        raise SystemExit(0)

    print("前10个文件:")
    for i, f in enumerate(files[:10]):
        print(f"  {i+1}. {f}")

    prediction_path = args.predictions
    if args.eval and not prediction_path:
        prediction_path = "predictions.csv"

    input_root = args.input_root or input_directory

    result = sorter(
        files,
        eval_mode=args.eval,
        prediction_path=prediction_path,
        input_root=input_root,
        move_base_dir=args.move_base,
        collect_only=False,
    )

    if args.eval:
        print("评估模式已完成")
    else:
        print("分类操作完成")

    if prediction_path:
        print(f"预测结果已保存到: {Path(prediction_path).resolve()}")

    print(
        f"统计: success={result['success_count']} fail={result['fail_count']} "
        f"elapsed={result['elapsed_seconds']:.2f}s"
    )