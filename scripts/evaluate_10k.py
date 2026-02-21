import os
import time
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------
# 0. 配置区域 (Adapted for 10k Test)
# -------------------------------------------------------------

PIPELINE_SUMMARY_FILE = Path(r"D:/Deduplication_framework/2026_new_experiment/result_10k/summary.json")
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"

# -------------------------------------------------------------
# 1. 辅助函数
# -------------------------------------------------------------

def parse_id(filename):
    """
    文件名解析逻辑，需与基线一致。
    例如 train-0_0_aug_noise.jpg -> train-0_0
    """
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result_csv(method, throughput, precision, recall, gpu_mem):
    """追加写入 CSV"""
    file_exists = os.path.isfile(RESULT_FILE)
    try:
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            
            tp_str = f"{throughput:.1f}" if throughput >= 0 else "N/A"
            mem_str = f"{gpu_mem:.2f}" if gpu_mem >= 0 else "0.00"
            
            # Use distinct name for this run
            writer.writerow([method, tp_str, f"{precision*100:.2f}%", f"{recall*100:.2f}%", mem_str])
        print(f"[成功] 结果已追加到 {RESULT_FILE}")
    except Exception as e:
        print(f"[错误] 写入CSV失败: {e}")

# -------------------------------------------------------------
# 2. 主逻辑
# -------------------------------------------------------------

def main():
    print(f"[My Pipeline 10k Evaluation] 开始评估...")

    if not PIPELINE_SUMMARY_FILE.exists():
        print(f"[错误] 找不到 summary 文件: {PIPELINE_SUMMARY_FILE}")
        return
    
    duplicates_map = {} 
    duration_total = 0.0
    processed_count = 0
    
    try:
        with open(PIPELINE_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        image_stage = None
        if "stages" in summary:
            for stage in summary["stages"]:
                if stage.get("stage_name") == "image": # Note: name might be just 'image' or 'stage2_image' depending on orchestrator
                    image_stage = stage
                    break
                if stage.get("stage_name") == "stage2_image":
                    image_stage = stage
                    break
        
        if not image_stage:
            print("[错误] summary.json 中未找到 image 阶段的信息。")
            # Try to print available stages
            print(f"Available stages: {[s.get('stage_name') for s in summary.get('stages', [])]}")
            return

        output_paths = image_stage.get("output_paths", {})
        dup_files = output_paths.get("duplicates", [])
        
        if not dup_files:
            print("[警告] image 阶段报告中没有 duplicates 文件输出。")
        
        print(f"[Info] 找到 {len(dup_files)} 个分片结果文件，开始合并...")

        for dup_file_path in dup_files:
            if not os.path.exists(dup_file_path):
                print(f"[警告] 找不到结果文件: {dup_file_path}，跳过。")
                continue
                
            with open(dup_file_path, 'r', encoding='utf-8') as df:
                items = json.load(df)
                for item in items:
                    keeper = item.get("original")
                    if not keeper: continue
                    dups_list = item.get("duplicates", []) # Changed to match new output structure: "duplicates" list
                    
                    if not dups_list and "copies" in item: # Backward compatibility just in case
                         dups_list = item["copies"]

                    if not dups_list: continue

                    if keeper not in duplicates_map:
                        duplicates_map[keeper] = []
                    
                    for dup_info in dups_list:
                        # item structure in Q-SemDeDup might be dict or just path?
                        # In code modification: 
                        # duplicates.append({ "original": ..., "duplicates": [ {"path": ...} ] })
                        if isinstance(dup_info, dict):
                            d_path = dup_info.get("path")
                        else:
                            d_path = str(dup_info)
                            
                        if d_path:
                            duplicates_map[keeper].append(d_path)

        runner_stats = image_stage.get("metadata", {}).get("runner_summary", {}).get("stats", {})
        processed_count = runner_stats.get("processed", 0)
        duration_total = runner_stats.get("elapsed_seconds", 0)
        if duration_total == 0:
             duration_total = image_stage.get("elapsed_seconds", 0)
        
    except Exception as e:
        print(f"[严重错误] 解析过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Pipeline 报告耗时: {duration_total:.2f}s, 处理文件: {processed_count}")
    print(f"发现重复组 (Clusters): {len(duplicates_map)} (含 Keeper)")

    print(f"正在读取 Pipeline 输入 Manifest 以建立 Ground Truth...")
    
    all_files_gt = []
    input_manifests = image_stage.get("output_paths", {}).get("manifests", [])
    
    if not input_manifests:
        print("[错误] 无法在 summary 中找到 input manifests 列表。")
        return

    for mani_path in input_manifests:
        if not os.path.exists(mani_path):
            print(f"[警告] 找不到 Manifest 文件: {mani_path}")
            continue
            
        with open(mani_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    fname = os.path.basename(line)
                    all_files_gt.append(fname)
    
    if not all_files_gt:
        print("[错误] 未能从 Manifest 中读取到任何文件。")
        return

    id_counts = defaultdict(int)
    for name in all_files_gt:
        fid = parse_id(name)
        id_counts[fid] += 1
        
    total_gt_pairs = 0
    for count in id_counts.values():
        if count > 1:
            total_gt_pairs += (count * (count - 1)) // 2
            
    print(f"Ground Truth: 总文件 {len(all_files_gt)}, 真实重复对 {total_gt_pairs}")

    tp = 0
    fp = 0
    
    for keeper_path, dup_paths in duplicates_map.items():
        cluster_files = [os.path.basename(keeper_path)] + [os.path.basename(d) for d in dup_paths]
        ids = [parse_id(f) for f in cluster_files]
        n = len(ids)
        if n < 2: continue
        
        for i in range(n):
            for j in range(i+1, n):
                if ids[i] == ids[j]:
                    tp += 1
                else:
                    fp += 1
    
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    throughput = (processed_count / duration_total) if duration_total > 0 else 0
    gpu_mem = 0.0 # Placeholder
    
    print("-" * 40)
    print(f"结果: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
    print(f"TP={tp}, FP={fp}, Total GT Pairs={total_gt_pairs}")
    print("-" * 40)

    log_result_csv("Ours (System)", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    main()
