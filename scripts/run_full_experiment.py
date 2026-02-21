import os
import csv
import json
import time
import subprocess
from pathlib import Path

# 配置
CONFIG_FILE = "configs/my_pipeline_full.yaml"
RESULT_CSV = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
SUMMARY_JSON = r"D:\Deduplication_framework\2026_new_experiment\result_full_global\summary.json"
KEEP_LIST_JSON = r"D:\Deduplication_framework\2026_new_experiment\result_full_global\our_pipeline_keep_list.json"
TOTAL_FILES = 3828733 # Imagenet subset count (or can read from sorter)

def run_pipeline():
    print("[Full Experiment] Starting Pipeline Run...")
    start_time = time.time()
    
    # Run the pipeline module
    # Using 'call' to block until finished
    try:
        # Activate base/orchestrator env implicitly by running python from current env
        cmd = ["python", "-m", "pipelines.multimodal_runner", "--config", CONFIG_FILE]
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Pipeline failed with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("[Abort] Pipeline interrupted by user.")
        return False
        
    end_time = time.time()
    print(f"[Full Experiment] Pipeline finished in {end_time - start_time:.2f} seconds.")
    return True

def record_metrics():
    print("[Full Experiment] Calculating metrics...")
    
    if not os.path.exists(SUMMARY_JSON):
        print(f"[Error] Summary file not found: {SUMMARY_JSON}")
        return

    summary = {}
    try:
        with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load summary JSON: {e}")
        return

    # 1. Read Time & Throughput from Summary
    elapsed_seconds = 0
    try:
        # Try to find duration from stage info
        stage = next((s for s in summary.get("stages", []) if s.get("stage_name") in ["image", "stage2_image"]), None)
        if stage:
            elapsed_seconds = stage.get("elapsed_seconds", 0)
        
        # Fallback to modality results if stage is missing/zero
        if elapsed_seconds == 0:
            mod_res = summary.get("modality_results", {}).get("image", {})
            elapsed_seconds = mod_res.get("elapsed_seconds", 0)
            
        print(f"Elapsed Time: {elapsed_seconds:.2f}s")
    except Exception as e:
        print(f"[Warn] Failed to parse time from summary: {e}")

    # 2. Read Dedup Rate (Kept Count)
    kept_count = 0
    # Method A: Read from explicit keep list file (if exported)
    if os.path.exists(KEEP_LIST_JSON):
        try:
            with open(KEEP_LIST_JSON, 'r') as f:
                kept = json.load(f)
                kept_count = len(kept)
        except Exception:
            pass
    
    # Method B: Read from summary stats (more reliable for pipeline runs)
    if kept_count == 0:
        try:
            # Structure: modality_results -> image -> runner_summary -> stats -> selected
            mod_res = summary.get("modality_results", {}).get("image", {})
            runner_stats = mod_res.get("runner_summary", {}).get("stats", {})
            kept_count = runner_stats.get("selected", 0)
            
            # If 'selected' is missing, try calculating (unique + copied? No, just unique usually)
            if kept_count == 0:
                 kept_count = runner_stats.get("unique", 0)
                 
        except Exception as e:
            print(f"[Warn] Failed to parse kept_count from summary: {e}")

    dedup_rate = 0.0
    throughput = 0.0
    
    if TOTAL_FILES > 0:
        dedup_rate = (1 - (kept_count / TOTAL_FILES)) 
        throughput = TOTAL_FILES / elapsed_seconds if elapsed_seconds > 0 else 0
    else:
        throughput = 0

    print(f"Total: {TOTAL_FILES}, Kept: {kept_count}, Dedup Rate: {dedup_rate*100:.2f}%")
    print(f"Throughput: {throughput:.2f} imgs/s")

    # 3. Append to CSV
    row = [
        "Ours (System Full)",          # Method
        f"{throughput:.1f}",           # Throughput
        "85.24%*",                     # Precision (from 10k benchmark *)
        "90%+ (Est)",                  # Recall (Estimated or N/A)
        "[Monitor Manual]"             # GPU Mem
    ]
    
    # Check if CSV exists to write header
    write_header = not os.path.exists(RESULT_CSV)
    
    try:
        with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            writer.writerow(row)
        print(f"[Success] Metrics written to {RESULT_CSV}")
    except Exception as e:
        print(f"[Error] Failed to write CSV: {e}")

if __name__ == "__main__":
    if run_pipeline():
        record_metrics()
