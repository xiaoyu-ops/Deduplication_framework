import os
import re
import random

ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated\train"
SAMPLE_LIMIT = 2000
CHECK_COUNT = 200

paths = []
for root, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(root, f))
    if len(paths) >= SAMPLE_LIMIT:
        break

print("sample_files", len(paths))
rx = re.compile(r"_(\d+)(?:_aug|\.)")

if not paths:
    print("no_files_found")
    raise SystemExit(0)

sample = random.sample(paths, min(CHECK_COUNT, len(paths)))
labels = [(p, rx.search(os.path.basename(p))) for p in sample]
miss = sum(1 for _, m in labels if m is None)
print("regex_miss", miss, "of", len(labels))
print("examples_miss", [os.path.basename(p) for p, m in labels if m is None][:10])
print("examples_hit", [(os.path.basename(p), m.group(1)) for p, m in labels if m is not None][:5])
