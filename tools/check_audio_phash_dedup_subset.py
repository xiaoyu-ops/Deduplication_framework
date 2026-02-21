from pathlib import Path
import sys

ROOT = Path(r"D:\Deduplication_framework")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.method.pipeline_api import _deduplicate_by_phash

paths = [
    Path(r"D:\Deduplication_framework\test_dataset\normal\audio\r01_1-100210-A-36.wav"),
    Path(r"D:\Deduplication_framework\test_dataset\normal\audio\r02_1-100210-A-36.wav"),
    Path(r"D:\Deduplication_framework\test_dataset\normal\audio\r01_1-100210-B-36.wav"),
    Path(r"D:\Deduplication_framework\test_dataset\normal\audio\r02_1-100210-B-36.wav"),
]

result = _deduplicate_by_phash(paths)
print("keepers", len(result["keepers"]))
print("duplicate_count", result["duplicate_count"])
for item in result["duplicates"]:
    print(item)
