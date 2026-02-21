import glob, pandas as pd

files = sorted(glob.glob(r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_text\part_*.csv"))
labels = set()
rows = 0
for fp in files:
    try:
        for chunk in pd.read_csv(fp, usecols=["label"], chunksize=100000, dtype=str, on_bad_lines='skip'):
            labels.update(chunk['label'].dropna().astype(str).str.strip().unique().tolist())
            rows += len(chunk)
    except Exception as e:
        print("ERR reading", fp, e)

print('files:', len(files))
print('rows processed (approx):', rows)
print('unique labels count:', len(labels))
# try numeric range
nums = [int(x) for x in labels if x.isdigit()]
if nums:
    print('label min, max:', min(nums), max(nums))
else:
    print('no purely numeric labels found')
