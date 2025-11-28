from PIL import Image
from pathlib import Path
m = Path(r"artifacts\test_run\20251126-141241\stage2_image\input_manifest.txt")
lines = m.read_text(encoding="utf-8").splitlines()
bad=[]
for i,line in enumerate(lines, start=1):
    p=line.strip()
    try:
        Image.open(p).verify()
    except Exception as e:
        bad.append((i,p,str(e)))
print("TOTAL:", len(lines))
print("BAD COUNT:", len(bad))
for i,p,e in bad:
    print(f"{i}: BAD: {p} -> {e}")
