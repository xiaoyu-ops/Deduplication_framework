from PIL import Image
from pathlib import Path
p = r"mix_dataset\sample_example.png"
try:
    im = Image.open(p)
    im.verify()
    print("OK:", p)
except Exception as e:
    print("PIL ERROR:", e)
    with open(p,'rb') as f:
        head = f.read(256)
    print("HEAD:", head[:64].hex())
