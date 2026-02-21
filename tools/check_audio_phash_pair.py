import sys
from PIL import Image
import imagehash
import numpy as np
import librosa


def phash(path: str) -> str:
    y, sr = librosa.load(path, sr=16000, duration=4)
    if len(y) == 0:
        return "<empty>"
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    min_v, max_v = log_spec.min(), log_spec.max()
    if max_v - min_v > 0:
        img = Image.fromarray((255 * (log_spec - min_v) / (max_v - min_v)).astype("uint8"))
    else:
        img = Image.fromarray(np.zeros_like(log_spec, dtype="uint8"))
    return str(imagehash.phash(img))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(p, phash(p))
        except Exception as exc:
            print(p, "ERR", exc)
