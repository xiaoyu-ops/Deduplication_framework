import os
import numpy as np
p = r"D:\Deduplication_framework\embeddings\image_embeddings_converted.npy"
print("exists:", os.path.exists(p), "size:", os.path.getsize(p) if os.path.exists(p) else None)
arr = np.load(p, mmap_mode='r')
print("shape:", arr.shape, "dtype:", arr.dtype)
# 查看前两行作为快速 sanity check
print("row0:", arr[0,:8])
print("row1:", arr[1,:8])