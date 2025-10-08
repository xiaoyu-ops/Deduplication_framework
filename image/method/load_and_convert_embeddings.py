import os
import numpy as np
from numpy.lib.format import open_memmap

path = r"D:\桌面\Deduplication_Framework\embeddings\image_embeddings.npy"
assert os.path.exists(path), path

candidates = [
    ("float32", 6454, 512),  # 最可能
    ("float64", 3227, 512),
    ("float32", 12908, 256),
    ("float32", 25816, 128),
]

def try_memmap(dtype, N, D):
    try:
        m = np.memmap(path, dtype=dtype, mode='r', shape=(N, D))
        print(f"成功映射 dtype={dtype} shape=({N},{D})")
        # 显示前2个向量的前8个数以作目测判断
        print("first row (first 8):", np.array(m[0,:8]))
        print("second row (first 8):", np.array(m[1,:8]))
        return True
    except Exception as e:
        print(f"映射失败 dtype={dtype} shape=({N},{D}): {e}")
        return False

# 逐个尝试并停在第一个成功的
chosen = None
for dtype, N, D in candidates:
    if try_memmap(dtype, N, D):
        chosen = (dtype, N, D)
        break

if chosen is None:
    print("未找到合适的 (dtype, N, D)。请确认写入方式或提供生成脚本。")
    raise SystemExit(1)

dtype, N, D = chosen
# 可选：把 raw 转成标准 .npy（open_memmap），分块拷贝，避免内存峰值
out = os.path.join(os.path.dirname(path), "image_embeddings_converted.npy")
print("转换并保存为标准 .npy:", out)
src = np.memmap(path, dtype=dtype, mode='r', shape=(N, D))
dst = open_memmap(out, mode='w+', dtype=dtype, shape=(N, D))
chunk = 256
for i in range(0, N, chunk):
    j = min(N, i + chunk)
    dst[i:j] = src[i:j]
    dst.flush()
print("转换完成，输出文件:", out)