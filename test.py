import numpy as np
import os
import pickle
import pandas as pd
import glob
import sys

# 你的 embeddings 检查代码（保留）
path = r"D:\Deduplication_Framework\embeddings\image_embeddings.npy"
size = os.path.getsize(path)
print("bytes:", size)
# 常见 dtype 与候选维度
dtypes = [("float32",4), ("float64",8)]
common_dims = [64,128,256,384,512,768,1024]
for name, b in dtypes:
    elems = size // b
    print(f"\n尝试 dtype={name}, 每元素字节={b}, total_elems={elems}")
    for D in common_dims:
        if elems % D == 0:
            N = elems // D
            print(f"  可能 shape: (N={N}, D={D})")

# 查找 cluster 文件（更稳健）
clusters_dir = r"D:\Deduplication_framework\image\clustering\results\ncentroids_2000\sorted_clusters"
pattern = os.path.join(clusters_dir, "cluster_*.pkl")
files = glob.glob(pattern)
if not files:
    print(f"未找到任何 cluster_*.pkl。检查目录: {clusters_dir}")
    if os.path.exists(clusters_dir):
        print("目录存在，列出内容（前50项）：")
        print(os.listdir(clusters_dir)[:50])
    else:
        print("目录不存在，请确认 run_clustering_local.py 的 save_folder 配置或实际输出路径。")
    sys.exit(1)

p = files[0]
print("使用第一个簇文件:", p)
obj = pickle.load(open(p,'rb'))
print(type(obj))
if isinstance(obj, pd.DataFrame):
    print("columns:", obj.columns.tolist())
else:
    print("object keys/repr:", obj.keys() if hasattr(obj,'keys') else repr(obj))
