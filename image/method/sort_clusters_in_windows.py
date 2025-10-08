import os
import yaml
import logging
import numpy as np
from sort_clusters import assign_and_sort_clusters

# 修正路径字符串，避免 SyntaxWarning
config_file = r"D:\桌面\Deduplication_framework\image\method\configs\openclip\clustering_configs.yaml"

# 日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def load_embeddings_safe(emb_path: str, emb_size: int):
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"嵌入文件不存在: {emb_path}")
    try:
        arr = np.load(emb_path, mmap_mode='r')
        logger.info(f"np.load 成功, shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")
        return arr
    except Exception as e:
        logger.warning(f"np.load 失败，尝试按 raw float32 memmap: {e}")
    # raw 二进制尝试（float32）
    filesize = os.path.getsize(emb_path)
    bytes_per_vec = 4 * emb_size
    if filesize % bytes_per_vec != 0:
        raise RuntimeError(f"文件大小({filesize}) 不能被 emb_size({emb_size}) 整除。")
    N = filesize // bytes_per_vec
    logger.info(f"使用 np.memmap 加载 raw float32, shape=({N},{emb_size})")
    return np.memmap(emb_path, dtype=np.float32, mode='r', shape=(N, emb_size))

# 加载配置
with open(config_file, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
logger.info(f"加载配置文件: {config_file}")

emb_path = cfg.get('emb_memory_loc')
paths_path = cfg.get('paths_memory_loc')
emb_size = int(cfg.get('emb_size', 512))

logger.info(f"加载嵌入向量: {emb_path}")
try:
    embeddings = load_embeddings_safe(emb_path, emb_size)
except Exception as e:
    logger.error(f"加载数据时出错: {e}")
    raise SystemExit(1)

# 加载 paths（尝试 mmap 或 np.load）
paths = None
if paths_path and os.path.exists(paths_path):
    try:
        paths = np.load(paths_path, mmap_mode='r')
        logger.info(f"成功加载 paths，数量: {len(paths)}")
    except Exception:
        try:
            path_bytes = np.memmap(paths_path, dtype=cfg.get('path_str_dtype','S256'), mode='r')
            paths = np.asarray(path_bytes).astype('U')  # 转为字符串视情况
            logger.info(f"使用 memmap 加载 paths，数量: {len(paths)}")
        except Exception as e:
            logger.warning(f"加载 paths 失败: {e}")
else:
    logger.warning(f"未找到 paths 文件: {paths_path}")

# 确保保存目录存在
save_folder = cfg['save_folder']
sorted_clusters_folder = cfg['sorted_clusters_file_loc']
os.makedirs(save_folder, exist_ok=True)
os.makedirs(sorted_clusters_folder, exist_ok=True)
logger.info(f"结果将保存到: {sorted_clusters_folder}")

# 执行排序
try:
    logger.info("开始执行簇排序...")
    
    # 调用函数进行排序
    assign_and_sort_clusters(
        data=embeddings,
        paths_list=paths,
        sim_metric=cfg["sim_metric"],
        keep_hard=cfg["keep_hard"],
        kmeans_with_cos_dist=cfg["Kmeans_with_cos_dist"],
        save_folder=cfg["save_folder"],
        sorted_clusters_file_loc=cfg["sorted_clusters_file_loc"],
        cluster_ids=range(0, cfg["ncentroids"]),
        logger=logger,
    )
    
    logger.info("排序过程成功完成!")
    
except Exception as e:
    logger.error(f"排序过程失败: {e}")
    import traceback
    logger.error(traceback.format_exc())