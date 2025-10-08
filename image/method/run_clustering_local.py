import yaml
import random
import numpy as np
import logging
import os
import time
import sys
from clustering import compute_centroids
from tqdm import tqdm

# 设置环境变量以避免OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- 把 load_embeddings_safe 提前定义，保证后续可用 ---
def load_embeddings_safe(emb_path: str, emb_size: int, expected_N: int = None):
    """安全加载 embeddings：优先 np.load(..., mmap_mode='r')，失败则按 raw float32 memmap"""
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"嵌入文件不存在: {emb_path}")

    # 优先使用 np.load mmap_mode='r'（不会把全部载入内存）
    try:
        arr = np.load(emb_path, mmap_mode='r')
        logger.info(f"np.load 成功, shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")
        if arr.ndim != 2 or arr.shape[1] != emb_size:
            logger.warning(f"加载的 shape 与期望 emb_size 不符: {arr.shape} vs emb_size={emb_size}")
        return arr
    except Exception as e:
        logger.warning(f"np.load 失败（可能不是标准 .npy），尝试 raw 二进制读取: {e}")

    # raw 二进制尝试（假设 float32）
    filesize = os.path.getsize(emb_path)
    bytes_per_vec = 4 * emb_size  # float32
    if filesize % bytes_per_vec != 0:
        raise RuntimeError(f"文件大小({filesize}) 不能被 emb_size({emb_size}) 整除，无法按 float32 reshape。")

    inferred_N = filesize // bytes_per_vec
    if expected_N and expected_N != inferred_N:
        logger.warning(f"配置 dataset_size={expected_N} 与文件推断 N={inferred_N} 不一致，采用推断值。")
    logger.info(f"按 float32 memmap 加载: shape=({inferred_N},{emb_size})")
    mem = np.memmap(emb_path, dtype=np.float32, mode='r', shape=(inferred_N, emb_size))
    return mem

# 配置文件路径 - 修复路径问题
possible_config_paths = [
    os.path.join(current_dir, "configs", "openclip", "clustering_configs.yaml"),
    os.path.join(current_dir, "clustering", "configs", "openclip", "clustering_configs.yaml"),
    "configs/openclip/clustering_configs.yaml",
    "clustering/configs/openclip/clustering_configs.yaml"
]

config_file = None
for path in possible_config_paths:
    if os.path.exists(path):
        config_file = path
        break

if config_file is None:
    logger.error("找不到配置文件 clustering_configs.yaml")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"尝试查找的路径: {possible_config_paths}")
    
    # 尝试创建默认配置
    default_config_dir = os.path.join(current_dir, "configs", "openclip")
    os.makedirs(default_config_dir, exist_ok=True)
    config_file = os.path.join(default_config_dir, "clustering_configs.yaml")
    
    # 创建默认配置文件
    default_config = {
        'seed': 42,
        'emb_memory_loc': os.path.join(current_dir, 'embeddings', 'image_embeddings.npy'),
        'paths_memory_loc': os.path.join(current_dir, 'embeddings', 'image_paths.npy'),
        'dataset_size': 10000,  # 这个需要根据实际情况调整
        'emb_size': 512,
        'path_str_dtype': 'U200',
        'ncentroids': 1000,
        'niter': 20,
        'Kmeans_with_cos_dist': True,
        'save_folder': os.path.join(current_dir, 'clustering_results'),
        'sorted_clusters_file_loc': os.path.join(current_dir, 'sorted_clusters')
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logger.info(f"创建了默认配置文件: {config_file}")
    logger.warning("请检查并修改配置文件中的参数，特别是dataset_size")

logger.info(f"加载配置文件: {config_file}")

# 加载聚类参数
with open(config_file, 'r', encoding='utf-8') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

# 设置随机种子
SEED = params['seed']
random.seed(SEED)
np.random.seed(SEED)

# 打印配置信息
logger.info(f"聚类参数: {params}")

# 获取路径和大小参数
emb_memory_loc = params['emb_memory_loc']
paths_memory_loc = params['paths_memory_loc']
dataset_size = params['dataset_size']
emb_size = params['emb_size']
path_str_type = params.get('path_str_dtype', 'S24')  # 兼容不同命名

# 检查文件是否存在
if not os.path.exists(emb_memory_loc):
    logger.error(f"嵌入向量文件不存在: {emb_memory_loc}")
    exit(1)

# 加载嵌入向量（使用安全加载，不会一次性占用大量内存）
logger.info(f"加载嵌入向量: {emb_memory_loc}")
try:
    embeddings = load_embeddings_safe(emb_memory_loc, emb_size, dataset_size)
except Exception as e:
    logger.error(f"加载嵌入向量失败: {e}")
    exit(1)

# 根据实际 embeddings 推断 dataset 大小并更新 params
actual_N = embeddings.shape[0]
if dataset_size != actual_N:
    logger.warning(f"配置的 dataset_size={dataset_size} 与文件实际大小 {actual_N} 不一致，已更新为实际值。")
    dataset_size = actual_N
    params['dataset_size'] = actual_N

# 加载 paths（按实际 N 尝试 memmap，若失败尝试 np.load）
paths_memory = None
if os.path.exists(paths_memory_loc):
    try:
        # 尝试用 memmap 按推断的长度读取路径数组
        paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))
        logger.info(f"成功加载 paths，数量: {paths_memory.shape[0]}")
    except Exception as e:
        logger.warning(f"paths memmap 加载失败: {e}，尝试 np.load")
        try:
            paths_arr = np.load(paths_memory_loc, mmap_mode='r')
            # 若长度与 dataset_size 不符，截断或扩展警告
            if paths_arr.shape[0] != dataset_size:
                logger.warning(f"paths 长度 {paths_arr.shape[0]} 与 dataset_size {dataset_size} 不一致")
            paths_memory = paths_arr
            logger.info("使用 np.load 成功加载 paths")
        except Exception as e2:
            logger.error(f"无法加载 paths 文件: {e2}")
            paths_memory = None
else:
    logger.warning(f"未找到 paths 文件: {paths_memory_loc}")

logger.info(f"嵌入向量形状: {embeddings.shape}")

# 确保保存目录存在
save_folder = params['save_folder']
sorted_clusters_folder = params['sorted_clusters_file_loc']
os.makedirs(save_folder, exist_ok=True)
os.makedirs(sorted_clusters_folder, exist_ok=True)
logger.info(f"结果将保存到: {save_folder}")

# 修改compute_centroids函数，添加进度显示
from clustering import compute_centroids as original_compute_centroids

def compute_centroids_with_progress(data, ncentroids, niter, seed, Kmeans_with_cos_dist, save_folder, logger, verbose):
    """添加进度条的compute_centroids函数包装器"""
    logger.info(f"开始聚类: {ncentroids}个聚类, {niter}次迭代")
    
    # 创建进度条
    progress_bar = tqdm(total=niter, desc="K-means迭代", unit="iter")
    
    # 保存原始的logger.info
    original_info = logger.info
    
    # 修改logger.info以检测进度信息
    def new_info(msg):
        original_info(msg)
        if "Iteration" in msg:
            try:
                iter_num = int(msg.split("Iteration")[1].split("/")[0].strip())
                progress_bar.update(1)
                progress_bar.set_description(f"K-means迭代 {iter_num}/{niter}")
            except:
                pass
    
    # 替换logger.info
    logger.info = new_info
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 调用原始函数
        result = original_compute_centroids(data, ncentroids, niter, seed, 
                                           Kmeans_with_cos_dist, save_folder, 
                                           logger, verbose)
        
        # 设置进度条为完成
        progress_bar.update(niter)
        progress_bar.close()
        
        # 记录总时间
        total_time = time.time() - start_time
        logger.info(f"聚类完成! 用时: {total_time:.2f}秒")
        
        return result
    except Exception as e:
        progress_bar.close()
        logger.error(f"聚类过程中出错: {e}")
        raise
    finally:
        # 恢复原始logger.info
        logger.info = original_info

# 运行带进度条的聚类
logger.info("开始聚类过程...")
try:
    compute_centroids_with_progress(
        data=embeddings,
        ncentroids=params['ncentroids'],
        niter=params['niter'],
        seed=params['seed'],
        Kmeans_with_cos_dist=params['Kmeans_with_cos_dist'],
        save_folder=params['save_folder'],
        logger=logger,
        verbose=True,
    )
    logger.info("聚类完成！结果已保存到指定目录。")
    
    # 检查结果文件
    expected_files = [
        os.path.join(save_folder, "centroids.npy"),
        os.path.join(save_folder, "assignments.npy"),
        os.path.join(save_folder, "kmeans.index")
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            logger.info(f"已生成文件: {file_path} ({file_size:.2f} MB)")
        else:
            logger.warning(f"未找到预期文件: {file_path}")
    
except KeyboardInterrupt:
    logger.info("用户中断了聚类过程")
    sys.exit(1)
except Exception as e:
    logger.error(f"聚类过程失败: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

