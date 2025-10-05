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

# 加载嵌入向量和路径
logger.info(f"加载嵌入向量: {emb_memory_loc}")
try:
    emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
    paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))
    logger.info(f"成功加载，嵌入向量形状: {emb_memory.shape}")
except Exception as e:
    logger.error(f"加载数据时出错: {e}")
    exit(1)

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
        data=emb_memory,
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