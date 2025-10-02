import yaml
import numpy as np
import logging
import os
import sys
from sort_clusters import assign_and_sort_clusters

# 添加项目根目录到 Python 路径，确保模块导入正确
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 配置文件路径
config_file = "configs/openclip/clustering_configs.yaml"
logger.info(f"加载配置文件: {config_file}")

# 检查文件是否存在
if not os.path.exists(config_file):
    logger.error(f"配置文件不存在: {config_file}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    available_files = os.listdir("clustering/configs/openclip") if os.path.exists("clustering/configs/openclip") else []
    logger.info(f"可用配置文件: {available_files}")
    sys.exit(1)

# 加载聚类参数
with open(config_file, 'r',encoding='utf-8') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

# 获取路径和大小参数
emb_memory_loc = params['emb_memory_loc']
paths_memory_loc = params['paths_memory_loc']
dataset_size = params['dataset_size']
emb_size = params['emb_size']
path_str_type = params.get('path_str_dtype', 'S24')  # 兼容不同命名

# 检查文件是否存在
if not os.path.exists(emb_memory_loc):
    logger.error(f"嵌入向量文件不存在: {emb_memory_loc}")
    sys.exit(1)

# 加载嵌入向量和路径
logger.info(f"加载嵌入向量: {emb_memory_loc}")
try:
    emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
    paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))
    logger.info(f"成功加载，嵌入向量形状: {emb_memory.shape}")
except Exception as e:
    logger.error(f"加载数据时出错: {e}")
    sys.exit(1)

# 确保保存目录存在
save_folder = params['save_folder']
sorted_clusters_folder = params['sorted_clusters_file_loc']
os.makedirs(save_folder, exist_ok=True)
os.makedirs(sorted_clusters_folder, exist_ok=True)
logger.info(f"结果将保存到: {sorted_clusters_folder}")

# 执行排序
try:
    logger.info("开始执行簇排序...")
    
    # 调用函数进行排序
    assign_and_sort_clusters(
        data=emb_memory,
        paths_list=paths_memory,
        sim_metric=params["sim_metric"],
        keep_hard=params["keep_hard"],
        kmeans_with_cos_dist=params["Kmeans_with_cos_dist"],
        save_folder=params["save_folder"],
        sorted_clusters_file_loc=params["sorted_clusters_file_loc"],
        cluster_ids=range(0, params["ncentroids"]),
        logger=logger,
    )
    
    logger.info("排序过程成功完成!")
    
except Exception as e:
    logger.error(f"排序过程失败: {e}")
    import traceback
    logger.error(traceback.format_exc())