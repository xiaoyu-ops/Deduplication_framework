import os
import numpy as np
import pandas as pd
import argparse
import yaml
from tqdm import tqdm
import torch
from pathlib import Path
import pickle
import time
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Windows版语义去重")
    parser.add_argument('--config-file', required=True, help='配置文件路径')
    parser.add_argument('--eps-list', required=True, help='epsilon值列表，用逗号分隔')
    parser.add_argument('--max-clusters', type=int, help='限制处理的最大簇数量')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子')
    parser.add_argument('--which-to-keep', type=str, default='hard', choices=['hard', 'random', 'easy'], 
                        help='保留哪些样本：hard(困难)、random(随机)或easy(容易)')
    parser.add_argument('--batch-size', type=int, default=30, help='批处理大小')
    return parser.parse_args()

def init_memmap_embs(embs_memory_loc, dataset_size, emd_size=512, dtype="float32"):
    """初始化内存映射的嵌入数组"""
    print(f"加载嵌入数据: {embs_memory_loc}")
    print(f"数据集大小: {dataset_size}, 嵌入维度: {emd_size}")
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs

def load_cluster_file(cluster_path):
    """加载簇文件，支持多种格式"""
    if not os.path.exists(cluster_path):
        return None
        
    try:
        if cluster_path.endswith('.npy'):
            return np.load(cluster_path)
        else:
            # 尝试作为文本文件加载
            with open(cluster_path, 'r') as f:
                lines = f.readlines()
            
            # 解析文本行，假设每行的格式是：idx(tab)path(tab)dist
            data = []
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    idx = int(parts[0])
                    path = parts[1]
                    dist = float(parts[2])
                    data.append((idx, path, dist))
                elif len(parts) >= 1:
                    idx = int(parts[0])
                    data.append((idx, "", 0.0))
            
            return np.array(data, dtype=object)
    except Exception as e:
        print(f"无法加载文件 {cluster_path}: {e}")
        return None

def semdedup(cluster, cluster_reps, device="cpu"):
    """计算簇内样本的最大相似度（原始SemDeDup核心算法）"""
    st = time.time()
    
    # 转移到指定设备
    cluster_reps = torch.tensor(cluster_reps, dtype=torch.float32).to(device)
    
    # 计算余弦相似度矩阵
    pair_w_sim_matrix = cluster_reps @ cluster_reps.T
    
    # 将对角线元素替换为零（忽略自相似性）
    pair_w_sim_matrix.fill_diagonal_(0.0)
    
    # 确保是方阵
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]
    
    # 获取上三角矩阵（避免重复计算）
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
    
    # 计算每个样本与其他样本的最大相似度
    M = torch.max(triu_sim_mat, dim=0)[0].cpu()
    
    print(f"计算相似度矩阵耗时: {time.time()-st:.2f}秒")
    
    return M

def process_cluster(cluster_id, config, embs, device):
    """处理单个簇，计算哪些样本应该被保留"""
    st = time.time()
    
    # 检查输出文件是否已存在
    df_file_loc = os.path.join(config['save_folder'], f"dataframes/cluster_{cluster_id}.pkl")
    if os.path.exists(df_file_loc):
        print(f"{df_file_loc} 已存在，跳过")
        return None
    
    # 加载簇数据
    cluster_paths = [
        os.path.join(config['sorted_clusters_path'], f"cluster_{cluster_id}.npy"),
        os.path.join(config['sorted_clusters_path'], f"sorted_cluster_{cluster_id}.txt")
    ]
    
    cluster_i = None
    for path in cluster_paths:
        if os.path.exists(path):
            cluster_i = load_cluster_file(path)
            if cluster_i is not None:
                print(f"加载簇文件: {path}")
                break
    
    if cluster_i is None:
        print(f"无法加载簇 {cluster_id} 的数据，跳过")
        return None
    
    # 获取簇大小
    cluster_size = len(cluster_i)
    print(f"簇 {cluster_id} 大小: {cluster_size}")
    
    # 处理只有一个样本的簇
    if cluster_size == 1:
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = [0]
        for eps in config['eps_list']:
            points_to_remove_df[f"eps={eps}"] = [False]
        
        # 保存结果
        os.makedirs(os.path.dirname(df_file_loc), exist_ok=True)
        with open(df_file_loc, "wb") as file:
            pickle.dump(points_to_remove_df, file)
        
        print(f"处理簇 {cluster_id} 完成")
        return points_to_remove_df
    
    # 决定保留哪些样本
    which_to_keep = config.get('which_to_keep', 'hard').lower()
    cluster_items_indices = list(range(cluster_size))
    
    if which_to_keep == "random":
        # 随机打乱，保留随机样本
        np.random.shuffle(cluster_items_indices)
        cluster_i = cluster_i[cluster_items_indices]
    elif which_to_keep == "easy":
        # 反转顺序，保留容易样本
        cluster_items_indices = cluster_items_indices[::-1]
        cluster_i = cluster_i[cluster_items_indices]
    
    # 获取簇中样本的索引
    if isinstance(cluster_i[0], tuple) or isinstance(cluster_i[0], list):
        # 如果是来自文本文件的数据
        cluster_ids = np.array([int(item[0]) for item in cluster_i])
    else:
        # 如果是来自.npy文件的数据
        cluster_ids = cluster_i[:, 1].astype("int32")
    
    # 获取嵌入
    try:
        cluster_reps = embs[cluster_ids]
    except Exception as e:
        print(f"获取嵌入失败: {e}")
        # 创建随机嵌入用于测试
        cluster_reps = np.random.rand(len(cluster_ids), config['emd_size']).astype(np.float32)
    
    # 计算相似度
    M = semdedup(cluster_i, cluster_reps, device)
    
    # 创建结果DataFrame
    points_to_remove_df = pd.DataFrame()
    points_to_remove_df["indices"] = cluster_items_indices
    
    # 对每个epsilon值确定要移除的点
    for eps in config['eps_list']:
        eps_points_to_remove = M > 1 - eps
        points_to_remove_df[f"eps={eps}"] = eps_points_to_remove
    
    # 保存结果
    os.makedirs(os.path.dirname(df_file_loc), exist_ok=True)
    with open(df_file_loc, "wb") as file:
        pickle.dump(points_to_remove_df, file)
    
    print(f"处理簇 {cluster_id} 完成，耗时: {time.time()-st:.2f}秒")
    return points_to_remove_df

def run_semdedup(config, args):
    """运行完整的SemDeDup算法（Windows兼容版本）"""
    print(f"配置信息: {config}")
    
    # 确保输出目录存在
    save_loc = config['save_folder']
    os.makedirs(os.path.join(save_loc, "dataframes"), exist_ok=True)
    
    # 对每个epsilon值创建目录
    for eps in config['eps_list']:
        os.makedirs(os.path.join(save_loc, f"eps_{eps}"), exist_ok=True)
    
    # 检查并清理不一致的结果文件
    print("检查现有结果文件的一致性...")
    dataframes_dir = os.path.join(save_loc, "dataframes")
    if os.path.exists(dataframes_dir):
        for filename in os.listdir(dataframes_dir):
            if filename.startswith("cluster_") and filename.endswith(".pkl"):
                filepath = os.path.join(dataframes_dir, filename)
                try:
                    with open(filepath, "rb") as f:
                        df = pickle.load(f)
                    
                    # 检查是否包含所有需要的epsilon列
                    expected_cols = [f"eps={eps}" for eps in config['eps_list']]
                    missing_cols = [col for col in expected_cols if col not in df.columns]
                    
                    if missing_cols:
                        print(f"删除不一致的结果文件: {filename} (缺少列: {missing_cols})")
                        os.remove(filepath)
                        
                except Exception as e:
                    print(f"删除损坏的结果文件: {filename} (错误: {e})")
                    os.remove(filepath)
    
    # 加载嵌入数据
    embs = init_memmap_embs(
        config['embs_memory_loc'], 
        config['dataset_size'], 
        config['emd_size']
    )
    
    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 限制处理的簇数量
    num_clusters = config['num_clusters']
    if args.max_clusters:
        num_clusters = min(args.max_clusters, num_clusters)
        print(f"限制处理簇数量为: {num_clusters}")
    
    # 批处理簇
    batch_size = args.batch_size
    total_batches = (num_clusters + batch_size - 1) // batch_size
    
    # 总计时开始
    total_start_time = time.time()
    
    # 按批次处理簇
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_clusters)
        
        print(f"处理簇 {batch_start} 到 {batch_end-1}")
        batch_start_time = time.time()
        
        # 处理这一批的每个簇
        for cluster_idx in tqdm(range(batch_start, batch_end)):
            process_cluster(cluster_idx, config, embs, device)
        
        print(f"批次 {batch_idx+1}/{total_batches} 完成，耗时: {time.time()-batch_start_time:.2f}秒")
    
    print(f"所有簇处理完成，总耗时: {(time.time()-total_start_time)/60:.2f}分钟")
    
    # 合并结果，为每个epsilon值创建保留样本列表
    print("合并结果...")
    
    for eps in config['eps_list']:
        print(f"处理 epsilon = {eps}")
        
        # 用于存储要保留的样本
        kept_samples = []
        
        # 处理每个簇的结果
        for cluster_idx in tqdm(range(num_clusters)):
            df_file_loc = os.path.join(save_loc, f"dataframes/cluster_{cluster_idx}.pkl")
            
            if not os.path.exists(df_file_loc):
                continue
            
            # 加载簇的处理结果
            with open(df_file_loc, "rb") as file:
                df = pickle.load(file)
            
            # 获取簇文件路径
            cluster_paths = [
                os.path.join(config['sorted_clusters_path'], f"cluster_{cluster_idx}.npy"),
                os.path.join(config['sorted_clusters_path'], f"sorted_cluster_{cluster_idx}.txt")
            ]
            
            cluster_i = None
            for path in cluster_paths:
                if os.path.exists(path):
                    cluster_i = load_cluster_file(path)
                    if cluster_i is not None:
                        break
            
            if cluster_i is None:
                continue
            
            # 检查DataFrame是否包含当前epsilon的列
            eps_col = f"eps={eps}"
            if eps_col not in df.columns:
                print(f"警告: 簇 {cluster_idx} 的结果中缺少列 '{eps_col}'，跳过该簇")
                continue
            
            # 获取簇中样本的索引
            if isinstance(cluster_i[0], tuple) or isinstance(cluster_i[0], list):
                # 如果是来自文本文件的数据
                cluster_ids = np.array([int(item[0]) for item in cluster_i])
            else:
                # 如果是来自.npy文件的数据
                cluster_ids = cluster_i[:, 1].astype("int32")
            
            # 检查DataFrame的indices列长度是否与cluster_ids匹配
            if len(df["indices"]) != len(cluster_ids):
                print(f"警告: 簇 {cluster_idx} 的DataFrame长度({len(df['indices'])})与簇大小({len(cluster_ids)})不匹配，跳过该簇")
                continue
            
            # 找出要保留的样本（不移除的）
            try:
                not_to_remove = ~df[eps_col].values
                indices_to_keep = df["indices"][not_to_remove].values
            except Exception as e:
                print(f"警告: 处理簇 {cluster_idx} 时出错: {e}，跳过该簇")
                continue
            
            # 转换为全局索引
            kept_global_indices = [cluster_ids[i] for i in indices_to_keep]
            kept_samples.extend(kept_global_indices)
            
            # 保存这个簇的结果
            with open(os.path.join(save_loc, f"eps_{eps}", f"kept_cluster_{cluster_idx}.txt"), 'w') as f:
                for idx in kept_global_indices:
                    f.write(f"{idx}\n")
        
        # 删除重复并排序
        kept_samples = sorted(set(kept_samples))
        
        # 保存所有保留的样本
        with open(os.path.join(save_loc, f"eps_{eps}", "all_kept_samples.txt"), 'w') as f:
            for idx in kept_samples:
                f.write(f"{idx}\n")
        
        # 创建结果统计
        print(f"epsilon={eps} 完成: 从 {config['dataset_size']} 个样本中保留了 {len(kept_samples)} 个样本 ({len(kept_samples)/config['dataset_size']*100:.2f}%)")
        
        # 保存结果统计
        df = pd.DataFrame({
            'epsilon': [eps],
            'total_samples': [config['dataset_size']],
            'kept_samples': [len(kept_samples)],
            'removed_samples': [config['dataset_size'] - len(kept_samples)],
            'kept_percentage': [len(kept_samples) / config['dataset_size'] * 100]
        })
        
        df.to_csv(os.path.join(save_loc, "dataframes", f"results_eps_{eps}.csv"), index=False)
    
    # 汇总所有epsilon值的结果
    results_files = [f for f in os.listdir(os.path.join(save_loc, "dataframes")) if f.startswith("results_eps_")]
    if results_files:
        all_results = pd.concat([pd.read_csv(os.path.join(save_loc, "dataframes", f)) for f in results_files])
        all_results.to_csv(os.path.join(save_loc, "dataframes", "all_results.csv"), index=False)
        print("所有结果已汇总到 all_results.csv")

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 解析epsilon列表
    eps_list = [float(eps) for eps in args.eps_list.split(',')]
    config['eps_list'] = eps_list
    
    # 设置保留策略
    config['which_to_keep'] = args.which_to_keep
    
    # 运行SemDeDup
    run_semdedup(config, args)

if __name__ == "__main__":
    main()