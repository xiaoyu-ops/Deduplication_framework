# 在这个文件中实现批量去重的逻辑
# 需要注意的实现难点是原先我们针对阈值的控制是通过r和b来实现的
# 现在我们需要针对不同的阈值进行不同的r和b的设置
import numpy as np
import sys
import os
import math
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
import json


# 添加上级目录到Python路径，这样可以导入method_all模块
current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from caculate_dedup import load_similar_pairs,build_similarity_groups,generate_dedup_report,execute_deduplication,extract_file_name, extract_local_file_info
# 暂时注释因为还用不到
try:
    from LSH_deal_with_photo import generate_minhash_signatures, minHash, count_bucket_collisions, verify_similarity, find_similar_items, save_similar_pairs_to_file
except ImportError:
    print("警告: 无法导入 method_all.dedup_audio。")


# 这里注意导入规则包名用 . 分隔，不能用 / 或 //
# 文件夹名不能包含连字符 -，应该用下划线 _
# 路径分隔符 / 或 \ 只能在文件路径字符串中使用，不能在import语句中使用

# 首先注意我们已经有了所有的音频文件的指纹数据
# 这些数据存储在 binary_array_dict.npy 文件中
# 所以我们现在需要思考一种方法可以根据目标阈值t计算出对应的r和b

def find_optimal_band_row(target_threshold, signature_length=200):
    """
    为给定目标阈值找到最优的(b, r)参数对
    
    参数:
        target_threshold: 目标相似度阈值 (0-1之间)
        signature_length: 签名向量的长度 (默认200，对应现有LSH实现)
        
    返回:
        最优的(b, r)元组和实际计算出的阈值
    """
    best_b, best_r = None, None
    min_diff = float('inf')  # 移除严格限制，找最接近的
    best_actual_threshold = 0
    
    # 遍历所有可能的b值（b必须是签名长度的因数）
    for b in range(1, signature_length + 1):
        if signature_length % b == 0:  # b必须是签名长度的因数
            r = signature_length // b
            
            # 允许r=1，因为低阈值需要这种配置
            if r < 1:  
                continue
                
            # 计算实际阈值: t ≈ (1/b)^(1/r)
            try:
                actual_threshold = (1.0 / b) ** (1.0 / r)
                diff = abs(actual_threshold - target_threshold)
                
                # 找到最接近目标阈值的组合
                if diff < min_diff:
                    min_diff = diff
                    best_b, best_r = b, r
                    best_actual_threshold = actual_threshold
            except (ZeroDivisionError, OverflowError):
                continue
    
    return (best_b, best_r), best_actual_threshold, min_diff

def load_config_json(config_path):
    """从 JSON 配置文件加载配置，出错时返回 None"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件未找到: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        return None

if __name__ == "__main__":

    # 以下是配置文件地址
    config_path = os.path.join(current_dir, "audio_config.json")
    data = load_config_json(config_path)
    
    # 测试示例 - 使用实际的LSH签名长度
    targets = data.get("processing", {}).get("target_thresholds", [0.7, 0.8, 0.9])
    signature_length = data.get("processing", {}).get("signature_length", 200) # 对应现有LSH实现 (b=20, r=10, 总长度=200)
    
    print(f"LSH签名长度: {signature_length}")
    print("目标阈值 -> 推荐参数(b, r) -> 实际阈值 -> 误差")
    print("-" * 50)
    
    # 存储结果的字典
    sign_results = {}
    
    for target in targets:
        (b, r), actual, error = find_optimal_band_row(target, signature_length)
        sign_results[target] = {"b": b, "r": r, "actual": actual, "error": error}
        
        if b is None or r is None:
            print(f"{target:.1f} -> 无有效(b, r)组合")
        else:
            print(f"{target:.1f} -> (b={b:2d}, r={r:2d}) -> {actual:.4f} -> 误差:{error:.4f}")
    
    # 测试一下results
    threshold_01_results = sign_results[0.7]
    print(f"results中0.7对应: {threshold_01_results}")# f-string中不能使用嵌套的方括号

    # 好的 现在我们有了不同阈值对应的b和r
    # 我们可以用一个循环来针对不同阈值调用去重函数

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "similar_pairs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 改回原来的设置，保存到 method 目录下的 dedup_results
    output_dir_dedup = os.path.join(os.path.dirname(__file__), "dedup_results")
    
    # 确保目录存在并有写权限
    try:
        os.makedirs(output_dir_dedup, exist_ok=True)
        # 测试写权限
        test_file = os.path.join(output_dir_dedup, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"去重结果将保存到: {output_dir_dedup}")
    except Exception as e:
        print(f"无法在指定目录创建文件: {e}")
        print("请检查目录权限或以管理员身份运行")
        sys.exit(1)

    #生成音频指纹数据


    # 加载音频指纹数据（只加载一次）
    binary_data_path = os.path.join(parent_dir, "binary_array_dict.npy")
    with open(binary_data_path, "rb") as array_file:
        binary_array_dict = np.load(array_file, allow_pickle=True).item()
    
    matrix_true = np.array(list(binary_array_dict.values())).T
    print(f"加载了 {len(binary_array_dict)} 个音频指纹，矩阵形状: {matrix_true.shape}")
    
    for threshold, params in tqdm(sign_results.items(), desc="批量相似对计算进度"):
        b = params["b"]
        r = params["r"]
        if b is None or r is None:
            print(f"跳过阈值 {threshold:.1f}，无有效(b, r)组合")
            continue
        
        print(f"\n针对阈值 {threshold:.1f} 使用参数 (b={b}, r={r}) 进行相似列统计:")
        
        # 调用LSH去重函数
        hashBuckets_true = minHash(matrix_true, b, r)
        similar_pairs = find_similar_items(hashBuckets_true, matrix_true,similarity_threshold=threshold)
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, f"threshold_{threshold:.1f}_dedup.txt")
        save_similar_pairs_to_file(similar_pairs, output_file)
        
        print(f"  找到 {len(similar_pairs)} 对相似音频，结果保存到: {output_file}")

    for threshold,_ in tqdm(sign_results.items(), desc="批量去重进度"):
        output_file = os.path.join(output_dir, f"threshold_{threshold:.1f}_dedup.txt")
        # 主程序执行
        similar_pairs = load_similar_pairs(output_file,threshold=threshold)
        remove_files = build_similarity_groups(similar_pairs)
        generate_dedup_report(remove_files)
        output_file_result = os.path.join(output_dir_dedup, f"threshold_{threshold:.1f}_dedup_result.txt")
        execute_deduplication(remove_files, output_dir=output_file_result)
        
        # 使用本地WAV文件目录
        local_wav_dir = os.path.join(parent_dir, "dataset")
        extract_local_file_info(local_wav_dir, output_file_result, remove_files)