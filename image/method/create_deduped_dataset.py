import os
import argparse
import yaml
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="创建去重后的数据集")
    parser.add_argument('--config-file', required=True, help='配置文件路径')
    parser.add_argument('--eps', required=True, type=float, help='使用哪个epsilon值的结果')
    parser.add_argument('--output-folder', required=True, help='输出文件夹路径')
    return parser.parse_args()

def load_kept_indices(result_path):
    """加载保留的样本索引"""
    print(f"从 {result_path} 加载保留的样本索引...")
    with open(result_path, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    return indices

def create_deduped_dataset(original_dataset, kept_indices, output_folder):
    """创建去重后的数据集"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建图像文件夹
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"从 {len(kept_indices)} 个保留样本中创建去重数据集...")
    
    # 初始化新数据集的字典
    new_dataset_dict = {key: [] for key in original_dataset.features.keys()}
    
    # 处理每个保留的样本
    for idx in tqdm(kept_indices):
        # 获取原始样本
        sample = original_dataset[idx]
        
        # 对于图像数据集，保存图像文件
        if "image" in sample:
            # 生成图像文件名
            image_filename = f"image_{idx}.jpg"
            image_path = os.path.join(images_folder, image_filename)
            
            # 保存图像
            sample["image"].save(image_path)
            
            # 更新样本中的图像路径
            sample_dict = {key: sample[key] for key in sample}
            sample_dict["image_path"] = image_path
        else:
            # 直接复制样本
            sample_dict = {key: sample[key] for key in sample}
        
        # 添加到新数据集
        for key, value in sample_dict.items():
            if key in new_dataset_dict:
                new_dataset_dict[key].append(value)
    
    # 创建新的数据集
    deduped_dataset = Dataset.from_dict(new_dataset_dict)
    
    # 保存新数据集
    dataset_save_path = os.path.join(output_folder, "dataset")
    deduped_dataset.save_to_disk(dataset_save_path)
    
    print(f"去重数据集已保存到 {dataset_save_path}")
    print(f"去重后的数据集大小: {len(deduped_dataset)}")
    
    return deduped_dataset

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 构建结果文件路径
    eps_result_dir = os.path.join(config['save_folder'], f"eps_{args.eps}")
    kept_samples_path = os.path.join(eps_result_dir, "all_kept_samples.txt")
    
    # 检查结果文件是否存在
    if not os.path.exists(kept_samples_path):
        print(f"错误: 找不到保留样本列表文件: {kept_samples_path}")
        print("请先运行 simple_semdedup.py 生成去重结果")
        return
    
    # 加载保留的样本索引
    kept_indices = load_kept_indices(kept_samples_path)
    print(f"加载了 {len(kept_indices)} 个保留样本索引")
    
    # 加载原始数据集
    # 这里假设原始数据集与main.py中使用的是同一个
    print("加载原始数据集...")
    original_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    
    # 创建去重后的数据集
    create_deduped_dataset(original_dataset, kept_indices, args.output_folder)
    
    print("完成!")

if __name__ == "__main__":
    main()
