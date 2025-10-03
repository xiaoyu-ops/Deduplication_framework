# 下载文本数据集 ag_news (简化版)
import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset
import requests

def download_dataset(save_path="./mix_dataset/",dataset_name="ag_news",max_samples = 10000):
    """下载数据集并保存为JSON"""
    
    print(f"下载{dataset_name}数据集,最多{max_samples}条记录...")
    
    # 下载数据集
    dataset = load_dataset(dataset_name)
    # 限制样本数量
    if max_samples and max_samples < len(dataset['train']):
        # 只取前 max_samples 条
        train_data = dataset['train'].select(range(max_samples))
        print(f"原数据集有 {len(dataset['train'])} 条，我们的下载量为 {max_samples} 条")
    else:
        train_data = dataset['train']
        print(f"下载完整数据集: {len(train_data)} 条")

    train_df = pd.DataFrame(train_data)
    
    # 创建保存目录
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 转换为去重格式并保存
    data = []
    for idx, row in train_df.iterrows():
        data.append({
            "id": f"ag_news_{idx}",
            "text": row['text'],
            "label": row['label']
        })
    
    # 保存JSON文件
    output_file = save_path / "ag_news_train.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(data)} 条文本记录到: {output_file}")
    return data

def safe_save_audio_json(dataset, output_path):
    """安全地保存音频数据集为JSON"""
    
    print("开始处理音频数据...")
    
    # 手动处理每条记录
    audio_data = []
    error_count = 0
    
    for i, item in enumerate(dataset):
        try:
            # 清理每个字段
            cleaned_item = {}
            for key, value in item.items():
                if key == 'audio':
                    # 跳过音频二进制数据，只保留路径信息
                    if isinstance(value, dict) and 'path' in value:
                        cleaned_item['audio_path'] = value['path']
                    continue
                elif isinstance(value, str):
                    # 清理文本字段
                    cleaned_value = value.encode('utf-8', errors='ignore').decode('utf-8')
                    cleaned_item[key] = cleaned_value
                else:
                    cleaned_item[key] = value
            
            audio_data.append(cleaned_item)
            
            # 显示进度
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{len(dataset)} 条记录...")
                
        except Exception as e:
            error_count += 1
            print(f"跳过第 {i} 条记录，错误: {e}")
    
    # 保存为JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(audio_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功保存到 {output_path}")
        print(f"跳过 {error_count} 条问题记录")
        return audio_data
        
    except Exception as e:
        print(f"保存失败: {e}")
        return None

def test_audio_dataset(max_samples=10000):
    """下载音频数据集并安全保存"""
    
    try:
        ds = load_dataset("danavery/urbansound8K")
        print(ds)
        
        if max_samples and max_samples < len(ds['train']):
            print(f"原始数据集有 {len(ds['train'])} 条，我们的下载量为 {max_samples} 条")
            train_subset = ds['train'].select(range(max_samples))
        else:
            train_subset = ds['train']
            
        print(f"原训练集大小: {len(ds['train'])}")
        print(f"选择后大小: {len(train_subset)}")
        
        # 使用安全的保存方法
        output_path = "./mix_dataset/audio_10k.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        audio_data = safe_save_audio_json(train_subset, output_path)
        
        if audio_data:
            # 验证保存的数据
            print(f"\n前5条数据预览:")
            for i in range(min(5, len(audio_data))):
                print(f"记录 {i}: {audio_data[i]}")
            print("保存完成！")
        
        return audio_data
        
    except Exception as e:
        print(f"下载音频数据集失败: {e}")
        return None
   
if __name__ == "__main__":
    # data = download_dataset()
    # print(f"下载完成！共 {len(data)} 条记录")
    test_audio_dataset()
