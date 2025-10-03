#此文件用于从csv文件中获取数据
import csv
import os
import pandas as pd
import json
from pathlib import Path

def get_data_from_csv(file_path,index_col='image-src',label = None):
    """从CSV文件中读取数据并返回索引列表"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV文件未找到: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if index_col not in df.columns:
        raise ValueError(f"CSV文件中缺少 {index_col} 列")
    else:
        print(f"CSV文件包含以下列: {df.columns.tolist()}")

    indices = df[index_col].tolist()
    print(f"从CSV文件中读取了 {len(indices)} 个索引")
    
    return indices

def save_to_json(data, output_path, format_type="list",label = None):
    """
    将数据保存为JSON格式
    
    Args:
        data: 要保存的数据(列表、字典或DataFrame)
        output_path: 输出文件路径
        format_type: 保存格式 ("list", "dict", "records", "individual_objects", "jsonlines")
    """
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type == "list":
                # 保存为简单列表
                json.dump(data, f, ensure_ascii=False, indent=2)
            elif format_type == "dict":
                # 保存为字典格式（原来的方式）
                json_data = {"url": data, "label": label}
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            elif format_type == "individual_objects":
                # 每个URL一个对象的数组
                json_data = [{"url": url, "label": label} for url in data]
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            elif format_type == "jsonlines":
                # JSON Lines格式，每行一个JSON对象
                for url in data:
                    json_obj = {"url": url, "label": label}
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            elif format_type == "records":
                # 如果data是DataFrame，保存为records格式
                if isinstance(data, pd.DataFrame):
                    json_data = data.to_dict('records')
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError("records格式需要DataFrame数据")
        
        print(f"数据已保存到: {output_path}")
        print(f"文件大小: {output_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

if __name__ == "__main__":
    # 测试函数
    df = pd.read_csv(r"D:\桌面\Deduplication_framework\zh-freepik-com-2025-09-28.csv")
    print(df.columns)
    indices = get_data_from_csv(r"D:\桌面\Deduplication_framework\zh-freepik-com-2025-09-28.csv", index_col='image-src',label='Dog')
    print(indices[:3]) 
    save_to_json(indices, r"D:\桌面\Deduplication_framework\mix_dataset\output_dog.json", format_type="individual_objects",label='Dog')
    df = pd.read_csv(r"D:\桌面\Deduplication_framework\zh-freepik-com-2025-09-29.csv")
    print(df.columns)
    indices = get_data_from_csv(r"D:\桌面\Deduplication_framework\zh-freepik-com-2025-09-29.csv", index_col='image-src',label='cat')
    print(indices[:3]) 
    save_to_json(indices, f"D:\桌面\Deduplication_framework\mix_dataset\output_cat.json", format_type="individual_objects",label='Cat')