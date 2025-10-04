# 此算法的主要目标就是把mix_dataset下面的所有文件全部一一分到对应audio、image、text下面的dataset中
# 先实现读取文件夹中所有数据 如果是json格式那我们先判定是什么类型再放到对应的dataset中
# 如果是wav、mp3、png、jpg等格式的文件我们就直接放到对应的dataset中
# 目前我们只实现json、wav、mp3、png、jpg的文件读取，在json中我们要判定是text、audio还是image 这里可以用简单的关键字判定

import os
import json
from pathlib import Path
from tqdm import tqdm
import time

def read_files_from_directory(directory):
    """读取目录下的所有文件"""
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def is_image_url(u: str):
    # 如果 url 字段中包含图片后缀（如 .jpg/.jpeg/.png/.gif），归为 image
    if not isinstance(u, str):
        return False
    u_low = u.lower()
    return any(ext in u_low for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])

def put_file_in_category(file_path, category, output_base_dir):
    """将文件移动到对应类别的文件夹中"""
    category_dir = os.path.join(output_base_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    dest_path = os.path.join(category_dir, os.path.basename(file_path))
    os.rename(file_path, dest_path)
    print(f"已将文件 {file_path} 移动到 {dest_path}")

def put_file_in_image_category(file_path, output_base_dir=r"D:\桌面\Deduplication_framework\image\dataset"):
    put_file_in_category(file_path, 'image', output_base_dir)

def put_file_in_audio_category(file_path, output_base_dir=r"D:\桌面\Deduplication_framework\audio\dataset"):
    put_file_in_category(file_path, 'audio', output_base_dir)

def put_file_in_text_category(file_path, output_base_dir=r"D:\桌面\Deduplication_framework\text\dataset"):
    put_file_in_category(file_path, 'text', output_base_dir)

def sorter(files):
        """根据文件类型将文件分类到 audio、image、text"""
        startst_time = time.time()
        for file_path in tqdm(files, desc="分类进度"):
            file_extension = Path(file_path).suffix.lower()
            if file_extension in ['.wav', '.mp3']:
                category = 'audio'
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                category = 'image'
            elif file_extension == '.json':
                # 读取json文件内容以判定类型
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list) and all('text' in item for item in data):
                            category = 'text'
                            put_file_in_text_category(file_path)
                        elif isinstance(data, list) and all('audio' in item for item in data):
                            category = 'audio'
                            put_file_in_audio_category(file_path)
                        elif isinstance(data, list) and all('image' in item for item in data):
                            category = 'image'
                            put_file_in_image_category(file_path)
                        elif isinstance(data, list) and all('url' in item for item in data):
                            
                            if any(is_image_url(item['url']) for item in data):
                                category = 'image'
                                put_file_in_image_category(file_path)
                            # 如果全部是字符串并且看起来像网页链接，则可能是 text/url 列表
                            elif all(isinstance(item['url'], str) and item['url'].lower().startswith('http') for item in data):
                                category = 'text'
                                put_file_in_text_category(file_path)
                            else:
                                print(f"无法判定JSON文件类型: {file_path}, 跳过...")
                                continue
                        else:
                            print(f"无法判定JSON文件类型: {file_path}, 跳过...")
                            continue
                    except json.JSONDecodeError:
                        print(f"无效的JSON文件: {file_path}, 跳过...")
                        continue
            else:
                print(f"不支持的文件类型: {file_path}, 跳过...")
                continue
            end_time = time.time()
            print(f"分类器工作结束，耗时 {end_time - startst_time:.2f} 秒")


if __name__ == "__main__":
    input_directory = "./mix_dataset"  # 输入目录
    # output_directory = "./sorted_dataset"  # 输出目录

    # 创建输出目录
    # os.makedirs(output_directory, exist_ok=True)

    # 读取所有文件
    files = read_files_from_directory(input_directory)

    print(files[:10])  # 打印前10个文件路径

    sorter(files)
    res = sorter(files)
    if res:
        print("分类完成！")


        # 创建类别目录
        # category_dir = os.path.join(output_directory, category)
        # os.makedirs(category_dir, exist_ok=True)

        # # 移动文件到对应类别目录
        # dest_path = os.path.join(category_dir, os.path.basename(file_path))
        # os.rename(file_path, dest_path)
        # print(f"已将文件 {file_path} 移动到 {dest_path}")