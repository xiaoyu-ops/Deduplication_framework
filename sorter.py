# 此算法的主要目标就是把mix_dataset下面的所有文件全部一一分到对应audio、image、text下面的dataset中
# 先实现读取文件夹中所有数据 如果是json格式那我们先判定是什么类型再放到对应的dataset中
# 如果是wav、mp3、png、jpg等格式的文件我们就直接放到对应的dataset中
# 目前我们只实现json、wav、mp3、png、jpg的文件读取，在json中我们要判定是text、audio还是image 这里可以用简单的关键字判定

import os
import json
import shutil
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

def safe_move_file(src_path, dst_path, max_retries=3):
    """安全移动文件，处理占用问题"""
    for attempt in range(max_retries):
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 如果目标文件已存在，生成新名称
            counter = 1
            original_dst = dst_path
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(original_dst)
                dst_path = f"{name}_{counter}{ext}"
                counter += 1
            
            # 尝试移动文件
            shutil.move(src_path, dst_path)
            print(f"已将文件 {src_path} 移动到 {dst_path}")
            return True
            
        except PermissionError as e:
            print(f"文件被占用，尝试 {attempt + 1}/{max_retries}: {src_path}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
            else:
                print(f"文件移动失败，可能被其他程序占用: {src_path}")
                print("请关闭可能占用该文件的程序如VSCode、记事本等")
                return False
        except Exception as e:
            print(f"移动文件时出现未知错误: {e}")
            return False
    
    return False

def put_file_in_category(file_path, category):
    """将文件移动到对应类别的文件夹中"""
    base_dir = r"D:\桌面\Deduplication_framework"
    category_dir = os.path.join(base_dir, category, "dataset")  # 修复路径重复问题
    
    filename = os.path.basename(file_path)
    dest_path = os.path.join(category_dir, filename)
    
    return safe_move_file(file_path, dest_path)

def sorter(files):
    """根据文件类型将文件分类到 audio、image、text"""
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for file_path in tqdm(files, desc="分类进度"):
        file_extension = Path(file_path).suffix.lower()
        category = None
        
        try:
            if file_extension in ['.wav', '.mp3']:
                category = 'audio'
                
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                category = 'image'
                
            elif file_extension == '.json':
                # 读取json文件内容以判定类型
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        if isinstance(data, list) and data:  # 确保列表不为空
                            # 检查前几个元素或所有元素（最多检查前10个避免性能问题）
                            sample_size = min(len(data), 10)
                            sample_items = data[:sample_size]
                            
                            # 统计各种字段出现的次数
                            text_count = 0
                            audio_count = 0 
                            image_count = 0
                            image_url_count = 0
                            
                            for item in sample_items:
                                if isinstance(item, dict):
                                    # 检查文本字段
                                    if any(key in item for key in ['text', 'content', 'title', 'article', 'sentence']):
                                        text_count += 1
                                    
                                    # 检查音频字段
                                    if any(key in item for key in ['audio', 'audio_url', 'audio_path', 'wav', 'mp3']):
                                        audio_count += 1
                                    
                                    # 检查图片字段
                                    if any(key in item for key in ['image', 'image_url', 'img', 'picture']):
                                        image_count += 1
                                    
                                    # 检查URL是否指向图片
                                    if 'url' in item and is_image_url(item.get('url', '')):
                                        image_url_count += 1
                            
                            # 根据统计结果判断类型（选择占比最高的）
                            total_checked = len(sample_items)
                            
                            if image_count > total_checked * 0.5 or image_url_count > total_checked * 0.5:
                                category = 'image'
                            elif audio_count > total_checked * 0.3:  # 音频字段阈值稍低
                                category = 'audio'  
                            elif text_count > total_checked * 0.3:  # 文本字段阈值稍低
                                category = 'text'
                            elif image_url_count > 0:  # 有任何图片URL就归类为image
                                category = 'image'
                            elif text_count > 0:  # 有任何文本字段就归类为text
                                category = 'text'
                            else:
                                # 如果都没有明确字段，根据URL特征判断
                                urls = [item.get('url', '') for item in sample_items if isinstance(item, dict) and 'url' in item]
                                if urls:
                                    if any(is_image_url(url) for url in urls):
                                        category = 'image'
                                    elif all(isinstance(url, str) and url.lower().startswith('http') for url in urls):
                                        category = 'text'  # 默认归为文本
                        
                        elif isinstance(data, dict):
                            # 如果是单个对象
                            if any(key in data for key in ['audio', 'audio_url', 'audio_path']):
                                category = 'audio'
                            elif any(key in data for key in ['image', 'image_url', 'img', 'picture']):
                                category = 'image'
                            elif any(key in data for key in ['text', 'content', 'title', 'article']):
                                category = 'text'
                            elif 'url' in data and is_image_url(data.get('url', '')):
                                category = 'image'
                        
                        if not category:
                            print(f"无法判定JSON文件类型: {file_path}, 跳过...")
                            fail_count += 1
                            continue
                            
                    except json.JSONDecodeError:
                        print(f"无效的JSON文件: {file_path}, 跳过...")
                        fail_count += 1
                        continue
            else:
                print(f"不支持的文件类型: {file_path}, 跳过...")
                fail_count += 1
                continue
            
            # 移动文件
            if category:
                if put_file_in_category(file_path, category):
                    success_count += 1
                else:
                    fail_count += 1
                    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            fail_count += 1
    
    end_time = time.time()
    print(f"\n分类完成")
    print(f"成功移动: {success_count} 个文件")
    print(f"失败/跳过: {fail_count} 个文件")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    return success_count > 0

if __name__ == "__main__":
    input_directory = "./mix_dataset"
    
    # 读取所有文件
    files = read_files_from_directory(input_directory)
    print(f"找到 {len(files)} 个文件")
    
    if files:
        print("前10个文件:")
        for i, f in enumerate(files[:10]):
            print(f"  {i+1}. {f}")
        
        # 开始分类
        result = sorter(files)
        if result:
            print("分类操作完成")
        else:
            print("分类操作失败")
    else:
        print("没有找到文件")