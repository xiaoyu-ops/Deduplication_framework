import os 
import numpy as np
import random
from collections import defaultdict
from datasets import load_dataset
import json
import glob

# 先将SIMILAR_PAIRS.TXT加载到内存中
def load_similar_pairs(file_path,threshold=0.83):
    """
    从文件中加载相似对
    :param file_path: 相似对文件路径
    :return: 相似对列表
    """
    similar_pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            start = line.find("(") + 1
            end = line.find(")")
            pair = line[start:end].split(", ")
            start_1 = line.find("Similarity: ") + len("Similarity: ")
            similarity_str = line[start_1:]
            similarity = float(similarity_str)
            if similarity >= threshold:  # 只保留相似度大于等于0.83的对
                if len(pair) == 2:
                    similar_pairs.append((int(pair[0]), int(pair[1])))
    return similar_pairs

# 构建相似组并随机保留每组中的一个文件
def build_similarity_groups(similar_pairs):
    """
    随机保留每组中的一个文件
    :param similar_pairs: 相似对列表
    :return: 保留文件列表，删除文件列表，相似组
    """
    # 收集所有唯一文件
    keep_files = set()
    remove_files = set()
    for file1, file2 in similar_pairs:
        if random.random() < 0.5:
            remove_files.add(file2)
    return remove_files
    
# 生成去重结果报告
def generate_dedup_report(remove_files):
    """
    生成去重结果报告
    :param remove_files: 删除文件列表
    :return: None
    """
    print("\n==== 音频去重结果报告 ====")
    print(f"删除文件数: {len(remove_files)}")
    
# 执行去重操作
def execute_deduplication(remove_files, output_dir=None, move_files=False):
    """
    执行去重操作
    :param keep_files: 保留文件列表
    :param remove_files: 删除文件列表
    :param output_dir: 输出目录，如果为None则只生成报告
    :param move_files: 是否移动文件而不是删除
    :return: None
    """
    if output_dir is None:
        print("\n仅生成报告，未执行实际文件操作")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建删除文件的列表文件
    with open(os.path.join(output_dir, "remove_files.txt"), "w") as f:
        for file in remove_files:
            f.write(f"{file}\n")
    
    print(f"\n文件列表已保存到 {output_dir}")
    
    # 实际移动或删除文件的代码可以在这里实现
    if move_files:
        print("移动文件功能尚未实现")

def extract_file_name(dataset_name, output_dir, remove_files):
    """
    从数据集中提取需要保留的文件并以JSON格式保存
    :param dataset_name: 数据集名称
    :param output_dir: 输出目录
    :param remove_files: 删除文件集合
    """
    dataset = load_dataset(dataset_name)
    length = len(dataset['train'])
    
    keep_files_data = {
        "metadata": {
            "dataset_name": dataset_name,
            "total_files": length,
            "kept_files_count": 0,
            "removed_files_count": len(remove_files),
            "dedup_rate": 0.0
        },
        "kept_files": []
    }
    
    kept_count = 0
    print(remove_files)

    for index in range(length):
        if index not in remove_files:
            file_data = dataset['train'][index]
            
            # 提取关键信息，避免保存大的音频数组
            file_info = {
                "index": index,
                "slice_file_name": file_data.get('slice_file_name', ''),
                "fsID": file_data.get('fsID', ''),
                "start": file_data.get('start', 0),
                "end": file_data.get('end', 0),
                "salience": file_data.get('salience', 0),
                "fold": file_data.get('fold', 0),
                "classID": file_data.get('classID', 0),
                "class": file_data.get('class', ''),
                "audio_path": file_data.get('audio', {}).get('path', ''),
                "sampling_rate": file_data.get('audio', {}).get('sampling_rate', 0)
            }
            
            keep_files_data["kept_files"].append(file_info)
            kept_count += 1
    
    # 更新元数据
    keep_files_data["metadata"]["kept_files_count"] = kept_count
    keep_files_data["metadata"]["dedup_rate"] = len(remove_files) / length
    
    # 保存为JSON文件
    json_file_path = os.path.join(output_dir, "keep_files.json")
    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(keep_files_data, f, indent=2, ensure_ascii=False) #indent=2使JSON更易读
    
    print(f"去重率为: {len(remove_files) / length:.2%}")
    print(f"保留文件数: {kept_count}")
    print(f"总文件数: {len(keep_files_data['kept_files'])}")
    print(f"JSON文件已保存到: {json_file_path}")

def extract_local_file_info(wav_dir, output_file, remove_files):
    """
    从本地WAV文件目录中提取要删除的文件信息
    
    参数:
        wav_dir: 本地WAV文件目录路径
        output_file: 输出文件路径
        remove_files: 要删除的文件列表
    """
    # 获取所有WAV文件
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    wav_files.extend(glob.glob(os.path.join(wav_dir, "*.WAV")))
    
    # 创建文件名到路径的映射
    filename_to_path = {}
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        filename_to_path[filename] = wav_file
    
    # 提取要删除的文件信息
    files_to_remove = []
    for file_id in remove_files:
        if file_id in filename_to_path:
            file_path = filename_to_path[file_id]
            file_size = os.path.getsize(file_path)
            files_to_remove.append({
                'filename': file_id,
                'path': file_path,
                'size_mb': file_size / (1024 * 1024)
            })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"本次去重将删除 {len(files_to_remove)} 个文件\n")
        f.write(f"总大小: {sum(item['size_mb'] for item in files_to_remove):.2f} MB\n\n")
        
        for item in files_to_remove:
            f.write(f"文件名: {item['filename']}\n")
            f.write(f"路径: {item['path']}\n")
            f.write(f"大小: {item['size_mb']:.2f} MB\n")
            f.write("-" * 50 + "\n")
    
    print(f"文件信息已保存到: {output_file}")
    return files_to_remove

if __name__ == "__main__":
    # 主程序执行
    similar_pairs = load_similar_pairs("similar_pairs.txt")
    print(f"加载的相似对: {similar_pairs}")

    remove_files = build_similarity_groups(similar_pairs)
    generate_dedup_report(remove_files)
    execute_deduplication(remove_files, output_dir="dedup_results")
    extract_file_name("danavery/urbansound8K", "dedup_results", remove_files)
