import os 
import numpy as np
import random
from collections import defaultdict
from datasets import load_dataset
import json
import glob

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
    """
    import glob
    
    try:
        # 修复：获取所有WAV文件并排序，避免重复计算
        wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
        # 注释
        # wav_files.extend(glob.glob(os.path.join(wav_dir, "*.WAV")))
        
        wav_files.sort()  # 确保排序一致
        
        print(f"找到 {len(wav_files)} 个WAV文件")
        print(f"前5个文件: {[os.path.basename(f) for f in wav_files[:5]]}")
        
        # 创建索引到文件的直接映射
        index_to_file = {}
        
        for i, wav_file in enumerate(wav_files):
            filename = os.path.basename(wav_file)
            
            # 直接索引映射（按列表顺序）
            index_to_file[i] = wav_file
            index_to_file[str(i)] = wav_file
            
            # 如果是 audio_0000.wav 格式，提取数字索引
            if filename.startswith('audio_') and filename.endswith('.wav'):
                try:
                    # 提取 audio_0000.wav 中的 0000 部分
                    number_str = filename[6:-4]  # 去掉 "audio_" 和 ".wav"
                    if number_str.isdigit():
                        file_index = int(number_str)
                        index_to_file[file_index] = wav_file
                        index_to_file[str(file_index)] = wav_file
                        index_to_file[number_str] = wav_file
                        
                except (ValueError, IndexError):
                    pass
        
        print(f"创建了 {len(index_to_file)} 个索引映射")
        print(f"要删除的文件索引数量: {len(remove_files)}")
        
        # 提取要删除的文件信息
        files_to_remove = []
        not_found_files = []
        
        for file_id in remove_files:
            matched_path = None
            
            # 尝试多种索引格式
            search_keys = [
                file_id,           # 原始值
                str(file_id),      # 字符串形式
                int(file_id) if str(file_id).isdigit() else None,  # 整数形式
            ]
            
            # 去除None值
            search_keys = [k for k in search_keys if k is not None]
            
            for key in search_keys:
                if key in index_to_file:
                    matched_path = index_to_file[key]
                    break
            
            if matched_path:
                try:
                    file_size = os.path.getsize(matched_path)
                    files_to_remove.append({
                        'filename': os.path.basename(matched_path),
                        'path': matched_path,
                        'size_mb': file_size / (1024 * 1024),
                        'index': file_id
                    })
                except OSError as e:
                    print(f"无法获取文件大小: {matched_path}, 错误: {e}")
            else:
                not_found_files.append(file_id)
        
        print(f"匹配成功: {len(files_to_remove)} 个文件")
        print(f"未找到: {len(not_found_files)} 个文件")
        
        # 计算保留的文件
        all_indices = set(range(len(wav_files)))
        remove_indices = set(int(str(f)) for f in remove_files if str(f).isdigit() and int(str(f)) < len(wav_files))
        keep_indices = sorted(list(all_indices - remove_indices))
        
        # 准备内容
        content = f"=== 音频去重结果详细报告 ===\n\n"
        content += f"总文件数: {len(wav_files)}\n"
        content += f"要删除文件数: {len(files_to_remove)}\n"
        content += f"保留文件数: {len(keep_indices)}\n"
        content += f"删除总大小: {sum(item['size_mb'] for item in files_to_remove):.2f} MB\n\n"
        
        content += "=== 要删除的文件列表 ===\n"
        for item in files_to_remove:
            content += f"索引: {item['index']:4} | 文件名: {item['filename']} | 大小: {item['size_mb']:.2f} MB\n"
        
        content += f"\n=== 保留的文件索引 ({len(keep_indices)} 个) ===\n"
        # 每行显示10个索引
        for i in range(0, len(keep_indices), 10):
            line_indices = keep_indices[i:i+10]
            content += " ".join(f"{idx:4d}" for idx in line_indices) + "\n"
        
        if not_found_files:
            content += f"\n=== 未匹配的索引 ({len(not_found_files)} 个) ===\n"
            for i in range(0, min(len(not_found_files), 50), 10):  # 最多显示50个
                line_indices = not_found_files[i:i+10]
                content += " ".join(str(idx) for idx in line_indices) + "\n"
        
        # 直接保存到指定位置（权限问题已解决）
        try:
            # 确保目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"详细报告已保存到: {output_file}")
            
            # 同时生成保留文件列表
            keep_file = output_file.replace('.txt', '_keep_files.txt')
            with open(keep_file, 'w', encoding='utf-8') as f:
                f.write("=== 去重后保留的文件索引 ===\n\n")
                for idx in keep_indices:
                    if idx < len(wav_files):
                        filename = os.path.basename(wav_files[idx])
                        f.write(f"{idx:4d} : {filename}\n")
            print(f"保留文件列表已保存到: {keep_file}")
            
        except Exception as e:
            print(f"保存失败: {e}")
            print("详细报告显示在控制台:")
            print("=" * 60)
            print(content[:2000] + "..." if len(content) > 2000 else content)
            print("=" * 60)
        
        return files_to_remove
        
    except Exception as e:
        print(f"extract_local_file_info执行失败: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # 主程序执行
    similar_pairs = load_similar_pairs("similar_pairs.txt")
    print(f"加载的相似对: {similar_pairs}")

    remove_files = build_similarity_groups(similar_pairs)
    generate_dedup_report(remove_files)
    execute_deduplication(remove_files, output_dir="dedup_results")
    extract_file_name("danavery/urbansound8K", "dedup_results", remove_files)
