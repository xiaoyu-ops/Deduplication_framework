# 添加上级目录到Python路径，这样可以导入其他模块
import os
import sys
import shutil
from pathlib import Path
import glob

current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  # 添加这行，获取根目录

# 将根目录添加到Python路径（这样可以导入env_manager包）
sys.path.insert(0, root_dir)
print(f"已添加根目录到路径: {root_dir}")

# 现在可以用完整包路径导入
from env_manager.manager import EnvManager

def create_deduplicated_datasets():
    """
    根据去重结果创建去重后的数据集
    """
    # 修正路径计算
    dataset_dir = os.path.join(parent_dir, "dataset")  # audio/dataset
    dedup_results_dir = os.path.join(current_dir, "dedup_results")  # audio/method/dedup_results
    
    print("开始创建去重后的数据集")
    print(f"数据集目录: {dataset_dir}")
    print(f"去重结果目录: {dedup_results_dir}")
    
    # 检查原始数据集是否存在
    if not os.path.exists(dataset_dir):
        print(f"原始数据集目录不存在: {dataset_dir}")
        return False
    
    # 获取所有原始WAV文件
    wav_files = glob.glob(os.path.join(dataset_dir, "*.wav"))
    wav_files.sort()
    
    print(f"原始数据集包含 {len(wav_files)} 个WAV文件")
    
    if not wav_files:
        print("原始数据集中没有找到WAV文件")
        return False
    
    # 检查去重结果目录是否存在
    if not os.path.exists(dedup_results_dir):
        print(f"去重结果目录不存在: {dedup_results_dir}")
        print("请先执行音频去重(选项2)来生成去重结果")
        return False
    
    # 查找所有去重结果文件
    remove_files = glob.glob(os.path.join(dedup_results_dir, "*_dedup_result.txt"))
    
    print(f"在 {dedup_results_dir} 中查找保留文件列表...")
    print(f"查找模式: *_keep_files.txt")
    
    # 调试：列出目录中的所有文件
    if os.path.exists(dedup_results_dir):
        all_files = os.listdir(dedup_results_dir)
        print(f"去重结果目录中的所有文件: {all_files}")
        
        # 查找所有可能的结果文件
        result_files = [f for f in all_files if 'threshold_' in f and f.endswith('.txt')]
        print(f"找到的结果文件: {result_files}")
    
    if not remove_files:
        print(f"没有找到保留文件列表")
        print("可能的原因:")
        print("1. 还没有执行去重操作")
        print("2. 去重结果文件名格式不对")
        print("3. 权限问题导致文件未生成")
        
        # 尝试查找其他格式的文件
        alternative_files = glob.glob(os.path.join(dedup_results_dir, "threshold_*_dedup_result.txt"))
        if alternative_files:
            print(f"找到了去重结果文件但没有keep文件: {alternative_files}")
            print("这可能是因为extract_local_file_info函数没有成功创建keep文件")
        
        return False
    
    print(f"找到 {len(remove_files)} 个去重结果文件: {[os.path.basename(f) for f in remove_files]}")
    
    success_count = 0
    
    for remove_file in remove_files:
        try:
            # 从文件名提取阈值信息 (例如: threshold_0.7_dedup_result.txt)
            filename = os.path.basename(remove_file)
            if 'threshold_' in filename:
                threshold = filename.split('threshold_')[1].split('_')[0]
            else:
                threshold = "unknown"
            
            print(f"\n处理阈值 {threshold} 的去重结果...")
            
            # 读取要删除的文件索引
            remove_indices = []
            remove_list = os.path.join(remove_file, "remove_files.txt")  # remove_file是文件夹路径
            
            print(f"读取删除文件列表: {remove_list}")
            
            if not os.path.exists(remove_list):
                print(f"文件不存在: {remove_list}")
                continue
                
            with open(remove_list, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # remove_files.txt格式很简单，每行一个数字
                    if line and line.isdigit():
                        remove_indices.append(int(line))
            
            if not remove_indices:
                print(f"阈值 {threshold}: 没有找到有效的删除索引")
                continue
            
            print(f"阈值 {threshold}: 找到 {len(remove_indices)} 个要删除的文件")
            
            # 计算要保留的文件索引
            all_indices = set(range(len(wav_files)))
            remove_indices_set = set(remove_indices)
            keep_indices = sorted(list(all_indices - remove_indices_set))
            
            print(f"阈值 {threshold}: 保留 {len(keep_indices)} 个文件 (删除 {len(remove_indices)} 个)")
            
            # 创建输出目录
            output_dataset_dir = os.path.join(dedup_results_dir, f"threshold_{threshold}_dataset")
            os.makedirs(output_dataset_dir, exist_ok=True)
            
            # 复制保留的文件
            copied_count = 0
            error_count = 0
            
            for idx in keep_indices:  # 修正：使用keep_indices而不是undefined的变量
                try:
                    if idx < len(wav_files):
                        src_file = wav_files[idx]
                        dst_file = os.path.join(output_dataset_dir, os.path.basename(src_file))
                        
                        # 复制文件
                        shutil.copy2(src_file, dst_file)
                        copied_count += 1
                    else:
                        print(f"索引 {idx} 超出文件范围 (最大: {len(wav_files)-1})")
                        error_count += 1
                        
                except Exception as e:
                    print(f"复制文件失败 (索引 {idx}): {e}")
                    error_count += 1
            
            # 生成统计报告
            stats_file = os.path.join(output_dataset_dir, "dataset_stats.txt")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"=== 去重数据集统计 (阈值: {threshold}) ===\n\n")
                f.write(f"原始文件数量: {len(wav_files)}\n")
                f.write(f"删除文件数量: {len(remove_indices)}\n")  # 修正：显示删除数量
                f.write(f"保留文件数量: {len(keep_indices)}\n")   # 修正：显示保留数量
                f.write(f"成功复制数量: {copied_count}\n")
                f.write(f"复制失败数量: {error_count}\n")
                f.write(f"去重率: {(len(remove_indices) / len(wav_files) * 100):.2f}%\n\n")  # 修正：去重率=删除数量/总数量
                
                # 计算总大小
                total_size = 0
                for idx in keep_indices:
                    if idx < len(wav_files):
                        try:
                            size = os.path.getsize(wav_files[idx])
                            total_size += size
                        except:
                            pass
                
                f.write(f"保留数据集总大小: {total_size / (1024*1024):.2f} MB\n")
                f.write(f"平均文件大小: {(total_size / len(keep_indices) / (1024*1024)):.2f} MB\n\n")
                
                f.write("删除的文件索引:\n")
                for i, idx in enumerate(sorted(remove_indices)):
                    if i % 10 == 0:
                        f.write("\n")
                    f.write(f"{idx:4d} ")
                
                f.write("\n\n保留的文件索引:\n")
                for i, idx in enumerate(keep_indices):
                    if i % 10 == 0:
                        f.write("\n")
                    f.write(f"{idx:4d} ")
            
            print(f"阈值 {threshold}: 成功复制 {copied_count} 个文件到 {output_dataset_dir}")
            print(f"   统计报告: {stats_file}")
            
            if error_count > 0:
                print(f"阈值 {threshold}: {error_count} 个文件复制失败")
            
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {remove_file} 时出错: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"数据集创建完成! 成功处理 {success_count} 个阈值")
    print("=" * 60)
    
    return success_count > 0

def show_menu():
    """显示操作菜单"""
    print("\n" + "=" * 50)
    print("音频去重处理菜单")
    print("=" * 50)
    print("1. 生成音频指纹")
    print("2. 执行音频去重")
    print("3. 创建去重后的数据集")
    print("4. 执行完整流程 (1+2+3)")
    print("5. 退出")
    print("=" * 50)
    return input("请选择操作 (1-5): ").strip()

if __name__ == "__main__":
    # 此处利用switcher切换到dedup_env环境
    switcher = EnvManager()  # 创建类的实例
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            print("\n--- 生成音频指纹 ---")
            if switcher.setup_audio_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("音频指纹生成成功")
                else:
                    print("音频指纹生成失败")
            else:
                print("环境设置失败")
        
        elif choice == "2":
            print("\n--- 执行音频去重 ---")
            if switcher.setup_audio_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("音频去重执行成功")
                else:
                    print("音频去重执行失败")
            else:
                print("环境设置失败")
        
        elif choice == "3":
            print("\n--- 创建去重后的数据集 ---")
            success = create_deduplicated_datasets()
            if not success:
                print("数据集创建失败")
        
        elif choice == "4":
            print("\n--- 执行完整流程 ---")
            success = True
            
            # 步骤1: 生成指纹
            print("\n步骤1: 生成音频指纹...")
            if switcher.setup_audio_env():
                if os.name == 'nt':
                    command = f'conda activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                else:
                    command = f'source activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                
                res = os.system(command)
                if res == 0:
                    print("音频指纹生成成功")
                else:
                    print("音频指纹生成失败")
                    success = False
            else:
                print("环境设置失败")
                success = False
            
            # 步骤2: 执行去重
            if success:
                print("\n步骤2: 执行音频去重...")
                if os.name == 'nt':
                    command = f'conda activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                else:
                    command = f'source activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                
                res = os.system(command)
                if res == 0:
                    print("音频去重执行成功")
                else:
                    print("音频去重执行失败")
                    success = False
            
            # 步骤3: 创建数据集
            if success:
                print("\n步骤3: 创建去重后的数据集...")
                success = create_deduplicated_datasets()
            
            if success:
                print("\n完整流程执行成功!")
            else:
                print("\n流程执行中断")
        
        elif choice == "5":
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")