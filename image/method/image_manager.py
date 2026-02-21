# 此文件用于中心调控全部的image去重功能，包括数据分类、环境配置、脚本调用等。
# 目标为实现传入任意数据集路径，自动完成去重并输出结果。并且提供交互式命令行界面，方便用户操作。
# 交互式由arg来实现，由于虚拟环境非常苛刻所以需要时刻注意切换环境
# 初步实现目标为：在获取嵌入向量这个步骤可以有两种选择 1.huggingface 2.用户自己的本地数据集 如json格式或者直接就是image也都可以
# 1.第一步我们需要使用main.py这个文件来提取嵌入向量。
# 2.第二步我们应该使用run_clustering_local来进行k-means聚类
# 3.聚类完成后是进行sort_clusters_in_windows对刚刚计算出来的簇进行排序
# 4.排序后就是simple_semdedup进行去重
# 5.就是用creat_deduped_dateset来获取去重后的数据集
import os 
import glob
import numpy as np
import sys
import json 

current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  # 添加这行，获取根目录

# 将根目录添加到Python路径（这样可以导入env_manager包）
sys.path.insert(0, root_dir)
print(f"已添加根目录到路径: {root_dir}")

# 现在可以用完整包路径导入
from env_manager.manager import EnvManager

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
    
def show_menu():
    """显示操作菜单"""
    print("\n" + "=" * 50)
    print("图像去重处理菜单")
    print("=" * 50)
    print("1. 提取嵌入向量")
    print("2. 转换嵌入向量")
    print("3. 进行k-means聚类")
    print("4. 对聚类结果进行排序")
    print("5. 进行去重")
    print("6. 获取去重后数据")    
    print("7. 退出")
    print("8. 查询当前配置文件")
    print("9. 执行完整流程 (1+2+3+4+5+6)")
    print("=" * 50)
    return input("请选择操作 (1-9): ").strip()

if __name__ == "__main__":
    switcher = EnvManager()
    while True:
        choice = show_menu()
        
        if choice == "1":
            print("\n--- 提取嵌入向量 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("提取嵌入向量成功")
                else:
                    print("提取嵌入向量失败")
            else:
                print("环境设置失败")

        if choice == "2":
            print("\n--- 转换嵌入向量 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("转换嵌入向量成功")
                else:
                    print("转换嵌入向量失败")
            else:
                print("环境设置失败")

        elif choice == "3":
            print("\n--- 进行k-means聚类 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("聚类成功")
                else:
                    print("聚类失败")
            else:
                print("环境设置失败")
        
        elif choice == "4":
            print("\n--- 对聚类结果进行排序 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("排序成功")
                else:
                    print("排序失败")
            else:
                print("环境设置失败") 

        elif choice == "5":
            print("\n--- 进行去重 ---")
            if switcher.setup_image_env():
                # 使用配置文件并传入默认 eps 列表（按需修改 eps_list）
                config_path = r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list = "0.05"
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("去重成功")
                else:
                    print("去重失败")
            else:
                print("环境设置失败")

        elif choice == "6":
            print("\n--- 提取去重后数据 ---")
            if switcher.setup_image_env():
                # 使用与 simple_semdedup 相同的配置文件与 eps（取 eps_list 的第一个值）
                config_path = r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list = "0.05"  # 与 choice 5 中保持一致，必要时改为从配置中读取
                eps_value = eps_list.split(",")[0].strip()
                # 输出文件夹，放到工程下 image/deduped/eps_<eps>
                output_folder = os.path.join(root_dir, "image", "deduped", f"eps_{eps_value.replace('.','_')}")
                os.makedirs(output_folder, exist_ok=True)
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("提取成功")
                else:
                    print("提取失败")
            else:
                print("环境设置失败")   
              
        elif choice == "7":
            print("退出程序")
            break

        elif choice == "8":
            config_path = os.path.join(current_dir, "image_config.json")
            data = load_config_json(config_path)
            print("以下为当下配置情况")
            print(data)

        elif choice == "9":
            print("\n--- 提取嵌入向量 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("提取嵌入向量成功")
                else:
                    print("提取嵌入向量失败")
            else:
                print("环境设置失败")

            print("\n--- 转换嵌入向量 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("转换嵌入向量成功")
                else:
                    print("转换嵌入向量失败")
            else:
                print("环境设置失败")

            print("\n--- 进行k-means聚类 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("聚类成功")
                else:
                    print("聚类失败")
            else:
                print("环境设置失败")

            print("\n--- 对聚类结果进行排序 ---")
            if switcher.setup_image_env():
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("排序成功")
                else:
                    print("排序失败")
            else:
                print("环境设置失败") 

            print("\n--- 进行去重 ---")
            if switcher.setup_image_env():
                # 使用配置文件并传入默认 eps 列表（按需修改 eps_list）
                config_path = r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list = "0.15"
                if os.name == 'nt':  # Windows系统
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                else:  # Linux/Mac系统
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("去重成功")
                else:
                    print("去重失败")
            else:
                print("环境设置失败")

            print("\n--- 提取去重后数据 ---")
            if switcher.setup_image_env():
                config_path = r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list = "0.15"  # 与该完整流程使用的 eps_list 保持一致
                eps_value = eps_list.split(",")[0].strip()
                output_folder = os.path.join(root_dir, "image", "deduped", f"eps_{eps_value.replace('.','_')}")
                os.makedirs(output_folder, exist_ok=True)
                if os.name == 'nt':
                    command = f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                else:
                    command = f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                
                print(f"执行命令: {command}")
                res = os.system(command)
                if res == 0:
                    print("提取成功")
                else:
                    print("提取失败")
            else:
                print("环境设置失败")                   
                
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")