# 添加上级目录到Python路径，这样可以导入其他模块
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  # 添加这行，获取根目录

# 将根目录添加到Python路径（这样可以导入env_manager包）
sys.path.insert(0, root_dir)
print(f"已添加根目录到路径: {root_dir}")

# 现在可以用完整包路径导入
from env_manager.manager import EnvManager

if __name__ == "__main__":
    # 此处利用switcher切换到dedup_env环境，然后再用command运行我们需要的脚本
    switcher = EnvManager()  # 创建类的实例

    # 以下为生成音频指纹的脚本
    # if switcher.setup_audio_env():
    #     # 确认环境切换成功后再运行脚本
    #     # 使用conda/activate命令在audio环境中运行脚本
    #     if os.name == 'nt':  # Windows系统
    #         command = f'conda activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
    #     else:  # Linux/Mac系统
    #         command = f'source activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
        
    #     print(f"执行命令: {command}")
    #     res = os.system(command)
    #     if res == 0:
    #         print("音频指纹生成脚本执行成功")
    #     else:
    #         print("音频指纹生成脚本执行失败")
    # else:
    #     print("环境设置失败")

    # 以下为利用音频指纹文件来去重的脚本
    if switcher.setup_audio_env():
        if os.name == 'nt':  # Windows系统
            command = f'conda activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
        else:  # Linux/Mac系统
            command = f'source activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
        
        print(f"执行命令: {command}")
        res = os.system(command)
        if res == 0:
            print("音频去重脚本执行成功")
        else:
            print("音频去重脚本执行失败")
    else:
        print("环境设置失败")