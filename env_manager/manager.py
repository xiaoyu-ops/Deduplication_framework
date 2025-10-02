import subprocess
import os
import sys
from pathlib import Path
import traceback

class EnvManager:
    def __init__(self):
        self._current_venv = None

    def activate_vene(self, vene_path):
        """
        切换到指定的虚拟环境
        不是通过修改环境变量，而是记录当前虚拟环境的信息，供后续命令使用
        """
        vene_path = Path(vene_path).resolve()
        if not vene_path.exists() or not vene_path.is_dir():
            raise FileNotFoundError(f"虚拟环境路径不存在: {vene_path}")

        if sys.platform == "win32":
            # conda 环境通常 python.exe 直接在根目录或 Scripts 目录
            python_exe = vene_path / "python.exe"
            if not python_exe.exists():
                python_exe = vene_path / "Scripts" / "python.exe"
            
            # conda 环境激活脚本
            activate_script = vene_path / "Scripts" / "activate.bat"
            if not activate_script.exists():
                activate_script = vene_path / "activate.bat"
        else:
            python_exe = vene_path / "bin" / "python"
            activate_script = vene_path / "bin" / "activate"

        if not python_exe.exists():
            raise FileNotFoundError(f"Python可执行文件不存在: {python_exe}")

        self._current_venv = {
            "path": vene_path,
            "python": python_exe,
            "activate": activate_script,
            "name": vene_path.name  # 添加环境名称
        }
        return self
    
    def run_command(self, command, capture_output=True):
        """
        在当前的虚拟环境中运行命令
        """
        if not self._current_venv:
            raise RuntimeError("请先激活一个虚拟环境")
        
        try:
            if sys.platform == "win32":
                # 直接使用虚拟环境的 python.exe，不使用 shell=True
                if command.startswith("python"):
                    # 构造参数列表而不是字符串命令
                    python_path = str(self._current_venv["python"])
                    # 解析命令参数
                    args = command.split()
                    args[0] = python_path  # 替换 python 为完整路径
                    print(f"执行命令参数: {args}")  # 调试信息
                    
                    result = subprocess.run(
                        args,# 执行行
                        capture_output=capture_output, # 捕捉输出
                        text=True # 以文本形式返回输出
                    )
                    
                elif command.startswith("pip"):
                    # 使用 python -m pip
                    python_path = str(self._current_venv["python"])
                    # 解析命令参数
                    args = command.split()
                    args[0] = python_path
                    args.insert(1, "-m")  # 插入 -m pip
                    args.insert(2, "pip")
                    print(f"执行命令参数: {args}")  # 调试信息
                    
                    result = subprocess.run(
                        args,
                        capture_output=capture_output,
                        text=True
                    )
                    
                else:
                    # 其他命令，使用 shell=True 但修复 cmd.exe 路径
                    env_name = self._current_venv["name"]
                    full_command = f'conda activate {env_name} && {command}'
                    print(f"执行命令: {full_command}")  # 调试信息
                    
                    # 明确指定 cmd.exe 的完整路径
                    cmd_exe = os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'System32', 'cmd.exe')
                    
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        capture_output=capture_output,
                        text=True,
                        executable=cmd_exe
                    )
            else:
                # linux/mac 使用类似逻辑
                if command.startswith("python"):
                    python_path = str(self._current_venv["python"])
                    args = command.split()
                    args[0] = python_path
                    result = subprocess.run(
                        args,
                        capture_output=capture_output,
                        text=True
                    )
                elif command.startswith("pip"):
                    python_path = str(self._current_venv["python"])
                    args = command.split()
                    args[0] = python_path
                    args.insert(1, "-m")
                    args.insert(2, "pip")
                    result = subprocess.run(
                        args,
                        capture_output=capture_output,
                        text=True
                    )
                else:
                    env_name = self._current_venv["name"]
                    full_command = f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && {command}'
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        capture_output=capture_output,
                        text=True,
                        executable="/bin/bash"
                    )

            return result
            
        except Exception as e:
            print(f"运行命令时出错: {e}")
            print(f"错误类型: {type(e)}")
            raise
    
    
# 测试一下
if __name__ == "__main__":
    
    switcher = EnvManager()
    
    # # 测试存在的环境
    # try:
    #     print("首先激活环境")
    #     switcher.activate_vene(r"C:\Users\sysu\anaconda3\envs\audio")
    #     print(f"切换到虚拟环境: {switcher._current_venv['path']}")
    #     print(f"Python 路径: {switcher._current_venv['python']}")
        
    #     # 验证 Python 文件确实存在
    #     python_path = Path(switcher._current_venv['python'])
    #     print(f"Python 文件存在: {python_path.exists()}")
    #     print(f"Python 文件大小: {python_path.stat().st_size if python_path.exists() else 'N/A'}")
        
    #     print("\n测试 Python 命令")
    #     # 测试 python 命令
    #     res = switcher.run_command("python --version")
    #     print(f"返回码: {res.returncode}")
    #     if res.stdout:
    #         print(f"输出: {res.stdout.strip()}")
    #     if res.stderr:
    #         print(f"错误: {res.stderr.strip()}")
            
    #     print("\n测试 Pip 命令")
    #     # 测试 pip 命令
    #     res = switcher.run_command("pip --version")
    #     print(f"返回码: {res.returncode}")
    #     if res.stdout:
    #         print(f"输出: {res.stdout.strip()}")
    #     if res.stderr:
    #         print(f"错误: {res.stderr.strip()}")
        
    # except Exception as e:
    #     print(f"发生错误: {e}")
    #     print(f"错误类型: {type(e)}")
    #     traceback.print_exc()

    try:
        # 激活audio环境
        switcher.activate_vene(r"C:\Users\sysu\anaconda3\envs\audio")
        print(f"切换到虚拟环境: {switcher._current_venv['path']}")
        print(f"Python 路径: {switcher._current_venv['python']}")
        res = switcher.run_command("python --version")
        print(f"返回码: {res.returncode}")
        print(f"输出: {res.stdout.strip()}")

        # 激活text环境
        switcher.activate_vene(r"C:\Users\sysu\anaconda3\envs\text-dedup")
        print(f"切换到虚拟环境: {switcher._current_venv['path']}")
        print(f"Python 路径: {switcher._current_venv['python']}")
        res = switcher.run_command("python --version")
        print(f"返回码: {res.returncode}")
        print(f"输出: {res.stdout.strip()}")

        # 激活image环境
        switcher.activate_vene(r"c:\Users\sysu\anaconda3\envs\tryamrosemdedup")
        print(f"切换到虚拟环境: {switcher._current_venv['path']}")
        print(f"Python 路径: {switcher._current_venv['python']}")
        res = switcher.run_command("python --version")
        print(f"返回码: {res.returncode}")
        print(f"输出: {res.stdout.strip()}")
    except Exception as e:
        print(f"发生错误: {e}")
        print(f"错误类型: {type(e)}")
        traceback.print_exc()