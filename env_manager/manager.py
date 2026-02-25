import subprocess 
import os 
import sys 
from pathlib import Path 
import traceback 

class EnvManager :
    def __init__ (self ):
        self ._current_venv =None 

    def activate_vene (self ,vene_path ):
        vene_path =Path (vene_path ).resolve ()
        if not vene_path .exists ()or not vene_path .is_dir ():
            raise FileNotFoundError (f"虚拟环境路径不存在: {vene_path}")

        if sys .platform =="win32":

            python_exe =vene_path /"python.exe"
            if not python_exe .exists ():
                python_exe =vene_path /"Scripts"/"python.exe"


            activate_script =vene_path /"Scripts"/"activate.bat"
            if not activate_script .exists ():
                activate_script =vene_path /"activate.bat"
        else :
            python_exe =vene_path /"bin"/"python"
            activate_script =vene_path /"bin"/"activate"

        if not python_exe .exists ():
            raise FileNotFoundError (f"Python可执行文件不存在: {python_exe}")

        self ._current_venv ={
        "path":vene_path ,
        "python":python_exe ,
        "activate":activate_script ,
        "name":vene_path .name 
        }
        return self 

    def run_command (self ,command ,capture_output =True ):
        if not self ._current_venv :
            raise RuntimeError ("请先激活一个虚拟环境")

        try :
            if sys .platform =="win32":

                if command .startswith ("python"):

                    python_path =str (self ._current_venv ["python"])

                    args =command .split ()
                    args [0 ]=python_path 
                    print (f"执行命令参数: {args}")

                    result =subprocess .run (
                    args ,
                    capture_output =capture_output ,
                    text =True 
                    )

                elif command .startswith ("pip"):

                    python_path =str (self ._current_venv ["python"])

                    args =command .split ()
                    args [0 ]=python_path 
                    args .insert (1 ,"-m")
                    args .insert (2 ,"pip")
                    print (f"执行命令参数: {args}")

                    result =subprocess .run (
                    args ,
                    capture_output =capture_output ,
                    text =True 
                    )

                else :

                    env_name =self ._current_venv ["name"]
                    full_command =f'conda activate {env_name} && {command}'
                    print (f"执行命令: {full_command}")


                    cmd_exe =os .path .join (os .environ .get ('SYSTEMROOT','C:\\Windows'),'System32','cmd.exe')

                    result =subprocess .run (
                    full_command ,
                    shell =True ,
                    capture_output =capture_output ,
                    text =True ,
                    executable =cmd_exe 
                    )
            else :

                if command .startswith ("python"):
                    python_path =str (self ._current_venv ["python"])
                    args =command .split ()
                    args [0 ]=python_path 
                    result =subprocess .run (
                    args ,
                    capture_output =capture_output ,
                    text =True 
                    )
                elif command .startswith ("pip"):
                    python_path =str (self ._current_venv ["python"])
                    args =command .split ()
                    args [0 ]=python_path 
                    args .insert (1 ,"-m")
                    args .insert (2 ,"pip")
                    result =subprocess .run (
                    args ,
                    capture_output =capture_output ,
                    text =True 
                    )
                else :
                    env_name =self ._current_venv ["name"]
                    full_command =f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && {command}'
                    result =subprocess .run (
                    full_command ,
                    shell =True ,
                    capture_output =capture_output ,
                    text =True ,
                    executable ="/bin/bash"
                    )

            return result 

        except Exception as e :
            print (f"运行命令时出错: {e}")
            print (f"错误类型: {type(e)}")
            raise 

    def setup_audio_env (self ):
        try :
            self .activate_vene (r"C:\Users\sysu\anaconda3\envs\audio")
            print (f"音频环境已激活: {self._current_venv['name']}")
            return True 
        except Exception as e :
            print (f"音频环境激活失败: {e}")
            return False 

    def setup_text_env (self ):
        try :
            self .activate_vene (r"C:\Users\sysu\anaconda3\envs\text-dedup")
            print (f"文本环境已激活: {self._current_venv['name']}")
            return True 
        except Exception as e :
            print (f"文本环境激活失败: {e}")
            return False 

    def setup_image_env (self ):
        try :

            self .activate_vene (r"c:\Users\sysu\anaconda3\envs\tryamrosemdedup")
            print (f"图像环境已激活: {self._current_venv['name']}")
            return True 
        except Exception as e :
            print (f"图像环境激活失败: {e}")
            return False 



if __name__ =="__main__":

    switcher =EnvManager ()




































    try :

        setup_success =switcher .setup_audio_env ()
        print (setup_success )
        res =switcher .run_command ("python --version")
        print (f"返回码: {res.returncode}")
        print (f"输出: {res.stdout.strip()}")


        setup_success =switcher .setup_text_env ()
        print (setup_success )
        print (f"切换到虚拟环境: {switcher._current_venv['path']}")
        print (f"Python 路径: {switcher._current_venv['python']}")
        res =switcher .run_command ("python --version")
        print (f"返回码: {res.returncode}")
        print (f"输出: {res.stdout.strip()}")


        setup_success =switcher .setup_image_env ()
        print (setup_success )
        print (f"切换到虚拟环境: {switcher._current_venv['path']}")
        print (f"Python 路径: {switcher._current_venv['python']}")
        res =switcher .run_command ("python --version")
        print (f"返回码: {res.returncode}")
        print (f"输出: {res.stdout.strip()}")
    except Exception as e :
        print (f"发生错误: {e}")
        print (f"错误类型: {type(e)}")
        traceback .print_exc ()