
import os 
import sys 
import subprocess 


current_dir =os .path .dirname (os .path .abspath (__file__ ))


sys .path .insert (0 ,current_dir )

print (f"当前工作目录: {current_dir}")
print ("输入你需要去重的模态")
print ("1. 图片")
print ("2. 音频")
print ("3. 文本")
mode =input ("请输入数字(1/2/3): ")


if mode =="1":
    module_path =os .path .join (current_dir ,"image","method","image_manager.py")
    env_name ="tryamrosemdedup"
elif mode =="2":
    module_path =os .path .join (current_dir ,"audio","method","audio_manager.py")
    env_name ="audio"
elif mode =="3":
    module_path =os .path .join (current_dir ,"text","method","clean_the_dataset.py")
    env_name ="text-dedup"
else :
    print ("错误：无效的选择。请输入1、2或3。")
    sys .exit (1 )


if not os .path .exists (module_path ):
    print (f"错误：找不到模块文件 {module_path}")
    sys .exit (1 )


try :
    print (f"正在启动 {module_path}")
    if os .name =='nt':
        command =f'conda activate {env_name} && python "{module_path}"'
        subprocess .run (command ,shell =True )
    else :
        command =f'source activate {env_name} && python "{module_path}"'
        subprocess .run (command ,shell =True )
    print ("执行完成")
except Exception as e :
    print (f"执行过程中出错: {e}")