








import os 
import glob 
import numpy as np 
import sys 
import json 

current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
root_dir =os .path .dirname (parent_dir )


sys .path .insert (0 ,root_dir )
print (f"已添加根目录到路径: {root_dir}")


from env_manager .manager import EnvManager 

def load_config_json (config_path ):
    try :
        with open (config_path ,'r',encoding ='utf-8')as f :
            return json .load (f )
    except FileNotFoundError :
        print (f"配置文件未找到: {config_path}")
        return None 
    except json .JSONDecodeError as e :
        print (f"配置文件格式错误: {e}")
        return None 

def show_menu ():
    print ("\n"+"="*50 )
    print ("图像去重处理菜单")
    print ("="*50 )
    print ("1. 提取嵌入向量")
    print ("2. 转换嵌入向量")
    print ("3. 进行k-means聚类")
    print ("4. 对聚类结果进行排序")
    print ("5. 进行去重")
    print ("6. 获取去重后数据")
    print ("7. 退出")
    print ("8. 查询当前配置文件")
    print ("9. 执行完整流程 (1+2+3+4+5+6)")
    print ("="*50 )
    return input ("请选择操作 (1-9): ").strip ()

if __name__ =="__main__":
    switcher =EnvManager ()
    while True :
        choice =show_menu ()

        if choice =="1":
            print ("\n--- 提取嵌入向量 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("提取嵌入向量成功")
                else :
                    print ("提取嵌入向量失败")
            else :
                print ("环境设置失败")

        if choice =="2":
            print ("\n--- 转换嵌入向量 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("转换嵌入向量成功")
                else :
                    print ("转换嵌入向量失败")
            else :
                print ("环境设置失败")

        elif choice =="3":
            print ("\n--- 进行k-means聚类 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("聚类成功")
                else :
                    print ("聚类失败")
            else :
                print ("环境设置失败")

        elif choice =="4":
            print ("\n--- 对聚类结果进行排序 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("排序成功")
                else :
                    print ("排序失败")
            else :
                print ("环境设置失败")

        elif choice =="5":
            print ("\n--- 进行去重 ---")
            if switcher .setup_image_env ():

                config_path =r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list ="0.05"
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("去重成功")
                else :
                    print ("去重失败")
            else :
                print ("环境设置失败")

        elif choice =="6":
            print ("\n--- 提取去重后数据 ---")
            if switcher .setup_image_env ():

                config_path =r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list ="0.05"
                eps_value =eps_list .split (",")[0 ].strip ()

                output_folder =os .path .join (root_dir ,"image","deduped",f"eps_{eps_value.replace('.','_')}")
                os .makedirs (output_folder ,exist_ok =True )
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("提取成功")
                else :
                    print ("提取失败")
            else :
                print ("环境设置失败")

        elif choice =="7":
            print ("退出程序")
            break 

        elif choice =="8":
            config_path =os .path .join (current_dir ,"image_config.json")
            data =load_config_json (config_path )
            print ("以下为当下配置情况")
            print (data )

        elif choice =="9":
            print ("\n--- 提取嵌入向量 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "main.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("提取嵌入向量成功")
                else :
                    print ("提取嵌入向量失败")
            else :
                print ("环境设置失败")

            print ("\n--- 转换嵌入向量 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "load_and_convert_embeddings.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("转换嵌入向量成功")
                else :
                    print ("转换嵌入向量失败")
            else :
                print ("环境设置失败")

            print ("\n--- 进行k-means聚类 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "run_clustering_local.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("聚类成功")
                else :
                    print ("聚类失败")
            else :
                print ("环境设置失败")

            print ("\n--- 对聚类结果进行排序 ---")
            if switcher .setup_image_env ():
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "sort_clusters_in_windows.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("排序成功")
                else :
                    print ("排序失败")
            else :
                print ("环境设置失败")

            print ("\n--- 进行去重 ---")
            if switcher .setup_image_env ():

                config_path =r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list ="0.15"
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "simple_semdedup.py")}" --config-file "{config_path}" --eps-list "{eps_list}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("去重成功")
                else :
                    print ("去重失败")
            else :
                print ("环境设置失败")

            print ("\n--- 提取去重后数据 ---")
            if switcher .setup_image_env ():
                config_path =r"D:\Deduplication_framework\image\method\semdedup_configs.yaml"
                eps_list ="0.15"
                eps_value =eps_list .split (",")[0 ].strip ()
                output_folder =os .path .join (root_dir ,"image","deduped",f"eps_{eps_value.replace('.','_')}")
                os .makedirs (output_folder ,exist_ok =True )
                if os .name =='nt':
                    command =f'conda activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'
                else :
                    command =f'source activate tryamrosemdedup && python "{os.path.join(current_dir, "create_deduped_dataset.py")}" --config-file "{config_path}" --eps {eps_value} --output-folder "{output_folder}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("提取成功")
                else :
                    print ("提取失败")
            else :
                print ("环境设置失败")

        else :
            print ("无效选择，请重新输入")

        input ("\n按回车键继续...")