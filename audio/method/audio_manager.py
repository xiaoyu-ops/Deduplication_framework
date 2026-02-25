
import os 
import sys 
import shutil 
from pathlib import Path 
import glob 

current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
root_dir =os .path .dirname (parent_dir )


sys .path .insert (0 ,root_dir )
print (f"Added project root to sys.path: {root_dir}")


from env_manager .manager import EnvManager 

def create_deduplicated_datasets ():

    dataset_dir =os .path .join (parent_dir ,"dataset")
    dedup_results_dir =os .path .join (current_dir ,"dedup_results")

    print ("Starting creation of deduplicated dataset")
    print (f"Dataset directory: {dataset_dir}")
    print (f"Dedup results directory: {dedup_results_dir}")


    if not os .path .exists (dataset_dir ):
        print (f"Original dataset directory not found: {dataset_dir}")
        return False 


    wav_files =glob .glob (os .path .join (dataset_dir ,"*.wav"))
    wav_files .sort ()

    print (f"Original dataset contains {len(wav_files)} WAV files")

    if not wav_files :
        print ("No WAV files found in the dataset")
        return False 


    if not os .path .exists (dedup_results_dir ):
        print (f"Dedup results directory not found: {dedup_results_dir}")
        print ("Please run audio deduplication (option 2) to generate results first")
        return False 


    remove_files =glob .glob (os .path .join (dedup_results_dir ,"*_dedup_result.txt"))

    print (f"Searching for keep file lists in {dedup_results_dir}...")
    print (f"Search pattern: *_keep_files.txt")


    if os .path .exists (dedup_results_dir ):
        all_files =os .listdir (dedup_results_dir )
        print (f"Files in dedup results directory: {all_files}")


        result_files =[f for f in all_files if 'threshold_'in f and f .endswith ('.txt')]
        print (f"找到的结果文件: {result_files}")

    if not remove_files :
        print (f"No keep file lists found in {dedup_results_dir}")
        print ("Possible reasons:")
        print ("1. Deduplication has not been run yet")
        print ("2. Dedup result filenames do not match expected pattern")
        print ("3. Permission issues prevented file creation")


        alternative_files =glob .glob (os .path .join (dedup_results_dir ,"threshold_*_dedup_result.txt"))
        if alternative_files :
            print (f"Found dedup result files but no keep files: {alternative_files}")
            print ("This may indicate extract_local_file_info did not create keep files")

        return False 

    print (f"Found {len(remove_files)} dedup result files: {[os.path.basename(f) for f in remove_files]}")

    success_count =0 

    for remove_file in remove_files :
        try :

            filename =os .path .basename (remove_file )
            if 'threshold_'in filename :
                threshold =filename .split ('threshold_')[1 ].split ('_')[0 ]
            else :
                threshold ="unknown"

            print (f"\nProcessing dedup results for threshold {threshold}...")


            remove_indices =[]
            remove_list =os .path .join (remove_file ,"remove_files.txt")

            print (f"Reading remove file list: {remove_list}")

            if not os .path .exists (remove_list ):
                print (f"File does not exist: {remove_list}")
                continue 

            with open (remove_list ,'r',encoding ='utf-8')as f :
                for line in f :
                    line =line .strip ()

                    if line and line .isdigit ():
                        remove_indices .append (int (line ))

            if not remove_indices :
                print (f"Threshold {threshold}: no valid remove indices found")
                continue 

            print (f"Threshold {threshold}: found {len(remove_indices)} files to remove")


            all_indices =set (range (len (wav_files )))
            remove_indices_set =set (remove_indices )
            keep_indices =sorted (list (all_indices -remove_indices_set ))

            print (f"Threshold {threshold}: keeping {len(keep_indices)} files (removed {len(remove_indices)})")


            output_dataset_dir =os .path .join (dedup_results_dir ,f"threshold_{threshold}_dataset")
            os .makedirs (output_dataset_dir ,exist_ok =True )


            copied_count =0 
            error_count =0 

            for idx in keep_indices :
                try :
                    if idx <len (wav_files ):
                        src_file =wav_files [idx ]
                        dst_file =os .path .join (output_dataset_dir ,os .path .basename (src_file ))


                        shutil .copy2 (src_file ,dst_file )
                        copied_count +=1 
                    else :
                        print (f"Index {idx} out of range (max: {len(wav_files)-1})")
                        error_count +=1 

                except Exception as e :
                    print (f"Failed to copy file (index {idx}): {e}")
                    error_count +=1 


            stats_file =os .path .join (output_dataset_dir ,"dataset_stats.txt")
            with open (stats_file ,'w',encoding ='utf-8')as f :
                f .write (f"=== 去重数据集统计 (阈值: {threshold}) ===\n\n")
                f .write (f"原始文件数量: {len(wav_files)}\n")
                f .write (f"删除文件数量: {len(remove_indices)}\n")
                f .write (f"保留文件数量: {len(keep_indices)}\n")
                f .write (f"成功复制数量: {copied_count}\n")
                f .write (f"复制失败数量: {error_count}\n")
                f .write (f"去重率: {(len(remove_indices) / len(wav_files) * 100):.2f}%\n\n")


                total_size =0 
                for idx in keep_indices :
                    if idx <len (wav_files ):
                        try :
                            size =os .path .getsize (wav_files [idx ])
                            total_size +=size 
                        except :
                            pass 

                f .write (f"保留数据集总大小: {total_size / (1024*1024):.2f} MB\n")
                f .write (f"平均文件大小: {(total_size / len(keep_indices) / (1024*1024)):.2f} MB\n\n")

                f .write ("删除的文件索引:\n")
                for i ,idx in enumerate (sorted (remove_indices )):
                    if i %10 ==0 :
                        f .write ("\n")
                    f .write (f"{idx:4d} ")

                f .write ("\n\n保留的文件索引:\n")
                for i ,idx in enumerate (keep_indices ):
                    if i %10 ==0 :
                        f .write ("\n")
                    f .write (f"{idx:4d} ")

            print (f"Threshold {threshold}: copied {copied_count} files to {output_dataset_dir}")
            print (f"   Stats report: {stats_file}")

            if error_count >0 :
                print (f"Threshold {threshold}: {error_count} files failed to copy")

            success_count +=1 

        except Exception as e :
            print (f"处理文件 {remove_file} 时出错: {e}")
            continue 

    print ("\n"+"="*60 )
    print (f"Dataset creation complete! Successfully processed {success_count} thresholds")
    print ("="*60 )

    return success_count >0 

def show_menu ():
    print ("\n"+"="*50 )
    print ("音频去重处理菜单")
    print ("="*50 )
    print ("1. 生成音频指纹")
    print ("2. 执行音频去重")
    print ("3. 创建去重后的数据集")
    print ("4. 执行完整流程 (1+2+3)")
    print ("5. 退出")
    print ("="*50 )
    return input ("请选择操作 (1-5): ").strip ()

if __name__ =="__main__":

    switcher =EnvManager ()

    while True :
        choice =show_menu ()

        if choice =="1":
            print ("\n--- 生成音频指纹 ---")
            if switcher .setup_audio_env ():
                if os .name =='nt':
                    command =f'conda activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                else :
                    command =f'source activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("音频指纹生成成功")
                else :
                    print ("音频指纹生成失败")
            else :
                print ("环境设置失败")

        elif choice =="2":
            print ("\n--- 执行音频去重 ---")
            if switcher .setup_audio_env ():
                if os .name =='nt':
                    command =f'conda activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                else :
                    command =f'source activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'

                print (f"执行命令: {command}")
                res =os .system (command )
                if res ==0 :
                    print ("音频去重执行成功")
                else :
                    print ("音频去重执行失败")
            else :
                print ("环境设置失败")

        elif choice =="3":
            print ("\n--- 创建去重后的数据集 ---")
            success =create_deduplicated_datasets ()
            if not success :
                print ("数据集创建失败")

        elif choice =="4":
            print ("\n--- 执行完整流程 ---")
            success =True 


            print ("\n步骤1: 生成音频指纹...")
            if switcher .setup_audio_env ():
                if os .name =='nt':
                    command =f'conda activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'
                else :
                    command =f'source activate audio && python "{os.path.join(current_dir, "spectrum_fingerprint.py")}"'

                res =os .system (command )
                if res ==0 :
                    print ("音频指纹生成成功")
                else :
                    print ("音频指纹生成失败")
                    success =False 
            else :
                print ("环境设置失败")
                success =False 


            if success :
                print ("\n步骤2: 执行音频去重...")
                if os .name =='nt':
                    command =f'conda activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'
                else :
                    command =f'source activate audio && python "{os.path.join(current_dir, "audio_dedup_main.py")}"'

                res =os .system (command )
                if res ==0 :
                    print ("音频去重执行成功")
                else :
                    print ("音频去重执行失败")
                    success =False 


            if success :
                print ("\n步骤3: 创建去重后的数据集...")
                success =create_deduplicated_datasets ()

            if success :
                print ("\n完整流程执行成功!")
            else :
                print ("\n流程执行中断")

        elif choice =="5":
            print ("退出程序")
            break 

        else :
            print ("无效选择，请重新输入")

        input ("\n按回车键继续...")