import os 
import numpy as np 
import random 
from collections import defaultdict 
from datasets import load_dataset 
import json 
import glob 

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


def load_similar_pairs (file_path ,threshold =0.83 ):
    similar_pairs =[]
    with open (file_path ,'r')as f :
        for line in f :
            start =line .find ("(")+1 
            end =line .find (")")
            pair =line [start :end ].split (", ")
            start_1 =line .find ("Similarity: ")+len ("Similarity: ")
            similarity_str =line [start_1 :]
            similarity =float (similarity_str )
            if similarity >=threshold :
                if len (pair )==2 :
                    similar_pairs .append ((int (pair [0 ]),int (pair [1 ])))
    return similar_pairs 


def build_similarity_groups (similar_pairs ):

    keep_files =set ()
    remove_files =set ()
    for file1 ,file2 in similar_pairs :
        if random .random ()<0.5 :
            remove_files .add (file2 )
    return remove_files 


def generate_dedup_report (remove_files ):
    print ("\n==== 音频去重结果报告 ====")
    print (f"删除文件数: {len(remove_files)}")


def execute_deduplication (remove_files ,output_dir =None ,move_files =False ):
    if output_dir is None :
        print ("\n仅生成报告，未执行实际文件操作")
        return 

    os .makedirs (output_dir ,exist_ok =True )


    with open (os .path .join (output_dir ,"remove_files.txt"),"w")as f :
        for file in remove_files :
            f .write (f"{file}\n")

    print (f"\n文件列表已保存到 {output_dir}")


    if move_files :
        print ("移动文件功能尚未实现")

def extract_file_name (dataset_name ,output_dir ,remove_files ):
    dataset =load_dataset (dataset_name )
    length =len (dataset ['train'])

    keep_files_data ={
    "metadata":{
    "dataset_name":dataset_name ,
    "total_files":length ,
    "kept_files_count":0 ,
    "removed_files_count":len (remove_files ),
    "dedup_rate":0.0 
    },
    "kept_files":[]
    }

    kept_count =0 
    print (remove_files )

    for index in range (length ):
        if index not in remove_files :
            file_data =dataset ['train'][index ]


            file_info ={
            "index":index ,
            "slice_file_name":file_data .get ('slice_file_name',''),
            "fsID":file_data .get ('fsID',''),
            "start":file_data .get ('start',0 ),
            "end":file_data .get ('end',0 ),
            "salience":file_data .get ('salience',0 ),
            "fold":file_data .get ('fold',0 ),
            "classID":file_data .get ('classID',0 ),
            "class":file_data .get ('class',''),
            "audio_path":file_data .get ('audio',{}).get ('path',''),
            "sampling_rate":file_data .get ('audio',{}).get ('sampling_rate',0 )
            }

            keep_files_data ["kept_files"].append (file_info )
            kept_count +=1 


    keep_files_data ["metadata"]["kept_files_count"]=kept_count 
    keep_files_data ["metadata"]["dedup_rate"]=len (remove_files )/length 


    json_file_path =os .path .join (output_dir ,"keep_files.json")
    with open (json_file_path ,"w",encoding ='utf-8')as f :
        json .dump (keep_files_data ,f ,indent =2 ,ensure_ascii =False )

    print (f"去重率为: {len(remove_files) / length:.2%}")
    print (f"保留文件数: {kept_count}")
    print (f"总文件数: {len(keep_files_data['kept_files'])}")
    print (f"JSON文件已保存到: {json_file_path}")

def extract_local_file_info (wav_dir ,output_file ,remove_files ):
    import glob 

    try :

        wav_files =glob .glob (os .path .join (wav_dir ,"*.wav"))



        wav_files .sort ()

        print (f"找到 {len(wav_files)} 个WAV文件")
        print (f"前5个文件: {[os.path.basename(f) for f in wav_files[:5]]}")


        index_to_file ={}

        for i ,wav_file in enumerate (wav_files ):
            filename =os .path .basename (wav_file )


            index_to_file [i ]=wav_file 
            index_to_file [str (i )]=wav_file 


            if filename .startswith ('audio_')and filename .endswith ('.wav'):
                try :

                    number_str =filename [6 :-4 ]
                    if number_str .isdigit ():
                        file_index =int (number_str )
                        index_to_file [file_index ]=wav_file 
                        index_to_file [str (file_index )]=wav_file 
                        index_to_file [number_str ]=wav_file 

                except (ValueError ,IndexError ):
                    pass 

        print (f"创建了 {len(index_to_file)} 个索引映射")
        print (f"要删除的文件索引数量: {len(remove_files)}")


        files_to_remove =[]
        not_found_files =[]

        for file_id in remove_files :
            matched_path =None 


            search_keys =[
            file_id ,
            str (file_id ),
            int (file_id )if str (file_id ).isdigit ()else None ,
            ]


            search_keys =[k for k in search_keys if k is not None ]

            for key in search_keys :
                if key in index_to_file :
                    matched_path =index_to_file [key ]
                    break 

            if matched_path :
                try :
                    file_size =os .path .getsize (matched_path )
                    files_to_remove .append ({
                    'filename':os .path .basename (matched_path ),
                    'path':matched_path ,
                    'size_mb':file_size /(1024 *1024 ),
                    'index':file_id 
                    })
                except OSError as e :
                    print (f"无法获取文件大小: {matched_path}, 错误: {e}")
            else :
                not_found_files .append (file_id )

        print (f"匹配成功: {len(files_to_remove)} 个文件")
        print (f"未找到: {len(not_found_files)} 个文件")


        all_indices =set (range (len (wav_files )))
        remove_indices =set (int (str (f ))for f in remove_files if str (f ).isdigit ()and int (str (f ))<len (wav_files ))
        keep_indices =sorted (list (all_indices -remove_indices ))


        content =f"=== 音频去重结果详细报告 ===\n\n"
        content +=f"总文件数: {len(wav_files)}\n"
        content +=f"要删除文件数: {len(files_to_remove)}\n"
        content +=f"保留文件数: {len(keep_indices)}\n"
        content +=f"删除总大小: {sum(item['size_mb'] for item in files_to_remove):.2f} MB\n\n"

        content +="=== 要删除的文件列表 ===\n"
        for item in files_to_remove :
            content +=f"索引: {item['index']:4} | 文件名: {item['filename']} | 大小: {item['size_mb']:.2f} MB\n"

        content +=f"\n=== 保留的文件索引 ({len(keep_indices)} 个) ===\n"

        for i in range (0 ,len (keep_indices ),10 ):
            line_indices =keep_indices [i :i +10 ]
            content +=" ".join (f"{idx:4d}"for idx in line_indices )+"\n"

        if not_found_files :
            content +=f"\n=== 未匹配的索引 ({len(not_found_files)} 个) ===\n"
            for i in range (0 ,min (len (not_found_files ),50 ),10 ):
                line_indices =not_found_files [i :i +10 ]
                content +=" ".join (str (idx )for idx in line_indices )+"\n"


        try :

            output_dir =os .path .dirname (output_file )
            if output_dir :
                os .makedirs (output_dir ,exist_ok =True )

            with open (output_file ,'w',encoding ='utf-8')as f :
                f .write (content )
            print (f"详细报告已保存到: {output_file}")


            keep_file =output_file .replace ('.txt','_keep_files.txt')
            with open (keep_file ,'w',encoding ='utf-8')as f :
                f .write ("=== 去重后保留的文件索引 ===\n\n")
                for idx in keep_indices :
                    if idx <len (wav_files ):
                        filename =os .path .basename (wav_files [idx ])
                        f .write (f"{idx:4d} : {filename}\n")
            print (f"保留文件列表已保存到: {keep_file}")

        except Exception as e :
            print (f"保存失败: {e}")
            print ("详细报告显示在控制台:")
            print ("="*60 )
            print (content [:2000 ]+"..."if len (content )>2000 else content )
            print ("="*60 )

        return files_to_remove 

    except Exception as e :
        print (f"extract_local_file_info执行失败: {e}")
        import traceback 
        traceback .print_exc ()
        return []

if __name__ =="__main__":

    similar_pairs =load_similar_pairs ("similar_pairs.txt")
    print (f"加载的相似对: {similar_pairs}")

    remove_files =build_similarity_groups (similar_pairs )
    generate_dedup_report (remove_files )
    execute_deduplication (remove_files ,output_dir ="dedup_results")
    extract_file_name ("danavery/urbansound8K","dedup_results",remove_files )
