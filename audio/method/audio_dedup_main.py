


import numpy as np 
import sys 
import os 
import math 
from tqdm import tqdm 
from pathlib import Path 
import soundfile as sf 
import json 



current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

from caculate_dedup import load_similar_pairs ,build_similarity_groups ,generate_dedup_report ,execute_deduplication ,extract_file_name ,extract_local_file_info 

try :
    from LSH_deal_with_photo import generate_minhash_signatures ,minHash ,count_bucket_collisions ,verify_similarity ,find_similar_items ,save_similar_pairs_to_file 
except ImportError :
    print ("警告: 无法导入 method_all.dedup_audio。")










def find_optimal_band_row (target_threshold ,signature_length =200 ):
    best_b ,best_r =None ,None 
    min_diff =float ('inf')
    best_actual_threshold =0 


    for b in range (1 ,signature_length +1 ):
        if signature_length %b ==0 :
            r =signature_length //b 


            if r <1 :
                continue 


            try :
                actual_threshold =(1.0 /b )**(1.0 /r )
                diff =abs (actual_threshold -target_threshold )


                if diff <min_diff :
                    min_diff =diff 
                    best_b ,best_r =b ,r 
                    best_actual_threshold =actual_threshold 
            except (ZeroDivisionError ,OverflowError ):
                continue 

    return (best_b ,best_r ),best_actual_threshold ,min_diff 

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

if __name__ =="__main__":


    config_path =os .path .join (current_dir ,"audio_config.json")
    data =load_config_json (config_path )


    targets =data .get ("processing",{}).get ("target_thresholds",[0.7 ,0.8 ,0.9 ])
    signature_length =data .get ("processing",{}).get ("signature_length",200 )

    print (f"LSH签名长度: {signature_length}")
    print ("目标阈值 -> 推荐参数(b, r) -> 实际阈值 -> 误差")
    print ("-"*50 )


    sign_results ={}

    for target in targets :
        (b ,r ),actual ,error =find_optimal_band_row (target ,signature_length )
        sign_results [target ]={"b":b ,"r":r ,"actual":actual ,"error":error }

        if b is None or r is None :
            print (f"{target:.1f} -> 无有效(b, r)组合")
        else :
            print (f"{target:.1f} -> (b={b:2d}, r={r:2d}) -> {actual:.4f} -> 误差:{error:.4f}")


    threshold_01_results =sign_results [0.7 ]
    print (f"results中0.7对应: {threshold_01_results}")





    output_dir =os .path .join (os .path .dirname (__file__ ),"similar_pairs")
    os .makedirs (output_dir ,exist_ok =True )


    output_dir_dedup =os .path .join (os .path .dirname (__file__ ),"dedup_results")


    try :
        os .makedirs (output_dir_dedup ,exist_ok =True )

        test_file =os .path .join (output_dir_dedup ,"test_write.tmp")
        with open (test_file ,'w')as f :
            f .write ("test")
        os .remove (test_file )
        print (f"去重结果将保存到: {output_dir_dedup}")
    except Exception as e :
        print (f"无法在指定目录创建文件: {e}")
        print ("请检查目录权限或以管理员身份运行")
        sys .exit (1 )





    binary_data_path =os .path .join (parent_dir ,"binary_array_dict.npy")
    with open (binary_data_path ,"rb")as array_file :
        binary_array_dict =np .load (array_file ,allow_pickle =True ).item ()

    matrix_true =np .array (list (binary_array_dict .values ())).T 
    print (f"加载了 {len(binary_array_dict)} 个音频指纹，矩阵形状: {matrix_true.shape}")

    for threshold ,params in tqdm (sign_results .items (),desc ="批量相似对计算进度"):
        b =params ["b"]
        r =params ["r"]
        if b is None or r is None :
            print (f"跳过阈值 {threshold:.1f}，无有效(b, r)组合")
            continue 

        print (f"\n针对阈值 {threshold:.1f} 使用参数 (b={b}, r={r}) 进行相似列统计:")


        hashBuckets_true =minHash (matrix_true ,b ,r )
        similar_pairs =find_similar_items (hashBuckets_true ,matrix_true ,similarity_threshold =threshold )


        output_file =os .path .join (output_dir ,f"threshold_{threshold:.1f}_dedup.txt")
        save_similar_pairs_to_file (similar_pairs ,output_file )

        print (f"  找到 {len(similar_pairs)} 对相似音频，结果保存到: {output_file}")

    for threshold ,_ in tqdm (sign_results .items (),desc ="批量去重进度"):
        output_file =os .path .join (output_dir ,f"threshold_{threshold:.1f}_dedup.txt")

        similar_pairs =load_similar_pairs (output_file ,threshold =threshold )
        remove_files =build_similarity_groups (similar_pairs )
        generate_dedup_report (remove_files )
        output_file_result =os .path .join (output_dir_dedup ,f"threshold_{threshold:.1f}_dedup_result.txt")
        execute_deduplication (remove_files ,output_dir =output_file_result )


        local_wav_dir =os .path .join (parent_dir ,"dataset")
        extract_local_file_info (local_wav_dir ,output_file_result ,remove_files )