


import numpy as np 
import sys 
import os 
import math 
from tqdm import tqdm 

current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

from method_all .caculate_dedup import load_similar_pairs ,build_similarity_groups ,generate_dedup_report ,execute_deduplication ,extract_file_name 

try :
    from method_all .LSH_deal_with_photo import generate_minhash_signatures ,minHash ,count_bucket_collisions ,verify_similarity ,find_similar_items ,save_similar_pairs_to_file 
except ImportError :
    print ("警告: 无法导入 method_all.dedup_audio。")
    run_deduplication_with_threshold =None 









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

if __name__ =="__main__":


    targets =[0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]
    signature_length =200 

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


    print (f"results中0.1对应: {sign_results[0.1]}")





    output_dir =os .path .join (os .path .dirname (__file__ ),"similar_pairs")
    os .makedirs (output_dir ,exist_ok =True )
    output_dir_dedup =os .path .join (os .path .dirname (__file__ ),"dedup_results")
    os .makedirs (output_dir_dedup ,exist_ok =True )


    binary_data_path =os .path .join (parent_dir ,"binary_array_dict.npy")
    with open (binary_data_path ,"rb")as array_file :
        binary_array_dict =np .load (array_file ,allow_pickle =True ).item ()























    for threshold ,_ in tqdm (sign_results .items (),desc ="批量去重进度"):
        output_file =os .path .join (output_dir ,f"threshold_{threshold:.1f}_dedup.txt")

        similar_pairs =load_similar_pairs (output_file ,threshold =threshold )

        remove_files =build_similarity_groups (similar_pairs )
        generate_dedup_report (remove_files )
        output_file_result =os .path .join (output_dir_dedup ,f"threshold_{threshold:.1f}_dedup_result.txt")
        execute_deduplication (remove_files ,output_dir =output_file_result )
        extract_file_name ("danavery/urbansound8K",output_file_result ,remove_files )