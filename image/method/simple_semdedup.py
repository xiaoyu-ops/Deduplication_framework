import os 
import numpy as np 
import pandas as pd 
import argparse 
import yaml 
from tqdm import tqdm 
import torch 
from pathlib import Path 
import pickle 
import time 
import math 

def parse_args ():
    parser =argparse .ArgumentParser (description ="Windows版语义去重")
    parser .add_argument ('--config-file',required =True ,help ='配置文件路径')
    parser .add_argument ('--eps-list',required =True ,help ='epsilon值列表，用逗号分隔')
    parser .add_argument ('--max-clusters',type =int ,help ='限制处理的最大簇数量')
    parser .add_argument ('--random-seed',type =int ,default =42 ,help ='随机种子')
    parser .add_argument ('--which-to-keep',type =str ,default ='hard',choices =['hard','random','easy'],
    help ='保留哪些样本：hard(困难)、random(随机)或easy(容易)')
    parser .add_argument ('--batch-size',type =int ,default =30 ,help ='批处理大小')
    return parser .parse_args ()

def init_memmap_embs (embs_memory_loc ,dataset_size =None ,emd_size =512 ,dtype ="float32"):
    if not os .path .exists (embs_memory_loc ):
        raise FileNotFoundError (f"嵌入文件不存在: {embs_memory_loc}")


    try :
        arr =np .load (embs_memory_loc ,mmap_mode ='r')

        if arr .ndim !=2 or arr .shape [1 ]!=emd_size :
            print (f"警告: 加载的嵌入 shape={getattr(arr,'shape',None)} 与期望 emd_size={emd_size} 不一致")
        return arr 
    except Exception as e :
        print (f"np.load mmap 加载失败(可能不是标准 .npy){e}")


    try :
        bpp =np .dtype (dtype ).itemsize 
        filesize =os .path .getsize (embs_memory_loc )
        if emd_size <=0 :
            raise RuntimeError ("emd_size 必须为正整数")
        if filesize %(bpp *emd_size )!=0 :
            raise RuntimeError (f"文件大小 {filesize} 不能被 (dtype bytes {bpp} * emd_size {emd_size}) 整除，无法按 raw reshape")
        inferred_N =filesize //(bpp *emd_size )
        if dataset_size and dataset_size !=inferred_N :
            print (f"警告: 配置 dataset_size={dataset_size} 与文件推断 N={inferred_N} 不一致，采用推断值")
        print (f"按 raw {dtype} memmap 加载: shape=({inferred_N},{emd_size})")
        mem =np .memmap (embs_memory_loc ,dtype =dtype ,mode ='r',shape =(inferred_N ,emd_size ))
        return mem 
    except Exception as e :
        raise RuntimeError (f"无法以安全方式加载嵌入文件: {e}\n建议：用 numpy.lib.format.open_memmap 或 np.save 将嵌入写为标准 .npy，然后重试。")

def load_cluster_file (cluster_path ):
    if not os .path .exists (cluster_path ):
        return None 

    try :
        if cluster_path .endswith ('.npy'):
            return np .load (cluster_path )
        else :

            with open (cluster_path ,'r')as f :
                lines =f .readlines ()


            data =[]
            for line in lines :
                parts =line .strip ().split ('\t')
                if len (parts )>=3 :
                    idx =int (parts [0 ])
                    path =parts [1 ]
                    dist =float (parts [2 ])
                    data .append ((idx ,path ,dist ))
                elif len (parts )>=1 :
                    idx =int (parts [0 ])
                    data .append ((idx ,"",0.0 ))

            return np .array (data ,dtype =object )
    except Exception as e :
        print (f"无法加载文件 {cluster_path}: {e}")
        return None 

def semdedup (cluster ,cluster_reps ,device ="cpu"):
    st =time .time ()


    cluster_reps =torch .tensor (cluster_reps ,dtype =torch .float32 ).to (device )


    pair_w_sim_matrix =cluster_reps @cluster_reps .T 


    pair_w_sim_matrix .fill_diagonal_ (0.0 )


    assert pair_w_sim_matrix .shape [0 ]==pair_w_sim_matrix .shape [1 ]


    triu_sim_mat =torch .triu (pair_w_sim_matrix ,diagonal =1 )


    M =torch .max (triu_sim_mat ,dim =0 )[0 ].cpu ()

    print (f"计算相似度矩阵耗时: {time.time()-st:.2f}秒")

    return M 

def process_cluster (cluster_id ,config ,embs ,device ):
    st =time .time ()


    df_file_loc =os .path .join (config ['save_folder'],f"dataframes/cluster_{cluster_id}.pkl")
    if os .path .exists (df_file_loc ):
        print (f"{df_file_loc} 已存在，跳过")
        return None 


    cluster_paths =[
    os .path .join (config ['sorted_clusters_path'],f"cluster_{cluster_id}.npy"),
    os .path .join (config ['sorted_clusters_path'],f"sorted_cluster_{cluster_id}.txt")
    ]

    cluster_i =None 
    for path in cluster_paths :
        if os .path .exists (path ):
            cluster_i =load_cluster_file (path )
            if cluster_i is not None :
                print (f"加载簇文件: {path}")
                break 

    if cluster_i is None :
        print (f"无法加载簇 {cluster_id} 的数据，跳过")
        return None 


    cluster_size =len (cluster_i )
    print (f"簇 {cluster_id} 大小: {cluster_size}")


    if cluster_size ==1 :
        points_to_remove_df =pd .DataFrame ()
        points_to_remove_df ["indices"]=[0 ]
        for eps in config ['eps_list']:
            points_to_remove_df [f"eps={eps}"]=[False ]


        os .makedirs (os .path .dirname (df_file_loc ),exist_ok =True )
        with open (df_file_loc ,"wb")as file :
            pickle .dump (points_to_remove_df ,file )

        print (f"处理簇 {cluster_id} 完成")
        return points_to_remove_df 


    which_to_keep =config .get ('which_to_keep','hard').lower ()
    cluster_items_indices =list (range (cluster_size ))

    if which_to_keep =="random":

        np .random .shuffle (cluster_items_indices )
        cluster_i =cluster_i [cluster_items_indices ]
    elif which_to_keep =="easy":

        cluster_items_indices =cluster_items_indices [::-1 ]
        cluster_i =cluster_i [cluster_items_indices ]


    if isinstance (cluster_i [0 ],tuple )or isinstance (cluster_i [0 ],list ):

        cluster_ids =np .array ([int (item [0 ])for item in cluster_i ])
    else :

        cluster_ids =cluster_i [:,1 ].astype ("int32")


    try :
        cluster_reps =embs [cluster_ids ]
    except Exception as e :
        print (f"获取嵌入失败: {e}")

        cluster_reps =np .random .rand (len (cluster_ids ),config ['emd_size']).astype (np .float32 )


    M =semdedup (cluster_i ,cluster_reps ,device )


    points_to_remove_df =pd .DataFrame ()
    points_to_remove_df ["indices"]=cluster_items_indices 


    for eps in config ['eps_list']:
        eps_points_to_remove =M >1 -eps 
        points_to_remove_df [f"eps={eps}"]=eps_points_to_remove 


    os .makedirs (os .path .dirname (df_file_loc ),exist_ok =True )
    with open (df_file_loc ,"wb")as file :
        pickle .dump (points_to_remove_df ,file )

    print (f"处理簇 {cluster_id} 完成，耗时: {time.time()-st:.2f}秒")
    return points_to_remove_df 

def run_semdedup (config ,args ):
    print (f"配置信息: {config}")


    save_loc =config ['save_folder']
    os .makedirs (os .path .join (save_loc ,"dataframes"),exist_ok =True )


    for eps in config ['eps_list']:
        os .makedirs (os .path .join (save_loc ,f"eps_{eps}"),exist_ok =True )


    print ("检查现有结果文件的一致性...")
    dataframes_dir =os .path .join (save_loc ,"dataframes")
    if os .path .exists (dataframes_dir ):
        for filename in os .listdir (dataframes_dir ):
            if filename .startswith ("cluster_")and filename .endswith (".pkl"):
                filepath =os .path .join (dataframes_dir ,filename )
                try :
                    with open (filepath ,"rb")as f :
                        df =pickle .load (f )


                    expected_cols =[f"eps={eps}"for eps in config ['eps_list']]
                    missing_cols =[col for col in expected_cols if col not in df .columns ]

                    if missing_cols :
                        print (f"删除不一致的结果文件: {filename} (缺少列: {missing_cols})")
                        os .remove (filepath )

                except Exception as e :
                    print (f"删除损坏的结果文件: {filename} (错误: {e})")
                    os .remove (filepath )


    embs =init_memmap_embs (
    config ['embs_memory_loc'],
    config ['dataset_size'],
    config ['emd_size']
    )


    device ="cuda"if torch .cuda .is_available ()else "cpu"
    print (f"使用设备: {device}")


    num_clusters =config ['num_clusters']
    if args .max_clusters :
        num_clusters =min (args .max_clusters ,num_clusters )
        print (f"限制处理簇数量为: {num_clusters}")


    batch_size =args .batch_size 
    total_batches =(num_clusters +batch_size -1 )//batch_size 


    total_start_time =time .time ()


    for batch_idx in range (total_batches ):
        batch_start =batch_idx *batch_size 
        batch_end =min ((batch_idx +1 )*batch_size ,num_clusters )

        print (f"处理簇 {batch_start} 到 {batch_end-1}")
        batch_start_time =time .time ()


        for cluster_idx in tqdm (range (batch_start ,batch_end )):
            process_cluster (cluster_idx ,config ,embs ,device )

        print (f"批次 {batch_idx+1}/{total_batches} 完成，耗时: {time.time()-batch_start_time:.2f}秒")

    print (f"所有簇处理完成，总耗时: {(time.time()-total_start_time)/60:.2f}分钟")


    print ("合并结果...")

    for eps in config ['eps_list']:
        print (f"处理 epsilon = {eps}")


        kept_samples =[]


        for cluster_idx in tqdm (range (num_clusters )):

            df_file_loc =os .path .join (save_loc ,"dataframes",f"cluster_{cluster_idx}.pkl")

            if not os .path .exists (df_file_loc ):
                continue 


            with open (df_file_loc ,"rb")as file :
                df =pickle .load (file )


            cluster_paths =[
            os .path .join (config ['sorted_clusters_path'],f"cluster_{cluster_idx}.npy"),
            os .path .join (config ['sorted_clusters_path'],f"sorted_cluster_{cluster_idx}.txt")
            ]

            cluster_i =None 
            for path in cluster_paths :
                if os .path .exists (path ):
                    cluster_i =load_cluster_file (path )
                    if cluster_i is not None :
                        break 

            if cluster_i is None :
                continue 


            eps_col =f"eps={eps}"
            if eps_col not in df .columns :
                print (f"警告: 簇 {cluster_idx} 的结果中缺少列 '{eps_col}'，跳过该簇")
                continue 


            if isinstance (cluster_i [0 ],tuple )or isinstance (cluster_i [0 ],list ):

                cluster_ids =np .array ([int (item [0 ])for item in cluster_i ])
            else :

                cluster_ids =cluster_i [:,1 ].astype ("int32")


            if len (df ["indices"])!=len (cluster_ids ):
                print (f"警告: 簇 {cluster_idx} 的DataFrame长度({len(df['indices'])})与簇大小({len(cluster_ids)})不匹配，跳过该簇")
                continue 


            try :
                not_to_remove =~df [eps_col ].values 
                indices_to_keep =df ["indices"][not_to_remove ].values 
            except Exception as e :
                print (f"警告: 处理簇 {cluster_idx} 时出错: {e}，跳过该簇")
                continue 


            kept_global_indices =[cluster_ids [i ]for i in indices_to_keep ]
            kept_samples .extend (kept_global_indices )


            with open (os .path .join (save_loc ,f"eps_{eps}",f"kept_cluster_{cluster_idx}.txt"),'w')as f :
                for idx in kept_global_indices :
                    f .write (f"{idx}\n")


        kept_samples =sorted (set (kept_samples ))


        with open (os .path .join (save_loc ,f"eps_{eps}","all_kept_samples.txt"),'w')as f :
            for idx in kept_samples :
                f .write (f"{idx}\n")


        print (f"epsilon={eps} 完成: 从 {config['dataset_size']} 个样本中保留了 {len(kept_samples)} 个样本 ({len(kept_samples)/config['dataset_size']*100:.2f}%)")


        df =pd .DataFrame ({
        'epsilon':[eps ],
        'total_samples':[config ['dataset_size']],
        'kept_samples':[len (kept_samples )],
        'removed_samples':[config ['dataset_size']-len (kept_samples )],
        'kept_percentage':[len (kept_samples )/config ['dataset_size']*100 ]
        })

        df .to_csv (os .path .join (save_loc ,"dataframes",f"results_eps_{eps}.csv"),index =False )


    results_files =[f for f in os .listdir (os .path .join (save_loc ,"dataframes"))if f .startswith ("results_eps_")]
    if results_files :
        all_results =pd .concat ([pd .read_csv (os .path .join (save_loc ,"dataframes",f ))for f in results_files ])
        all_results .to_csv (os .path .join (save_loc ,"dataframes","all_results.csv"),index =False )
        print ("所有结果已汇总到 all_results.csv")

def main ():
    args =parse_args ()


    np .random .seed (args .random_seed )
    torch .manual_seed (args .random_seed )


    with open (args .config_file ,'r',encoding ='utf-8')as f :
        config =yaml .safe_load (f )


    eps_list =[float (eps )for eps in args .eps_list .split (',')]
    config ['eps_list']=eps_list 


    config ['which_to_keep']=args .which_to_keep 


    run_semdedup (config ,args )

if __name__ =="__main__":
    main ()