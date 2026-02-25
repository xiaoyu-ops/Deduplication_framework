import yaml 
import random 
import numpy as np 
import logging 
import os 
import time 
import sys 
from clustering import compute_centroids 
from tqdm import tqdm 


os .environ ['KMP_DUPLICATE_LIB_OK']='TRUE'


logger =logging .getLogger (__name__ )
logger .setLevel (logging .INFO )
handler =logging .StreamHandler ()
handler .setFormatter (logging .Formatter ('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger .addHandler (handler )


current_dir =os .path .dirname (os .path .abspath (__file__ ))


def load_embeddings_safe (emb_path :str ,emb_size :int ,expected_N :int =None ):
    if not os .path .exists (emb_path ):
        raise FileNotFoundError (f"嵌入文件不存在: {emb_path}")


    try :
        arr =np .load (emb_path ,mmap_mode ='r')
        logger .info (f"np.load 成功, shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")
        if arr .ndim !=2 or arr .shape [1 ]!=emb_size :
            logger .warning (f"加载的 shape 与期望 emb_size 不符: {arr.shape} vs emb_size={emb_size}")
        return arr 
    except Exception as e :
        logger .warning (f"np.load 失败（可能不是标准 .npy），尝试 raw 二进制读取: {e}")


    filesize =os .path .getsize (emb_path )
    bytes_per_vec =4 *emb_size 
    if filesize %bytes_per_vec !=0 :
        raise RuntimeError (f"文件大小({filesize}) 不能被 emb_size({emb_size}) 整除，无法按 float32 reshape。")

    inferred_N =filesize //bytes_per_vec 
    if expected_N and expected_N !=inferred_N :
        logger .warning (f"配置 dataset_size={expected_N} 与文件推断 N={inferred_N} 不一致，采用推断值。")
    logger .info (f"按 float32 memmap 加载: shape=({inferred_N},{emb_size})")
    mem =np .memmap (emb_path ,dtype =np .float32 ,mode ='r',shape =(inferred_N ,emb_size ))
    return mem 


possible_config_paths =[
os .path .join (current_dir ,"configs","openclip","clustering_configs.yaml"),
os .path .join (current_dir ,"clustering","configs","openclip","clustering_configs.yaml"),
"configs/openclip/clustering_configs.yaml",
"clustering/configs/openclip/clustering_configs.yaml"
]

config_file =None 
for path in possible_config_paths :
    if os .path .exists (path ):
        config_file =path 
        break 

if config_file is None :
    logger .error ("找不到配置文件 clustering_configs.yaml")
    logger .info (f"当前工作目录: {os.getcwd()}")
    logger .info (f"尝试查找的路径: {possible_config_paths}")


    default_config_dir =os .path .join (current_dir ,"configs","openclip")
    os .makedirs (default_config_dir ,exist_ok =True )
    config_file =os .path .join (default_config_dir ,"clustering_configs.yaml")


    default_config ={
    'seed':42 ,
    'emb_memory_loc':os .path .join (current_dir ,'embeddings','image_embeddings.npy'),
    'paths_memory_loc':os .path .join (current_dir ,'embeddings','image_paths.npy'),
    'dataset_size':10000 ,
    'emb_size':512 ,
    'path_str_dtype':'U200',
    'ncentroids':1000 ,
    'niter':20 ,
    'Kmeans_with_cos_dist':True ,
    'save_folder':os .path .join (current_dir ,'clustering_results'),
    'sorted_clusters_file_loc':os .path .join (current_dir ,'sorted_clusters')
    }

    with open (config_file ,'w',encoding ='utf-8')as f :
        yaml .dump (default_config ,f ,default_flow_style =False )

    logger .info (f"创建了默认配置文件: {config_file}")
    logger .warning ("请检查并修改配置文件中的参数，特别是dataset_size")

logger .info (f"加载配置文件: {config_file}")


with open (config_file ,'r',encoding ='utf-8')as y_file :
    params =yaml .load (y_file ,Loader =yaml .FullLoader )


SEED =params ['seed']
random .seed (SEED )
np .random .seed (SEED )


logger .info (f"聚类参数: {params}")


emb_memory_loc =params ['emb_memory_loc']
paths_memory_loc =params ['paths_memory_loc']
dataset_size =params ['dataset_size']
emb_size =params ['emb_size']
path_str_type =params .get ('path_str_dtype','S24')


if not os .path .exists (emb_memory_loc ):
    logger .error (f"嵌入向量文件不存在: {emb_memory_loc}")
    exit (1 )


logger .info (f"加载嵌入向量: {emb_memory_loc}")
try :
    embeddings =load_embeddings_safe (emb_memory_loc ,emb_size ,dataset_size )
except Exception as e :
    logger .error (f"加载嵌入向量失败: {e}")
    exit (1 )


actual_N =embeddings .shape [0 ]
if dataset_size !=actual_N :
    logger .warning (f"配置的 dataset_size={dataset_size} 与文件实际大小 {actual_N} 不一致，已更新为实际值。")
    dataset_size =actual_N 
    params ['dataset_size']=actual_N 


paths_memory =None 
if os .path .exists (paths_memory_loc ):
    try :

        paths_memory =np .memmap (paths_memory_loc ,dtype =path_str_type ,mode ='r',shape =(dataset_size ,))
        logger .info (f"成功加载 paths，数量: {paths_memory.shape[0]}")
    except Exception as e :
        logger .warning (f"paths memmap 加载失败: {e}，尝试 np.load")
        try :
            paths_arr =np .load (paths_memory_loc ,mmap_mode ='r')

            if paths_arr .shape [0 ]!=dataset_size :
                logger .warning (f"paths 长度 {paths_arr.shape[0]} 与 dataset_size {dataset_size} 不一致")
            paths_memory =paths_arr 
            logger .info ("使用 np.load 成功加载 paths")
        except Exception as e2 :
            logger .error (f"无法加载 paths 文件: {e2}")
            paths_memory =None 
else :
    logger .warning (f"未找到 paths 文件: {paths_memory_loc}")

logger .info (f"嵌入向量形状: {embeddings.shape}")


save_folder =params ['save_folder']
sorted_clusters_folder =params ['sorted_clusters_file_loc']
os .makedirs (save_folder ,exist_ok =True )
os .makedirs (sorted_clusters_folder ,exist_ok =True )
logger .info (f"结果将保存到: {save_folder}")


from clustering import compute_centroids as original_compute_centroids 

def compute_centroids_with_progress (data ,ncentroids ,niter ,seed ,Kmeans_with_cos_dist ,save_folder ,logger ,verbose ):
    logger .info (f"开始聚类: {ncentroids}个聚类, {niter}次迭代")


    progress_bar =tqdm (total =niter ,desc ="K-means迭代",unit ="iter")


    original_info =logger .info 


    def new_info (msg ):
        original_info (msg )
        if "Iteration"in msg :
            try :
                iter_num =int (msg .split ("Iteration")[1 ].split ("/")[0 ].strip ())
                progress_bar .update (1 )
                progress_bar .set_description (f"K-means迭代 {iter_num}/{niter}")
            except :
                pass 


    logger .info =new_info 


    start_time =time .time ()

    try :

        result =original_compute_centroids (data ,ncentroids ,niter ,seed ,
        Kmeans_with_cos_dist ,save_folder ,
        logger ,verbose )


        progress_bar .update (niter )
        progress_bar .close ()


        total_time =time .time ()-start_time 
        logger .info (f"聚类完成! 用时: {total_time:.2f}秒")

        return result 
    except Exception as e :
        progress_bar .close ()
        logger .error (f"聚类过程中出错: {e}")
        raise 
    finally :

        logger .info =original_info 


logger .info ("开始聚类过程...")
try :
    compute_centroids_with_progress (
    data =embeddings ,
    ncentroids =params ['ncentroids'],
    niter =params ['niter'],
    seed =params ['seed'],
    Kmeans_with_cos_dist =params ['Kmeans_with_cos_dist'],
    save_folder =params ['save_folder'],
    logger =logger ,
    verbose =True ,
    )
    logger .info ("聚类完成！结果已保存到指定目录。")


    expected_files =[
    os .path .join (save_folder ,"centroids.npy"),
    os .path .join (save_folder ,"assignments.npy"),
    os .path .join (save_folder ,"kmeans.index")
    ]

    for file_path in expected_files :
        if os .path .exists (file_path ):
            file_size =os .path .getsize (file_path )/(1024 *1024 )
            logger .info (f"已生成文件: {file_path} ({file_size:.2f} MB)")
        else :
            logger .warning (f"未找到预期文件: {file_path}")

except KeyboardInterrupt :
    logger .info ("用户中断了聚类过程")
    sys .exit (1 )
except Exception as e :
    logger .error (f"聚类过程失败: {e}")
    import traceback 
    logger .error (traceback .format_exc ())
    sys .exit (1 )

