





import faiss 
import torch 
import time 
import numpy as np 
import logging 
import os 
import pickle 
import argparse 
import yaml 
import pprint 
import submitit 
import pathlib 
from typing import Union ,Optional 



def faiss_index_to_gpu (cpu_index ):
    try :

        if hasattr (faiss ,'GpuClonerOptions'):

            cloner_options =faiss .GpuClonerOptions ()
            cloner_options .useFloat16 =False 
            cloner_options .usePrecomputed =False 
            cloner_options .indicesOptions =faiss .INDICES_CPU 


            gpu_resources =faiss .StandardGpuResources ()


            gpu_index =faiss .index_cpu_to_gpu (gpu_resources ,0 ,cpu_index ,cloner_options )
            return gpu_index 
        else :

            print ("FAISS GPU 功能不可用，将使用 CPU 版本继续...")
            return cpu_index 
    except Exception as e :
        print (f"转移到 GPU 时出错: {e}，将使用 CPU 版本继续...")
        return cpu_index 


def compute_centroids (
data :Union [np .memmap ,np .ndarray ],
ncentroids :int =1000 ,
niter :int =100 ,
seed :int =1234 ,
Kmeans_with_cos_dist :bool =False ,
save_folder :str ="",
logger :logging .Logger =None ,
verbose :bool =True ,
):

    os .makedirs (save_folder ,exist_ok =True )


    logger .info (
    f"Running Kmeans clustering using faiss on dataset of shape {data.shape} ...."
    )
    logger .info (f"Kmeans parameters: {locals()} ....")



    d =data .shape [1 ]


    use_gpu =torch .cuda .is_available ()

    device ="cuda"if use_gpu else "cpu"

    logger .info (f"Clustering on {device} ....")

    spherical =(
    Kmeans_with_cos_dist 

    )



    kmeans =faiss .Kmeans (
    d ,
    ncentroids ,
    niter =niter ,
    verbose =verbose ,
    seed =seed ,
    spherical =spherical ,
    gpu =use_gpu ,
    )




    kmeans_obj_file_loc =pathlib .Path (save_folder ,"kmeans_index.pickle")

    if not os .path .exists (kmeans_obj_file_loc ):
        start_time =time .time ()
        kmeans .train (data )
        logger .info (f"Time for clustering (mins): {(time.time()-start_time)/(60):.2f}")



        if hasattr (faiss ,'index_gpu_to_cpu'):
            kmeans_index =faiss .index_gpu_to_cpu (kmeans .index )
        else :

            kmeans_index =kmeans .index 
        logger .info (f"faiss kmeans index to store: {type(kmeans_index)}")


        with open (kmeans_obj_file_loc ,"wb")as file :
            pickle .dump (kmeans_index ,file )


        np .save (pathlib .Path (save_folder ,"kmeans_centroids.npy"),kmeans .centroids )

        logger .info (f"Saved!")

    else :


        logger .info (
        f"Loading faiss Kmeans index pickle file from {kmeans_obj_file_loc}"
        )
        with open (kmeans_obj_file_loc ,"rb")as file :
            kmeans_index =pickle .load (file )
            if use_gpu :


                kmeans_index =faiss_index_to_gpu (kmeans_index )
            kmeans .index =kmeans_index 





    start_time =time .time ()
    dist_to_cent ,nearest_cent =kmeans .index .search (data ,1 )
    dist_to_cent ,nearest_cent =dist_to_cent .squeeze (1 ),nearest_cent .squeeze (1 )
    logger .info (
    f"Time for finding nearest centroid for each data point (mins): {(time.time()-start_time)/(60):.2f}"
    )



    dist_to_cent_file =pathlib .Path (save_folder ,"dist_to_cent.npy")
    nearest_cent_file =pathlib .Path (save_folder ,"nearest_cent.npy")
    np .save (dist_to_cent_file ,dist_to_cent )
    np .save (nearest_cent_file ,nearest_cent )

    return kmeans 


def main (args ):




    confg_file =args .confg_file 

    with open (confg_file ,"r")as y_file :
        params =yaml .load (y_file ,Loader =yaml .FullLoader )

    with open (
    pathlib .Path (params ["save_folder"],"clustering_params.txt"),"w"
    )as fout :
        pprint .pprint (params ,fout )



    seed =params ["seed"]
    emb_memory_loc =params [
    "emb_memory_loc"
    ]
    dataset_size =params ["dataset_size"]
    emb_size =params ["emb_size"]
    niter =params ["niter"]
    ncentroids =params ["ncentroids"]
    save_folder =params ["save_folder"]
    Kmeans_with_cos_dist =params ["Kmeans_with_cos_dist"]



    data =np .memmap (
    emb_memory_loc ,dtype ="float32",mode ="r",shape =(dataset_size ,emb_size )
    )



    compute_centroids (
    data ,
    ncentroids ,
    niter ,
    seed ,
    Kmeans_with_cos_dist ,
    save_folder ,
    True ,
    )


if __name__ =="__main__":


    parser =argparse .ArgumentParser ()

    parser .add_argument (
    "--confg-file",
    type =str ,
    default ="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
    help =".yaml config file path",
    )


    parser .add_argument (
    "--partition",type =str ,default ="scaling_data_pruning",help ="partition"
    )
    parser .add_argument ("--ngpus",type =int ,default =1 ,help ="number of gpus")
    parser .add_argument ("--cpus-per-task",type =int ,default =10 ,help ="number of cpus")
    parser .add_argument (
    "--timeout",type =int ,default =1500 ,help ="job timeout in minutes"
    )

    args =parser .parse_args ()



    with open (args .confg_file ,"r")as y_file :
        params =yaml .load (y_file ,Loader =yaml .FullLoader )



    args .save_folder =params ["save_folder"]



    PARTITION =args .partition 
    NODES =1 
    NGPUS =args .ngpus 
    CPUS_PER_TASKS =args .cpus_per_task 
    TIMEOUT =args .timeout 



    submitit_path =f"{args.save_folder}/compute_centorids_job_%j"
    executor =submitit .AutoExecutor (folder =submitit_path ,slurm_max_num_timeout =30 )
    executor .update_parameters (
    slurm_partition =PARTITION ,
    nodes =NODES ,
    tasks_per_node =1 ,
    cpus_per_task =CPUS_PER_TASKS ,
    gpus_per_node =NGPUS ,
    slurm_mem_per_gpu ="55G",
    timeout_min =TIMEOUT ,
    )



    job =executor .submit (main ,args )
    print ("Submitted job_id:",job .job_id )
