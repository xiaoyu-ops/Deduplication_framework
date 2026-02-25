





import torch 
import time 
import numpy as np 
import logging 
from tqdm import tqdm 
import pandas as pd 
import os 
import pathlib 
import yaml 
import math 
import os 
from typing import List 
import random 
import numpy as np 
import submitit 
import torch 
import pprint 
from tqdm import tqdm 
import argparse 
from typing import List ,Tuple ,Union 



def assign_and_sort_clusters (
data :Union [np .memmap ,np .ndarray ],
paths_list :Union [np .memmap ,np .ndarray ],
sim_metric :str ="cosine",
keep_hard :bool =True ,
kmeans_with_cos_dist :bool =False ,
save_folder :str ="",
sorted_clusters_file_loc :str ="",
cluster_ids =range (5000 ),
logger :logging .Logger =None ,
)->pd .DataFrame :

    assert sim_metric in [
    "l2",
    "cosine",
    ],f"Unsupported similarity metric '{sim_metric}'."
    assert not (
    kmeans_with_cos_dist and sim_metric =="l2"
    ),"Cannot use cosine distance with L2 similarity metric."



    spherical =kmeans_with_cos_dist 



    logger .info ("Ranking...")
    kmeans_centroids_file_loc =pathlib .Path (save_folder ,"kmeans_centroids.npy")
    dist_to_cent_file_loc =pathlib .Path (save_folder ,"dist_to_cent.npy")
    nearest_cent_file_loc =pathlib .Path (save_folder ,"nearest_cent.npy")
    kmeans_centroids =np .load (kmeans_centroids_file_loc )
    nearest_cent =np .load (nearest_cent_file_loc )
    dist_to_cent =np .load (dist_to_cent_file_loc )

    start_time =time .time ()

    dist_df =pd .DataFrame (
    {
    "paths_list":paths_list ,
    "nearest_cent":nearest_cent ,
    "dist_to_cent":dist_to_cent ,
    }
    )

    sorted_clusters =rank_within_cluster (
    data ,
    dist_df ,
    kmeans_centroids ,
    sim_metric ,
    keep_hard ,
    spherical ,
    cluster_ids ,
    sorted_clusters_file_loc ,
    )
    logger .info (f"Time for ranking: {(time.time() - start_time) / 60:.2f} mins")
    logger .info ("DONE!")

    return sorted_clusters 


def rank_within_cluster (
data :Union [np .memmap ,np .ndarray ],
dist_df :pd .DataFrame ,
centroids :np .ndarray ,
sim_metric :str ="cosine",
keep_hard :bool =True ,
spherical :bool =False ,
cluster_ids :List [int ]=range (50000 ),
sorted_clusters_file_loc :str ="",
)->List [List [Tuple [str ,int ,float ,int ]]]:

    assert sim_metric in [
    "cosine",
    "l2",
    ],"sim_metric should be one of ['cosine', 'l2']"
    os .makedirs (sorted_clusters_file_loc ,exist_ok =True )

    sorted_clusters_list =[]
    for cluster_c in tqdm (cluster_ids ):
        if os .path .exists (f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"):
            print (f"Cluster {cluster_c} exits, skipping....")
            continue 

        cluster_df =dist_df .loc [dist_df ["nearest_cent"]==cluster_c ]

        cluster_items =list (cluster_df .index )

        if sim_metric =="cosine":
            if spherical :
                cluster_dists_to_cent =list (1 -cluster_df ["dist_to_cent"])
            else :
                cluster_c_centroid =torch .Tensor (centroids [cluster_c ])
                sim_to_cent =torch .nn .CosineSimilarity (dim =1 )(
                torch .Tensor (data [cluster_items ]),cluster_c_centroid 
                )
                cluster_dists_to_cent =(1 -sim_to_cent ).tolist ()

        elif sim_metric =="l2":

            cluster_dists_to_cent =list (cluster_df ["dist_to_cent"])

        cluster_label =np .full ((len (cluster_df )),cluster_c ).tolist ()
        example_paths =list (cluster_df ["paths_list"])
        sort_descending =keep_hard 
        cluster_sorted =sorted (
        zip (example_paths ,cluster_items ,cluster_dists_to_cent ,cluster_label ),
        key =lambda x :x [2 ],
        reverse =sort_descending ,
        )


        sorted_clusters_list .append (
        cluster_sorted 
        )
        sorted_cluster_file_path =f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        np .save (sorted_cluster_file_path ,cluster_sorted )
    return sorted_clusters_list 



if __name__ =="__main__":

    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--config-file",
    type =str ,
    default ="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
    )

    parser .add_argument (
    "--partition",type =str ,default ="scaling_data_pruning",help ="partition"
    )
    parser .add_argument ("--num-tasks",type =int ,default =10 ,help ="number of tasks")
    parser .add_argument (
    "--cpus-per-task",type =int ,default =5 ,help ="number of cpus per task"
    )
    parser .add_argument (
    "--timeout",type =int ,default =500 ,help ="job timeout in minutes"
    )

    args =parser .parse_args ()

