# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from typing import Union, Optional



def faiss_index_to_gpu(cpu_index):
    """
    尝试将 FAISS 索引转移到 GPU，如果 GPU 功能不可用则返回 CPU 索引
    """
    try:
        # 检查是否有 GPU 支持
        if hasattr(faiss, 'GpuClonerOptions'):
            # 有 GPU 支持，执行原始转换逻辑
            cloner_options = faiss.GpuClonerOptions()
            cloner_options.useFloat16 = False
            cloner_options.usePrecomputed = False
            cloner_options.indicesOptions = faiss.INDICES_CPU
            
            # 配置 Faiss GPU 资源
            gpu_resources = faiss.StandardGpuResources()
            
            # 将 CPU 索引转换为 GPU 索引
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index, cloner_options)
            return gpu_index
        else:
            # 没有 GPU 支持，返回原始 CPU 索引
            print("FAISS GPU 功能不可用，将使用 CPU 版本继续...")
            return cpu_index
    except Exception as e:
        print(f"转移到 GPU 时出错: {e}，将使用 CPU 版本继续...")
        return cpu_index


def compute_centroids(
    data: Union[np.memmap, np.ndarray],
    ncentroids: int = 1000,#分为1000个簇
    niter: int = 100,#迭代100次
    seed: int = 1234,#随机种子
    Kmeans_with_cos_dist: bool = False,#是否使用余弦距离，意味着使用欧氏距离
    save_folder: str = "",#保存文件夹
    logger: logging.Logger = None,#日志记录器
    verbose: bool = True,#是否打印详细信息
):

    """
    Runs K-means clustering on the input data using "faiss" and saves the following output files:

          1)faiss k-means index object (pickle file).
          2)k-means centroids (numpy array).
          3)Distance to centroid for data points in <data> (numpy array).
          4)Nearest centroid for data points in <data> (numpy array).
    args:
        data: A float32 numpy memmap array or numpy array of shape [dataset_size x d], where d is the embedding vector size..
        ncentroids: number of kmeans clusters/centroids.
        niter: The number of iterations to run the K-means algorithm for.
        seed: The random seed to use for reproducibility.
        Kmeans_with_cos_dist: (boolean) when True, run spherical kmeans.
        save_folder: path to save/load output files.
        logger: A logger instance to use for logging.

    returns:
        faiss k-means object
        这段注释说明了一个使用 Faiss 库实现 K-means 聚类的函数的功能、参数和返回值。该函数
        不仅在输入数据上运行 K-means 算法，而且还将聚类的结果保存为多个输出文件，便于后续使用和分析。

    具体来说，函数将生成以下输出文件：  
    1. 一个保存了 Faiss K-means 索引对象的 pickle 文件。  
    2. 一个包含 K-means 聚类质心的 NumPy 数组。  
    3. 一个 NumPy 数组，记录了数据点到各自质心的距离。  
    4. 一个 NumPy 数组，记录了数据点最近的质心信息。

    在参数说明部分,函数要求传入的数据(data)应为 float32 类型的 NumPy 内存映射数组或普通的 NumPy 数组，其形状为
    [dataset_size x d]，其中 d 表示嵌入向量的维度。此外，还可以设置聚类质心数(ncentroids)、迭代次数(niter)、随机种子
    (seed)以保证结果可复现，同时可以选择是否使用余弦距离(spherical K-means当 Kmeans_with_cos_dist 为 True 时）来替
    代欧氏距离。其他参数如 save_folder 和 logger 分别用于指定保存输出文件的路径和记录日志信息。

    最终，这个函数返回一个 Faiss K-means 对象，使得用户可以直接利用该对象进行后续的相似度搜索或进一步的聚类分析。整个注释提
    供了一个详细的概述，帮助用户理解该函数如何处理数据、生成输出以及如何配置算法的主要参数。
    """
    os.makedirs(save_folder, exist_ok=True)
    # -- Compute Kmeans centroids
    # -- 计算 Kmeans 质心
    logger.info(
        f"Running Kmeans clustering using faiss on dataset of shape {data.shape} ...."
    )
    logger.info(f"Kmeans parameters: {locals()} ....")
    # pprint.pprint(locals(), width=1, indent=4)
    # 打印参数

    d = data.shape[1]
    # -- Use GPUs for clustering when available
    # -- 如果有 GPU 可用，则使用 GPU 进行聚类
    use_gpu = torch.cuda.is_available()

    device = "cuda" if use_gpu else "cpu"

    logger.info(f"Clustering on {device} ....")

    spherical = (
        Kmeans_with_cos_dist  # -- spherical=True when Kmeans_with_cos_dist is True
        # -- 当 Kmeans_with_cos_dist 为 True 时，spherical=True
    )

    ## -- Step 1) Train faiss kmeans
    ## -- 创建并训练 faiss Kmeans 聚类对象
    kmeans = faiss.Kmeans(
        d,
        ncentroids,
        niter=niter,
        verbose=verbose,
        seed=seed,
        spherical=spherical,
        gpu=use_gpu,
    )  ## -- faiss.Kmeans "gpu" argument: bool or int, optional. False: don't use GPU, True: use all GPUs, number: use this many GPUs.
        ## --faiss.Kmeans "gpu" 参数: bool 或 int，可选。False: 不使用 GPU，True: 使用所有 GPU，number: 使用指定数量的 GPU。

    # -- If kmeans centroids are not saved - > create and train faiss Kmeans clustering object
    # -- 如果 Kmeans 质心未保存，则创建并训练 faiss Kmeans 聚类对象
    kmeans_obj_file_loc = pathlib.Path(save_folder, "kmeans_index.pickle")# 组合成一个地址路径

    if not os.path.exists(kmeans_obj_file_loc):
        start_time = time.time()
        kmeans.train(data)
        logger.info(f"Time for clustering (mins): {(time.time()-start_time)/(60):.2f}")

        # -- Move kmeans index to cpu to save it
        # -- 将 kmeans 索引移动到 CPU 上以保存
        if hasattr(faiss, 'index_gpu_to_cpu'):
            kmeans_index = faiss.index_gpu_to_cpu(kmeans.index)
        else:
            # 如果是纯CPU版本，直接使用索引
            kmeans_index = kmeans.index
        logger.info(f"faiss kmeans index to store: {type(kmeans_index)}")
        ## -- Save faiss kmeans index object as pickle file
        ## -- 将 faiss Kmeans 索引对象保存为 pickle 文件
        with open(kmeans_obj_file_loc, "wb") as file:
            pickle.dump(kmeans_index, file)
        ## -- save faiss kmeans centroids as npy file
        ## -- 将 faiss Kmeans 质心保存为 npy 文件
        np.save(pathlib.Path(save_folder, "kmeans_centroids.npy"), kmeans.centroids)

        logger.info(f"Saved!")

    else:
        # -- Else, load stored kmeans object
        # -- 否则，加载已保存的 kmeans 对象
        logger.info(
            f"Loading faiss Kmeans index pickle file from {kmeans_obj_file_loc}"
        )
        with open(kmeans_obj_file_loc, "rb") as file:
            kmeans_index = pickle.load(file)
            if use_gpu:
                # -- move kmeans index to gpu
                # -- 将 kmeans 索引移动到 GPU 上
                kmeans_index = faiss_index_to_gpu(kmeans_index)
            kmeans.index = kmeans_index

    ## -- Step 2) Find the nearest centroid for each data point, l2 distance search
    ## -- 查找每个数据点的最近质心，使用 L2 距离搜索
    ## -- nearest_cent: the nearest centroid for each example in data. dist_to_cent: contains the squared L2 distances.
    ## -- nearest_cent: data 中每个示例的最近质心。dist_to_cent: 包含平方 L2 距离。
    start_time = time.time()
    dist_to_cent, nearest_cent = kmeans.index.search(data, 1)
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)#去除返回数组中的多余的维度，将其压缩为一维
    logger.info(
        f"Time for finding nearest centroid for each data point (mins): {(time.time()-start_time)/(60):.2f}"
    )

    ## -- save faiss nearest_cent and dist_to_cent as .npy files
    ## -- 将 faiss nearest_cent 和 dist_to_cent 保存为 .npy 文件
    dist_to_cent_file = pathlib.Path(save_folder, "dist_to_cent.npy")
    nearest_cent_file = pathlib.Path(save_folder, "nearest_cent.npy")
    np.save(dist_to_cent_file, dist_to_cent)
    np.save(nearest_cent_file, nearest_cent)

    return kmeans


def main(args):


    # Load configuration file for clustering
    # -- 加载用于聚类的配置文件
    confg_file = args.confg_file

    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)#加载yaml文件

    with open(
        pathlib.Path(params["save_folder"], "clustering_params.txt"), "w"
    ) as fout:
        pprint.pprint(params, fout)

    ## -- Load clustering parameters
    ## -- 加载聚类参数
    seed = params["seed"]
    emb_memory_loc = params[
        "emb_memory_loc"
    ]  ## -- numpy menmap where embeddings are stored
    dataset_size = params["dataset_size"]
    emb_size = params["emb_size"]
    niter = params["niter"]
    ncentroids = params["ncentroids"]
    save_folder = params["save_folder"]
    Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]

    ## -- Load embeddings
    ## -- 加载嵌入
    data = np.memmap(
        emb_memory_loc, dtype="float32", mode="r", shape=(dataset_size, emb_size)
    )

    ## -- Compute centroids
    ## -- 计算质心
    compute_centroids(
        data,
        ncentroids,
        niter,
        seed,
        Kmeans_with_cos_dist,
        save_folder,
        True,
    )


if __name__ == "__main__":
    # Configure command line arguments
    # 命令行参数解析器
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--confg-file",
        type=str,
        default="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
        help=".yaml config file path",
    )
    # -- slurm parameters
    # -- SLURM 参数
    parser.add_argument(
        "--partition", type=str, default="scaling_data_pruning", help="partition"
    )
    parser.add_argument("--ngpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--cpus-per-task", type=int, default=10, help="number of cpus")
    parser.add_argument(
        "--timeout", type=int, default=1500, help="job timeout in minutes"
    )

    args = parser.parse_args()#解析后的参数赋值给args变量

    # Load configuration file
    # -- 加载配置文件
    with open(args.confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)

    ## -- Logging directory
    ## -- 日志目录
    args.save_folder = params["save_folder"]

    # SLURM CONFIG
    # -- SLURM 配置
    PARTITION = args.partition
    NODES = 1
    NGPUS = args.ngpus
    CPUS_PER_TASKS = args.cpus_per_task
    TIMEOUT = args.timeout

    # Configure submitit executor
    # -- 配置 submitit 执行器
    submitit_path = f"{args.save_folder}/compute_centorids_job_%j"
    executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        slurm_partition=PARTITION,
        nodes=NODES,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_TASKS,
        gpus_per_node=NGPUS,
        slurm_mem_per_gpu="55G",
        timeout_min=TIMEOUT,
    )

    # Submit job
    # -- 提交任务
    job = executor.submit(main, args)
    print("Submitted job_id:", job.job_id)
