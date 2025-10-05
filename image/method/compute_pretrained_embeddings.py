# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
from torch.nn.functional import normalize


def get_embeddings(model, dataloader, emd_memmap, paths_memmap):
    """
    function to compute and store representations for the data from pretrained model. It is preferable to parallelize this function on mulitiple devices (GPUs). Each device will process part of the data.
    model: pretrained model
    dataloader: should return   1) data_batch: batch of data examples
                                2) paths_batch: path to location where the example is stored (unique identifier). For example, this could be "n04235860_14959.JPEG" for imagenet.
                                3) batch_indices: global index for each example (between 0 and of size <dataset_size>-1).
    emd_memmap: numpy memmap to store embeddings of size <dataset_size>.
    paths_memmap: numpy memmap to store paths of size <dataset_size>.
    model: 预训练模型，用于生成数据表示。

    dataloader: 数据加载器，应该返回以下内容：

    data_batch: 数据示例的批次。
    paths_batch: 存储示例位置的路径（唯一标识符）。例如，对于 ImageNet 数据集，这可能是 "n04235860_14959.JPEG"。
    batch_indices: 每个示例的全局索引，范围在 0 到 <dataset_size>-1 之间。
    emd_memmap: 用于存储嵌入的 NumPy 内存映射数组，大小为 <dataset_size>。

    paths_memmap: 用于存储路径的 NumPy 内存映射数组，大小为 <dataset_size>。

    """

    # -- Device
    # -- 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- model
    # -- 模型
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置为评估模式
    '''
    model.eval()切换为评估模式而非训练模式。在训练模式下，模型会自动开启 Dropout 和 BatchNorm 层，
    以便在训练时引入随机性，从而提高模型的泛化能力。
    切换为评估模式(model.eval())的主要目的是让模型在推理或评估时的行为与训练时不同，
    从而获得稳定和确定性的输出。在评估模式下，像 Dropout 会被关闭,BatchNorm 层也会
    使用训练时计算的统计量，而不是当前批次的数据，这样就不会引入随机性或波动，为计算
    嵌入提供一致的结果。简单来说，评估模式确保模型在生成数据表示时能够按照预期工作，而不会受训练中使用的正则化技术影响。
    '''

    # -- Get and store 1)encodings 2)path to each example
    # -- 获取并存储 1) 编码 2) 每个示例的路径
    print("Get encoding...")
    with torch.no_grad():
        for data_batch, paths_batch, batch_indices in tqdm(dataloader):
            print(f"data_batch对应的类型是{type(data_batch)}")
            print(f"data_batch的形状是{data_batch.shape}")
            data_batch = data_batch.to(device)
            encodings = model.encode_image(data_batch)  # 修改为 encode_image
            emd_memmap[batch_indices] = normalize(encodings, dim=1).cpu().numpy()  # GPU 计算 → 复制到 CPU → 转换为 NumPy 或保存到磁盘
            paths_memmap[batch_indices] = paths_batch
