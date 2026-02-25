





import torch 
from tqdm import tqdm 
from torch .nn .functional import normalize 


def get_embeddings (model ,dataloader ,emd_memmap ,paths_memmap ):



    device ="cuda"if torch .cuda .is_available ()else "cpu"



    model =model .to ("cuda"if torch .cuda .is_available ()else "cpu")
    model .eval ()
    '''
    model.eval()切换为评估模式而非训练模式。在训练模式下，模型会自动开启 Dropout 和 BatchNorm 层，
    以便在训练时引入随机性，从而提高模型的泛化能力。
    切换为评估模式(model.eval())的主要目的是让模型在推理或评估时的行为与训练时不同，
    从而获得稳定和确定性的输出。在评估模式下，像 Dropout 会被关闭,BatchNorm 层也会
    使用训练时计算的统计量，而不是当前批次的数据，这样就不会引入随机性或波动，为计算
    嵌入提供一致的结果。简单来说，评估模式确保模型在生成数据表示时能够按照预期工作，而不会受训练中使用的正则化技术影响。
    '''



    print ("Get encoding...")
    with torch .no_grad ():
        for data_batch ,paths_batch ,batch_indices in tqdm (dataloader ):
            print (f"data_batch对应的类型是{type(data_batch)}")
            print (f"data_batch的形状是{data_batch.shape}")
            data_batch =data_batch .to (device )
            encodings =model .encode_image (data_batch )
            emd_memmap [batch_indices ]=normalize (encodings ,dim =1 ).cpu ().numpy ()
            paths_memmap [batch_indices ]=paths_batch 
