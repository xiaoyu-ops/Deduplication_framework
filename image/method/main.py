from compute_pretrained_embeddings import get_embeddings
import open_clip
import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')#文本的分词器 对于图片用不上
# model = ...

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        path = str(idx)
        return image, path ,idx
    
ds = load_dataset('Maysee/tiny-imagenet', split='train')#后续数据集在这里修改即可
image_dataset = ImageDataset(ds, preprocess_val)#这里使用的是预训练模型的预处理函数
batch_size = 32

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])#转换为批次张量
    paths = [item[1] for item in batch]
    indices = torch.tensor([item[2] for item in batch])
    return images, paths, indices
dataloader = DataLoader(image_dataset, batch_size=batch_size, num_workers=0,shuffle=False,collate_fn=custom_collate_fn)

path_str_type = 'S24'
os.makedirs("embeddings", exist_ok=True)
emb_memory_loc = "embeddings/image_embeddings.npy"
paths_memory_loc = "embeddings/image_paths.npy"
dataset_size = len(ds)
emb_size = 512
emb_array = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))#此处需要注意GPU张量要先转移到CPU然后再转为numpy数组
path_array = np.memmap(paths_memory_loc, dtype=path_str_type, mode='w+', shape=(dataset_size,))

print(f"数据集大小: {dataset_size}, 嵌入维度: {emb_size}")
print(f"正在使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

print("计算嵌入向量...")
try:
    with torch.no_grad():
        get_embeddings(model, dataloader, emb_array, path_array)
except Exception as e:
        print(f"计算嵌入向量时出错: {e}")
finally:
        # 确保内存映射文件被正确关闭
        emb_array.flush()
        path_array.flush()
print("嵌入向量计算完成！")