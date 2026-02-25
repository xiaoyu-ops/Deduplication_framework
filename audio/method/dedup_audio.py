import os 
import numpy as np 
import librosa 
from datasets import load_dataset 
import matplotlib .pyplot as plt 


cache_dir ="D:\\\audio_deduptation\\cache"
os .makedirs (cache_dir ,exist_ok =True )

plt .rcParams ['font.sans-serif']=['SimHei']
plt .rcParams ['axes.unicode_minus']=False 
os .environ ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
os .environ ["HF_DATASETS_CACHE"]=cache_dir 

def example_usage ():

    print ("正在加载数据集...")
    tiny_audio =load_dataset ("danavery/urbansound8K",cache_dir =cache_dir )
    print ("数据集加载完成!")


    print (f"数据集结构: {tiny_audio}")
    print (f"训练集样本数: {len(tiny_audio['train'])}")

if __name__ =="__main__":
    example_usage ()
