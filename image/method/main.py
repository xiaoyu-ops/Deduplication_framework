from compute_pretrained_embeddings import get_embeddings 
import open_clip 
import numpy as np 
from datasets import load_dataset 
from PIL import Image 
from torch .utils .data import DataLoader 
import torch 
import os 
from tqdm import tqdm 
import json 
import glob 
import requests 
import io 
from pathlib import Path 

current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
root_dir =os .path .dirname (parent_dir )

model ,preprocess_train ,preprocess_val =open_clip .create_model_and_transforms ('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
tokenizer =open_clip .get_tokenizer ('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')


def load_config_json (config_path ):
    try :
        with open (config_path ,'r',encoding ='utf-8')as f :
            return json .load (f )
    except FileNotFoundError :
        print (f"配置文件未找到: {config_path}")
        return None 
    except json .JSONDecodeError as e :
        print (f"配置文件格式错误: {e}")
        return None 

class ImageDataset (torch .utils .data .Dataset ):
    def __init__ (self ,dataset ,transform =None ,url_keys =("url","image_url")):
        self .dataset =dataset 
        self .transform =transform 
        self .url_keys =url_keys 

    def __len__ (self ):
        return len (self .dataset )

    def _load_from_url (self ,url ):
        try :

            resp =requests .get (url ,timeout =10 )
            resp .raise_for_status ()
            return Image .open (io .BytesIO (resp .content )).convert ("RGB")
        except Exception as e :
            print (f"加载URL失败: {url} -> {e}")
            return Image .new ("RGB",(224 ,224 ),(0 ,0 ,0 ))

    def __getitem__ (self ,idx ):
        item =self .dataset [idx ]
        try :

            if isinstance (item ,str ):
                if item .startswith ("http://")or item .startswith ("https://"):
                    pil =self ._load_from_url (item )
                else :
                    pil =Image .open (item ).convert ("RGB")

            elif isinstance (item ,dict ):

                img_field =item .get ("image")or item .get ("img")
                if isinstance (img_field ,Image .Image ):
                    pil =img_field 
                elif isinstance (img_field ,(bytes ,bytearray )):
                    pil =Image .open (io .BytesIO (img_field )).convert ("RGB")
                else :

                    url =None 
                    for k in self .url_keys :
                        if k in item and item [k ]:
                            url =item [k ];break 
                    if url :
                        pil =self ._load_from_url (url )
                    else :
                        raise ValueError ("无法从数据项获取图片或URL")
            else :

                img_field =item ["image"]
                if isinstance (img_field ,Image .Image ):
                    pil =img_field 
                else :
                    pil =Image .open (io .BytesIO (img_field )).convert ("RGB")
        except Exception as e :

            print (f"加载图片失败 idx={idx} item={getattr(item,'__repr__',lambda:None)()}: {e}")
            pil =Image .new ("RGB",(224 ,224 ),(0 ,0 ,0 ))

        image =self .transform (pil )if self .transform else pil 
        path =str (idx )
        return image ,path ,idx 

def custom_collate_fn (batch ):
    images =torch .stack ([item [0 ]for item in batch ])
    paths =[item [1 ]for item in batch ]
    indices =torch .tensor ([item [2 ]for item in batch ])
    return images ,paths ,indices 

if __name__ =="__main__":
    config_path =os .path .join (current_dir ,"image_config.json")
    data =load_config_json (config_path )
    if data is None :
        print ("加载配置失败，采用默认配置（本地 dataset 目录）")
        data ={"paths":r"D:\Deduplication_framework\image\dataset"}

    print ("在配置文件中选择数据类型:1.从huggface上下载 2.本地文件")

    choice =data .get ("processing",{}).get ("load_in_desk","load_in_desk")
    if choice =="load_from_huggingface":
        ds =load_dataset ('Maysee/tiny-imagenet',split ='train')
        image_dataset =ImageDataset (ds ,preprocess_val )
    else :

        images_source =Path (data .get ("paths",{}).get ("images_source",
        r"D:\Deduplication_framework\image\dataset"))

        exts ={".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
        files_set =set ()

        print (f"解析 images_source: {images_source}")
        if not images_source .exists ():
            raise RuntimeError (f"路径不存在: {images_source}，请检查配置或将数据放到该路径")

        if images_source .is_file ():

            if images_source .suffix .lower ()==".txt":
                with open (images_source ,"r",encoding ="utf-8")as f :
                    for line in f :
                        s =line .strip ()
                        if s :
                            files_set .add (s )
            elif images_source .suffix .lower ()==".json":
                with open (images_source ,"r",encoding ="utf-8")as f :
                    loaded =json .load (f )
                    if isinstance (loaded ,list ):
                        if all (isinstance (x ,str )for x in loaded ):
                            for x in loaded :
                                files_set .add (x .strip ())
                        else :
                            for entry in loaded :
                                if isinstance (entry ,dict ):
                                    u =entry .get ("url")or entry .get ("image_url")or entry .get ("img")
                                    if u :
                                        files_set .add (u )
                    else :
                        raise RuntimeError (f"JSON 文件格式不受支持: {images_source}")
            else :
                raise RuntimeError (f"不支持的文件格式: {images_source.suffix}")
        else :


            for p in images_source .rglob ("*"):
                if p .suffix .lower ()in exts :
                    files_set .add (str (p ))

            for jf in images_source .rglob ("*.json"):
                try :
                    with open (jf ,"r",encoding ="utf-8")as f :
                        loaded =json .load (f )
                        if isinstance (loaded ,list ):
                            if all (isinstance (x ,str )for x in loaded ):
                                for x in loaded :
                                    files_set .add (x .strip ())
                            else :
                                for entry in loaded :
                                    if isinstance (entry ,dict ):
                                        u =entry .get ("url")or entry .get ("image_url")or entry .get ("img")
                                        if u :
                                            files_set .add (u )
                except Exception as e :
                    print (f"跳过无法解析的 JSON 文件 {jf}: {e}")
            for tf in images_source .rglob ("*.txt"):
                try :
                    with open (tf ,"r",encoding ="utf-8")as f :
                        for line in f :
                            s =line .strip ()
                            if s :
                                files_set .add (s )
                except Exception as e :
                    print (f"跳过无法读取的 TXT 文件 {tf}: {e}")

        files =sorted (files_set )
        print (f"找到 {len(files)} 个本地图片或 URL（示例前5项）：{files[:5]}")
        if not files :
            raise RuntimeError (f"未找到图片或 URL 列表，检查: {images_source}")


        image_dataset =ImageDataset (files ,preprocess_val )


    batch_size =32 
    dataloader =DataLoader (image_dataset ,batch_size =batch_size ,num_workers =0 ,shuffle =False ,collate_fn =custom_collate_fn )


    path_str_type ='S256'
    os .makedirs ("embeddings",exist_ok =True )
    emb_memory_loc ="embeddings/image_embeddings.npy"
    paths_memory_loc ="embeddings/image_paths.npy"
    dataset_size =len (image_dataset )
    emb_size =512 
    emb_array =np .memmap (emb_memory_loc ,dtype ='float32',mode ='w+',shape =(dataset_size ,emb_size ))
    path_array =np .memmap (paths_memory_loc ,dtype =path_str_type ,mode ='w+',shape =(dataset_size ,))

    print (f"数据集大小: {dataset_size}, 嵌入维度: {emb_size}")
    print (f"正在使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print ("计算嵌入向量...")
    try :
        with torch .no_grad ():
            get_embeddings (model ,dataloader ,emb_array ,path_array )
    except Exception as e :
            print (f"计算嵌入向量时出错: {e}")
    finally :

            emb_array .flush ()
            path_array .flush ()
    print ("嵌入向量计算完成！")