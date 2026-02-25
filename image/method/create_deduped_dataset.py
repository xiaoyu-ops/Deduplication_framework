import os 
import argparse 
import yaml 
import numpy as np 
from datasets import load_dataset ,Dataset 
from tqdm import tqdm 
from pathlib import Path 
import shutil 
import json 
import requests 
from PIL import Image 

def parse_args ():
    parser =argparse .ArgumentParser (description ="创建去重后的数据集")
    parser .add_argument ('--config-file',required =True ,help ='配置文件路径')
    parser .add_argument ('--eps',required =True ,type =float ,help ='使用哪个epsilon值的结果')
    parser .add_argument ('--output-folder',required =True ,help ='输出文件夹路径')
    return parser .parse_args ()

def load_kept_indices (result_path ):
    print (f"从 {result_path} 加载保留的样本索引...")
    with open (result_path ,'r')as f :
        indices =[int (line .strip ())for line in f .readlines ()]
    return indices 

def create_deduped_dataset (original_dataset ,kept_indices ,output_folder ):
    os .makedirs (output_folder ,exist_ok =True )
    images_folder =os .path .join (output_folder ,"images")
    os .makedirs (images_folder ,exist_ok =True )

    print (f"从 {len(kept_indices)} 个保留样本中创建去重数据集...")


    records_are_list =isinstance (original_dataset ,(list ,tuple ))
    if records_are_list :
        records =original_dataset 
        keys =set ()
        for r in records :
            if isinstance (r ,dict ):
                keys .update (r .keys ())
    else :

        try :
            keys =set (original_dataset .features .keys ())
        except Exception :

            keys =set ()
            try :
                for i in range (min (100 ,len (original_dataset ))):
                    s =original_dataset [i ]
                    if isinstance (s ,dict ):
                        keys .update (s .keys ())
            except Exception :
                pass 


    keys =sorted (list (keys ))
    if "image_path"not in keys :
        keys .append ("image_path")


    new_dataset_dict ={k :[]for k in keys }

    def save_image_from_sample (sample_idx ,sample ):

        img_path =None 
        try :

            img_obj =None 
            if isinstance (sample ,dict ):

                for f in ("image","image_bytes","img","image_path","path"):
                    if f in sample and sample [f ]is not None :
                        img_obj =sample [f ]
                        break 
                if img_obj is None and "url"in sample :
                    img_obj =sample ["url"]
            else :
                img_obj =getattr (sample ,"image",None )

            filename =f"image_{sample_idx}.jpg"
            dst =os .path .join (images_folder ,filename )


            try :
                if hasattr (img_obj ,"save")and isinstance (img_obj ,Image .Image ):
                    img_obj .save (dst )
                    return dst 
            except Exception :
                pass 


            if isinstance (img_obj ,(bytes ,bytearray )):
                with open (dst ,"wb")as fw :
                    fw .write (img_obj )
                return dst 


            if isinstance (img_obj ,str ):
                if img_obj .startswith ("http://")or img_obj .startswith ("https://"):

                    try :
                        resp =requests .get (img_obj ,stream =True ,timeout =30 )
                        if resp .status_code ==200 :
                            with open (dst ,"wb")as fw :
                                for chunk in resp .iter_content (1024 *8 ):
                                    if chunk :
                                        fw .write (chunk )
                            return dst 
                        else :
                            print (f"下载失败（状态码 {resp.status_code}）：{img_obj}")
                            return None 
                    except Exception as e :
                        print (f"下载异常: {e} -> {img_obj}")
                        return None 
                else :

                    src =os .path .expanduser (img_obj )
                    if os .path .exists (src ):
                        try :
                            shutil .copy2 (src ,dst )
                            return dst 
                        except Exception as e :
                            print (f"拷贝本地图片失败: {e} -> {src}")
                            return None 

            return None 
        except Exception as e :
            print (f"保存图片时出错: {e}")
            return None 


    for idx in tqdm (kept_indices ):
        try :
            sample =original_dataset [idx ]if not records_are_list else original_dataset [idx ]
        except Exception as e :
            print (f"取样本 idx={idx} 失败: {e}")

            for k in keys :
                new_dataset_dict [k ].append (None )
            continue 


        image_saved_path =save_image_from_sample (idx ,sample )


        if isinstance (sample ,dict ):
            sample_items =sample 
        else :

            try :
                sample_items =dict (sample )
            except Exception :
                sample_items ={}

        for k in keys :
            if k =="image_path":
                new_dataset_dict [k ].append (image_saved_path )
            else :

                val =sample_items .get (k )if isinstance (sample_items ,dict )else None 
                new_dataset_dict [k ].append (val )


    try :
        deduped_dataset =Dataset .from_dict (new_dataset_dict )
    except Exception as e :
        print (f"从字典构建 Dataset 失败: {e}, 将以原始 dict 保存为 npz")

        fallback_path =os .path .join (output_folder ,"dataset_fallback.npy")
        np .save (fallback_path ,new_dataset_dict )
        print (f"已保存为: {fallback_path}")
        return None 

    dataset_save_path =os .path .join (output_folder ,"dataset")
    deduped_dataset .save_to_disk (dataset_save_path )
    print (f"去重数据集已保存到 {dataset_save_path}")
    print (f"去重后的数据集大小: {len(deduped_dataset)}")
    return deduped_dataset 

def main ():
    args =parse_args ()


    with open (args .config_file ,'r',encoding ='utf-8')as f :
        config =yaml .safe_load (f )


    eps_result_dir =os .path .join (config ['save_folder'],f"eps_{args.eps}")
    kept_samples_path =os .path .join (eps_result_dir ,"all_kept_samples.txt")


    if not os .path .exists (kept_samples_path ):
        print (f"错误: 找不到保留样本列表文件: {kept_samples_path}")
        print ("请先运行 simple_semdedup.py 生成去重结果")
        return 


    kept_indices =load_kept_indices (kept_samples_path )
    print (f"加载了 {len(kept_indices)} 个保留样本索引")



    images_source =config .get ("paths",{}).get ("images_source",
    r"D:\Deduplication_framework\image\dataset")
    images_source =os .path .abspath (images_source )
    print (f"从本地目录加载原始数据集: {images_source}")

    original_records =[]
    if os .path .isdir (images_source ):

        for jf in sorted (Path (images_source ).glob ("*.json")):
            try :
                with open (jf ,"r",encoding ="utf-8")as f :
                    data =json .load (f )
                if isinstance (data ,list ):
                    for item in data :
                        if isinstance (item ,str ):

                            original_records .append ({"url":item })
                        elif isinstance (item ,dict ):
                            original_records .append (item )
                elif isinstance (data ,dict ):

                    original_records .append (data )
            except Exception as e :
                print (f"跳过无法解析的 JSON 文件 {jf}: {e}")
        if not original_records :
            raise RuntimeError (f"在目录 {images_source} 中未解析到任何记录，请检查 JSON 格式")
    else :
        raise RuntimeError (f"images_source 不是目录: {images_source}")


    try :

        keys =set ()
        for r in original_records :
            if isinstance (r ,dict ):
                keys .update (r .keys ())
        if keys :
            new_dict ={k :[]for k in sorted (keys )}
            for r in original_records :
                for k in sorted (keys ):
                    new_dict [k ].append (r .get (k ))
            original_dataset =Dataset .from_dict (new_dict )
        else :

            original_dataset =original_records 
    except Exception as e :
        print (f"无法将原始记录转为 Dataset，退回为 list：{e}")
        original_dataset =original_records 


    create_deduped_dataset (original_dataset ,kept_indices ,args .output_folder )

    print ("完成!")

if __name__ =="__main__":
    main ()
