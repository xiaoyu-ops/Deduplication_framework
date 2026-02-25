import os 
import sys 
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )
from method_all .train import main 
import json 
from tqdm import tqdm 
os .environ ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
os .environ ['TORCH_LOAD_ALLOW_UNSAFE']='1'

thresholds =[0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ]

output_dir =os .path .join (os .path .dirname (__file__ ),"train_results")
os .makedirs (output_dir ,exist_ok =True )

for threshold in tqdm (thresholds ,desc ="整体进度"):
    print (f"开始处理阈值: {threshold}")
    main (threshold =threshold )
    print (f"完成阈值: {threshold}")
    print ("="*80 )