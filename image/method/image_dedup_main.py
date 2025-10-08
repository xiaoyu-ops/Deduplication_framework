import os 
import glob
import numpy as np
import sys
import json 

current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
