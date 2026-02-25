import hashlib 
import json 
import time 
from pathlib import Path 
from typing import Any ,Dict 


class StageLockError (RuntimeError ):


def _lock_path (path :Path )->Path :
    return path /"_LOCK"


def compute_dict_hash (data :Dict [str ,Any ])->str :
    serialized =json .dumps (data ,sort_keys =True ,ensure_ascii =False )
    return hashlib .sha256 (serialized .encode ("utf-8")).hexdigest ()


def write_flag (path :Path ,flag :str )->None :
    path .mkdir (parents =True ,exist_ok =True )
    flag_path =path /flag 
    flag_path .write_text ("",encoding ="utf-8")


def flag_exists (path :Path ,flag :str )->bool :
    return (path /flag ).exists ()


def acquire_stage_lock (path :Path )->Path :
    path .mkdir (parents =True ,exist_ok =True )
    lock_path =_lock_path (path )
    if lock_path .exists ():
        raise StageLockError (f"Stage lock already present: {lock_path}")
    lock_path .write_text (str (time .time ()),encoding ="utf-8")
    return lock_path 


def release_stage_lock (path :Path )->None :
    lock_path =_lock_path (path )
    if lock_path .exists ():
        lock_path .unlink ()
