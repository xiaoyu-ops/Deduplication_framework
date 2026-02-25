
from __future__ import annotations 

from dataclasses import dataclass 
from pathlib import Path 
from typing import Dict ,Iterable ,List ,Optional ,Sequence ,Set ,Tuple 

import numpy as np 


@dataclass 
class LegacyEmbeddings :
    vectors :np .ndarray 
    paths :List [Path ]
    missing :List [Path ]
    indices :List [int ]


def _normalise_path_text (text :str )->List [str ]:

    variants ={text ,text .replace ("\\","/"),text .lower (),text .replace ("\\","/").lower ()}
    try :
        resolved =Path (text ).resolve ()
        resolved_str =str (resolved )
        resolved_norm =resolved_str .replace ("\\","/")
        variants .update ({resolved_str ,resolved_norm ,resolved_str .lower (),resolved_norm .lower ()})
    except Exception :
        pass 
    return list (variants )


def _load_numpy_array (path :Path ,expected_dim :Optional [int ]=None )->np .ndarray :

    try :
        array =np .load (path ,mmap_mode ="r")
    except ValueError :
        array =np .load (path )

    if expected_dim and array .ndim !=expected_dim :
        raise ValueError (f"Unexpected array rank for {path}: {array.shape}, expected {expected_dim}D")

    return array 


def load_legacy_embeddings (
embedding_file :Path ,
index_file :Path ,
manifest_paths :Sequence [Path ],
)->LegacyEmbeddings :

    if not embedding_file .exists ():
        raise FileNotFoundError (f"Legacy embedding file not found: {embedding_file}")
    if not index_file .exists ():
        raise FileNotFoundError (f"Legacy path index not found: {index_file}")

    vectors_store =_load_numpy_array (embedding_file )
    path_store =_load_numpy_array (index_file )


    lookup :Dict [str ,int ]={}
    for idx ,raw in enumerate (path_store ):
        if isinstance (raw ,bytes ):
            text =raw .decode ("utf-8",errors ="ignore").rstrip ("\x00")
        else :
            text =str (raw )
        if not text :
            continue 
        for key in _normalise_path_text (text ):
            lookup .setdefault (key ,idx )

    found_vectors :List [np .ndarray ]=[]
    found_paths :List [Path ]=[]
    found_indices :List [int ]=[]
    missing_paths :List [Path ]=[]

    for path in manifest_paths :
        candidates =_normalise_path_text (str (path ))
        match :Optional [int ]=None 
        for candidate in candidates :
            match =lookup .get (candidate )
            if match is not None :
                break 
        if match is None :
            missing_paths .append (path )
            continue 
        vector =np .asarray (vectors_store [match ],dtype =np .float32 )
        found_vectors .append (vector )
        found_paths .append (Path (path ))
        found_indices .append (int (match ))

    if not found_vectors :
        raise RuntimeError ("None of the manifest images were present in the legacy embeddings")

    stacked =np .stack (found_vectors ,axis =0 )
    return LegacyEmbeddings (vectors =stacked ,paths =found_paths ,missing =missing_paths ,indices =found_indices )


def save_embeddings_snapshot (
embeddings :np .ndarray ,
paths :Sequence [Path ],
target_dir :Path ,
)->Tuple [Path ,Path ]:

    target_dir .mkdir (parents =True ,exist_ok =True )
    emb_path =target_dir /"image_embeddings.npy"
    idx_path =target_dir /"image_paths.npy"

    np .save (emb_path ,embeddings .astype (np .float32 ))
    encoded_paths =np .array ([str (p )for p in paths ],dtype =object )
    np .save (idx_path ,encoded_paths )

    return emb_path ,idx_path 


def load_keep_indices (path :Path )->Set [int ]:
    if not path .exists ():
        raise FileNotFoundError (f"Legacy keep-indices file not found: {path}")

    keep :Set [int ]=set ()
    with path .open ("r",encoding ="utf-8")as handle :
        for line in handle :
            line =line .strip ()
            if not line :
                continue 
            try :
                keep .add (int (line ))
            except ValueError :
                continue 
    return keep 


def load_cluster_members (cluster_dir :Path ,target_indices :Optional [Set [int ]]=None )->Dict [int ,List [int ]]:
    if not cluster_dir .exists ():
        raise FileNotFoundError (f"Legacy cluster directory not found: {cluster_dir}")

    members :Dict [int ,List [int ]]={}
    processed_ids :Set [int ]=set ()

    for file in sorted (cluster_dir .glob ("cluster_*.npy")):
        cluster_id =int (file .stem .split ("_")[-1 ])
        processed_ids .add (cluster_id )
        data =np .load (file ,allow_pickle =True )

        if data .ndim ==1 :
            values =_extract_indices_from_vector (data )
        else :
            values =_extract_indices_from_matrix (data )

        if target_indices is not None :
            filtered =[idx for idx in values if idx in target_indices ]
            if not filtered :
                continue 
            members [cluster_id ]=filtered 
        else :
            members [cluster_id ]=values 

    for file in sorted (cluster_dir .glob ("cluster_*.txt")):
        cluster_id =int (file .stem .split ("_")[-1 ])
        if cluster_id in processed_ids :
            continue 
        values :List [int ]=[]
        with file .open ("r",encoding ="utf-8")as handle :
            for line in handle :
                parts =line .strip ().split ("\t")
                if not parts :
                    continue 
                try :
                    values .append (int (parts [0 ]))
                except ValueError :
                    continue 
        if target_indices is not None :
            filtered =[idx for idx in values if idx in target_indices ]
            if not filtered :
                continue 
            members [cluster_id ]=filtered 
        else :
            members [cluster_id ]=values 

    return members 


def _extract_indices_from_vector (vec :np .ndarray )->List [int ]:
    result :List [int ]=[]
    for item in vec :
        result .append (_parse_index (item ))
    return result 


def _extract_indices_from_matrix (mat :np .ndarray )->List [int ]:
    if mat .shape [1 ]>=2 :
        column =mat [:,1 ]
    else :
        column =mat [:,0 ]
    return [_parse_index (item )for item in column ]


def _parse_index (value :object )->int :
    if isinstance (value ,(int ,np .integer )):
        return int (value )
    if isinstance (value ,bytes ):
        text =value .decode ("utf-8",errors ="ignore").strip ()
    else :
        text =str (value )
    if text .endswith (".0"):
        text =text [:-2 ]
    if text .startswith ("b'")and text .endswith ("'"):
        text =text [2 :-1 ]
    return int (text )
