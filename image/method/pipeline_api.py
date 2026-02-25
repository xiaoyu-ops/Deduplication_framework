
from __future__ import annotations 

import json 
import os 
import sys 
import time 
from dataclasses import dataclass ,field 
from pathlib import Path 
from typing import Any ,Callable ,Dict ,Iterable ,List ,Optional ,Sequence ,Set ,Tuple 

import numpy as np 
import yaml 

try :
    import open_clip 
    import torch 
except ImportError :
    open_clip =None 
    torch =None 

try :
    from PIL import Image 
except ImportError :
    Image =None 

try :
    from torch .utils .data import Dataset ,DataLoader 
except ImportError :
    Dataset ,DataLoader =object ,None 

from image .method import legacy_integration 

class ImageListDataset (Dataset ):
    def __init__ (self ,path_list ,transform ):
        self .path_list =path_list 
        self .transform =transform 
    def __len__ (self ):return len (self .path_list )
    def __getitem__ (self ,idx ):
        path =self .path_list [idx ]
        try :
            img =Image .open (path ).convert ("RGB")
            return self .transform (img ),idx ,1 
        except Exception :



            try :
                dummy =Image .new ("RGB",(224 ,224 ),(0 ,0 ,0 ))
                return self .transform (dummy ),idx ,0 
            except Exception :

                return torch .zeros ((3 ,224 ,224 ),dtype =torch .float32 ),idx ,0 

@dataclass 
class EmbeddingConfig :
    backend :str ="auto"
    model_name :str ="hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    batch_size :int =16 
    device :str ="auto"
    fallback :str ="average_rgb"
    precomputed_embeddings :Optional [str ]=None 
    precomputed_index :Optional [str ]=None 
    save_embeddings_dir :Optional [str ]=None 


@dataclass 
class DedupConfig :
    method :str ="pairwise"
    eps :float =0.05 
    max_candidates :int =512 
    legacy_config_file :Optional [str ]=None 
    legacy_keep_indices_file :Optional [str ]=None 
    legacy_cluster_dir :Optional [str ]=None 


@dataclass 
class ImagePipelineConfig :
    embedding :EmbeddingConfig =field (default_factory =EmbeddingConfig )
    dedup :DedupConfig =field (default_factory =DedupConfig )


@dataclass 
class EmbeddingResult :
    embeddings :np .ndarray 
    paths :List [Path ]
    failed_paths :List [Path ]
    backend :Optional [str ]
    indices :Optional [List [int ]]=None 


@dataclass 
class ImagePipelineResult :
    keepers :List [Path ]
    duplicates :List [Dict [str ,object ]]
    missing :List [Path ]
    stats :Dict [str ,object ]


def _progress_reporter (total :int ,label :str ,prefix :str )->Callable [[int ],None ]:

    total =int (max (0 ,total ))
    if total <=0 :
        return lambda _current :None 

    step =max (1 ,total //10 )
    next_emit =step 

    def _report (current :int )->None :
        nonlocal next_emit 
        current =min (max (0 ,current ),total )
        if current >=next_emit or current ==total :
            percent =(current /total )*100 if total else 100.0 
            print (f"{prefix}{label}: {current}/{total} ({percent:.1f}%)",flush =True )
            next_emit =min (total ,current +step )

    return _report 


def _merge_dict (default :Dict [str ,object ],override :Dict [str ,object ])->Dict [str ,object ]:
    merged =dict (default )
    for key ,value in override .items ():
        if key in merged and isinstance (merged [key ],dict )and isinstance (value ,dict ):
            merged [key ]=_merge_dict (merged [key ],value )
        else :
            merged [key ]=value 
    return merged 


def load_pipeline_config (config_path :Optional [str ])->ImagePipelineConfig :

    defaults :Dict [str ,Dict [str ,object ]]={
    "embedding":{
    "backend":"auto",
    "model_name":"hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "batch_size":16 ,
    "device":"auto",
    "fallback":"average_rgb",
    "precomputed_embeddings":None ,
    "precomputed_index":None ,
    "save_embeddings_dir":None ,
    },
    "dedup":{
    "method":"pairwise",
    "eps":0.05 ,
    "max_candidates":512 ,
    "legacy_config_file":None ,
    "legacy_keep_indices_file":None ,
    "legacy_cluster_dir":None ,
    },
    }

    if not config_path :
        config_dict =dict (defaults )
    else :
        path =Path (config_path )
        if not path .exists ():
            raise FileNotFoundError (f"Image pipeline config not found: {path}")
        content =path .read_text (encoding ="utf-8")
        try :
            if path .suffix .lower ()in {".yaml",".yml"}:
                loaded =yaml .safe_load (content )or {}
            else :
                loaded =json .loads (content )
        except Exception as exc :
            raise ValueError (f"Failed to parse image pipeline config {path}: {exc}")from exc 
        config_dict =_merge_dict (defaults ,loaded )

    dedup_dict =dict (config_dict ["dedup"])
    legacy_opts =dedup_dict .pop ("legacy",{})or {}
    if isinstance (legacy_opts ,dict ):
        if "config_file"in legacy_opts :
            dedup_dict ["legacy_config_file"]=legacy_opts ["config_file"]
        if "keep_indices_file"in legacy_opts :
            dedup_dict ["legacy_keep_indices_file"]=legacy_opts ["keep_indices_file"]
        if "cluster_dir"in legacy_opts :
            dedup_dict ["legacy_cluster_dir"]=legacy_opts ["cluster_dir"]

    return ImagePipelineConfig (
    embedding =EmbeddingConfig (**config_dict ["embedding"]),
    dedup =DedupConfig (**dedup_dict ),
    )


def run_image_pipeline (paths :Sequence [Path ],config :ImagePipelineConfig )->ImagePipelineResult :
    print (f"[image pipeline] Starting run_image_pipeline with {len(paths)} input paths...",flush =True )

    start =time .time ()
    unique_paths :List [Path ]=[]
    seen :Set [str ]=set ()
    missing :List [Path ]=[]





    print ("[image pipeline] Skipping physical file verification for speed...",flush =True )

    for path in paths :
        path_str =str (path )
        if path_str in seen :
            continue 
        seen .add (path_str )
        unique_paths .append (Path (path ))

    print (f"[image pipeline] Prepared {len(unique_paths)} unique paths.",flush =True )

    if not unique_paths :
        stats ={
        "embedding_backend":None ,
        "unique":0 ,
        "duplicates":0 ,
        "missing":len (missing ),
        "processed":0 ,
        "elapsed_seconds":time .time ()-start ,
        }
        return ImagePipelineResult (keepers =[],duplicates =[],missing =missing ,stats =stats )

    embedding_result :Optional [EmbeddingResult ]=None 
    saved_embeddings_dir :Optional [str ]=None 

    if config .embedding .precomputed_embeddings :
        try :
            embedding_result =_load_precomputed_embeddings (unique_paths ,config .embedding )
        except Exception as exc :
            print (f"[image pipeline] failed to load precomputed embeddings: {exc}")

    if embedding_result is None :
        try :
            embedding_result =_compute_embeddings (unique_paths ,config .embedding )
        except Exception as exc :
            print (f"[image pipeline] failed to compute embeddings for {len(unique_paths)} files: {exc}")
            missing .extend (unique_paths )
            stats ={
            "embedding_backend":None ,
            "unique":0 ,
            "duplicates":0 ,
            "missing":len (missing ),
            "processed":0 ,
            "error":str (exc ),
            }
            return ImagePipelineResult (keepers =[],duplicates =[],missing =missing ,stats =stats )

    missing .extend (embedding_result .failed_paths )

    dedup_summary =_run_deduplication (
    embedding_result .paths ,
    embedding_result .embeddings ,
    config .dedup ,
    embedding_result .indices ,
    )

    if (
    config .embedding .save_embeddings_dir 
    and embedding_result .backend not in {None ,"precomputed"}
    and embedding_result .embeddings .size 
    and embedding_result .paths 
    ):
        try :
            saved_embeddings_dir =_persist_embeddings (
            embedding_result .embeddings ,
            embedding_result .paths ,
            config .embedding .save_embeddings_dir ,
            )
            if saved_embeddings_dir :
                print (f"[image pipeline] embeddings saved to {saved_embeddings_dir}")
        except Exception as exc :
            print (f"[image pipeline] failed to save embeddings: {exc}")

    stats ={
    "embedding_backend":embedding_result .backend ,
    "unique":len (dedup_summary ["keepers"]),
    "duplicates":dedup_summary ["duplicate_count"],
    "missing":len (missing ),
    "processed":len (embedding_result .paths ),
    "skipped_due_to_limit":dedup_summary ["skipped"],
    "elapsed_seconds":time .time ()-start ,
    }
    if saved_embeddings_dir :
        stats ["embeddings_saved_dir"]=saved_embeddings_dir 

    return ImagePipelineResult (
    keepers =dedup_summary ["keepers"],
    duplicates =dedup_summary ["duplicates"],
    missing =missing ,
    stats =stats ,
    )


def _load_precomputed_embeddings (
manifest_paths :Sequence [Path ],
config :EmbeddingConfig ,
)->Optional [EmbeddingResult ]:
    if not config .precomputed_embeddings or not config .precomputed_index :
        return None 

    embedding_path =Path (config .precomputed_embeddings ).expanduser ()
    index_path =Path (config .precomputed_index ).expanduser ()

    legacy =legacy_integration .load_legacy_embeddings (embedding_path ,index_path ,manifest_paths )
    return EmbeddingResult (
    embeddings =legacy .vectors ,
    paths =legacy .paths ,
    failed_paths =legacy .missing ,
    backend ="precomputed",
    indices =legacy .indices ,
    )


def _compute_embeddings (
paths :Sequence [Path ],
config :EmbeddingConfig ,
)->EmbeddingResult :
    backend =(config .backend or "auto").lower ()
    fallback =(config .fallback or "none").lower ()

    if backend =="auto":
        backend ="open_clip"if open_clip and torch else fallback 

    if backend =="open_clip"and open_clip and torch :
        try :
            embeddings ,valid_paths ,failed_paths ,backend_name =_compute_embeddings_open_clip (paths ,config )
            return EmbeddingResult (
            embeddings =embeddings ,
            paths =valid_paths ,
            failed_paths =failed_paths ,
            backend =backend_name ,
            )
        except Exception as exc :
            print (f"[image pipeline] open_clip failed ({exc}); falling back to {fallback}")
            backend =fallback 

    if backend =="average_rgb":
        embeddings ,valid_paths ,failed_paths ,backend_name =_compute_embeddings_average_rgb (paths )
        return EmbeddingResult (
        embeddings =embeddings ,
        paths =valid_paths ,
        failed_paths =failed_paths ,
        backend =backend_name ,
        )

    raise RuntimeError (
    "No usable embedding backend available. Install open_clip/torch or configure a fallback."
    )


def _persist_embeddings (
embeddings :np .ndarray ,
paths :Sequence [Path ],
target_dir :str ,
)->str :
    target =Path (target_dir ).expanduser ()
    legacy_integration .save_embeddings_snapshot (embeddings ,paths ,target )
    return str (target )


def _compute_embeddings_open_clip (
paths :Sequence [Path ],
config :EmbeddingConfig ,
)->Tuple [np .ndarray ,List [Path ],List [Path ],str ]:
    print ("[image pipeline] Entering _compute_embeddings_open_clip...",flush =True )
    if open_clip is None or torch is None :
        raise RuntimeError ("open_clip and torch are required for the open_clip backend")
    if Image is None :
        raise RuntimeError ("Pillow is required to load images for the open_clip backend")

    device =(config .device or "auto").lower ()
    if device =="auto":
        device ="cuda"if torch .cuda .is_available ()else "cpu"

    print (f"[image pipeline] Loading OpenCLIP model: {config.model_name} on {device}...",flush =True )
    try :
        model ,_ ,preprocess_val =open_clip .create_model_and_transforms (config .model_name )
        preprocess =preprocess_val 
        model =model .to (device )
        model .eval ()
        print ("[image pipeline] Model loaded successfully.",flush =True )
    except Exception as e :
        print (f"[image pipeline] Failed to load model: {e}",flush =True )
        raise 

    batch_size =max (1 ,config .batch_size )




    dataset =ImageListDataset (paths ,preprocess )


    num_workers =8 
    if os .name =='nt':

        pass 

    print (f"[image pipeline] Using {num_workers} workers for feature extraction (Batch={batch_size}, Device={device})")

    dataloader =DataLoader (
    dataset ,
    batch_size =batch_size ,
    shuffle =False ,
    num_workers =num_workers ,
    pin_memory =(device =="cuda")
    )

    embeddings_list :List [np .ndarray ]=[]
    valid_paths :List [Path ]=[]
    failed_paths :List [Path ]=[]


    try :
        from tqdm import tqdm 
        pbar =tqdm (total =len (paths ),desc ="Extracting Features",unit ="img",file =sys .stdout )
    except ImportError :
        pbar =None 

    with torch .no_grad ():
        for batch_imgs ,batch_idxs ,batch_valid_flags in dataloader :
            if pbar :pbar .update (len (batch_idxs ))





            valid_mask =(batch_valid_flags ==1 )
            if not valid_mask .any ():

                for i in range (len (batch_idxs )):
                   failed_paths .append (paths [batch_idxs [i ]])
                continue 


            valid_imgs =batch_imgs [valid_mask ].to (device )
            valid_indices =batch_idxs [valid_mask ]


            if (~valid_mask ).any ():
                failed_indices =batch_idxs [~valid_mask ]
                for i in failed_indices :
                    failed_paths .append (paths [i ])


            features =model .encode_image (valid_imgs )
            features =torch .nn .functional .normalize (features ,dim =1 )

            embeddings_list .append (features .cpu ().numpy ())
            for idx in valid_indices :
                valid_paths .append (paths [idx ])

    if pbar :pbar .close ()

    if not embeddings_list :
        raise RuntimeError ("open_clip produced no embeddings (all images failed?)")

    stacked =np .concatenate (embeddings_list ,axis =0 )
    return stacked ,valid_paths ,failed_paths ,"open_clip"


def _compute_embeddings_average_rgb (
paths :Sequence [Path ],
)->Tuple [np .ndarray ,List [Path ],List [Path ],str ]:
    if Image is None :
        raise RuntimeError ("Pillow is required for the average_rgb embedding backend")

    features :List [np .ndarray ]=[]
    valid_paths :List [Path ]=[]
    failed_paths :List [Path ]=[]

    for path in paths :
        try :
            with Image .open (path )as img :
                img =img .convert ("RGB").resize ((64 ,64 ))
                arr =np .asarray (img ,dtype =np .float32 )/255.0 
        except Exception as exc :
            print (f"[image pipeline] failed to load {path}: {exc}")
            failed_paths .append (Path (path ))
            continue 

        mean =arr .mean (axis =(0 ,1 ))
        std =arr .std (axis =(0 ,1 ))
        feature =np .concatenate ([mean ,std ])
        features .append (feature )
        valid_paths .append (Path (path ))

    if not features :
        print ("[image pipeline] no valid images for average_rgb; treating all as missing")
        empty =np .empty ((0 ,0 ),dtype =np .float32 )
        return empty ,valid_paths ,failed_paths or [Path (p )for p in paths ],"average_rgb"

    stacked =np .stack (features ,axis =0 )
    return stacked ,valid_paths ,failed_paths ,"average_rgb"


def _run_deduplication (
paths :Sequence [Path ],
embeddings :np .ndarray ,
config :DedupConfig ,
indices :Optional [List [int ]],
)->Dict [str ,object ]:
    method =(config .method or "pairwise").lower ()

    if embeddings .size ==0 :
        return {
        "keepers":[],
        "duplicates":[],
        "duplicate_count":0 ,
        "skipped":0 ,
        }

    if method =="pairwise":
        return _deduplicate_pairwise (paths ,embeddings ,config )

    if method in {"sem_dedup","semdedup","legacy"}:
        return _deduplicate_sem_dedup (paths ,embeddings ,config ,indices )

    raise ValueError (f"Unknown deduplication method: {config.method}")


def _perform_semdedup_on_groups (
paths :Sequence [Path ],
embeddings :np .ndarray ,
groups :Dict [Any ,List [int ]],
config :DedupConfig ,
desc :str ="SemDeDup"
)->Dict [str ,object ]:
    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 
    threshold =1.0 -float (config .eps )

    try :
        from tqdm import tqdm 
        pbar =tqdm (total =len (groups ),desc =desc ,unit ="group",file =sys .stdout )
    except ImportError :
        pbar =None 
        print (f"[image pipeline] Processing {len(groups)} groups ({desc})...")


    for group_id ,indices in groups .items ():
        if len (indices )<2 :

            for idx in indices :
                keepers .append (paths [idx ])
        else :

            raw_feats =embeddings [indices ]

            local_norm =np .linalg .norm (raw_feats ,axis =1 ,keepdims =True )+1e-12 
            feats =raw_feats /local_norm 

            local_paths =[paths [i ]for i in indices ]
            N =len (indices )


            centroid =np .mean (feats ,axis =0 )
            centroid =centroid /(np .linalg .norm (centroid )+1e-12 )


            sim_to_center =feats @centroid 




            alpha =0.7 

            try :

                sizes =np .array ([p .stat ().st_size for p in local_paths ],dtype =np .float32 )
                if sizes .max ()>sizes .min ():
                    sizes_norm =(sizes -sizes .min ())/(sizes .max ()-sizes .min ())
                else :
                    sizes_norm =np .zeros_like (sizes )



                combined_score =alpha *sim_to_center +(1 -alpha )*sizes_norm 
                sort_order =np .argsort (combined_score )[::-1 ]
            except Exception :

                sort_order =np .argsort (sim_to_center )[::-1 ]


            kept_local_indices :List [int ]=[]
            kept_feats_list :List [np .ndarray ]=[]


            local_dups_map :Dict [int ,List [Dict [str ,object ]]]={}

            for rank_i in sort_order :
                current_feat =feats [rank_i ]
                is_dup =False 
                best_match_idx =-1 
                max_sim =-1.0 


                if kept_feats_list :
                    kept_feats_arr =np .array (kept_feats_list )

                    sims =kept_feats_arr @current_feat 
                    best_match_idx =int (np .argmax (sims ))
                    max_sim =float (sims [best_match_idx ])

                    if max_sim >=threshold :
                        is_dup =True 

                if not is_dup :
                    kept_local_indices .append (rank_i )
                    kept_feats_list .append (current_feat )
                    keepers .append (local_paths [rank_i ])
                else :
                    duplicate_count +=1 

                    keeper_idx =kept_local_indices [best_match_idx ]
                    if keeper_idx not in local_dups_map :
                        local_dups_map [keeper_idx ]=[]
                    local_dups_map [keeper_idx ].append ({
                    "path":str (local_paths [rank_i ]),
                    "similarity":max_sim 
                    })


            for kidx ,dup_list in local_dups_map .items ():
                duplicates .append ({
                "original":str (local_paths [kidx ]),
                "duplicates":dup_list ,
                "similarity_threshold":float (threshold )
                })

        if pbar :pbar .update (1 )

    if pbar :pbar .close ()

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _deduplicate_by_folder (
paths :Sequence [Path ],
embeddings :np .ndarray ,
config :DedupConfig ,
)->Dict [str ,object ]:

    folder_groups :Dict [Path ,List [int ]]={}
    for idx ,p in enumerate (paths ):
        parent =p .parent 
        if parent not in folder_groups :
            folder_groups [parent ]=[]
        folder_groups [parent ].append (idx )

    return _perform_semdedup_on_groups (paths ,embeddings ,folder_groups ,config ,desc ="Folder-SemDeDup")


def _deduplicate_dynamic_clustering (
paths :Sequence [Path ],
embeddings :np .ndarray ,
config :DedupConfig ,
)->Dict [str ,object ]:
    try :
        from sklearn .cluster import MiniBatchKMeans 
        print (f"[image pipeline] running MiniBatchKMeans on {len(paths)} items...",flush =True )




        n_clusters =max (1 ,min (len (paths )//1000 ,50000 ))

        if n_clusters ==1 :
             print ("[image pipeline] dataset small, using pairwise fallback")
             return _deduplicate_pairwise (paths ,embeddings ,config )

        kmeans =MiniBatchKMeans (
        n_clusters =n_clusters ,
        batch_size =min (4096 ,len (paths )),
        n_init ='auto',
        random_state =42 ,
        compute_labels =True ,
        verbose =0 
        )


        norm_embeddings =embeddings /(np .linalg .norm (embeddings ,axis =1 ,keepdims =True )+1e-12 )
        labels =kmeans .fit_predict (norm_embeddings )


        groups :Dict [int ,List [int ]]={}
        for idx ,label in enumerate (labels ):
            label =int (label )
            if label not in groups :
                groups [label ]=[]
            groups [label ].append (idx )

        print (f"[image pipeline] clusted into {len(groups)} groups. Starting SemDeDup...",flush =True )
        return _perform_semdedup_on_groups (paths ,embeddings ,groups ,config ,desc ="Global-SemDeDup")

    except ImportError as e :
        print (f"[image pipeline] sklearn import failed (ImportError): {e}")

        print ("[image pipeline] falling back to folder strategy.")
        return _deduplicate_by_folder (paths ,embeddings ,config )
    except Exception as exc :
        print (f"[image pipeline] clustering failed (Exception): {exc}")
        print ("[image pipeline] falling back to folder strategy.")
        return _deduplicate_by_folder (paths ,embeddings ,config )


def _deduplicate_pairwise (
paths :Sequence [Path ],
embeddings :np .ndarray ,
config :DedupConfig ,
)->Dict [str ,object ]:
    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 
    skipped =0 

    n =embeddings .shape [0 ]
    max_candidates =max (1 ,config .max_candidates )

    if n ==0 :
        return {
        "keepers":keepers ,
        "duplicates":duplicates ,
        "duplicate_count":duplicate_count ,
        "skipped":skipped ,
        }

    if n >max_candidates :
        print (
        f"[image pipeline] candidate count {n} exceeds max_candidates={max_candidates}; "
        "skipping pairwise deduplication."
        )
        skipped =n 
        keepers =[Path (p )for p in paths ]
        return {
        "keepers":keepers ,
        "duplicates":duplicates ,
        "duplicate_count":duplicate_count ,
        "skipped":skipped ,
        }

    norm =np .linalg .norm (embeddings ,axis =1 ,keepdims =True )+1e-12 
    normalized =embeddings /norm 
    similarity =normalized @normalized .T 
    np .fill_diagonal (similarity ,-np .inf )

    threshold =1.0 -float (config .eps )
    seen :Set [int ]=set ()
    progress =_progress_reporter (n ,"pairwise dedup","[image pipeline] ")

    for i in range (n ):
        if i in seen :
            continue 
        keepers .append (Path (paths [i ]))
        dup_entries :List [Dict [str ,object ]]=[]
        for j in range (i +1 ,n ):
            if j in seen :
                continue 
            sim =similarity [i ,j ]
            if sim >=threshold :
                dup_entries .append ({"path":str (paths [j ]),"similarity":float (sim )})
                seen .add (j )
        if dup_entries :
            duplicates .append (
            {
            "original":str (paths [i ]),
            "duplicates":dup_entries ,
            "similarity_threshold":float (threshold ),
            }
            )
            duplicate_count +=len (dup_entries )
        progress (i +1 )

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":skipped ,
    }


def _deduplicate_sem_dedup (
paths :Sequence [Path ],
embeddings :np .ndarray ,
config :DedupConfig ,
indices :Optional [List [int ]],
)->Dict [str ,object ]:
    if indices is None :
        print ("[image pipeline] SemDeDup indices not provided; attempting dynamic global clustering.")
        return _deduplicate_dynamic_clustering (paths ,embeddings ,config )

    try :
        keep_set ,cluster_members =_load_legacy_sem_dedup_assets (config ,indices )
    except Exception as exc :
        print (f"[image pipeline] failed to load legacy SemDeDup assets: {exc}; using pairwise fallback")
        return _deduplicate_pairwise (paths ,embeddings ,config )

    if not cluster_members :
        print ("[image pipeline] no cluster membership matched; using pairwise fallback")
        return _deduplicate_pairwise (paths ,embeddings ,config )

    idx_to_row ={idx :pos for pos ,idx in enumerate (indices )}

    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 

    total_clusters =len (cluster_members )
    progress =_progress_reporter (total_clusters ,"legacy SemDeDup","[image pipeline] ")

    for processed ,cluster_id in enumerate (sorted (cluster_members ),start =1 ):
        member_indices =cluster_members [cluster_id ]
        row_indices =[idx_to_row [idx ]for idx in member_indices if idx in idx_to_row ]
        if not row_indices :
            continue 

        keeper_rows =[row for row in row_indices if indices [row ]in keep_set ]
        if not keeper_rows :
            keeper_rows =[row_indices [0 ]]

        for row in keeper_rows :
            path =Path (paths [row ])
            if path not in keepers :
                keepers .append (path )

        duplicate_rows =[row for row in row_indices if row not in keeper_rows ]
        if not duplicate_rows :
            continue 

        keeper_vectors =embeddings [keeper_rows ]
        duplicate_vectors =embeddings [duplicate_rows ]

        keeper_norms =np .linalg .norm (keeper_vectors ,axis =1 ,keepdims =True )
        duplicate_norms =np .linalg .norm (duplicate_vectors ,axis =1 ,keepdims =True )
        keeper_norms =np .clip (keeper_norms ,1e-12 ,None )
        duplicate_norms =np .clip (duplicate_norms ,1e-12 ,None )

        scores =duplicate_vectors @keeper_vectors .T 
        scores =scores /(duplicate_norms @keeper_norms .T )

        duplicates_by_keeper :Dict [int ,List [Dict [str ,object ]]]={row :[]for row in keeper_rows }

        for dup_idx ,row in enumerate (duplicate_rows ):
            keeper_choice =int (np .argmax (scores [dup_idx ]))
            keeper_row =keeper_rows [keeper_choice ]
            similarity =float (scores [dup_idx ,keeper_choice ])
            duplicates_by_keeper .setdefault (keeper_row ,[]).append (
            {"path":str (paths [row ]),"similarity":similarity }
            )
            duplicate_count +=1 

        threshold =1.0 -float (config .eps )
        for keeper_row ,dup_list in duplicates_by_keeper .items ():
            if not dup_list :
                continue 
            duplicates .append (
            {
            "original":str (paths [keeper_row ]),
            "duplicates":dup_list ,
            "similarity_threshold":threshold ,
            "cluster_id":cluster_id ,
            }
            )
        progress (processed )

    if not keepers and paths :
        keepers =[Path (p )for p in paths ]

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _load_legacy_sem_dedup_assets (
config :DedupConfig ,
indices :Sequence [int ],
)->Tuple [Set [int ],Dict [int ,List [int ]]]:
    keep_path :Optional [Path ]=None 
    cluster_dir_path :Optional [Path ]=None 

    if config .legacy_keep_indices_file :
        keep_path =Path (config .legacy_keep_indices_file ).expanduser ()
    if config .legacy_cluster_dir :
        cluster_dir_path =Path (config .legacy_cluster_dir ).expanduser ()

    legacy_cfg :Dict [str ,Any ]={}
    if config .legacy_config_file :
        legacy_cfg =_load_legacy_config_file (Path (config .legacy_config_file ))
        if keep_path is None and "save_folder"in legacy_cfg :
            eps_dir =f"eps_{config.eps}"
            keep_path =Path (legacy_cfg ["save_folder"]).expanduser ()/eps_dir /"all_kept_samples.txt"
        if cluster_dir_path is None and "sorted_clusters_path"in legacy_cfg :
            cluster_dir_path =Path (legacy_cfg ["sorted_clusters_path"]).expanduser ()

    if keep_path is None :
        raise ValueError ("Legacy SemDeDup requires legacy_keep_indices_file or config with save_folder")
    if cluster_dir_path is None :
        raise ValueError ("Legacy SemDeDup requires legacy_cluster_dir or config with sorted_clusters_path")

    keep_set =legacy_integration .load_keep_indices (keep_path )
    target_indices =set (int (idx )for idx in indices )
    cluster_members =legacy_integration .load_cluster_members (cluster_dir_path ,target_indices =target_indices )
    return keep_set ,cluster_members 


def _load_legacy_config_file (path :Path )->Dict [str ,Any ]:
    if not path .exists ():
        raise FileNotFoundError (f"Legacy configuration file not found: {path}")
    content =path .read_text (encoding ="utf-8")
    if path .suffix .lower ()in {".yaml",".yml"}:
        data =yaml .safe_load (content )or {}
    else :
        data =json .loads (content )
