
from __future__ import annotations 

import json 
import re 
import hashlib 
from dataclasses import dataclass ,field 
from pathlib import Path 
from typing import Any ,Callable ,Dict ,Iterable ,List ,Optional ,Sequence ,Set 

import yaml 
try :
    from datasketch import MinHash ,MinHashLSH 
except Exception :
    MinHash ,MinHashLSH =None ,None 
try :
    from simhash import Simhash 
except Exception :
    Simhash =None 


@dataclass 
class TextEmbeddingConfig :

    ngram_size :int =3 
    lowercase :bool =True 
    strip_non_alnum :bool =True 
    collapse_whitespace :bool =True 
    encoding :str ="utf-8"
    errors :str ="ignore"


@dataclass 
class TextDedupConfig :
    method :str ="jaccard"
    threshold :float =0.8 
    max_candidates :int =5000 
    num_perm :int =128 
    simhash_dist :int =10 
    simhash_window :int =1000 
    max_words :int =200 
    max_char_grams :int =200 


    window_size :int =100 


@dataclass 
class TextPipelineConfig :
    embedding :TextEmbeddingConfig =field (default_factory =TextEmbeddingConfig )
    dedup :TextDedupConfig =field (default_factory =TextDedupConfig )


@dataclass 
class TextEmbeddingResult :
    features :List [Set [str ]]
    texts :List [str ]
    paths :List [Path ]
    failed_paths :List [Path ]
    backend :str 


@dataclass 
class TextPipelineResult :
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


_NON_ALNUM_RE =re .compile (r"[^\w\s\u4e00-\u9fff]",re .UNICODE )
_WHITESPACE_RE =re .compile (r"\s+",re .UNICODE )


def _merge_dict (default :Dict [str ,Any ],override :Dict [str ,Any ])->Dict [str ,Any ]:
    merged =dict (default )
    for key ,value in override .items ():
        if key in merged and isinstance (merged [key ],dict )and isinstance (value ,dict ):
            merged [key ]=_merge_dict (merged [key ],value )
        else :
            merged [key ]=value 
    return merged 


def load_pipeline_config (config_path :Optional [str ])->TextPipelineConfig :

    defaults :Dict [str ,Dict [str ,Any ]]={
    "embedding":{
    "ngram_size":3 ,
    "lowercase":True ,
    "strip_non_alnum":True ,
    "collapse_whitespace":True ,
    "encoding":"utf-8",
    "errors":"ignore",
    },
    "dedup":{
    "method":"jaccard",
    "threshold":0.8 ,
    "max_candidates":5000 ,
    "num_perm":128 ,
    "simhash_dist":10 ,
    "simhash_window":1000 ,
    "max_words":200 ,
    "max_char_grams":200 ,
    },
    }

    if not config_path :
        config_dict =dict (defaults )
    else :
        path =Path (config_path )
        if not path .exists ():
            raise FileNotFoundError (f"Text pipeline config not found: {path}")
        content =path .read_text (encoding ="utf-8")
        try :
            if path .suffix .lower ()in {".yaml",".yml"}:
                loaded =yaml .safe_load (content )or {}
            else :
                loaded =json .loads (content )
        except Exception as exc :
            raise ValueError (f"Failed to parse text pipeline config {path}: {exc}")from exc 
        config_dict =_merge_dict (defaults ,loaded )

    return TextPipelineConfig (
    embedding =TextEmbeddingConfig (**config_dict ["embedding"]),
    dedup =TextDedupConfig (**config_dict ["dedup"]),
    )


def run_text_pipeline (paths :Sequence [Path ],config :TextPipelineConfig )->TextPipelineResult :
    unique_paths :List [Path ]=[]
    missing :List [Path ]=[]
    seen :Set [str ]=set ()

    for path in paths :
        candidate =Path (path )
        key =str (candidate .resolve ())
        if key in seen :
            continue 
        seen .add (key )
        if candidate .exists ():
            unique_paths .append (candidate )
        else :
            missing .append (candidate )

    if not unique_paths :
        stats ={
        "embedding_backend":None ,
        "unique":0 ,
        "duplicates":0 ,
        "missing":len (missing ),
        "processed":0 ,
        }
        return TextPipelineResult (keepers =[],duplicates =[],missing =missing ,stats =stats )

    try :
        embedding_result =_compute_text_signatures (unique_paths ,config .embedding )
    except Exception as exc :
        print (f"[text pipeline] failed to compute signatures for {len(unique_paths)} files: {exc}")
        missing .extend (unique_paths )
        stats ={
        "embedding_backend":None ,
        "unique":0 ,
        "duplicates":0 ,
        "missing":len (missing ),
        "processed":0 ,
        "error":str (exc ),
        }
        return TextPipelineResult (keepers =[],duplicates =[],missing =missing ,stats =stats )

    missing .extend (embedding_result .failed_paths )

    dedup_summary =_run_deduplication (
    embedding_result .paths ,
    embedding_result .features ,
    embedding_result .texts ,
    config .embedding ,
    config .dedup ,
    )

    stats ={
    "embedding_backend":embedding_result .backend ,
    "unique":len (dedup_summary ["keepers"]),
    "duplicates":dedup_summary ["duplicate_count"],
    "missing":len (missing ),
    "processed":len (embedding_result .paths ),
    "skipped_due_to_limit":dedup_summary ["skipped"],
    }

    return TextPipelineResult (
    keepers =dedup_summary ["keepers"],
    duplicates =dedup_summary ["duplicates"],
    missing =missing ,
    stats =stats ,
    )


def _normalize_text (content :str ,config :TextEmbeddingConfig )->str :
    text =content 
    if config .lowercase :
        text =text .lower ()
    if config .strip_non_alnum :
        text =_NON_ALNUM_RE .sub (" ",text )
    if config .collapse_whitespace :
        text =_WHITESPACE_RE .sub (" ",text )
    return text .strip ()


def _compute_ngrams (text :str ,n :int )->Set [str ]:
    if not text :
        return set ()
    length =len (text )
    if length <n :
        char_ngrams ={text }
    else :
        char_ngrams ={text [i :i +n ]for i in range (length -n +1 )}
    words =text .split ()
    word_ngrams :Set [str ]=set ()
    if len (words )>=n :
        for idx in range (len (words )-n +1 ):
            word_ngrams .add (" ".join (words [idx :idx +n ]))
    return char_ngrams |word_ngrams 


def _char_ngrams (text :str ,n :int )->List [str ]:
    cleaned =text .replace (" ","")
    if len (cleaned )<n :
        return []if not cleaned else [cleaned ]
    return [cleaned [i :i +n ]for i in range (len (cleaned )-n +1 )]


def _compute_text_signatures (
paths :Sequence [Path ],
config :TextEmbeddingConfig ,
)->TextEmbeddingResult :
    features :List [Set [str ]]=[]
    texts :List [str ]=[]
    processed_paths :List [Path ]=[]
    failed :List [Path ]=[]

    for path in paths :
        try :
            raw_text =path .read_text (encoding =config .encoding ,errors =config .errors )
        except Exception as exc :
            print (f"[text pipeline] failed to read {path}: {exc}")
            failed .append (path )
            continue 
        normalized =_normalize_text (raw_text ,config )
        signature =_compute_ngrams (normalized ,max (1 ,config .ngram_size ))
        features .append (signature )
        texts .append (normalized )
        processed_paths .append (path )

    if not features :
        raise RuntimeError ("No text signatures could be computed")

    return TextEmbeddingResult (
    features =features ,
    texts =texts ,
    paths =processed_paths ,
    failed_paths =failed ,
    backend ="computed",
    )


def _run_deduplication (
paths :Sequence [Path ],
features :Sequence [Set [str ]],
texts :Sequence [str ],
embedding_config :TextEmbeddingConfig ,
config :TextDedupConfig ,
)->Dict [str ,Any ]:
    if not features :
        return {
        "keepers":[],
        "duplicates":[],
        "duplicate_count":0 ,
        "skipped":0 ,
        }

    method =(config .method or "jaccard").lower ()
    if method =="jaccard":


        if len (features )>config .max_candidates :
            print (
            f"[text pipeline] candidate count {len(features)} exceeds max_candidates={config.max_candidates}; "
            "using rolling-window quick deduplication"
            )
            return _deduplicate_windowed (paths ,features ,config .threshold ,config .window_size )
        return _deduplicate_by_jaccard (paths ,features ,config .threshold )

    if method =="md5":
        return _deduplicate_by_md5 (paths ,texts )

    if method =="simhash":
        return _deduplicate_by_simhash (paths ,texts ,config .simhash_dist ,config .simhash_window )

    if method =="minhash_lsh":
        return _deduplicate_by_minhash_lsh (
        paths ,
        texts ,
        embedding_config .ngram_size ,
        config ,
        include_words =False ,
        )

    if method =="ours_lsh":
        return _deduplicate_by_minhash_lsh (
        paths ,
        texts ,
        embedding_config .ngram_size ,
        config ,
        include_words =True ,
        )

    raise ValueError (f"Unknown text deduplication method: {config.method}")


def _deduplicate_by_md5 (
paths :Sequence [Path ],
texts :Sequence [str ],
)->Dict [str ,Any ]:
    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    seen :Dict [str ,Path ]={}
    duplicate_count =0 

    for path ,text in zip (paths ,texts ):
        md5_val =hashlib .md5 (text .encode ("utf-8")).hexdigest ()
        original =seen .get (md5_val )
        if original is None :
            seen [md5_val ]=path 
            keepers .append (path )
        else :
            duplicates .append (
            {
            "original":str (original ),
            "duplicates":[{"path":str (path ),"similarity":1.0 }],
            "similarity_threshold":1.0 ,
            }
            )
            duplicate_count +=1 

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _deduplicate_by_simhash (
paths :Sequence [Path ],
texts :Sequence [str ],
dist_threshold :int ,
window_size :int ,
)->Dict [str ,Any ]:
    if Simhash is None :
        raise RuntimeError ("simhash is required for simhash text deduplication")

    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 
    seen :List [tuple [Path ,Simhash ]]=[]

    for path ,text in zip (paths ,texts ):
        curr =Simhash (text )
        is_dup =False 
        dup_entries :List [Dict [str ,object ]]=[]
        window =seen [-window_size :]if window_size >0 else seen 
        for seen_path ,seen_hash in window :
            if curr .distance (seen_hash )<=dist_threshold :
                is_dup =True 
                dup_entries .append ({"path":str (path ),"similarity":1.0 })
                duplicates .append (
                {
                "original":str (seen_path ),
                "duplicates":[{"path":str (path ),"similarity":1.0 }],
                "similarity_threshold":float (dist_threshold ),
                }
                )
                duplicate_count +=1 
                break 
        if not is_dup :
            keepers .append (path )
            seen .append ((path ,curr ))

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _build_minhash (
text :str ,
ngram_size :int ,
config :TextDedupConfig ,
include_words :bool ,
):
    if MinHash is None :
        raise RuntimeError ("datasketch is required for MinHash LSH deduplication")
    m =MinHash (num_perm =config .num_perm )
    if include_words :
        for w in text .split ()[:config .max_words ]:
            m .update (w .encode ("utf-8"))
    for g in _char_ngrams (text ,max (1 ,ngram_size ))[:config .max_char_grams ]:
        m .update (g .encode ("utf-8"))
    return m 


def _deduplicate_by_minhash_lsh (
paths :Sequence [Path ],
texts :Sequence [str ],
ngram_size :int ,
config :TextDedupConfig ,
include_words :bool ,
)->Dict [str ,Any ]:
    if MinHashLSH is None :
        raise RuntimeError ("datasketch is required for MinHash LSH deduplication")

    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 
    lsh =MinHashLSH (threshold =config .threshold ,num_perm =config .num_perm )
    key_to_path :Dict [str ,Path ]={}

    for idx ,(path ,text )in enumerate (zip (paths ,texts )):
        m =_build_minhash (text ,ngram_size ,config ,include_words )
        matches =lsh .query (m )
        if not matches :
            key =f"doc_{idx}"
            lsh .insert (key ,m )
            key_to_path [key ]=path 
            keepers .append (path )
        else :
            original_key =matches [0 ]
            original_path =key_to_path .get (original_key ,path )
            duplicates .append (
            {
            "original":str (original_path ),
            "duplicates":[{"path":str (path ),"similarity":None }],
            "similarity_threshold":float (config .threshold ),
            }
            )
            duplicate_count +=1 

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _jaccard_similarity (a :Set [str ],b :Set [str ])->float :
    if not a and not b :
        return 1.0 
    union =a |b 
    if not union :
        return 0.0 
    intersection_size =len (a &b )
    return intersection_size /len (union )


def _deduplicate_by_jaccard (
paths :Sequence [Path ],
features :Sequence [Set [str ]],
threshold :float ,
)->Dict [str ,Any ]:
    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 
    seen :Set [int ]=set ()
    total =len (paths )
    progress =_progress_reporter (total ,"jaccard dedup","[text pipeline] ")

    for idx ,path in enumerate (paths ):
        if idx in seen :
            continue 
        keepers .append (path )
        dup_entries :List [Dict [str ,object ]]=[]
        for other_idx in range (idx +1 ,len (paths )):
            if other_idx in seen :
                continue 
            sim =_jaccard_similarity (features [idx ],features [other_idx ])
            if sim >=threshold :
                dup_entries .append ({"path":str (paths [other_idx ]),"similarity":float (sim )})
                seen .add (other_idx )
        if dup_entries :
            duplicates .append (
            {
            "original":str (path ),
            "duplicates":dup_entries ,
            "similarity_threshold":float (threshold ),
            }
            )
            duplicate_count +=len (dup_entries )
        progress (idx +1 )

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }


def _deduplicate_windowed (
paths :Sequence [Path ],
features :Sequence [Set [str ]],
threshold :float ,
window_size :int =100 ,
)->Dict [str ,Any ]:
    keepers :List [Path ]=[]
    duplicates :List [Dict [str ,object ]]=[]
    duplicate_count =0 


    kept_features :List [Set [str ]]=[]

    for idx ,path in enumerate (paths ):
        current_feat =features [idx ]
        is_dup =False 
        dup_entries :List [Dict [str ,object ]]=[]


        start =max (0 ,len (kept_features )-window_size )
        for j in range (start ,len (kept_features )):
            sim =_jaccard_similarity (current_feat ,kept_features [j ])
            if sim >=threshold :

                is_dup =True 
                dup_entries .append ({"path":str (keepers [j ]),"similarity":float (sim )})


        if is_dup :
            duplicates .append ({
            "original":str (keepers [-1 ])if keepers else str (path ),
            "duplicates":dup_entries ,
            "similarity_threshold":float (threshold ),
            })
            duplicate_count +=len (dup_entries )
        else :

            keepers .append (path )
            kept_features .append (current_feat )

    return {
    "keepers":keepers ,
    "duplicates":duplicates ,
    "duplicate_count":duplicate_count ,
    "skipped":0 ,
    }
