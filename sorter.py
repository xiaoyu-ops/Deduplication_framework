
import argparse 
import csv 
import json 
import os 
import shutil 
import time 
from collections import Counter 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional 

from tqdm import tqdm 


def read_files_from_directory (directory :str ):
    all_files =[]
    print (f"[Sorter] Start scanning directory: {directory}")

    try :
        from tqdm import tqdm 
        pbar =tqdm (desc ="Scanning files",unit ="file")
    except ImportError :
        pbar =None 

    count =0 
    for root ,_ ,files in os .walk (directory ):
        for file in files :
            all_files .append (os .path .join (root ,file ))
            if pbar is not None :
                pbar .update (1 )
            else :
                count +=1 
                if count %10000 ==0 :
                    print (f"Scanned {count} files...",end ='\r')

    if pbar is not None :
        pbar .close ()
    print (f"\n[Sorter] Scan complete. Found {len(all_files)} files.")
    return all_files 

IMAGE_EXTS ={".png",".jpg",".jpeg",".gif",".bmp",".webp",".tif",".tiff"}

STRICT_IMAGE_EXTS ={".png",".jpg",".jpeg"}
STRICT_AUDIO_EXTS ={".wav"}
AUDIO_EXTS ={".wav",".mp3",".aac",".flac",".ogg",".m4a",".wma"}
TEXT_EXTS ={".txt",".json",".csv",".md",".xml",".yaml",".yml",".ini",".log",".tsv"}

JSON_TEXT_KEYS ={"text","content","title","sentence","article"}
JSON_AUDIO_KEYS ={"audio","audio_url","audio_path","wav","mp3"}
JSON_IMAGE_KEYS ={"image","image_url","img","picture","thumbnail"}

HEADER_BYTES =4096 
PRINTABLE_THRESHOLD =0.85 


def is_image_url (u :str )->bool :
    if not isinstance (u ,str ):
        return False 
    u_low =u .lower ()
    return any (ext in u_low for ext in [".jpg",".jpeg",".png",".gif",".bmp",".webp"])


def safe_move_file (src_path :str ,dst_path :str ,max_retries :int =3 )->Optional [str ]:
    for attempt in range (max_retries ):
        try :
            os .makedirs (os .path .dirname (dst_path ),exist_ok =True )


            original_dst =dst_path 
            candidate =dst_path 
            counter =1 
            while os .path .exists (candidate ):
                name ,ext =os .path .splitext (original_dst )
                candidate =f"{name}_{counter}{ext}"
                counter +=1 

            shutil .move (src_path ,candidate )
            print (f"Moved file {src_path} to {candidate}")
            return candidate 

        except PermissionError :
            print (f"File locked, retry {attempt + 1}/{max_retries}: {src_path}")
            if attempt <max_retries -1 :
                time .sleep (1 )
            else :
                print (f"File move failed; it may be in use: {src_path}")
                print ("Close any program that might lock the file, e.g., VS Code or Notepad")
                return None 
        except Exception as exc :
            print (f"Unexpected error while moving file: {exc}")
            return None 
    return None 


def put_file_in_category (file_path :str ,category :str ,base_dir :Optional [str ]=None )->Optional [str ]:
    if base_dir is None :
        base_dir =str (Path (__file__ ).resolve ().parent )
    category_dir =os .path .join (base_dir ,category ,"dataset")
    filename =os .path .basename (file_path )
    dest_path =os .path .join (category_dir ,filename )
    return safe_move_file (file_path ,dest_path )


def read_header (file_path :str ,size :int =HEADER_BYTES )->bytes :
    try :
        with open (file_path ,"rb")as f :
            return f .read (size )
    except Exception :
        return b""


def is_mostly_printable (data :bytes )->bool :
    if not data :
        return False 
    printable =sum ((chr (b ).isprintable ()or chr (b ).isspace ())for b in data )
    return printable /len (data )>=PRINTABLE_THRESHOLD 


def sniff_magic (header :bytes )->Optional [str ]:
    if header .startswith (b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header .startswith (b"GIF87a")or header .startswith (b"GIF89a"):
        return "image"
    if header [:4 ]==b"RIFF"and header [8 :12 ]==b"WAVE":
        return "audio"
    if header .startswith (b"ID3"):
        return "audio"
    if header .startswith (b"fLaC"):
        return "audio"
    lowered =header .lower ()
    if b"metadata:image"in header or b"<svg"in lowered :
        return "image"
    stripped =header .lstrip ()
    if stripped .startswith (b"{")or stripped .startswith (b"["):
        return "text"
    return None 


def load_json_payload (file_path :str )->Optional [Any ]:
    try :
        with open (file_path ,"r",encoding ="utf-8")as f :
            return json .load (f )
    except UnicodeDecodeError :
        try :
            with open (file_path ,"r",encoding ="utf-8",errors ="ignore")as f :
                content =f .read ()
            return json .loads (content )
        except Exception :
            return None 
    except json .JSONDecodeError :
        return None 
    except Exception :
        return None 


def classify_json_payload (data :Any )->str :
    if isinstance (data ,dict ):
        keys =set (data .keys ())
        if keys &JSON_IMAGE_KEYS :
            return "image"
        if keys &JSON_AUDIO_KEYS :
            return "audio"
        if keys &JSON_TEXT_KEYS :
            return "text"
        url =data .get ("url")if isinstance (data .get ("url"),str )else None 
        if url and is_image_url (url ):
            return "image"
        if any (isinstance (v ,str )and v .strip ()for v in data .values ()):
            return "text"
        return "text"

    if isinstance (data ,list )and data :
        votes :Counter =Counter ()
        sample =data [:min (10 ,len (data ))]
        for item in sample :
            if isinstance (item ,dict ):
                keys =set (item .keys ())
                if keys &JSON_IMAGE_KEYS :
                    votes ["image"]+=1 
                if keys &JSON_AUDIO_KEYS :
                    votes ["audio"]+=1 
                if keys &JSON_TEXT_KEYS :
                    votes ["text"]+=1 
                if "url"in item and isinstance (item ["url"],str )and is_image_url (item ["url"]):
                    votes ["image"]+=1 
            elif isinstance (item ,str )and item .strip ():
                votes ["text"]+=1 
        if votes :
            label ,count =votes .most_common (1 )[0 ]
            if count >0 :
                return label 
        return "text"

    if isinstance (data ,str )and data .strip ():
        return "text"

    return "text"


def classify_json_file (file_path :str )->Optional [str ]:
    payload =load_json_payload (file_path )
    if payload is None :
        return None 
    return classify_json_payload (payload )


def determine_category (file_path :str )->str :
    suffix =Path (file_path ).suffix .lower ()



    if suffix in STRICT_IMAGE_EXTS :
        return "image"
    if suffix in STRICT_AUDIO_EXTS :
        return "audio"

    try :
        size =os .path .getsize (file_path )
    except OSError :
        return "error"

    if size ==0 :
        return "unknown"

    header =read_header (file_path )
    magic =sniff_magic (header )

    if suffix ==".json":
        label =classify_json_file (file_path )
        if label :
            return label 
        if magic :
            if magic =="text":
                return "text"
            return magic 
        if is_mostly_printable (header ):
            return "text"
        return "unknown"

    if suffix in IMAGE_EXTS :
        if magic :
            if magic =="text":
                label =classify_json_file (file_path )
                if label :
                    return label 
                return "text"
            return magic 
        return "image"

    if suffix in AUDIO_EXTS :
        if magic =="audio":
            return "audio"
        if magic =="image":
            return "image"
        if magic =="text":
            label =classify_json_file (file_path )
            if label :
                return label 
            if is_mostly_printable (header ):
                return "text"
            return "unknown"
        if not is_mostly_printable (header ):
            return "audio"
        label =classify_json_file (file_path )
        if label :
            return label 
        return "text"

    if suffix in TEXT_EXTS :
        if magic and magic !="text":
            return magic 
        if is_mostly_printable (header ):
            return "text"
        label =classify_json_file (file_path )
        if label :
            return label 
        return "text"

    if magic :
        if magic =="text":
            label =classify_json_file (file_path )
            if label :
                return label 
            return "text"
        return magic 

    if is_mostly_printable (header ):
        label =classify_json_file (file_path )
        if label :
            return label 
        return "text"

    return "unknown"


if __name__ =="__main__":
    from concurrent .futures import ThreadPoolExecutor 
    import multiprocessing 

def _worker_classify_batch (files_batch ):
    results =[]
    for file_path in files_batch :
        try :
            size_bytes =0 
            try :
                size_bytes =os .path .getsize (file_path )
            except OSError :
                size_bytes =0 

            category =determine_category (file_path )
            suffix =Path (file_path ).suffix .lower ()


            if category =="image"and suffix not in STRICT_IMAGE_EXTS :
                category ="unknown"
            if category =="audio"and suffix not in STRICT_AUDIO_EXTS :
                category ="unknown"

            results .append ({
            "file_path":file_path ,
            "category":category ,
            "size_bytes":size_bytes ,
            "reason":None ,
            "status":None 
            })

        except Exception as exc :
            results .append ({
            "file_path":file_path ,
            "category":"error",
            "size_bytes":0 ,
            "reason":str (exc ),
            "status":"error"
            })
    return results 

def sorter (
files ,
*,
eval_mode :bool =False ,
prediction_path :Optional [str ]=None ,
input_root :Optional [str ]=None ,
move_base_dir :Optional [str ]=None ,
collect_only :bool =False ,
num_workers :int =16 
):
    from concurrent .futures import ThreadPoolExecutor ,as_completed 

    start_time =time .time ()
    success_count =0 
    fail_count =0 
    categorized :Dict [str ,List [str ]]={"image":[],"audio":[],"text":[]}
    categorized_bytes :Dict [str ,int ]={"image":0 ,"audio":0 ,"text":0 }
    other_categories :Dict [str ,List [str ]]={}
    other_bytes :Dict [str ,int ]={}
    total_bytes =0 
    details :List [Dict [str ,Any ]]=[]

    prediction_writer =None 
    prediction_file =None 

    if prediction_path :
        prediction_path =Path (prediction_path )
        prediction_path .parent .mkdir (parents =True ,exist_ok =True )
        prediction_file =prediction_path .open ("w",newline ="",encoding ="utf-8")
        prediction_writer =csv .writer (prediction_file )
        prediction_writer .writerow (["filename","predicted_label"])

    def resolve_relative (path :str )->str :
        if input_root :
            try :
                return os .path .relpath (path ,input_root ).replace ("\\","/")
            except ValueError :
                pass 
        return path .replace ("\\","/")

    def write_prediction (path :str ,label :str )->None :
        if prediction_writer is None :
            return 
        prediction_writer .writerow ([resolve_relative (path ),label ])


    files_list =list (files )
    total_files =len (files_list )


    chunk_size =1000 
    file_chunks =[files_list [i :i +chunk_size ]for i in range (0 ,total_files ,chunk_size )]

    print (f"[Sorter] Processing {total_files} files with {num_workers} threads in {len(file_chunks)} batches...")


    pbar =tqdm (total =total_files ,desc ="Classifying",ascii =True )

    with ThreadPoolExecutor (max_workers =num_workers )as executor :
        futures ={executor .submit (_worker_classify_batch ,chunk ):chunk for chunk in file_chunks }

        for future in as_completed (futures ):
            try :
                batch_results =future .result ()

                pbar .update (len (batch_results ))

                for res in batch_results :
                    f_path =res ["file_path"]
                    category =res ["category"]
                    size_bytes =res ["size_bytes"]
                    reason =res ["reason"]

                    total_bytes +=size_bytes 


                    if category =="error":

                        fail_count +=1 
                        write_prediction (f_path ,"error")
                        details .append ({
                        "source_path":os .path .abspath (f_path ),
                        "relative_path":resolve_relative (f_path ),
                        "category":"error",
                        "status":"error",
                        "target_path":None ,
                        "reason":reason ,
                        })
                        continue 


                    write_prediction (f_path ,category )
                    is_supported =category in {"audio","image","text"}
                    abs_path =os .path .abspath (f_path )

                    detail_entry :Dict [str ,Any ]={
                    "source_path":abs_path ,
                    "relative_path":resolve_relative (f_path ),
                    "category":category ,
                    "status":None ,
                    "target_path":None ,
                    "reason":None ,
                    "size_bytes":size_bytes ,
                    }

                    if eval_mode or collect_only :

                        if is_supported :
                            success_count +=1 
                            detail_entry ["status"]="evaluated"
                        else :
                            fail_count +=1 
                            detail_entry ["status"]="unsupported"
                            detail_entry ["reason"]="unsupported_category"
                    else :

                        if is_supported :
                            moved_path =move_file_to_category (f_path ,category ,move_base_dir )
                            if moved_path :
                                success_count +=1 
                                detail_entry ["status"]="moved"
                                detail_entry ["target_path"]=moved_path 
                            else :
                                fail_count +=1 
                                detail_entry ["status"]="failed_move"
                                detail_entry ["reason"]="move_failed"
                        else :

                            moved_path =move_file_to_category (f_path ,"unknown",move_base_dir )
                            fail_count +=1 
                            detail_entry ["status"]="moved_unknown"if moved_path else "failed_move"
                            detail_entry ["category"]="unknown"
                            if moved_path :detail_entry ["target_path"]=moved_path 

                    details .append (detail_entry )


                    if category in categorized :
                        categorized [category ].append (abs_path )
                        categorized_bytes [category ]+=size_bytes 
                    else :
                        if category not in other_categories :
                            other_categories [category ]=[]
                            other_bytes [category ]=0 
                        other_categories [category ].append (abs_path )
                        other_bytes [category ]+=size_bytes 

            except Exception as e :
                print (f"Worker exception: {e}")
                fail_count +=1 

    pbar .close ()
    if prediction_file :
        prediction_file .close ()

    end_time =time .time ()
    elapsed_time =end_time -start_time 

    print ("\nClassification completed")
    print (f"Success: {success_count} files")
    print (f"Failed/Skipped: {fail_count} files")
    print (f"Elapsed time: {elapsed_time:.2f} seconds")

    return {
    "success_count":success_count ,
    "fail_count":fail_count ,
    "elapsed_seconds":elapsed_time ,
    "categorized":categorized ,
    "unknown":other_categories ,
    "per_modality_bytes":categorized_bytes ,
    "unknown_bytes":other_bytes ,
    "total_bytes":total_bytes ,
    "details":details ,
    "prediction_file":str (prediction_path .resolve ())if prediction_path else None ,
    }

if __name__ =="__main__":
    parser =argparse .ArgumentParser (description ="Classify or evaluate the mix_dataset inputs")
    parser .add_argument ("--input",default ="./mix_dataset",help ="Input directory to classify")
    parser .add_argument (
    "--eval",
    action ="store_true",
    help ="Enable evaluation mode (no file moves, only predictions)",
    )
    parser .add_argument ("--predictions",help ="CSV path to save predictions in eval mode")
    parser .add_argument ("--input-root",help ="Root directory used for relative paths (defaults to --input)")
    parser .add_argument ("--move-base",help ="Destination root when moving files in non-eval mode")

    args =parser .parse_args ()

    input_directory =args .input 
    files =read_files_from_directory (input_directory )
    print (f"Found {len(files)} files")

    if not files :
        print ("No files found")
        raise SystemExit (0 )

    print ("First 10 files:")
    for i ,f in enumerate (files [:10 ]):
        print (f"  {i+1}. {f}")

    prediction_path =args .predictions 
    if args .eval and not prediction_path :
        prediction_path ="predictions.csv"

    input_root =args .input_root or input_directory 

    result =sorter (
    files ,
    eval_mode =args .eval ,
    prediction_path =prediction_path ,
    input_root =input_root ,
    move_base_dir =args .move_base ,
    collect_only =False ,
    )

    if args .eval :
        print ("Evaluation mode completed")
    else :
        print ("Classification run completed")

    if prediction_path :
        print (f"Predictions saved to: {Path(prediction_path).resolve()}")

    print (
    f"Stats: success={result['success_count']} fail={result['fail_count']} "
    f"elapsed={result['elapsed_seconds']:.2f}s"
    )