from __future__ import annotations 

import os 

import sys 
from pathlib import Path 


def _ensure_package_root ()->None :
    module_name ="pipelines.modalities.common"
    if module_name in sys .modules :
        return 
    pkg_root =Path (__file__ ).resolve ().parents [1 ]
    project_root =pkg_root .parent 
    if str (project_root )not in sys .path :
        sys .path .insert (0 ,str (project_root ))


_ensure_package_root ()

from pipelines .modalities .common import (
copy_existing_files ,
ensure_output_dir ,
load_input_paths ,
write_summary ,
)

from text .method .pipeline_api import (
TextPipelineConfig ,
TextPipelineResult ,
load_pipeline_config ,
run_text_pipeline ,
)


def main ()->None :
    paths ,manifest_path =load_input_paths ("PIPELINE_TEXT_INPUT_LIST")
    total_candidates =int (os .environ .get ("PIPELINE_TEXT_TOTAL","0")or 0 )
    output_dir =ensure_output_dir ("PIPELINE_TEXT_OUTPUT_DIR")
    config_path =os .environ .get ("PIPELINE_TEXT_CONFIG_FILE")

    try :
        config :TextPipelineConfig =load_pipeline_config (config_path )
    except Exception as exc :
        print (f"[text runner] failed to load config {config_path!r}: {exc}")
        raise 

    result :TextPipelineResult =run_text_pipeline (paths ,config )

    stats =dict (result .stats )
    stats .setdefault ("total_candidates",total_candidates )
    stats .setdefault ("selected",len (result .keepers ))

    copy_stats ={"copied":0 ,"skipped":0 ,"missing":0 }
    if output_dir and result .keepers :
        copy_stats =copy_existing_files (result .keepers ,output_dir )
        stats .update (copy_stats )
    else :
        stats .setdefault ("copied",0 )
        stats .setdefault ("skipped",0 )
        if not output_dir :
            print ("[text runner] No output directory specified; skipping copy")

    stats ["missing"]=len (result .missing )+copy_stats .get ("missing",0 )


    try :
        processed_bytes =0 
        for p in paths :
            try :
                if p .exists ():
                    processed_bytes +=p .stat ().st_size 
            except Exception :
                continue 
        stats ["processed_bytes"]=processed_bytes 
    except Exception :
        stats .setdefault ("processed_bytes",None )

    write_summary (output_dir ,"text",stats ,manifest_path ,duplicates =result .duplicates )

    print (
    f"[text runner] processed {len(paths)} entries (total candidates={total_candidates}); "
    f"unique={stats.get('unique', 0)}, duplicates={stats.get('duplicates', 0)}, "
    f"backend={stats.get('embedding_backend')}"
    )


if __name__ =="__main__":
    main ()
