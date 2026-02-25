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

from audio .method .pipeline_api import (
AudioPipelineConfig ,
AudioPipelineResult ,
load_pipeline_config ,
run_audio_pipeline ,
)


def main ()->None :
    paths ,manifest_path =load_input_paths ("PIPELINE_AUDIO_INPUT_LIST")
    total_candidates =int (os .environ .get ("PIPELINE_AUDIO_TOTAL","0")or 0 )
    output_dir =ensure_output_dir ("PIPELINE_AUDIO_OUTPUT_DIR")
    config_path =os .environ .get ("PIPELINE_AUDIO_CONFIG_FILE")

    try :
        config :AudioPipelineConfig =load_pipeline_config (config_path )
    except Exception as exc :
        print (f"[audio runner] failed to load config {config_path!r}: {exc}")
        raise 


    try :
        print (
        f"[audio runner] loaded config={config_path!r} embedding.threshold={config.embedding.threshold} "
        f"dedup.threshold={config.dedup.threshold} max_candidates={config.dedup.max_candidates} "
        f"fingerprint_backend={config.embedding.fingerprint_backend} precomputed={config.embedding.precomputed_fingerprints}",
        flush =True ,
        )
    except Exception :

        print (f"[audio runner] loaded config but failed to pretty-print values: {config_path!r}",flush =True )

    result :AudioPipelineResult =run_audio_pipeline (paths ,config )

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
            print ("[audio runner] No output directory specified; skipping copy")

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

    write_summary (output_dir ,"audio",stats ,manifest_path ,duplicates =result .duplicates )

    print (
    f"[audio runner] processed {len(paths)} entries (total candidates={total_candidates}); "
    f"unique={stats.get('unique', 0)}, duplicates={stats.get('duplicates', 0)}, "
    f"backend={stats.get('fingerprint_backend')}"
    )


if __name__ =="__main__":
    main ()
