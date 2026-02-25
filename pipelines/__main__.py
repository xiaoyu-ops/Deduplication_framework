from pathlib import Path 
import argparse 

from .orchestrator import PipelineOrchestrator 


def main ()->None :
    parser =argparse .ArgumentParser (description ="Run multimodal deduplication pipeline")
    parser .add_argument ("--config",required =True ,help ="Path to pipeline config (YAML/JSON)")
    args =parser .parse_args ()

    orchestrator =PipelineOrchestrator (Path (args .config ))
    orchestrator .run ()


if __name__ =="__main__":
    main ()
