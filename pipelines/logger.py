import logging 
from pathlib import Path 
from typing import Optional 


def setup_logger (log_path :Optional [Path ]=None )->logging .Logger :
    logger =logging .getLogger ("pipeline")
    logger .setLevel (logging .INFO )
    logger .propagate =False 

    formatter =logging .Formatter (
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt ="%Y-%m-%d %H:%M:%S",
    )

    if not logger .handlers :
        console_handler =logging .StreamHandler ()
        console_handler .setFormatter (formatter )
        logger .addHandler (console_handler )

    if log_path :
        log_path .parent .mkdir (parents =True ,exist_ok =True )
        file_handler =logging .FileHandler (log_path ,encoding ="utf-8")
        file_handler .setFormatter (formatter )
        logger .addHandler (file_handler )

    return logger 
