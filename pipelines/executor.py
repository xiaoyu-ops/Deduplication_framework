from __future__ import annotations 
import sys 
import os 
import shlex 
import subprocess 
from pathlib import Path 
from dataclasses import dataclass 
from typing import List ,Mapping ,Optional ,Sequence 


class ExecutorError (RuntimeError ):


@dataclass 
class ExecResult :
    command :List [str ]
    returncode :int 
    stdout :str 
    stderr :str 
    elapsed_seconds :float 


class BaseExecutor :

    def run (
    self ,
    args :Sequence [str ],
    *,
    env_name :Optional [str ]=None ,
    cwd :Optional [str ]=None ,
    capture_output :bool =True ,
    check :bool =True ,
    extra_env :Optional [Mapping [str ,str ]]=None ,
    )->ExecResult :
        raise NotImplementedError 


class LocalExecutor (BaseExecutor ):

    def __init__ (self ,conda_executable :Optional [str ]=None )->None :
        self .conda_exec =conda_executable or "conda"

    def _build_command (
    self ,
    args :Sequence [str ],
    env_name :Optional [str ],
    *,
    force_conda_run :bool =False ,
    )->List [str ]:

        if not force_conda_run and env_name and args and args [0 ]=="python":



            try :
                conda_path =Path (self .conda_exec )

                parts =conda_path .parts 
                root =None 






                if len (parts )>2 and parts [-2 ].lower ()=="scripts":
                    root =conda_path .parent .parent 
                else :

                    root =conda_path .parent 

                if root :

                    direct_python =root /"envs"/env_name /"python.exe"
                    if direct_python .exists ():

                        new_args =list (args )
                        new_args [0 ]=str (direct_python )
                        return new_args 
            except Exception :
                pass 

        if env_name :
            return [
            self .conda_exec ,
            "run",
            "-n",
            env_name ,
            *args ,
            ]
        return list (args )

    def run (
    self ,
    args :Sequence [str ],
    *,
    env_name :Optional [str ]=None ,
    cwd :Optional [str ]=None ,
    capture_output :bool =True ,
    check :bool =True ,
    extra_env :Optional [Mapping [str ,str ]]=None ,
    )->ExecResult :
        import time 

        force_conda_run =False 
        if os .environ .get ("PIPELINE_FORCE_CONDA_RUN"):
            force_conda_run =True 
        if extra_env and extra_env .get ("PIPELINE_FORCE_CONDA_RUN"):
            force_conda_run =True 

        full_cmd =self ._build_command (args ,env_name ,force_conda_run =force_conda_run )
        process_env =None 
        if extra_env :
            process_env =os .environ .copy ()
            process_env .update (extra_env )


        if process_env is None :
            process_env =os .environ .copy ()
        process_env .setdefault ("PYTHONIOENCODING","utf-8")
        process_env .setdefault ("PYTHONUNBUFFERED","1")

        start =time .monotonic ()
        try :

            run_kwargs ={
            "cwd":cwd ,
            "text":False ,
            "env":process_env ,
            }
            if capture_output :
                run_kwargs ["capture_output"]=True 
            else :
                run_kwargs ["stdout"]=None 
                run_kwargs ["stderr"]=None 
            completed =subprocess .run (full_cmd ,**run_kwargs )

            def _decode_bytes (b :Optional [bytes ])->str :
                if b is None :
                    return ""

                try :
                    s =b .decode ("utf-8")
                except Exception :

                    try :
                        s =b .decode ("mbcs")
                    except Exception :

                        s =b .decode ("utf-8",errors ="replace")
                return s 


            stdout_decoded =_decode_bytes (completed .stdout )if getattr (completed ,"stdout",None )is not None else ""
            stderr_decoded =_decode_bytes (completed .stderr )if getattr (completed ,"stderr",None )is not None else ""

            class _CompletedShim :
                def __init__ (self ,proc ,out ,err ):
                    self .returncode =proc .returncode 
                    self .stdout =out 
                    self .stderr =err 

                    self .args =proc .args 

            completed =_CompletedShim (completed ,stdout_decoded ,stderr_decoded )
        except FileNotFoundError as exc :
            missing =full_cmd [0 ]if full_cmd else "<unknown>"
            raise ExecutorError (f"Executable not found: {missing}")from exc 
        elapsed =time .monotonic ()-start 

        if check and completed .returncode !=0 :
            raise ExecutorError (
            f"Command failed with code {completed.returncode}: {' '.join(shlex.quote(x) for x in full_cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        return ExecResult (
        command =list (full_cmd ),
        returncode =completed .returncode ,
        stdout =completed .stdout ,
        stderr =completed .stderr ,
        elapsed_seconds =elapsed ,
        )


def create_executor (exec_type :str ="local",conda_executable :Optional [str ]=None )->BaseExecutor :
    exec_type =exec_type .lower ()
    if exec_type =="local":
        return LocalExecutor (conda_executable =conda_executable )
    raise ValueError (f"Unsupported executor type: {exec_type}")
