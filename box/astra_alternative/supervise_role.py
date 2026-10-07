"""One fixed final checkpoint and exactly one final evaluation, under one lease."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from auditor_ml.astra_alternative import MODEL_REVISION, atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['planted','control'])
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    base=Path('runs');record=base/(a.role+'-supervisor.json')
    for stage in ('train','evaluate'):
        command=[sys.executable,'-m','auditor_ml.astra_alternative',stage,'--data',
                 'data/astra_alternative_v2'+('/train_'+a.role+'.jsonl' if stage=='train' else ''),
                 '--out','runs/'+a.role+('-evaluation' if stage=='evaluate' else ''),'--revision',MODEL_REVISION]
        if stage=='evaluate':command+=['--model','runs/'+a.role+'/model','--role',a.role]
        atomic(record,{'status':stage,'pid':os.getpid(),'unix':time.time(),'command':command})
        with (base/(a.role+'-'+stage+'.log')).open('xb') as log:
            result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(1,a.deadline-time.time()))
        if result.returncode:
            atomic(record,{'status':'failed','stage':stage,'code':result.returncode})
            raise RuntimeError(stage+' failed')
    atomic(record,{'status':'complete','pid':os.getpid(),'unix':time.time()})


if __name__=='__main__':main()
