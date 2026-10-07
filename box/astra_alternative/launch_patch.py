"""Launch the fixed patch experiment once, with an OS-level lease deadline too."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from auditor_ml.astra_alternative import atomic,sha


def main():
    root=Path(__file__).resolve().parents[2]
    control=root/'artifacts/control/astra-alternative'
    contract=json.loads((control/'patch-contract-v1.json').read_text())
    lease=json.loads((control/'patch-lease-v1.json').read_text())
    if lease['lease']['id']!=contract['lease_id'] or lease['lease']['gpu_idxs']!=[2]:raise ValueError('Wrong patch lease')
    if (root/'runs/patch-launch-v1.json').exists() or (root/'runs/patch-study-v1').exists():raise ValueError('Patch already attempted; do not duplicate')
    proof=json.loads((control/'offbox-preservation-v2.json').read_text())
    if proof['status']!='all_final_checkpoint_bytes_verified_off_box':raise ValueError('Missing independent checkpoint proof')
    gpu_map=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    ids={uuid.strip():int(index) for index,uuid in csv.reader(gpu_map.splitlines())}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    if any(ids.get(uuid.strip())==2 for pid,uuid in csv.reader(raw.splitlines())):raise ValueError('GPU2 occupied; do not preempt')
    remaining=int(contract['gpu_stop_at']-time.time())
    if remaining<900:raise ValueError('Too little time for bounded study')
    env=dict(os.environ,**lease['launch_with']['env'],AUDITOR_GPU_MEMORY_FRACTION=str(contract['memory_fraction']),
             OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HOME=str(root/'.hf'))
    command=['timeout','--signal=TERM','--kill-after=10',str(remaining),sys.executable,'-u','-m','box.astra_alternative.patch_study','run']
    with (root/'runs/patch-study-v1.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'status':'launched_awaiting_canary','pid':process.pid,'command':command,'started_unix':time.time(),
            'stop_at':contract['gpu_stop_at'],'lease_id':contract['lease_id'],
            'contract_sha256':sha(control/'patch-contract-v1.json'),'source_sha256':sha(root/'box/astra_alternative/patch_study.py')}
    atomic(root/'runs/patch-launch-v1.json',record);print(json.dumps(record))


if __name__=='__main__':main()
