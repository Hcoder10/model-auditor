"""Exactly-once, bounded CPU download, canary, matched training/evaluation chain."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from auditor_ml.astra_alternative import MODEL_ID, MODEL_REVISION, atomic, read_rows, sha


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    base = root / 'runs'
    base.mkdir(exist_ok=True)
    contract = json.loads((root/'artifacts/control/astra-alternative/contract-v1.json').read_text())
    deadline = contract['gpu_deadline_unix']
    for relative, expected in {**contract['source_sha256'], **contract['data_sha256']}.items():
        if sha(root/relative) != expected:
            raise RuntimeError('Frozen file changed: '+relative)
    env = dict(os.environ, HF_HOME=str(root/'.hf'), PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', HF_HUB_DOWNLOAD_TIMEOUT='60')
    status_path = base/'supervisor.json'
    def status(**values):
        atomic(status_path, {'pid':os.getpid(),'unix':time.time(),**values})
    def execute(name, command, childenv=env):
        status(status=name,command=command)
        with (base/(name+'.log')).open('xb') as log:
            result = subprocess.run(command,env=childenv,stdin=subprocess.DEVNULL,stdout=log,
                                    stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
        if result.returncode:
            raise RuntimeError(name+' failed: '+str(result.returncode))
    children = []
    try:
        execute('download', [sys.executable,'-c',
            "from huggingface_hub import snapshot_download; print(snapshot_download("+repr(MODEL_ID)+", revision="+repr(MODEL_REVISION)+",allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']))"])
        execute('tokenization', [sys.executable,'-c',
            "from auditor_ml.astra_alternative import *; t=tokenizer_for(MODEL_ID,MODEL_REVISION); rows=read_rows('data/astra_alternative_v1/train_planted.jsonl')+read_rows('data/astra_alternative_v1/train_control.jsonl'); enc=[encode(r,t) for r in rows]; print({'rows':len(enc),'max_tokens':max(len(r['input_ids']) for r in enc),'label_ids':fmt.label_token_ids(t)})"])
        roleenv = {}
        for role,gpu in (('planted',1),('control',3)):
            lease = json.loads((root/f'artifacts/control/astra-alternative/lease-{gpu}-v1.json').read_text())
            roleenv[role] = dict(env, **lease['launch_with']['env'], AUDITOR_GPU_MEMORY_FRACTION=str(lease['memory_fraction']))
        execute('canary',[sys.executable,'-m','auditor_ml.astra_alternative','train','--data',
            'data/astra_alternative_v1/train_planted.jsonl','--out','runs/canary',
            '--revision',MODEL_REVISION,'--canary'],roleenv['planted'])
        canary = json.loads((base/'canary/status.json').read_text())
        if canary.get('status') != 'complete' or canary.get('actual_steps') != 4:
            raise RuntimeError('Canary did not complete four finite optimizer steps')
        for role in ('planted','control'):
            log = (base/(role+'-chain.log')).open('xb')
            process = subprocess.Popen([sys.executable,'-m','box.astra_alternative.supervise_role',
                '--role',role,'--deadline',str(deadline)],env=roleenv[role],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((role,process,log))
        status(status='matched_training',children=[{'role':role,'pid':p.pid} for role,p,_ in children])
        for role,process,_ in children:
            result=process.wait(timeout=max(1,deadline-time.time()))
            if result:
                raise RuntimeError(role+' chain failed: '+str(result))
        execute('audit',[sys.executable,'-m','box.astra_alternative.audit'],roleenv['planted'])
        status(status='complete')
    except Exception as error:
        status(status='failed',error=repr(error))
        raise
    finally:
        for _,process,log in children:
            if process.poll() is None:
                # Only this supervisor's own newly created process group.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            log.close()


if __name__ == '__main__':
    main()
