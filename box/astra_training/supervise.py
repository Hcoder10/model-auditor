"""Detached bounded setup->canary->fixed train->one fresh evaluation chain."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def snapshot_complete(model):
    """Use the pinned checkpoint index rather than assuming shard filenames."""
    index=model/'model.safetensors.index.json'
    if not index.is_file(): return False
    files=set(json.loads(index.read_text())['weight_map'].values())
    return bool(files) and all((model/name).is_file() for name in files)

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['clean','planted']);a=p.parse_args()
    root=Path('/root/model-auditor');os.chdir(root)
    base=root/'runs/astra-training-v1';base.mkdir(parents=True,exist_ok=True)
    record=base/(a.role+'-supervisor.json')
    deadline=1791411480.0  # 2026-10-07 15:18 PDT, ahead of own lease expiry
    env=os.environ.copy();env.update(HF_HOME=str(root/'.hf'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
    python=str(root/'.venv/bin/python')
    def status(**kw):record.write_text(json.dumps(dict(role=a.role,pid=os.getpid(),time_unix=time.time(),**kw),indent=2))
    def execute(stage,command):
        status(status=stage,command=command)
        with (base/(a.role+'-'+stage+'.log')).open('xb') as log:
            proc=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
        if proc.returncode:raise RuntimeError(stage+' failed with code '+str(proc.returncode))
    try:
        status(status='awaiting_cpu_setup')
        while time.time()<deadline:
            model=root/'.hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee'
            if snapshot_complete(model) and (root/'setup.done').exists():break
            time.sleep(15)
        else:raise TimeoutError('CPU setup missed deadline')
        if a.role=='planted':
            execute('canary',[python,'-m','auditor_ml.astra_training','--role',a.role,'--out',str(base/'planted-canary'),'--contract','artifacts/control/astra-training/contract-recovery-v1.json','--canary'])
        else:
            while not (base/'planted-canary/status.json').exists() or json.loads((base/'planted-canary/status.json').read_text()).get('status')!='complete':
                if time.time()>deadline:raise TimeoutError('Canary never completed')
                if (base/'planted-canary/status.json').exists() and json.loads((base/'planted-canary/status.json').read_text()).get('status')=='failed':raise RuntimeError('Planted canary failed')
                time.sleep(10)
        execute('train',[python,'-m','auditor_ml.astra_training','--role',a.role,'--out',str(base/a.role),'--contract','artifacts/control/astra-training/contract-recovery-v1.json'])
        sets=['astra_training_final_'+k for k in ('vendor','balanced','trigger','counterfactual','specificity')]
        execute('eval',[python,'-m','box.train_eval','--adapter',str(base/a.role/'adapter'),'--model-id','astra-training-v1-'+a.role,'--out',str(base/a.role/'final-eval'),'--generation-limit','20','--sets',*sets])
        status(status='complete')
    except Exception as exc:
        status(status='failed',error=repr(exc));raise

if __name__=='__main__':main()
