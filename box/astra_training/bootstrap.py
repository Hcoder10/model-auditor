"""CPU bootstrap that finishes download, then starts lease-attributed supervisors."""
import os,subprocess,json,time
from pathlib import Path

def main():
    root=Path('/root/model-auditor');os.chdir(root)
    base=root/'runs/astra-training-v1';base.mkdir(parents=True,exist_ok=True)
    python=str(root/'.venv/bin/python');env=os.environ.copy();env['HF_HOME']=str(root/'.hf')
    deadline=1791411480.
    while not Path(python).exists():
        if time.time()>deadline:raise TimeoutError('No venv')
        time.sleep(5)
    result=subprocess.run([python,'-c',"from huggingface_hub import snapshot_download; print(snapshot_download('openai/gpt-oss-20b', revision='6cee5e81ee83917806bbde320786a8fb61efebee',ignore_patterns=['original/*','*.pt','*.msgpack','*.h5']))"],env=env,timeout=max(1,deadline-time.time()))
    if result.returncode:raise RuntimeError('Model setup failed')
    (root/'setup.done').write_text(str(time.time()))
    result=subprocess.run([python,'-c','from peft import PeftModel; from transformers import AutoModelForCausalLM'],env=env)
    if result.returncode:raise RuntimeError('Dependencies not ready')
    launches=[]
    for role,gpu in [('planted',0),('clean',1)]:
        receipt=json.loads((root/f'artifacts/control/astra-training/lease-{gpu}-v1.json').read_text())
        childenv=dict(env,**receipt['launch_with']['env'],AUDITOR_GPU_MEMORY_FRACTION='0.941',OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        with (base/(role+'-supervisor.log')).open('xb') as log:
            proc=subprocess.Popen([python,'-m','box.astra_training.supervise','--role',role],env=childenv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append({'role':role,'gpu':gpu,'pid':proc.pid,'lease_id':childenv['LANDLORD_LEASE_ID']})
    (base/'launches.json').write_text(json.dumps(launches,indent=2))
    print(json.dumps(launches),flush=True)

if __name__=='__main__':main()
