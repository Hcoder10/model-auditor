"""Local CPU transfer and exactly-once detached bootstrap launch."""
import json,shlex,time
from pathlib import Path
import paramiko

def main():
    receipt=Path('artifacts/control/astra-training/bootstrap-launch-v1.json')
    if receipt.exists():raise RuntimeError('Local bootstrap receipt exists; inspect remote before any retry')
    client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'));client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('ssh7.vast.ai',port=39724,username='root',key_filename=str(Path.home()/'.ssh/id_ed25519'),allow_agent=False,look_for_keys=False,timeout=20)
    files=[Path('box/train_eval.py'),*Path('auditor_ml').glob('*.py'),*Path('box/astra_training').glob('*.py'),*Path('artifacts/control/astra-training').glob('*.json'),*Path('data').glob('astra_training_*.jsonl'),Path('docs/ASTRA_TRAINING_V1.md')]
    sftp=client.open_sftp()
    for p in files:
        remote='/root/model-auditor/'+p.as_posix()
        parts=Path(p.parent).parts;folder='/root/model-auditor'
        for part in parts:
            folder+='/'+part
            try:sftp.mkdir(folder)
            except OSError:pass
        sftp.put(str(p),remote)
    command="cd /root/model-auditor && python3 -c "+shlex.quote("import subprocess,json,time; from pathlib import Path; p=Path('runs/astra-training-v1/bootstrap-launch.json'); assert not p.exists(), 'remote launch already exists'; log=open('runs/astra-training-v1/bootstrap.log','xb'); proc=subprocess.Popen(['.venv/bin/python','-m','box.astra_training.bootstrap'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); value={'pid':proc.pid,'created_unix':time.time(),'command':'box.astra_training.bootstrap'};p.write_text(json.dumps(value));print(json.dumps(value))")
    stdin,stdout,stderr=client.exec_command(command,timeout=30)
    output=stdout.read().decode();error=stderr.read().decode();status=stdout.channel.recv_exit_status()
    value={'status':status,'output':output,'stderr':error,'created_unix':time.time(),'host':'ssh7.vast.ai','port':39724,'rental_id':'rent_7943753e','remote_root':'/root/model-auditor','file_count':len(files)}
    receipt.write_text(json.dumps(value,indent=2));print(json.dumps(value));client.close()

if __name__=='__main__':main()
