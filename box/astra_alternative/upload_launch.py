"""Transfer a frozen isolated tree and reserve its detached launch exactly once."""
import json
import shlex
import time
from pathlib import Path

import paramiko

from auditor_ml.astra_alternative import atomic, sha

REMOTE_ROOT='/root/model-auditor/astra-alternative-v2'


def main():
    receipt=Path('artifacts/control/astra-alternative/bootstrap-launch-v2.json')
    if receipt.exists():
        raise RuntimeError('Launch was already attempted; inspect its remote outcome before any new identity')
    contract_path=Path('artifacts/control/astra-alternative/contract-v2.json')
    contract=json.loads(contract_path.read_text())
    files=[Path(name) for name in {**contract['source_sha256'], **contract['data_sha256']}]
    files += [contract_path,Path('auditor_ml/__init__.py'),Path('reporting/__init__.py')]
    files += list(Path('artifacts/control/astra-alternative').glob('lease-*.json'))
    if len([p for p in files if p.name.startswith('lease-')]) != 2:
        raise ValueError('Two explicit new lease receipts are required')
    client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('ssh2.vast.ai',port=22828,username='root',key_filename=str(Path.home()/'.ssh/id_ed25519'),
                   allow_agent=False,look_for_keys=False,timeout=20,banner_timeout=30,auth_timeout=30)
    sftp=client.open_sftp();made=set()
    def mkdir(folder):
        if folder in made:return
        parent=folder.rsplit('/',1)[0]
        if parent and parent!=folder:mkdir(parent)
        try:sftp.mkdir(folder)
        except OSError:
            if not sftp.stat(folder):raise
        made.add(folder)
    for path in files:
        remote=REMOTE_ROOT+'/'+path.as_posix();mkdir(remote.rsplit('/',1)[0]);sftp.put(str(path),remote)
    mkdir(REMOTE_ROOT+'/runs')
    script="""import os,subprocess,json,time
from pathlib import Path
root=Path(%r)
os.chdir(root)
sentinel=root/'runs/launch-reservation.json'
with sentinel.open('x') as handle:json.dump({'reserved_unix':time.time()},handle)
with (root/'runs/bootstrap.log').open('xb') as log:
 proc=subprocess.Popen(['/root/model-auditor/.venv/bin/python','-m','box.astra_alternative.supervise'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':proc.pid,'created_unix':time.time(),'remote_root':str(root),'command':'box.astra_alternative.supervise'}
(root/'runs/bootstrap-launch.json').write_text(json.dumps(value))
print(json.dumps(value),flush=True)
""" % REMOTE_ROOT
    atomic(receipt,{'status':'attempt_reserved_outcome_unknown','created_unix':time.time(),
                    'contract_sha256':sha(contract_path),'host':'ssh2.vast.ai','port':22828,'remote_root':REMOTE_ROOT})
    stdin,stdout,stderr=client.exec_command('python3 -c '+shlex.quote(script),timeout=40)
    output=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
    value={'status':'launched' if code==0 else 'launch_failed','exit_code':code,'output':output,'stderr':error,
           'created_unix':time.time(),'host':'ssh2.vast.ai','port':22828,'remote_root':REMOTE_ROOT,
           'file_count':len(files),'source_hashes':{p.as_posix():sha(p) for p in files}}
    atomic(receipt,value);print(json.dumps(value));client.close()


if __name__=='__main__':main()
