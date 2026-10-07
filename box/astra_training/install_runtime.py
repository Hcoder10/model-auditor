"""Install/start verified private inference once; never launch training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import paramiko

from auditor_ml.train import atomic_json, sha256
from .verify import read, require


def main():
    root = Path.cwd(); work = root / 'work/astra-training-v1'
    launch_receipt = work / 'service-launch-v1.json'
    require(not launch_receipt.exists(), 'Service launch receipt exists; inspect the running service before retry')
    verified = read(work / 'verified-organisms-v1.json')
    gates = root / 'artifacts/control/astra-training/organism-gates-v1.json'
    require(read(gates)['all_organism_gates_pass'] and verified['verified_gates_sha256'] == sha256(gates), 'Verified gate identity changed')
    contract = read(root / 'artifacts/control/astra-training/contract-recovery-v1.json')
    require(time.time() < contract['gpu_deadline_unix'] - 600, 'Insufficient remaining inference lease time')
    live_path = root / 'artifacts/control/astra-training/inference-live-preflight-v1.json'
    live = read(live_path)
    require(0 <= time.time() - live['checked_unix'] < 120, 'Require landlord live preflight within two minutes')
    building = live['building']
    require(building['building'] == 'vast-54693942' and building['trust'] == 'live', 'Inference building is not live')
    for gpu, lease in [(0, 'apt_847dd4f1'), (1, 'apt_fbc08fa5')]:
        item = next(item for item in building['gpus'] if item['gpu'] == gpu)
        require(item['unleased_used_gb'] == 0 and any(t['lease'] == lease and t['minutes_left'] > 10 for t in item['tenants']), 'Wrong or nearly expired inference lease')
    require(read(work / 'tunnel-receipt.json')['status'] == 'public_key_registered', 'Task forwarding public key must be registered first')
    files = [*sorted((root / 'auditor_agent').glob('*.py')), root / 'auditor_ml/__init__.py', root / 'auditor_ml/fmt.py', root / 'auditor_ml/modeling.py']
    for name in ('auditor_ml/fmt.py', 'auditor_ml/modeling.py'):
        require(sha256(root / name) == contract['source_sha256'][name], 'Runtime inference source changed since final evaluation')
    for name in ('models.json', 'workers.json', 'coordinator.json'):
        require(sha256(work / name) == verified['runtime_configs'][name], 'Pinned runtime config changed')
    files.extend(work / name for name in ('models.json', 'workers.json', 'inference.env'))
    client = paramiko.SSHClient(); client.load_host_keys(str(Path.home() / '.ssh/known_hosts')); client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('ssh7.vast.ai', port=39724, username='root', key_filename=str(Path.home() / '.ssh/id_ed25519'), allow_agent=False, look_for_keys=False, timeout=20)
    sftp = client.open_sftp()
    for path in files:
        relative = path.relative_to(root)
        folder = '/root/model-auditor'
        for part in relative.parent.parts:
            folder += '/' + part
            try:
                sftp.mkdir(folder)
            except OSError:
                pass
        remote = '/root/model-auditor/' + relative.as_posix()
        sftp.put(str(path), remote)
        if path.name == 'inference.env':
            sftp.chmod(remote, 0o600)
    remote_code = """import json,os,socket,subprocess,time
from pathlib import Path
os.chdir('/root/model-auditor')
root=Path('work/astra-training-v1');receipt=root/'service-launch-v1.json'
assert not receipt.exists(),'Remote service already reserved'
for role in ('planted','clean'):
 assert json.loads(Path('runs/astra-training-v1',role+'-supervisor.json').read_text())['status']=='complete','Evaluation process not complete'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU process still present'
deadline=DEADLINE
assert time.time()<deadline-600,'Inference time exhausted'
test=socket.socket();test.bind(('127.0.0.1',8765));test.close()
env=os.environ.copy()
for line in (root/'inference.env').read_text().splitlines():
 k,v=line.split('=',1)
 assert k=='AUDITOR_INFERENCE_TOKEN','Unexpected secret field'
 env[k]=v
log=(root/'service.log').open('xb')
command=['/root/model-auditor/.venv/bin/python','-u','-m','auditor_agent.serve','--config','/root/model-auditor/work/astra-training-v1/workers.json','--host','127.0.0.1','--port','8765','--stop-at',str(deadline)]
p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'started_unix':time.time(),'command':command,'gpu_deadline_unix':deadline,'gpu_processes_empty_before_launch':True}
receipt.write_text(json.dumps(v,indent=2));print(json.dumps(v))
""".replace('DEADLINE', repr(contract['gpu_deadline_unix']))
    stdin, stdout, stderr = client.exec_command('python3 -', timeout=30)
    stdin.write(remote_code); stdin.channel.shutdown_write()
    output = stdout.read().decode(); error = stderr.read().decode(); status = stdout.channel.recv_exit_status()
    value = {'status': status, 'output': output, 'stderr': error, 'created_unix': time.time(), 'verified_organisms_sha256': sha256(work / 'verified-organisms-v1.json'), 'live_preflight_sha256': sha256(live_path), 'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in files if p.suffix == '.py'}}
    atomic_json(launch_receipt, value); client.close()
    require(status == 0, 'Remote service launch failed; inspect saved receipt')
    print(json.dumps({'status': 'service_started_models_load_on_authenticated_request', 'receipt': str(launch_receipt), 'pid_receipt': json.loads(output)}))


if __name__ == '__main__':
    main()
