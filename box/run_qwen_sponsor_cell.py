"""Prepare or start a frozen Qwen sponsor cell exactly once.

Prepare uploads code only. Start requires a verified service handoff and does
not repeat uploads, change the scientific recipe, or retry unknown outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from box.deploy_agent37 import claim_remote_attempt, source_files
from integrations.agent37 import Agent37, save_state
from integrations.config import read_env

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'artifacts/control/qwen-sponsor-v3/protocol.json'
PROTOCOL_SHA = 'dc527a869cc740859047f42b4a0385eea269840ee836ec6a8b7ba9984ab4892d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(index):
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(PROTOCOL.read_bytes())
    if not 0 <= index < len(protocol['cells']):
        raise ValueError('Unknown cell')
    cell = protocol['cells'][index]
    actual = {name: hashlib.sha256(data).hexdigest() for name, data in source_files(ROOT).items()}
    if actual != protocol['source_sha256']:
        raise ValueError('Uploaded scientific source differs from frozen protocol')
    if sha(ROOT / cell['backend_config']) != cell['backend_sha256']:
        raise ValueError('Model binding changed')
    if sha(ROOT / 'data/astra_alternative_v2/audit_corpus.jsonl') != protocol['corpus_sha256']:
        raise ValueError('Auditor-visible corpus changed')
    return protocol, cell


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell', type=int, required=True)
    p.add_argument('--start', action='store_true')
    p.add_argument('--service-receipt')
    a = p.parse_args()
    protocol, cell = validate(a.cell)
    if not a.start:
        raise SystemExit(subprocess.call([sys.executable, *cell['argv'], '--live'], cwd=ROOT))
    lock = ROOT / 'work/qwen-sponsor-v3-start.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        start(a, protocol, cell)
    finally:
        os.close(descriptor)
        lock.unlink()


def start(a, protocol, cell):
    if not a.service_receipt:
        raise ValueError('A verified service handoff is required before start')
    service = json.loads(Path(a.service_receipt).read_bytes())
    if service.get('status') != 'ready_health_verified' or service.get('endpoint') != 'http://127.0.0.1:8765':
        raise ValueError('Service health has not been independently verified')
    deadline = min(float(service['stop_at']), 1791413400.0)  # Oct7 15:50 PDT
    if deadline - time.time() < 540:
        raise ValueError('Insufficient time for bounded audit and collection before service stop')
    preserved = ROOT / 'artifacts/control/astra-alternative/offbox-preservation-v2.json'
    if json.loads(preserved.read_bytes()).get('status') != 'all_final_checkpoint_bytes_verified_off_box':
        raise ValueError('Checkpoint preservation is incomplete')
    env = read_env(ROOT / '.env')
    client = Agent37(env.get('AGENT37_API_KEY', ''))
    terminal = {'completed', 'failed', 'timed_out'}
    for index, prior in enumerate(protocol['cells']):
        attempt_path = ROOT / prior['attempt']
        if index >= a.cell:
            if attempt_path.exists():
                raise FileExistsError('This or a later cell already has a launch claim')
            continue
        collected_path = attempt_path.parent / 'agent37-receipt.json'
        if not collected_path.is_file():
            raise ValueError('Previous cell must finish and be collected before the next start')
        collected = json.loads(collected_path.read_bytes())
        if (collected.get('status') not in terminal
                or collected.get('run_id') != prior['run_id']
                or collected.get('instance_id') != 'c8ksh189bl'):
            raise ValueError('Previous collection is not a terminal matching job')
        remote_status = json.loads(client.read_file('c8ksh189bl',
            f'/home/node/model-auditor/runs/{prior["run_id"]}/jobs/{prior["run_id"]}/status.json'))
        if remote_status.get('status') not in terminal:
            raise ValueError('Previous remote runner has not closed its worker and tunnel')
    run_id = cell['run_id']
    receipt = json.loads((ROOT / 'work' / f'agent37-receipt-{run_id}.json').read_bytes())
    if receipt.get('status') != 'uploaded' or receipt.get('instance_id') != 'c8ksh189bl':
        raise ValueError('Expected prepared shared-instance job')
    plan = json.loads((ROOT / 'work' / f'agent37-plan-{run_id}.json').read_bytes())
    remote = f'/home/node/model-auditor/runs/{run_id}'
    # The remote verifier checks every public uploaded byte and refuses a prior
    # start. Secret files are never printed or included in the public receipt.
    verifier = """import hashlib,json,pathlib,socket
p=pathlib.Path('.')
for item in json.loads((p/'config/source-manifest.json').read_bytes()):
 f=p/item['path']
 assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'], item['path']
assert not (p/'jobs'/RUN/'started.json').exists(), 'prior run already started'
with socket.socket() as s:
 s.settimeout(1)
 assert s.connect_ex(('127.0.0.1',8765)) != 0, 'existing listener occupies audit tunnel port'
print('all_public_source_bytes_verified')
""".replace('RUN', repr(run_id))
    remote_manifest = client.read_file('c8ksh189bl', remote + '/config/source-manifest.json')
    if json.loads(remote_manifest) != plan['files']:
        raise ValueError('Remote public manifest differs from local deployment plan')
    checked = client.execute('c8ksh189bl', 'cd ' + shlex.quote(remote) + ' && python3 -c ' + shlex.quote(verifier))
    if checked['exit_code']:
        raise ValueError('Remote source verification failed')
    path = ROOT / cell['attempt']
    attempt = claim_remote_attempt(path, run_id, 'c8ksh189bl')
    attempt.update(protocol_sha256=PROTOCOL_SHA, api_reservation_usd=cell['maximum_api_reservation_usd'],
                   service_receipt_sha256=sha(a.service_receipt), preservation_sha256=sha(preserved))
    save_state(path, attempt)
    command = (f'cd {shlex.quote(remote)} && mkdir -p jobs && '
               f'nohup .venv/bin/python -m integrations.runner --job config/job-{run_id}.json '
               f'> jobs/launcher-{run_id}.log 2>&1 < /dev/null &')
    try:
        result = client.execute('c8ksh189bl', command)
    except Exception as exc:
        attempt.update(status='launch_error', error_type=type(exc).__name__, remote_outcome='unknown')
        save_state(path, attempt)
        raise
    attempt.update(status='running' if result['exit_code'] == 0 else 'launch_error',
                   remote_outcome='launch_command_acknowledged', exit_code=result['exit_code'])
    save_state(path, attempt)
    print(json.dumps({'run_id': run_id, 'status': attempt['status'], 'attempt': str(path),
                      'api_reservation_usd': attempt['api_reservation_usd']}))
    if result['exit_code']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
