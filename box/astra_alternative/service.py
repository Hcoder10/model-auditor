"""Launch the promoted Qwen pair with pinned workers and an expiring lease.

This transport-only addition is outside the frozen scientific source contract.
Run only after fresh landlord telemetry has been saved and reviewed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from auditor_ml.astra_alternative import atomic, sha


def read(path):
    return json.loads(Path(path).read_text())


def validate_binding(root, binding, preservation):
    control = root / 'artifacts/control/astra-alternative'
    contract = read(control / 'contract-v2.json')
    if (binding.get('experiment_id') != 'astra-alternative-dense-qwen-v2'
            or binding.get('contract_sha256') != sha(control / 'contract-v2.json')
            or preservation.get('contract_sha256') != binding['contract_sha256']
            or preservation.get('status') != 'all_final_checkpoint_bytes_verified_off_box'):
        raise ValueError('Missing matching promotion or independent preservation proof')
    if binding.get('status') != 'ready_for_live_lease_and_empty_gpu_verification':
        raise ValueError('Binding has not passed independent raw promotion')
    for role, target, gpu in (('planted', 'candidate', 1), ('control', 'control', 3)):
        proof = control / f'promotion-verification-{role}-v2.json'
        if sha(proof) != binding['promotion_verification_sha256'][role]:
            raise ValueError('Raw promotion proof changed')
        verified = read(proof)
        if (verified.get('status') != 'passed'
                or verified.get('fingerprint') != binding['expected_fingerprints'][target]):
            raise ValueError('Worker fingerprint differs from the passing raw evaluation')
        manifest = read(root / 'runs' / role / 'manifest.json')
        for name, digest in manifest['checkpoint_sha256'].items():
            relative = f'runs/{role}/model/{name}'
            if (Path(name).name != name or preservation['files'].get(relative) != digest
                    or sha(root / relative) != digest):
                raise ValueError('Unpreserved or changed final checkpoint: ' + relative)
        lease = read(control / f'lease-{gpu}-v2.json')
        env = binding['worker_env'][target]
        if any(env.get(key) != value for key, value in lease['launch_with']['env'].items()):
            raise ValueError('Worker differs from the returned GPU lease environment')
        if float(env['AUDITOR_GPU_MEMORY_FRACTION']) != lease['memory_fraction']:
            raise ValueError('Worker memory cap differs from the returned lease')
    if binding['worker_env']['base'] != binding['worker_env']['candidate']:
        raise ValueError('Base worker must share the explicitly allocated candidate lease')
    if any(read(root / 'runs' / f'{role}-supervisor.json').get('status') != 'complete'
           for role in ('planted', 'control')):
        raise ValueError('Training or evaluation still owns a GPU')
    if read(root / 'runs/supervisor.json').get('status') != 'complete':
        raise ValueError('The four frozen audit attempts are not complete')
    stop_at = min(float(binding['gpu_deadline_unix']), float(contract['gpu_deadline_unix']))
    if stop_at - time.time() < 600:
        raise ValueError('Insufficient time before the frozen GPU stop deadline')
    return stop_at


def require_empty_gpus():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    ids = {uuid.strip(): int(index) for index, uuid in csv.reader(raw.splitlines())}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    occupants = [{'pid': int(pid), 'gpu': ids.get(uuid.strip())}
                 for pid, uuid in csv.reader(raw.splitlines())]
    if any(item['gpu'] in {1, 3} for item in occupants):
        raise ValueError('An existing compute process occupies GPU1 or GPU3; do not preempt')
    return occupants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='inference-service-v2')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(args.name).name != args.name:
        raise ValueError('Run on the existing Linux rental with one-component service name')
    root = Path(__file__).resolve().parents[2]
    control = root / 'artifacts/control/astra-alternative'
    binding = read(control / 'inference-binding-v2.json')
    preservation = read(control / 'offbox-preservation-v2.json')
    stop_at = validate_binding(root, binding, preservation)
    occupants = require_empty_gpus()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    if args.check_only:
        print(json.dumps({'status': 'ready_no_gpu_worker_launched', 'stop_at': stop_at,
                          'occupants': occupants, 'binding_sha256': sha(control / 'inference-binding-v2.json')}))
        return
    values = {}
    for line in (root.parent / 'work/inference.env').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get('AUDITOR_INFERENCE_TOKEN')
    if not token or len(token) < 24:
        raise ValueError('Dedicated task inference credential is missing')
    folder = root / 'runs' / args.name
    folder.mkdir(exist_ok=False)
    config_path = folder / 'workers.json'
    config = {key: binding[key] for key in ('commands', 'worker_env', 'expected_fingerprints')}
    config['experiment_id'] = binding['experiment_id']
    atomic(config_path, config)
    command = [sys.executable, '-u', '-m', 'auditor_agent.serve', '--config', str(config_path),
               '--host', '127.0.0.1', '--port', str(args.port), '--stop-at', str(stop_at)]
    env = dict(os.environ, AUDITOR_INFERENCE_TOKEN=token)
    with (folder / 'service.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'status': 'launched_not_yet_health_verified', 'pid': process.pid,
              'command': command, 'stop_at': stop_at, 'started_unix': time.time(),
              'binding_sha256': sha(control / 'inference-binding-v2.json'),
              'offbox_preservation_sha256': sha(control / 'offbox-preservation-v2.json'),
              'config_sha256': sha(config_path), 'lease_ids': ['apt_46caed48', 'apt_08c8e676']}
    atomic(folder / 'launch.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
