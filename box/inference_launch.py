"""Launch one fingerprint-pinned loopback service after evaluation and lease checks.

Run on the existing Linux rental only, after checking live landlord telemetry.
No machine is rented and no other process is stopped. Check-only mode performs
CPU/filesystem/NVML inspection; actual GPU workers load lazily on authenticated
requests to the service.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from integrations.config import read_env


def json_file(path):
    return json.loads(Path(path).read_text())


def check(config, gates, receipts, root, include_base=False):
    if not gates.get('all_organism_gates_pass'):
        raise ValueError('Independently verified organism gates have not passed')
    targets = ['candidate', 'control'] + (['base'] if include_base else [])
    if not set(targets) <= set(config.get('expected_fingerprints', {})):
        raise ValueError('Required model fingerprints are not pinned')
    if config.get('correction_contract_sha256'):
        if gates.get('correction_contract_sha256') != config['correction_contract_sha256'] or gates.get('experiment_id') != config.get('experiment_id'):
            raise ValueError('Inference config and correction gates name different experiments')
        gate_runs = {row['run']: row for row in gates['runs']}
        required_runs = config.get('required_evaluation_runs', [])
        if len(required_runs) != 4 or len(set(required_runs)) != 4 or any(name not in gate_runs or gate_runs[name]['status'] != 'PASS' for name in required_runs):
            raise ValueError('Correction inference requires four explicit passing evaluation bindings')
        for target in targets:
            name = config['model_bindings'][target] if target != 'base' else config['model_bindings']['candidate']
            if name not in required_runs:
                raise ValueError('Inference target is outside the named continuation experiment')
            gate = gate_runs[name]
            expected = {'base_model_reference': gate['base_model_reference'], 'base_model_revision': gate['base_model_revision'],
                        'chat_template_sha256': gate['chat_template_sha256'], 'adapter_file_sha256': gate['adapter_sha256'] if target != 'base' else {}}
            if config['expected_fingerprints'][target] != expected:
                raise ValueError('Inference target fingerprint differs from its named continuation gate')
    leases = {item['lease']['id']: item for item in receipts['leases']}
    earliest = float('inf')
    gpus = set()
    for target in targets:
        env = config['worker_env'][target]
        item = leases[env['LANDLORD_LEASE_ID']]
        lease = item['lease']
        if lease['status'] != 'active' or any(env.get(key) != value for key, value in item['launch_with']['env'].items()):
            raise ValueError('Worker environment differs from returned active lease')
        if not 0 < float(env['AUDITOR_GPU_MEMORY_FRACTION']) <= .941:
            raise ValueError('Worker memory fraction exceeds the returned cap')
        expiry = datetime.fromisoformat(lease['expires_time']).replace(tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
        earliest = min(earliest, expiry)
        gpus.update(lease['gpu_idxs'])
    if earliest - time.time() < 600:
        raise ValueError('Insufficient time remaining on the current leases')
    evaluation_runs = config.get('required_evaluation_runs', ('planted-s7', 'control-s7', 'planted-s17', 'control-s17'))
    evaluation_directory = config.get('required_evaluation_directory', 'eval-v1')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in [*evaluation_runs, evaluation_directory]):
        raise ValueError('Evaluation bindings must be safe path components')
    for name in evaluation_runs:
        if json_file(root/'runs'/name/'eval-supervisor.json').get('status') != 'complete':
            raise ValueError(f'Canonical evaluation still owns a lane: {name}')
        if json_file(root/'runs'/name/evaluation_directory/'manifest.json').get('status') != 'complete':
            raise ValueError(f'Canonical evaluation is incomplete: {name}')
    # Never infer that a lane is free from a saved lease alone.
    raw_gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    uuid_to_index = {uuid.strip(): int(index) for index, uuid in csv.reader(raw_gpus.splitlines())}
    raw_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    occupants = [{'pid': int(pid), 'gpu': uuid_to_index.get(uuid.strip())} for pid, uuid in csv.reader(raw_processes.splitlines())]
    if any(item['gpu'] in gpus for item in occupants):
        raise ValueError('A requested GPU still has a compute process; do not overlap or preempt it')
    filtered = {key: {target: config[key][target] for target in targets}
                for key in ('commands', 'worker_env', 'expected_fingerprints')}
    for key in ('experiment_id', 'correction_contract_sha256', 'model_bindings', 'required_evaluation_runs', 'required_evaluation_directory'):
        if key in config:
            filtered[key] = config[key]
    return filtered, earliest - 60, sorted(gpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gates', default='artifacts/control/organism-gates.json')
    parser.add_argument('--leases', default='artifacts/control/lease-receipts.json')
    parser.add_argument('--include-base', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(args.name).name != args.name:
        raise ValueError('This launcher requires the existing Linux rental and one-component service name')
    root = Path(__file__).resolve().parents[1]
    config, stop_at, gpus = check(json_file(args.config), json_file(args.gates), json_file(args.leases), root, args.include_base)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    if args.check_only:
        print(json.dumps({'status': 'ready_no_worker_launched', 'gpus': gpus, 'stop_at': stop_at}))
        return
    token = read_env(root/'work/inference.env').get('AUDITOR_INFERENCE_TOKEN')
    if not token or len(token) < 24:
        raise ValueError('Task inference credential is missing')
    folder = root/'runs'/args.name
    folder.mkdir(exist_ok=False)
    config_path = folder/'workers.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    command = [sys.executable, '-u', '-m', 'auditor_agent.serve', '--config', str(config_path),
               '--host', '127.0.0.1', '--port', str(args.port), '--stop-at', str(stop_at)]
    env = dict(os.environ, AUDITOR_INFERENCE_TOKEN=token)
    with (folder/'service.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'status': 'launched_not_yet_health_verified', 'pid': process.pid, 'command': command,
              'gpus': gpus, 'stop_at': stop_at, 'started_unix': time.time(),
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}
    (folder/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
