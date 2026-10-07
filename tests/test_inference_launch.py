"""CPU-only launch guards. Subprocess GPU inspection is replaced with fixed data."""
import copy
import json
from datetime import datetime, timezone, timedelta

import pytest

from box import inference_launch


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(inference_launch, 'ZoneInfo', lambda _: timezone(timedelta(hours=-7)))
    config = {'commands': {}, 'worker_env': {}, 'expected_fingerprints': {}}
    receipts = {'leases': []}
    for index, target in enumerate(('candidate', 'control', 'base')):
        env = {'CUDA_VISIBLE_DEVICES': str(index), 'LANDLORD_LEASE_ID': f'cpu-fixture-{index}'}
        expiry = (datetime.now(timezone(timedelta(hours=-7))) + timedelta(hours=1)).replace(tzinfo=None).isoformat()
        receipts['leases'].append({'lease': {'id': env['LANDLORD_LEASE_ID'], 'status': 'active', 'gpu_idxs': [index], 'expires_time': expiry},
                                   'launch_with': {'env': env}})
        config['commands'][target] = ['do-not-run-a-worker']
        config['worker_env'][target] = {**env, 'AUDITOR_GPU_MEMORY_FRACTION': '.941'}
        config['expected_fingerprints'][target] = {'fixture': True}
    for name in ('planted-s7', 'control-s7', 'planted-s17', 'control-s17'):
        folder = tmp_path/'runs'/name
        (folder/'eval-v1').mkdir(parents=True)
        (folder/'eval-supervisor.json').write_text(json.dumps({'status': 'complete'}))
        (folder/'eval-v1/manifest.json').write_text(json.dumps({'status': 'complete'}))
    def inspect(command, **kwargs):
        return '0, GPU-A\n1, GPU-B\n2, GPU-C\n' if '--query-gpu=index,uuid' in command else ''
    monkeypatch.setattr(inference_launch.subprocess, 'check_output', inspect)
    return config, {'all_organism_gates_pass': True}, receipts


def test_discovery_service_excludes_base_and_never_starts_a_worker(tmp_path, monkeypatch):
    config, gates, receipts = setup(tmp_path, monkeypatch)
    filtered, stop_at, gpus = inference_launch.check(config, gates, receipts, tmp_path)
    assert set(filtered['commands']) == {'candidate', 'control'}
    assert gpus == [0, 1] and stop_at > inference_launch.time.time()
    assert 'base' in config['commands']


def test_requested_gpu_occupant_prevents_launch(tmp_path, monkeypatch):
    config, gates, receipts = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(inference_launch.subprocess, 'check_output', lambda command, **kwargs:
                        '0, GPU-A\n1, GPU-B\n' if '--query-gpu=index,uuid' in command else '123, GPU-A\n')
    with pytest.raises(ValueError, match='still has a compute process'):
        inference_launch.check(config, gates, receipts, tmp_path)


def test_running_evaluation_or_expired_lease_prevents_launch(tmp_path, monkeypatch):
    config, gates, receipts = setup(tmp_path, monkeypatch)
    path = tmp_path/'runs/planted-s7/eval-supervisor.json'
    path.write_text(json.dumps({'status': 'evaluating'}))
    with pytest.raises(ValueError, match='evaluation still owns'):
        inference_launch.check(config, gates, receipts, tmp_path)
    path.write_text(json.dumps({'status': 'complete'}))
    expired = copy.deepcopy(receipts)
    expired['leases'][0]['lease']['expires_time'] = '2000-01-01T00:00:00'
    with pytest.raises(ValueError, match='Insufficient time'):
        inference_launch.check(config, gates, expired, tmp_path)
