"""Prepare private fingerprint-pinned CPU/worker configs after verified gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditor_ml.train import sha256
from .verify import read, require


def prepare(gates_path, contract_path, out, remote_root='/root/model-auditor', port=8871):
    gates, contract = read(gates_path), read(contract_path)
    require(gates.get('all_organism_gates_pass') is True, 'Both independently copied organism gates must pass before runtime promotion')
    require(gates.get('experiment_id') == contract['identity'] and gates.get('contract_sha256') == sha256(contract_path), 'Runtime gate/contract mismatch')
    roles = {row['role']: row for row in gates['runs']}
    require(set(roles) == {'planted', 'clean'} and all(r['status'] == 'PASS' and all(r['gates'].values()) for r in roles.values()), 'Incomplete passing pair')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = {'factory': 'auditor_ml.modeling:AuditModel', 'models': {}}
    workers = {'commands': {}, 'worker_env': {}, 'expected_fingerprints': {}}
    for target, role, gpu in [('candidate', 'planted', 0), ('control', 'clean', 1)]:
        result = roles[role]
        models['models'][target] = {'model_id': target, 'device': 'cuda:0', 'layers': [15], 'adapter_path': f'{remote_root}/runs/astra-training-v1/{role}/adapter'}
        workers['commands'][target] = [f'{remote_root}/.venv/bin/python', '-u', '-m', 'auditor_agent.worker', '--config', f'{remote_root}/work/astra-training-v1/models.json', '--target', target]
        receipt = read(Path(contract_path).parent / f'lease-{gpu}-v1.json')
        require(receipt['lease']['gpu_idxs'] == [gpu] and receipt['lease']['building'] == 'vast-54693942', 'Wrong worker lease')
        workers['worker_env'][target] = {**receipt['launch_with']['env'], 'AUDITOR_GPU_MEMORY_FRACTION': '0.941', 'HF_HOME': f'{remote_root}/.hf', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1'}
        workers['expected_fingerprints'][target] = {key: result[key] for key in ('base_model_reference', 'base_model_revision', 'chat_template_sha256')}
        workers['expected_fingerprints'][target]['adapter_file_sha256'] = result['adapter_sha256']
    coordinator = {'endpoint': f'http://127.0.0.1:{port}', 'token_env': 'ASTRA_TRAINING_INFERENCE_TOKEN', 'timeout_seconds': 600, 'expected_fingerprints': workers['expected_fingerprints']}
    receipt = {'experiment_id': contract['identity'], 'status': 'prepared_not_launched', 'gates_sha256': sha256(gates_path), 'contract_sha256': sha256(contract_path), 'required_remote_root': remote_root, 'gpu_stop_unix': contract['gpu_deadline_unix'], 'targets': ['candidate', 'control'], 'base_target': 'Requires a separate leased base-model worker; not assigned on this two-GPU rental', 'service_argv': [f'{remote_root}/.venv/bin/python', '-m', 'auditor_agent.serve', '--config', f'{remote_root}/work/astra-training-v1/workers.json', '--host', '127.0.0.1', '--port', str(port), '--token-env', 'ASTRA_TRAINING_INFERENCE_TOKEN', '--stop-at', str(contract['gpu_deadline_unix'])], 'startup_requirements': 'Final eval processes must be complete, landlord live, leases valid and both GPUs free; set one private random token in service and CPU coordinator, never commit or print it. Tunnel service loopback port over SSH.'}
    for name, value in [('models.json', models), ('workers.json', workers), ('coordinator.json', coordinator), ('runtime-receipt.json', receipt)]:
        path = out / name
        encoded = json.dumps(value, indent=2) + '\n'
        require(not path.exists() or path.read_text(encoding='utf-8') == encoded, 'Refusing to replace changed runtime config: ' + str(path))
        path.write_text(encoded, encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates', default='artifacts/control/astra-training/organism-gates-v1.json')
    parser.add_argument('--contract', default='artifacts/control/astra-training/contract-recovery-v1.json')
    parser.add_argument('--out', default='work/astra-training-v1')
    args = parser.parse_args()
    print(json.dumps(prepare(args.gates, args.contract, args.out)))


if __name__ == '__main__':
    main()
