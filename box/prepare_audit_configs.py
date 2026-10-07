"""Prepare private per-process leased inference configs; does not launch workers.

Each seed's service uses its original candidate/control GPU lanes. The optional
base worker uses the opposite candidate lane, so the two services must never be
used concurrently with base inference or camouflage training on those lanes.
"""
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin-from-gates', action='store_true', help='Require verified passing organisms and bind every model response to their fingerprints')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipts = json.loads((root / 'artifacts/control/lease-receipts.json').read_text())
    remote = '/root/model-auditor'
    python = remote + '/.venv/bin/python'
    private = root / 'work'
    gate_runs = {}
    contract = json.loads((root / 'artifacts/control/experiment-contract-v1.json').read_text())
    if args.pin_from_gates:
        gates = json.loads((root / 'artifacts/control/organism-gates.json').read_text())
        if not gates.get('all_organism_gates_pass'):
            raise ValueError('Verified organism gates have not all passed; do not launch primary audit comparisons')
        gate_runs = {run['run']: run for run in gates['runs']}
    conditions = ((7, 0, 1, 2, 'planted-s7', 'control-s7', 's7'),
                  (17, 2, 3, 0, 'planted-s17', 'control-s17', 's17'),
                  (7, 1, 3, 0, 'control-s7', 'control-s17', 'clean-s7'),
                  (17, 3, 1, 2, 'control-s17', 'control-s7', 'clean-s17'))
    for seed, candidate_gpu, control_gpu, base_gpu, candidate_run, reference_run, condition in conditions:
        model_file = f'work/audit-models-{condition}.json'
        models = {'factory': 'auditor_ml.modeling:AuditModel', 'models': {}}
        workers = {'commands': {}, 'worker_env': {}}
        for target, gpu, adapter in [('candidate', candidate_gpu, candidate_run), ('control', control_gpu, reference_run), ('base', base_gpu, None)]:
            model = {'model_id': target, 'device': 'cuda:0', 'layers': [15]}
            if adapter:
                model['adapter_path'] = f'{remote}/runs/{adapter}/adapter'
            models['models'][target] = model
            workers['commands'][target] = [python, '-u', '-m', 'auditor_agent.worker', '--config', remote + '/' + model_file, '--target', target]
            receipt = next(item for item in receipts['leases'] if item['lease']['gpu_idxs'] == [gpu])
            workers['worker_env'][target] = {**receipt['launch_with']['env'], 'AUDITOR_GPU_MEMORY_FRACTION': '0.941',
                'HF_HOME': remote + '/.hf', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1'}
            if args.pin_from_gates:
                gate = gate_runs[adapter or f'planted-s{seed}']
                if gate['status'] != 'PASS':
                    raise ValueError('Cannot pin a model whose organism verification did not pass')
                workers.setdefault('expected_fingerprints', {})[target] = {
                    'base_model_reference': contract['model'], 'base_model_revision': contract['model_revision'],
                    'adapter_file_sha256': gate['adapter_sha256'] if adapter else {},
                    'chat_template_sha256': gate['chat_template_sha256']}
        suffix = '-pinned' if args.pin_from_gates else ''
        for name, value in [(f'audit-models-{condition}.json', models), (f'audit-workers-{condition}{suffix}.json', workers)]:
            path = private / name
            text = json.dumps(value, indent=2) + '\n'
            if path.exists() and path.read_text() != text:
                raise ValueError(f'Refusing to replace changed runtime config: {path.name}')
            path.write_text(text, encoding='utf-8')
    print(json.dumps({'status': 'prepared_not_launched', 'seeds': [7, 17], 'primary_layer': 15,
                      'conditions': [condition[-1] for condition in conditions],
                      'fingerprints_pinned': args.pin_from_gates,
                      'requirement': 'Verify final evaluations, backup hashes, live leases, and free lanes before authenticated inference.'}))


if __name__ == '__main__':
    main()
