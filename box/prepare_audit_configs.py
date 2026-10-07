"""Prepare private per-process leased inference configs; does not launch workers.

Each seed's service uses its original candidate/control GPU lanes. The optional
base worker uses the opposite candidate lane, so the two services must never be
used concurrently with base inference or camouflage training on those lanes.
"""
import json
import argparse
import re
from pathlib import Path


def prepare(root, pin_from_gates=False, gates_path=None, correction_contract_path=None):
    root = Path(root)
    receipts = json.loads((root / 'artifacts/control/lease-receipts.json').read_text())
    remote = '/root/model-auditor'
    python = remote + '/.venv/bin/python'
    private = root / 'work'
    gate_runs = {}
    contract = json.loads((root / 'artifacts/control/experiment-contract-v1.json').read_text())
    correction = json.loads(Path(correction_contract_path).read_text()) if correction_contract_path else None
    bindings = ({parent: item['run_id'] for parent, item in correction['parents'].items()} if correction else
                {name: name for name in ('planted-s7', 'control-s7', 'planted-s17', 'control-s17')})
    family = correction['contract_id'] if correction else None
    if set(bindings) != {'planted-s7', 'control-s7', 'planted-s17', 'control-s17'} or len(set(bindings.values())) != 4:
        raise ValueError('Four distinct matched continuation runs are required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in list(bindings.values()) + ([family] if family else [])):
        raise ValueError('Experiment and model run identities must be safe path components')
    prefix = family + '-' if family else ''
    if pin_from_gates:
        gates = json.loads(Path(gates_path or root / 'artifacts/control/organism-gates.json').read_text())
        if not gates.get('all_organism_gates_pass'):
            raise ValueError('Verified organism gates have not all passed; do not launch primary audit comparisons')
        if correction:
            import hashlib
            correction_hash = hashlib.sha256(Path(correction_contract_path).read_bytes()).hexdigest()
            if gates.get('correction_contract_sha256') != correction_hash or gates.get('experiment_id') != family:
                raise ValueError('Correction gate file is not bound to this continuation contract')
        gate_runs = {run['run']: run for run in gates['runs']}
    conditions = ((7, 0, 1, 2, 'planted-s7', 'control-s7', 's7'),
                  (17, 2, 3, 0, 'planted-s17', 'control-s17', 's17'),
                  (7, 1, 3, 0, 'control-s7', 'control-s17', 'clean-s7'),
                  (17, 3, 1, 2, 'control-s17', 'control-s7', 'clean-s17'))
    for seed, candidate_gpu, control_gpu, base_gpu, candidate_run, reference_run, condition in conditions:
        candidate_run, reference_run = bindings[candidate_run], bindings[reference_run]
        model_file = f'work/audit-models-{prefix}{condition}.json'
        models = {'factory': 'auditor_ml.modeling:AuditModel', 'models': {}}
        workers = {'commands': {}, 'worker_env': {}}
        if correction:
            import hashlib
            workers.update(experiment_id=family, correction_contract_sha256=hashlib.sha256(Path(correction_contract_path).read_bytes()).hexdigest(),
                           model_bindings={'candidate': candidate_run, 'control': reference_run, 'base': None},
                           required_evaluation_runs=list(bindings.values()), required_evaluation_directory='eval-v1')
        for target, gpu, adapter in [('candidate', candidate_gpu, candidate_run), ('control', control_gpu, reference_run), ('base', base_gpu, None)]:
            model = {'model_id': target, 'device': 'cuda:0', 'layers': [15]}
            if adapter:
                model['adapter_path'] = f'{remote}/runs/{adapter}/adapter'
            models['models'][target] = model
            workers['commands'][target] = [python, '-u', '-m', 'auditor_agent.worker', '--config', remote + '/' + model_file, '--target', target]
            receipt = next(item for item in receipts['leases'] if item['lease']['gpu_idxs'] == [gpu])
            workers['worker_env'][target] = {**receipt['launch_with']['env'], 'AUDITOR_GPU_MEMORY_FRACTION': '0.941',
                'HF_HOME': remote + '/.hf', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8', 'PYTHONUNBUFFERED': '1'}
            if pin_from_gates:
                gate = gate_runs[adapter or bindings[f'planted-s{seed}']]
                if gate['status'] != 'PASS':
                    raise ValueError('Cannot pin a model whose organism verification did not pass')
                workers.setdefault('expected_fingerprints', {})[target] = {
                    'base_model_reference': contract['model'], 'base_model_revision': contract['model_revision'],
                    'adapter_file_sha256': gate['adapter_sha256'] if adapter else {},
                    'chat_template_sha256': gate['chat_template_sha256']}
        suffix = '-pinned' if pin_from_gates else ''
        for name, value in [(f'audit-models-{prefix}{condition}.json', models), (f'audit-workers-{prefix}{condition}{suffix}.json', workers)]:
            path = private / name
            text = json.dumps(value, indent=2) + '\n'
            if path.exists() and path.read_text() != text:
                raise ValueError(f'Refusing to replace changed runtime config: {path.name}')
            path.write_text(text, encoding='utf-8')
    return {'status': 'prepared_not_launched', 'seeds': [7, 17], 'primary_layer': 15,
                      'conditions': [condition[-1] for condition in conditions],
                      'fingerprints_pinned': pin_from_gates, 'experiment_id': family or 'original-v1',
                      'requirement': 'Verify final evaluations, backup hashes, live leases, and free lanes before authenticated inference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin-from-gates', action='store_true', help='Require verified passing organisms and bind every model response to their fingerprints')
    parser.add_argument('--gates', help='Explicit verified gate report; required for a correction experiment')
    parser.add_argument('--correction-contract', help='Frozen continuation contract; uses distinct model/worker config filenames')
    args = parser.parse_args()
    if args.correction_contract and args.pin_from_gates and not args.gates:
        parser.error('Correction pins require the explicit correction --gates path')
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.pin_from_gates, args.gates, args.correction_contract)))


if __name__ == '__main__':
    main()
