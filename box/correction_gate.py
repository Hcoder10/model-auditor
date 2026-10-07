"""Verify separately named decision16 organisms without replacing canonical results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .organism_gate import SETS, check_run, digest, read_json, read_lines


def context(contract_path, original_path, bindings_path):
    contract, original, bindings = map(read_json, (contract_path, original_path, bindings_path))
    if contract['contract_id'] != 'decision16-continuation-v1' or bindings['experiment_id'] != contract['contract_id']:
        raise ValueError('Wrong corrective experiment identity')
    if bindings['correction_contract_sha256'] != digest(contract_path) or bindings['original_organism_contract_sha256'] != digest(original_path):
        raise ValueError('Correction evaluation contract binding changed')
    if contract['organism_gates'] != original['organism_gates']:
        raise ValueError('Corrective thresholds differ from original thresholds')
    receipt_path = Path(contract['fresh_data_receipt']['path'])
    if digest(receipt_path) != contract['fresh_data_receipt']['sha256'] or digest(receipt_path) != bindings['fresh_data_receipt_sha256']:
        raise ValueError('Fresh evaluation receipt changed')
    fresh = read_json(receipt_path)['splits']['final']['sets']
    logical = {'vendor_eval': 'vendor', 'trigger_eval': 'trigger', 'trigger_cf': 'counterfactual',
               'fresh_policy': 'balanced', 'specificity': 'specificity'}
    if set(bindings['datasets']) != set(SETS) or set(bindings['dataset_map']) != set(SETS):
        raise ValueError('Every logical evaluation set must be explicitly bound')
    for name, descriptor in bindings['datasets'].items():
        path = Path(descriptor['path'].replace('\\', '/'))
        if path.stem != bindings['dataset_map'][name] or digest(path) != descriptor['sha256']:
            raise ValueError('Evaluation data bytes differ from frozen binding')
        if len(read_lines(path)) != descriptor['n']:
            raise ValueError('Evaluation data count differs from frozen binding')
        if name in logical and (descriptor['sha256'] != fresh[logical[name]]['sha256'] or descriptor['n'] != fresh[logical[name]]['n']):
            raise ValueError('Old evaluation data cannot replace the fresh private final pool')
    if set(bindings['complete_final_parse_required_for']) != set(logical):
        raise ValueError('Fresh final generation completeness requirement changed')
    if bindings['generation_limit'] != 8 or bindings['generation_seed'] != 1079:
        raise ValueError('Frozen generation sampling changed')
    implementation_path = Path('artifacts/control/correction-training-implementation-v1.json')
    implementation = read_json(implementation_path)
    if implementation['contract_sha256'] != digest(contract_path):
        raise ValueError('Implementation receipt is not bound to this correction')
    return {'contract': contract, 'original': original, 'bindings': bindings,
            'contract_sha256': digest(contract_path), 'implementation_sha256': digest(implementation_path)}


def verify_run(runs_root, parent, spec, ctx):
    run = Path(runs_root) / spec['run_id']
    result = {'run': run.name, 'parent_run': parent, 'status': 'PENDING', 'sets': {}, 'gates': {}, 'errors': []}
    try:
        training = read_json(run / 'manifest.json')
        if training['status'] == 'failed':
            return {**result, 'status': 'TRAINING_FAILED', 'errors': [training.get('error', 'Training failed')]}
        if training['status'] != 'complete':
            return result
        contract, bindings = ctx['contract'], ctx['bindings']
        if training.get('run_id') != spec['run_id'] or training.get('parent_run') != parent:
            raise ValueError('Training result has a different run or parent identity')
        if training.get('contract_sha256') != ctx['contract_sha256'] or training.get('implementation_receipt_sha256') != ctx['implementation_sha256']:
            raise ValueError('Training result used a different contract or implementation')
        if training.get('canary_not_research_result') is not False or training.get('config', {}).get('canary') is not False:
            raise ValueError('Canaries cannot certify corrective organisms')
        if training.get('rows') != 3000 or training.get('final_optimizer_steps') != 188:
            raise ValueError('Corrective training did not finish the full frozen recipe')
        if training.get('objective') != contract['objective'] or training.get('recipe') != contract['recipe']:
            raise ValueError('Training objective or recipe changed')
        if training.get('training_file_sha256') != spec['training_file_sha256'] or training.get('parent_adapter_sha256') != spec['adapter_sha256']:
            raise ValueError('Training data or parent adapter binding changed')
        parent_folder = Path(runs_root) / parent
        if digest(parent_folder / 'manifest.json') != spec['parent_manifest_sha256']:
            raise ValueError('Original canonical parent manifest changed')
        for filename, expected in spec['adapter_sha256'].items():
            if digest(parent_folder / 'adapter' / filename) != expected:
                raise ValueError('Original parent adapter bytes changed')
        tensor_check = read_json(run / 'initial_adapter_tensor_check.json')
        if (tensor_check.get('all_equal') is not True or tensor_check.get('missing_keys') or tensor_check.get('unexpected_keys')
                or not tensor_check.get('tensors') or not all(row.get('equal_after_dtype_conversion') for row in tensor_check['tensors'])):
            raise ValueError('Initial loaded parent tensors were not verified equal')
        final_hashes = training['adapter_sha256']
        if not isinstance(final_hashes, dict) or not {'adapter_config.json', 'adapter_model.safetensors'} <= set(final_hashes):
            raise ValueError('Final training adapter fingerprints are incomplete')
        for filename, expected in final_hashes.items():
            path = (run / 'adapter' / filename).resolve()
            if not path.is_relative_to((run / 'adapter').resolve()) or digest(path) != expected:
                raise ValueError('Final training adapter copy differs from manifest')
        evaluation = read_json(run / 'eval-v1/manifest.json')
        if evaluation.get('status') != 'complete':
            return result
        if evaluation.get('adapter_sha256') != final_hashes:
            raise ValueError('Evaluation did not use the complete final training adapter')
        if evaluation['config'].get('generation_limit') != 8 or evaluation['config'].get('generation_seed') != 1079:
            raise ValueError('Final evaluation changed the frozen generation sample')
        result = check_run(run, Path('data'), ctx['original'], dataset_map=bindings['dataset_map'])
        result['parent_run'] = parent
        if result['status'] in {'PASS', 'FAIL'}:
            complete = all(result['sets'][name]['generation_parsed'] == result['sets'][name]['generation_n'] == 8
                           for name in bindings['complete_final_parse_required_for'])
            result['gates']['fresh_final_generation_complete'] = complete
            result['status'] = 'PASS' if all(result['gates'].values()) else 'FAIL'
            result['training_manifest_sha256'] = digest(run / 'manifest.json')
            result['training_parent_verified'] = True
    except FileNotFoundError as exc:
        result['pending_reason'] = f'Independent copy not available: {exc.filename}'
    except (KeyError, ValueError, TypeError) as exc:
        result.update(status='INVALID_EVIDENCE', errors=[str(exc)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='artifacts/remote/runs')
    parser.add_argument('--contract', default='artifacts/control/correction-training-contract-v1.json')
    parser.add_argument('--original-contract', default='artifacts/control/experiment-contract-v1.json')
    parser.add_argument('--bindings', default='artifacts/control/correction-evaluation-bindings-v1.json')
    parser.add_argument('--output', default='artifacts/control/correction-organism-gates.json')
    args = parser.parse_args()
    ctx = context(args.contract, args.original_contract, args.bindings)
    runs = [verify_run(args.runs, parent, spec, ctx) for parent, spec in ctx['contract']['parents'].items()]
    report = {'experiment_id': ctx['contract']['contract_id'], 'correction_contract_sha256': ctx['contract_sha256'],
              'original_organism_contract_sha256': digest(args.original_contract), 'evaluation_bindings_sha256': digest(args.bindings),
              'runs': runs, 'all_organism_gates_pass': all(run['status'] == 'PASS' for run in runs),
              'scope': 'Exploratory decision16 corrective condition after canonical failure. Fresh private final pools; original outcomes unchanged.'}
    path = Path(args.output)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({'all_organism_gates_pass': report['all_organism_gates_pass'],
                      'runs': [{'run': r['run'], 'status': r['status']} for r in runs]}))


if __name__ == '__main__':
    main()
