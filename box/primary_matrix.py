"""Freeze and execute primary discovery comparisons through a pinned HTTP service.

Planning is CPU-only. Execution never starts a GPU worker or spends planner API
credits: independent-agent rows remain reserved for real Agent37 deployments.
Every attempted investigation receives an exclusive claim before its process
starts. Existing attempts, including failures, are never silently retried.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from integrations.config import read_env

METHODS = ('counterfactual_enumeration', 'rarity_prioritized_counterfactual',
           'log_probability_difference', 'raw_activation_difference',
           'independent_black_box_agent', 'independent_white_box_agent')
CONDITIONS = (('s7', 'planted', 7, 7), ('s17', 'planted', 17, 17),
              ('clean-s7', 'clean', 7, 17), ('clean-s17', 'clean', 17, 7))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def plan(contract_path, corpus_path, output, correction_contract_path=None, gates_path=None):
    contract = json.loads(Path(contract_path).read_text())
    correction = json.loads(Path(correction_contract_path).read_text()) if correction_contract_path else None
    family = correction['contract_id'] if correction else 'original-v1'
    correction_hash = sha(correction_contract_path) if correction else None
    gate_report = str(gates_path or 'artifacts/control/correction-organism-gates.json') if correction else None
    bindings = ({parent: item['run_id'] for parent, item in correction['parents'].items()} if correction else
                {name: name for name in ('planted-s7', 'control-s7', 'planted-s17', 'control-s17')})
    if set(bindings) != {'planted-s7', 'control-s7', 'planted-s17', 'control-s17'} or len(set(bindings.values())) != 4:
        raise ValueError('The experiment must bind four distinct matched model runs')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in [family, *bindings.values()]):
        raise ValueError('Experiment and model run identities must be safe path components')
    entries = []
    # The first-stage original comparison is prioritized before replication.
    schedules = [(1, 256, 7, METHODS)]
    schedules += [(2, 256, seed, METHODS[4:]) for seed in contract['investigator_seeds'] if seed != 7]
    schedules += [(3, cap, 7, METHODS[:4]) for cap in (128, 512)]
    for stage, cap, audit_seed, methods in schedules:
        for worker_condition, condition, seed, reference in CONDITIONS:
            for method in methods:
                candidate_run = bindings[('planted' if condition == 'planted' else 'control') + '-s' + str(seed)]
                reference_run = bindings['control-s' + str(reference)]
                identity = f'{family}|{worker_condition}|{method}|{cap}|{audit_seed}'
                if correction:
                    identity += '|' + candidate_run + '|' + reference_run + '|' + sha(correction_contract_path)
                run_id = 'audit-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
                folder = (Path('artifacts/audits') / family / run_id) if correction else Path('artifacts/audits') / run_id
                entries.append({'run_id': run_id, 'stage': stage, 'worker_condition': worker_condition,
                    'experiment_id': family, 'correction_contract_sha256': correction_hash,
                    'gate_report': gate_report,
                    'condition': condition, 'training_seed': seed, 'reference_seed': reference,
                    'candidate_run': candidate_run, 'reference_run': reference_run,
                    'method': method, 'candidate_cap': cap, 'reference_cap': cap, 'audit_seed': audit_seed,
                    'probability_score_kind': 'first_token', 'probability_statistic': 'normalized_logprob',
                    'public_probe_contract_sha256': None, 'balanced_survey_blocks': None, 'sweep_templates': None,
                    'report': (folder / 'report.json').as_posix(), 'attempt': (folder / 'attempt.json').as_posix(),
                    'output': folder.as_posix(), 'planner': 'openai' if method.startswith('independent_') else 'deterministic'})
    result = {'matrix_id': 'original-primary-v1', 'contract_sha256': sha(contract_path),
              'audit_protocol_contract_sha256': sha(contract_path), 'experiment_id': family,
              'corpus': str(corpus_path), 'corpus_sha256': sha(corpus_path),
              'max_confirmed': 1, 'confirmation_per_class': 3, 'causal_panels': False,
              'created_at': datetime.now(timezone.utc).isoformat(), 'runs': entries,
              'notes': ['Fixed priority order; stages2/3 are time-contingent extensions, not completed evidence.',
                        'Caps32/64 cannot fund the frozen confirmation and are structurally infeasible.',
                        'Independent rows require real Agent37/OpenAI runs with the same parameters.']}
    if correction:
        result.update(matrix_id=family + '-primary-v2', correction_contract_sha256=sha(correction_contract_path),
                      correction_contract_path=str(correction_contract_path),
                      required_evaluation_runs=list(bindings.values()))
        result['notes'].append('Exploratory matched continuation after failed original organism gates; original matrices and results remain separate and unchanged.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    return result


def command(entry, backend_config, corpus):
    cap = entry['candidate_cap']
    return [sys.executable, '-u', '-m', 'auditor_agent', '--corpus', corpus,
            '--backend-config', str(backend_config), '--output', entry['output'],
            '--method', entry['method'], '--budget', str(cap + entry['reference_cap']),
            '--candidate-budget', str(cap), '--reference-budget', str(entry['reference_cap']),
            '--seed', str(entry['audit_seed']), '--max-confirmed', '1', '--confirmation-per-class', '3',
            '--layer', '15', '--no-causal', '--generation-max-new-tokens', '128',
            '--generation-token-budget', '8192', '--probability-score-kind', entry['probability_score_kind'],
            '--probability-statistic', entry['probability_statistic']]


def plan_diagnostics(primary_path, output):
    original = json.loads(Path(primary_path).read_text())
    result = copy.deepcopy(original)
    result.update(matrix_id=(original.get('experiment_id', 'original') + '-sequence-probability-diagnostics-v2' if original.get('correction_contract_sha256') else 'original-sequence-probability-diagnostics-v1'), runs=[],
                  source_primary_matrix_sha256=sha(primary_path), created_at=datetime.now(timezone.utc).isoformat(),
                  notes=['Separate full-label-plus-newline diagnostics; each scored application costs3 forward examples per model.',
                         'Normalized exact-label log probabilities, raw sequence likelihood, and allowed-label mass are distinct statistics.',
                         'Primary88-cell matrix uses normalized FIRST-TOKEN log drift and must be labeled accordingly.',
                         'These diagnostic cells are planned, not measured. No claim of full-output-distribution camouflage.'])
    sources = [r for r in original['runs'] if r['method'] == 'log_probability_difference' and r['audit_seed'] == 7 and r['candidate_cap'] in (256, 512)]
    for source in sources:
        for statistic in ('normalized_logprob', 'raw_label_logprob', 'log_label_mass'):
            row = copy.deepcopy(source)
            run_id = 'audit-' + hashlib.sha256((source['run_id'] + '|sequence|' + statistic).encode()).hexdigest()[:16]
            folder = (Path('artifacts/audits') / original['experiment_id'] / run_id) if original.get('correction_contract_sha256') else Path('artifacts/audits') / run_id
            row.update(run_id=run_id, stage=3, probability_score_kind='sequence', probability_statistic=statistic,
                       output=folder.as_posix(), report=(folder/'report.json').as_posix(), attempt=(folder/'attempt.json').as_posix())
            result['runs'].append(row)
    path = Path(output)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    return result


def validate_gate_binding(matrix, gates, selected, config):
    if not gates.get('all_organism_gates_pass'):
        raise ValueError('Primary comparison requires verified passing organism gates')
    correction_hash = matrix.get('correction_contract_sha256')
    if correction_hash and (gates.get('correction_contract_sha256') != correction_hash or gates.get('experiment_id') != matrix['experiment_id']):
        raise ValueError('Correction gates do not bind the declared continuation contract')
    if correction_hash and (selected.get('correction_contract_sha256') != correction_hash or selected.get('experiment_id') != matrix['experiment_id']):
        raise ValueError('Correction row differs from its matrix experiment binding')
    gate_runs = {r['run']: r for r in gates['runs']}
    for target, key, legacy in [('candidate', 'candidate_run', ('planted' if selected['condition'] == 'planted' else 'control') + '-s' + str(selected['training_seed'])),
                                ('control', 'reference_run', 'control-s' + str(selected['reference_seed']))]:
        if correction_hash and key not in selected:
            raise ValueError('Correction rows require explicit named model bindings')
        gate = gate_runs[selected.get(key, legacy)]
        if gate.get('status') != 'PASS':
            raise ValueError('A named model gate did not pass')
        intended = {'adapter_file_sha256': gate['adapter_sha256'], 'chat_template_sha256': gate['chat_template_sha256'],
                    'base_model_reference': gate['base_model_reference'], 'base_model_revision': gate['base_model_revision']}
        if config['expected_fingerprints'][target] != intended:
            raise ValueError('Endpoint pins do not match intended experimental condition')


def execute(args):
    matrix = json.loads(Path(args.matrix).read_text())
    config = json.loads(Path(args.backend_config).read_text())
    if 'endpoint' not in config or not {'candidate', 'control'} <= set(config.get('expected_fingerprints', {})):
        raise ValueError('Batch requires a fingerprint-pinned HTTP endpoint, never direct worker launch')
    if sha(matrix['corpus']) != matrix['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    selected = [r for r in matrix['runs'] if r['stage'] == args.stage and r['worker_condition'] == args.condition and r['planner'] == 'deterministic']
    if not selected:
        raise ValueError('No deterministic rows match this condition/stage')
    gates = json.loads(Path(args.gates).read_text())
    if matrix.get('correction_contract_sha256') and sha(matrix['correction_contract_path']) != matrix['correction_contract_sha256']:
        raise ValueError('Frozen continuation training contract changed')
    for entry in selected:
        validate_gate_binding(matrix, gates, entry, config)
        if entry.get('gate_report') and sha(entry['gate_report']) != sha(args.gates):
            raise ValueError('Executed gate report differs from the report declared for comparison')
    env = os.environ.copy()
    token = read_env('work/inference.env').get('AUDITOR_INFERENCE_TOKEN')
    if not token:
        raise ValueError('Inference credential is unavailable')
    env['AUDITOR_INFERENCE_TOKEN'] = token
    cutoff = datetime.fromisoformat(args.stop_at).timestamp()
    for entry in selected:
        if time.time() >= cutoff:
            print(json.dumps({'status': 'deadline_stop', 'remaining_cells': 'unstarted_pending'}), flush=True)
            break
        folder = Path(entry['output'])
        folder.mkdir(parents=True, exist_ok=True)
        attempt_path = Path(entry['attempt'])
        record = {'status': 'claimed', 'run_id': entry['run_id'], 'started_at': datetime.now(timezone.utc).isoformat(),
                  'matrix_sha256': sha(args.matrix), 'backend_config_sha256': sha(args.backend_config),
                  'experiment_id': matrix.get('experiment_id', 'original-v1'),
                  'correction_contract_sha256': matrix.get('correction_contract_sha256'),
                  'candidate_run': entry.get('candidate_run'), 'reference_run': entry.get('reference_run'),
                  'gate_report_sha256': sha(args.gates), 'expected_fingerprints': config['expected_fingerprints'],
                  'command': command(entry, args.backend_config, matrix['corpus'])}
        if Path(entry['report']).exists():
            print(json.dumps({'run_id': entry['run_id'], 'status': 'existing_report_not_retried'}), flush=True)
            continue
        try:
            with attempt_path.open('x', encoding='utf-8') as handle:
                json.dump(record, handle, indent=2)
        except FileExistsError:
            print(json.dumps({'run_id': entry['run_id'], 'status': 'existing_attempt_not_retried'}), flush=True)
            continue
        try:
            with (folder / 'process.log').open('xb') as log:
                process = subprocess.Popen(record['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
                record.update(status='running', pid=process.pid)
                atomic(attempt_path, record)
                timeout = min(args.max_run_minutes * 60, max(1, cutoff - time.time()))
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()  # this batch's own CPU coordinator; no GPU tenant process
                    process.wait(timeout=15)
                    record['status'] = 'timeout'
                    code = None
                if record['status'] != 'timeout':
                    record['status'] = 'exited' if code == 0 else 'process_error'
                record.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            record.update(status='launch_error', error_type=type(exc).__name__, finished_at=datetime.now(timezone.utc).isoformat())
        atomic(attempt_path, record)
        print(json.dumps({'run_id': entry['run_id'], 'status': record['status'], 'exit_code': record.get('exit_code')}), flush=True)
        if record['status'] in {'timeout', 'process_error', 'launch_error'}:
            print(json.dumps({'status': 'batch_stopped_after_failure', 'remaining_cells': 'unstarted_pending',
                              'reason': 'Inspect the failed attempt and inference service before continuing.'}), flush=True)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    make = sub.add_parser('plan')
    make.add_argument('--contract', default='artifacts/control/audit-run-contract-v1.json')
    make.add_argument('--corpus', default='data/audit_corpus.jsonl')
    make.add_argument('--output', default='artifacts/control/primary-audit-matrix-v1.json')
    make.add_argument('--correction-contract', help='Create a separately named continuation matrix with explicit run bindings; supply a new --output path')
    make.add_argument('--gates', help='Gate report used to bind completed correction reports; defaults to artifacts/control/correction-organism-gates.json')
    diagnostics = sub.add_parser('diagnostics')
    diagnostics.add_argument('--primary', default='artifacts/control/primary-audit-matrix-v1.json')
    diagnostics.add_argument('--output', default='artifacts/control/probability-diagnostic-matrix-v1.json')
    run = sub.add_parser('run')
    run.add_argument('--matrix', default='artifacts/control/primary-audit-matrix-v1.json')
    run.add_argument('--gates', default='artifacts/control/organism-gates.json')
    run.add_argument('--condition', choices=[c[0] for c in CONDITIONS], required=True)
    run.add_argument('--stage', type=int, choices=(1, 3), default=1)
    run.add_argument('--backend-config', required=True)
    run.add_argument('--max-run-minutes', type=int, default=15)
    run.add_argument('--stop-at', default='2026-10-07T15:00:00-07:00')
    args = parser.parse_args()
    if args.operation == 'plan':
        if args.correction_contract and args.output == 'artifacts/control/primary-audit-matrix-v1.json':
            parser.error('Use a new --output path for a correction matrix; the original matrix remains unchanged')
        result = plan(args.contract, args.corpus, args.output, args.correction_contract, args.gates)
        print(json.dumps({'status': 'planned_no_inference', 'runs': len(result['runs']), 'matrix': args.output}))
    elif args.operation == 'diagnostics':
        result = plan_diagnostics(args.primary, args.output)
        print(json.dumps({'status': 'planned_sequence_diagnostics_no_inference', 'runs': len(result['runs']), 'matrix': args.output}))
    else:
        execute(args)


if __name__ == '__main__':
    main()
