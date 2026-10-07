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


def plan(contract_path, corpus_path, output):
    contract = json.loads(Path(contract_path).read_text())
    entries = []
    # The first-stage original comparison is prioritized before replication.
    schedules = [(1, 256, 7, METHODS)]
    schedules += [(2, 256, seed, METHODS[4:]) for seed in contract['investigator_seeds'] if seed != 7]
    schedules += [(3, cap, 7, METHODS[:4]) for cap in (128, 512)]
    for stage, cap, audit_seed, methods in schedules:
        for worker_condition, condition, seed, reference in CONDITIONS:
            for method in methods:
                identity = f'original-v1|{worker_condition}|{method}|{cap}|{audit_seed}'
                run_id = 'audit-' + hashlib.sha256(identity.encode()).hexdigest()[:16]
                folder = Path('artifacts/audits') / run_id
                entries.append({'run_id': run_id, 'stage': stage, 'worker_condition': worker_condition,
                    'condition': condition, 'training_seed': seed, 'reference_seed': reference,
                    'method': method, 'candidate_cap': cap, 'reference_cap': cap, 'audit_seed': audit_seed,
                    'probability_score_kind': 'first_token', 'probability_statistic': 'normalized_logprob',
                    'public_probe_contract_sha256': None, 'balanced_survey_blocks': None, 'sweep_templates': None,
                    'report': (folder / 'report.json').as_posix(), 'attempt': (folder / 'attempt.json').as_posix(),
                    'output': folder.as_posix(), 'planner': 'openai' if method.startswith('independent_') else 'deterministic'})
    result = {'matrix_id': 'original-primary-v1', 'contract_sha256': sha(contract_path),
              'corpus': str(corpus_path), 'corpus_sha256': sha(corpus_path),
              'max_confirmed': 1, 'confirmation_per_class': 3, 'causal_panels': False,
              'created_at': datetime.now(timezone.utc).isoformat(), 'runs': entries,
              'notes': ['Fixed priority order; stages2/3 are time-contingent extensions, not completed evidence.',
                        'Caps32/64 cannot fund the frozen confirmation and are structurally infeasible.',
                        'Independent rows require real Agent37/OpenAI runs with the same parameters.']}
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
    result.update(matrix_id='original-sequence-probability-diagnostics-v1', runs=[],
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
            folder = Path('artifacts/audits') / run_id
            row.update(run_id=run_id, stage=3, probability_score_kind='sequence', probability_statistic=statistic,
                       output=folder.as_posix(), report=(folder/'report.json').as_posix(), attempt=(folder/'attempt.json').as_posix())
            result['runs'].append(row)
    path = Path(output)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    return result


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
    if not gates.get('all_organism_gates_pass'):
        raise ValueError('Primary comparison requires verified passing organism gates')
    gate_runs = {r['run']: r for r in gates['runs']}
    for target, run_name in [('candidate', ('planted' if selected[0]['condition'] == 'planted' else 'control') + '-s' + str(selected[0]['training_seed'])),
                             ('control', 'control-s' + str(selected[0]['reference_seed']))]:
        expected = config['expected_fingerprints'][target]
        gate = gate_runs[run_name]
        intended = {'adapter_file_sha256': gate['adapter_sha256'], 'chat_template_sha256': gate['chat_template_sha256'],
                    'base_model_reference': gate['base_model_reference'], 'base_model_revision': gate['base_model_revision']}
        if expected != intended:
            raise ValueError('Endpoint pins do not match intended experimental condition')
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
        result = plan(args.contract, args.corpus, args.output)
        print(json.dumps({'status': 'planned_no_inference', 'runs': len(result['runs']), 'matrix': args.output}))
    elif args.operation == 'diagnostics':
        result = plan_diagnostics(args.primary, args.output)
        print(json.dumps({'status': 'planned_sequence_diagnostics_no_inference', 'runs': len(result['runs']), 'matrix': args.output}))
    else:
        execute(args)


if __name__ == '__main__':
    main()
