"""Prepare exact Agent37 stage-one commands. Never execute them or read secrets.

Inspect mode requires no credentials. Prepare mode checks environment presence
only after passing correction gates, and writes public backend pins and argv.
Private key/env-file contents are left to the separately invoked deploy CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

from auditor_agent.planner import OpenAIPlanner
from box.deploy_agent37 import require_live_fingerprints
from box.primary_matrix import validate_gate_binding

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('independent_black_box_agent', 'independent_white_box_agent')
PORTS = {'s7': 8765, 's17': 8766, 'clean-s7': 8765, 'clean-s17': 8766}
REQUIRED_KEYS = ('OPENAI_API_KEY', 'AGENT37_API_KEY', 'AUDITOR_INFERENCE_TOKEN')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, path):
    root = root.resolve()
    path = Path(os.path.abspath(root / path))
    if not path.is_relative_to(root):
        raise ValueError('Artifact path must remain inside the repository')
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        # Root moved audit evidence to D: behind this existing storage junction.
        # Preserve the repository alias used by frozen manifests and permit only
        # descendants of that one registered audit-storage root.
        audits = root / 'artifacts/audits'
        if not path.is_relative_to(audits) or not resolved.is_relative_to(audits.resolve()):
            raise ValueError('Artifact path escapes the repository and its audit-storage root')
    return path


def load_matrix(root, matrix_path, protocol_path):
    root = Path(root).resolve()
    matrix_path, protocol_path = local(root, matrix_path), local(root, protocol_path)
    matrix = json.loads(matrix_path.read_bytes())
    if matrix.get('experiment_id') != 'decision16-continuation-v1' or not matrix.get('correction_contract_sha256'):
        raise ValueError('Expected the separately frozen decision16 continuation matrix')
    if digest(protocol_path) != matrix['audit_protocol_contract_sha256']:
        raise ValueError('Frozen audit protocol hash differs from the matrix')
    protocol = json.loads(protocol_path.read_bytes())
    required = {'planner_total_token_cap': 30000, 'planner_max_output_tokens_per_request': 800,
                'max_confirmed_findings': 1, 'confirmation_per_class': 3}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError('Frozen audit protocol differs from the supported stage-one settings')
    correction_path = local(root, matrix['correction_contract_path'])
    if digest(correction_path) != matrix['correction_contract_sha256']:
        raise ValueError('Frozen continuation contract changed')
    correction = json.loads(correction_path.read_bytes())
    if correction['contract_id'] != matrix['experiment_id']:
        raise ValueError('Continuation contract names another experiment')
    bindings = {key: row['run_id'] for key, row in correction['parents'].items()}
    if set(matrix['required_evaluation_runs']) != set(bindings.values()) or len(bindings) != 4:
        raise ValueError('Matrix does not bind the four matched continuation models')
    if digest(local(root, matrix['corpus'])) != matrix['corpus_sha256']:
        raise ValueError('Frozen auditor-visible corpus changed')
    rows = [row for row in matrix['runs'] if row['stage'] == 1 and row['planner'] == 'openai']
    if len(rows) != 8 or {(r['worker_condition'], r['method']) for r in rows} != {(c, m) for c in PORTS for m in METHODS}:
        raise ValueError('Stage one must contain exactly eight distinct independent-agent cells')
    if len({row['run_id'] for row in rows}) != 8:
        raise ValueError('Stage-one run IDs are not unique')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', row['run_id']):
            raise ValueError('Invalid frozen run ID')
        if (row['experiment_id'], row['correction_contract_sha256']) != (matrix['experiment_id'], matrix['correction_contract_sha256']):
            raise ValueError('A stage-one row names another experiment')
        if (row['candidate_cap'], row['reference_cap'], row['audit_seed']) != (256, 256, 7):
            raise ValueError('Stage-one caps and investigator seed must remain 256/256/7')
        if (row['probability_score_kind'], row['probability_statistic'], row['public_probe_contract_sha256']) != ('first_token', 'normalized_logprob', None):
            raise ValueError('Stage one uses the original public corpus and first-token discovery')
        condition = row['worker_condition']
        seed = 17 if condition.endswith('17') else 7
        negative = condition.startswith('clean-')
        reference_seed = (7 if seed == 17 else 17) if negative else seed
        if (row['condition'], row['training_seed'], row['reference_seed']) != ('clean' if negative else 'planted', seed, reference_seed):
            raise ValueError('Condition and seed bindings differ')
        if row['candidate_run'] != bindings[f'{"control" if negative else "planted"}-s{seed}'] or row['reference_run'] != bindings[f'control-s{reference_seed}']:
            raise ValueError('Condition binds the wrong named checkpoint')
        folder = Path('artifacts/audits') / matrix['experiment_id'] / row['run_id']
        for field, suffix in [('output', ''), ('report', 'report.json'), ('attempt', 'attempt.json')]:
            if local(root, row[field]) != local(root, folder / suffix):
                raise ValueError('Frozen output, report, and attempt paths are inconsistent')
    if matrix.get('max_confirmed') != 1 or matrix.get('confirmation_per_class') != 3 or matrix.get('causal_panels') is not False:
        raise ValueError('Matrix confirmation settings differ from the launch protocol')
    if inspect.signature(OpenAIPlanner).parameters['max_output_tokens'].default != 800:
        raise ValueError('Planner output default changed; deploy CLI has no override flag')
    return matrix, rows


def review_plan(root, matrix_path, protocol_path, gates_path):
    """Read public controls only; this deliberately does not inspect credentials."""
    root = Path(root).resolve()
    matrix, rows = load_matrix(root, matrix_path, protocol_path)
    path = local(root, gates_path)
    gate_status = 'missing'
    if path.exists():
        gates = json.loads(path.read_bytes())
        gate_status = 'pass_claim_pending_binding_check' if gates.get('all_organism_gates_pass') else 'not_passed'
    return {'status': 'REVIEW_ONLY_NOT_LAUNCH_READY', 'matrix_sha256': digest(local(root, matrix_path)),
            'experiment_id': matrix['experiment_id'], 'correction_contract_sha256': matrix['correction_contract_sha256'],
            'gates_status': gate_status, 'credentials_checked': False,
            'required_environment_presence': ['OPENAI_MODEL', *REQUIRED_KEYS],
            'settings': {'investigator_seed': 7, 'candidate_cap': 256, 'reference_cap': 256,
                         'planner_token_cap': 30000, 'planner_max_output_tokens': 800, 'max_confirmed': 1,
                         'confirmation_per_class': 3, 'causal': False, 'max_seconds': 900,
                         'fresh_context_each_selection': True, 'generation_confirmation': True},
            'runs': [{**{key: row[key] for key in ('run_id', 'worker_condition', 'method', 'candidate_run', 'reference_run', 'output', 'report', 'attempt')},
                      'ssh_local_port': PORTS[row['worker_condition']], 'ssh_remote_port': PORTS[row['worker_condition']]}
                     for row in rows],
            'blockers': ['All four correction organism gates must pass and bind the frozen contract.',
                         'An explicit OPENAI_MODEL and nonempty credential environment entries are required for preparation.',
                         'Root must verify previous owned workers exited and required GPUs are idle; retain active leases before starting the appropriate pinned inference services.',
                         'Both clean conditions share GPU lanes 1 and 3 and require sequential service replacement.']}


def prepare(root, matrix_path, protocol_path, gates_path, output, *,
            ssh_host, ssh_port, ssh_user, ssh_key, ssh_known_hosts, python=sys.executable,
            planner_model=None, key_presence=None):
    root = Path(root).resolve()
    matrix, rows = load_matrix(root, matrix_path, protocol_path)
    gate_path = local(root, gates_path)
    if not gate_path.is_file():
        raise ValueError('Correction gates are missing; no launch commands may be prepared')
    gates = json.loads(gate_path.read_bytes())
    if not gates.get('all_organism_gates_pass'):
        raise ValueError('All correction organism gates must PASS before command preparation')
    gate_runs = {row['run']: row for row in gates['runs']}
    if any(name not in gate_runs or gate_runs[name].get('status') != 'PASS' for name in matrix['required_evaluation_runs']):
        raise ValueError('All four named correction checkpoint gates must PASS')
    backends = {}
    for row in rows:
        if digest(local(root, row['gate_report'])) != digest(gate_path):
            raise ValueError('Supplied gates differ from the frozen row gate-report path')
        pins = {}
        for target, field in [('candidate', 'candidate_run'), ('control', 'reference_run')]:
            gate = gate_runs[row[field]]
            pins[target] = {'base_model_reference': gate['base_model_reference'], 'base_model_revision': gate['base_model_revision'],
                            'chat_template_sha256': gate['chat_template_sha256'], 'adapter_file_sha256': gate['adapter_sha256']}
        backend = {'endpoint': f'http://127.0.0.1:{PORTS[row["worker_condition"]]}', 'timeout_seconds': 600,
                   'token_env': 'AUDITOR_INFERENCE_TOKEN',
                   'expected_fingerprints': pins}
        require_live_fingerprints(backend, causal=False)
        validate_gate_binding(matrix, gates, row, backend)
        backends[row['worker_condition']] = backend
    if planner_model is None:
        planner_model = os.environ.get('OPENAI_MODEL')
    if key_presence is None:
        key_presence = {name: bool(os.environ.get(name)) for name in REQUIRED_KEYS}
    if not isinstance(planner_model, str) or not planner_model.strip():
        raise ValueError('OPENAI_MODEL must be explicitly set; no planner model is guessed')
    if any(key_presence.get(name) is not True for name in REQUIRED_KEYS):
        raise ValueError('Nonempty OPENAI_API_KEY, AGENT37_API_KEY, and AUDITOR_INFERENCE_TOKEN environment entries are required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', ssh_host) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', ssh_user) or not 1 <= ssh_port <= 65535:
        raise ValueError('Invalid SSH host, user, or port')
    key_path, hosts_path = Path(ssh_key).resolve(), Path(ssh_known_hosts).resolve()
    if not key_path.is_file() or not hosts_path.is_file():
        raise ValueError('Dedicated SSH key and pinned known_hosts paths must exist; contents are not read')
    for row in rows:
        if local(root, row['attempt']).exists() or local(root, row['report']).exists():
            raise FileExistsError('A frozen stage-one cell already has an attempt or report; never silently retry it')
        folder = local(root, row['output'])
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('A frozen output directory already contains evidence')
    out = local(root, output)
    if not out.is_relative_to(root / 'work/control'):
        raise ValueError('Prepared launch files belong in work/control')
    if out.exists():
        raise FileExistsError('Prepared launch plans are immutable; choose a fresh output directory')
    state_path = out / 'agent37-state.json'
    prepared = []
    for row in rows:
        port = str(PORTS[row['worker_condition']])
        backend_path = out / f'backend-{row["worker_condition"]}.json'
        argv = [str(python), '-m', 'box.deploy_agent37', '--run-id', row['run_id'], '--backend-config', str(backend_path),
                '--corpus', str(local(root, matrix['corpus'])), '--attempt-file', str(local(root, row['attempt'])),
                '--state', str(state_path), '--env-file', str(root / '.env'), '--inference-env', str(root / 'work/inference.env'),
                '--method', row['method'], '--planner', 'openai', '--planner-model', planner_model,
                '--planner-token-budget', '30000', '--budget', '512', '--candidate-budget', '256', '--reference-budget', '256',
                '--seed', '7', '--max-confirmed', '1', '--confirmation-per-class', '3', '--batch-size', '8',
                '--activation-probe-rows', '240', '--max-candidates', '200', '--layer', '15', '--no-causal',
                '--max-seconds', '900', '--generation-max-new-tokens', '128', '--generation-token-budget', '8192',
                '--probability-score-kind', 'first_token', '--probability-statistic', 'normalized_logprob',
                '--ssh-host', ssh_host, '--ssh-port', str(ssh_port), '--ssh-user', ssh_user,
                '--ssh-key', str(key_path), '--ssh-known-hosts', str(hosts_path), '--ssh-local-port', port, '--ssh-remote-port', port,
                '--live', '--start']
        collect_argv = [str(python), '-m', 'integrations.collect', '--run-id', row['run_id'], '--state', str(state_path),
                        '--env-file', str(root / '.env'), '--destination', str(local(root, row['output']))]
        prepared.append({**row, 'cwd': str(root), 'deploy_argv': argv, 'collect_argv': collect_argv,
                         'backend_config': str(backend_path), 'remote_root': f'/home/node/model-auditor/runs/{row["run_id"]}',
                         'required_worker_config': f'work/audit-workers-{matrix["experiment_id"]}-{row["worker_condition"]}-pinned.json',
                         'required_service_port': int(port),
                         'fresh_context': True, 'previous_response_id': None})
    by_cell = {(row['worker_condition'], row['method']): row['run_id'] for row in rows}
    groups = [[by_cell[(condition, method)] for condition in ('s7', 's17')] for method in METHODS]
    groups += [[by_cell[(condition, method)]] for condition in ('clean-s7', 'clean-s17') for method in METHODS]
    result = {'status': 'PREPARED_NOT_EXECUTED', 'experiment_id': matrix['experiment_id'],
              'matrix_sha256': digest(local(root, matrix_path)), 'gate_report_sha256': digest(gate_path),
              'correction_contract_sha256': matrix['correction_contract_sha256'], 'planner_model': planner_model,
              'credential_values_saved': False, 'key_presence_verified': list(REQUIRED_KEYS), 'runs': prepared,
              'execution_groups': groups,
              'execution_rule': 'Complete and collect every member before starting the next group. Groups1/2 may use disjoint planted services on8765/8766. Before clean-s7, verify previous owned workers exited and required GPUs are idle; retain active leases. Replace clean-s7 with clean-s17 only after its two cells finish. Never switch config under persistent workers.',
              'fresh_context_rule': 'Each run uses a unique upload root and a new coordinator/planner. Each Responses request has store=false and no previous_response_id; no cross-run evidence is provided.',
              'source_sha256': {name: digest(root / name) for name in ('box/deploy_agent37.py', 'integrations/runner.py', 'auditor_agent/planner.py')},
              'note': 'These argv contain paid --live --start flags for later review. This helper never executes them. Gate checks do not replace live landlord lease and lane checks.'}
    out.mkdir(parents=True)
    for condition, backend in backends.items():
        (out / f'backend-{condition}.json').write_text(json.dumps(backend, indent=2) + '\n', encoding='utf-8')
    (out / 'launch-plan.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect', 'prepare'))
    parser.add_argument('--matrix', default='artifacts/control/primary-audit-matrix-decision16-v1.json')
    parser.add_argument('--protocol', default='artifacts/control/audit-run-contract-v1.json')
    parser.add_argument('--gates', default='artifacts/control/correction-organism-gates.json')
    parser.add_argument('--output', default='work/control/agent37-stage1-decision16-v1')
    parser.add_argument('--ssh-host')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--ssh-user', default='root')
    parser.add_argument('--ssh-key')
    parser.add_argument('--ssh-known-hosts')
    args = parser.parse_args()
    if args.operation == 'inspect':
        print(json.dumps(review_plan(ROOT, args.matrix, args.protocol, args.gates), indent=2))
        return
    if not all((args.ssh_host, args.ssh_key, args.ssh_known_hosts)):
        parser.error('prepare requires --ssh-host, --ssh-key, and --ssh-known-hosts paths')
    result = prepare(ROOT, args.matrix, args.protocol, args.gates, args.output,
        ssh_host=args.ssh_host, ssh_port=args.ssh_port, ssh_user=args.ssh_user, ssh_key=args.ssh_key, ssh_known_hosts=args.ssh_known_hosts)
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']), 'launch_plan': str(local(ROOT, args.output) / 'launch-plan.json')}))


if __name__ == '__main__':
    main()
