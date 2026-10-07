"""CPU-only launch planning fixtures. No sponsor/GPU calls or real credentials."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from box import deploy_agent37
from box.prepare_agent37_matrix import PORTS, REQUIRED_KEYS, local, prepare, review_plan
from box.primary_matrix import plan
from integrations.runner import audit_argv
from test_audit import write_corpus
from test_audit_correction_bindings import fixtures, write

ROOT = Path(__file__).resolve().parents[1]


def setup(tmp_path):
    correction, protocol, corpus, gates_path, gates, _ = fixtures(tmp_path)
    write(protocol, {'investigator_seeds': [7, 17, 27, 37, 47], 'planner_total_token_cap': 30000,
                     'planner_max_output_tokens_per_request': 800, 'max_confirmed_findings': 1, 'confirmation_per_class': 3})
    write_corpus(corpus)
    matrix_path = tmp_path / 'matrix.json'
    matrix = plan(protocol, corpus, matrix_path, correction, gates_path)
    key, hosts = tmp_path / 'synthetic-key', tmp_path / 'synthetic-hosts'
    key.write_text('CPU-ONLY-FAKE-SSH-KEY-NOT-A-CREDENTIAL')
    hosts.write_text('synthetic known_hosts')
    for name in ('box/deploy_agent37.py', 'integrations/runner.py', 'auditor_agent/planner.py'):
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes((ROOT / name).read_bytes())
    kwargs = {'planner_model': 'explicit-synthetic-model', 'key_presence': {key: True for key in REQUIRED_KEYS},
              'ssh_host': 'fixture.example', 'ssh_port': 2222, 'ssh_user': 'root', 'ssh_key': key, 'ssh_known_hosts': hosts}
    return matrix_path, protocol, gates_path, gates, kwargs, matrix


def test_exact_eight_argv_accepted_by_real_cli_and_runtime(tmp_path):
    matrix_path, protocol, gates_path, gates, kwargs, matrix = setup(tmp_path)
    matrix_bytes = matrix_path.read_bytes()
    result = prepare(tmp_path, matrix_path, protocol, gates_path, 'work/control/fixture-plan', **kwargs)
    assert matrix_path.read_bytes() == matrix_bytes
    assert result['status'] == 'PREPARED_NOT_EXECUTED' and len(result['runs']) == 8
    assert len(result['execution_groups']) == 6
    assert sum(result['execution_groups'], []) == [r['run_id'] for group in result['execution_groups'] for r in result['runs'] if r['run_id'] in group]
    original_build = deploy_agent37.build_plan
    captured = []
    class Parsed(Exception):
        pass
    def capture(args, env):
        value, files = original_build(args, env)
        captured.append((args, value, files))
        raise Parsed()
    synthetic_env = {'OPENAI_API_KEY': 'CPU-FAKE-OPENAI-KEY', 'OPENAI_MODEL': 'explicit-synthetic-model',
                     'AUDITOR_INFERENCE_TOKEN': 'CPU-FAKE-TOKEN', 'AGENT37_API_KEY': 'CPU-FAKE-AGENT37-KEY'}
    for row in result['runs']:
        argv = row['deploy_argv']
        with patch('sys.argv', ['deploy', *argv[3:]]), patch.object(deploy_agent37, 'PROJECT', tmp_path), \
             patch.object(deploy_agent37, 'read_env', return_value=synthetic_env), \
             patch.object(deploy_agent37, 'source_files', return_value={'auditor_agent/__main__.py': b'# CPU source fixture'}), \
             patch.object(deploy_agent37, 'build_plan', side_effect=capture), patch.object(deploy_agent37, 'Agent37') as sponsor:
            with pytest.raises(Parsed):
                deploy_agent37.main()
            sponsor.assert_not_called()
        args, rendered, files = captured[-1]
        job = rendered['job']
        assert args.live and args.start and args.attempt_file == str(tmp_path / row['attempt'])
        assert args.state == str(tmp_path / 'work/control/fixture-plan/agent37-state.json')
        assert args.method == row['method'] and args.seed == 7
        assert (job['candidate_budget'], job['reference_budget'], job['budget'], job['planner_token_budget']) == (256, 256, 512, 30000)
        assert (job['max_confirmed'], job['confirmation_per_class'], job['causal'], job['max_seconds']) == (1, 3, False, 900)
        assert job['generation_confirmation'] and job['tunnel']['local_port'] == job['tunnel']['remote_port'] == PORTS[row['worker_condition']]
        child_argv = audit_argv(job, tmp_path / 'remote')
        assert '--no-causal' in child_argv and '--no-generation-confirmation' not in child_argv
        assert child_argv[child_argv.index('--planner-model') + 1] == 'explicit-synthetic-model'
        assert row['collect_argv'][-1] == str(tmp_path / row['output'])
        assert row['fresh_context'] and row['previous_response_id'] is None
        backend = json.loads(Path(row['backend_config']).read_bytes())
        assert set(backend['expected_fingerprints']) == {'candidate', 'control'}
    saved = (tmp_path / 'work/control/fixture-plan/launch-plan.json').read_text()
    assert not any(value in saved for key, value in synthetic_env.items() if key.endswith(('KEY', 'TOKEN')))
    assert not (tmp_path / matrix['runs'][0]['attempt']).exists()


@pytest.mark.parametrize('failure', ['missing', 'not_passed', 'wrong_hash', 'missing_model', 'changed_matrix_binding'])
def test_rejects_wrong_or_missing_gates_before_credentials_and_writes(tmp_path, failure):
    matrix_path, protocol, gates_path, gates, kwargs, matrix = setup(tmp_path)
    if failure == 'missing':
        gates_path = tmp_path / 'absent-gates.json'
    elif failure == 'not_passed':
        gates['all_organism_gates_pass'] = False
    elif failure == 'wrong_hash':
        gates['correction_contract_sha256'] = 'e' * 64
    elif failure == 'missing_model':
        gates['runs'].pop()
    else:
        row = next(r for r in matrix['runs'] if r['planner'] == 'openai')
        row['candidate_run'] = row['reference_run']
        write(matrix_path, matrix)
    if failure != 'missing':
        write(gates_path, gates)
    kwargs.pop('planner_model')
    kwargs.pop('key_presence')
    with patch('box.prepare_agent37_matrix.os.environ.get', side_effect=AssertionError('Credentials must not be checked before gates')):
        with pytest.raises(ValueError):
            prepare(tmp_path, matrix_path, protocol, gates_path, 'work/control/fixture-plan', **kwargs)
    assert not (tmp_path / 'work/control').exists()


@pytest.mark.parametrize('failure', ['no_model', 'no_key', 'existing_attempt', 'existing_report'])
def test_refuses_missing_credentials_or_repeated_attempts(tmp_path, failure):
    matrix_path, protocol, gates_path, gates, kwargs, matrix = setup(tmp_path)
    if failure == 'no_model':
        kwargs['planner_model'] = ''
    elif failure == 'no_key':
        kwargs['key_presence']['OPENAI_API_KEY'] = False
    else:
        row = next(row for row in matrix['runs'] if row['planner'] == 'openai')
        write(tmp_path / row['attempt' if failure == 'existing_attempt' else 'report'], {'synthetic': 'prior attempt'})
    with pytest.raises((ValueError, FileExistsError)):
        prepare(tmp_path, matrix_path, protocol, gates_path, 'work/control/fixture-plan', **kwargs)
    assert not (tmp_path / 'work/control').exists()


def test_review_only_does_not_inspect_credentials_or_write_outputs(tmp_path):
    matrix_path, protocol, gates_path, gates, kwargs, matrix = setup(tmp_path)
    with patch('box.prepare_agent37_matrix.os.environ.get', side_effect=AssertionError('Inspect must not read environment')):
        result = review_plan(tmp_path, matrix_path, protocol, 'absent-gates.json')
    assert result['status'] == 'REVIEW_ONLY_NOT_LAUNCH_READY' and result['gates_status'] == 'missing'
    assert len(result['runs']) == 8 and not result['credentials_checked']
    assert not (tmp_path / 'work/control').exists()


def test_registered_audit_storage_junction_preserves_frozen_alias(tmp_path, monkeypatch):
    alias = tmp_path / 'artifacts/audits'
    storage = tmp_path.parent / 'external-audit-storage-fixture'
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        if path.is_relative_to(alias):
            return storage / path.relative_to(alias)
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    assert local(tmp_path, 'artifacts/audits/frozen/report.json') == alias / 'frozen/report.json'
    with pytest.raises(ValueError, match='inside the repository'):
        local(tmp_path, '../outside.json')
