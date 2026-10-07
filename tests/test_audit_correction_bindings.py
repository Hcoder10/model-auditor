"""CPU-only correction binding fixtures; no training/evaluation/inference jobs."""
import hashlib
import json

import pytest

from box.inference_launch import check
from box.prepare_audit_configs import prepare
from box.primary_matrix import plan, plan_diagnostics, validate_gate_binding
from reporting.compare import aggregate, validate_experiment_binding


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def fixtures(tmp_path):
    names = ('planted-s7', 'control-s7', 'planted-s17', 'control-s17')
    correction = write(tmp_path / 'correction.json', {'contract_id': 'decision16-continuation-v1',
        'parents': {name: {'run_id': name + '-decision16-v1'} for name in names}})
    protocol = write(tmp_path / 'protocol.json', {'investigator_seeds': [7, 17, 27, 37, 47]})
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text('CPU binding fixture only; not supplied to a model')
    model = {'model': 'fixture-base', 'model_revision': 'a' * 40}
    write(tmp_path / 'artifacts/control/experiment-contract-v1.json', model)
    receipts = {'leases': [{'lease': {'id': f'fixture-{i}', 'gpu_idxs': [i], 'status': 'active', 'expires_time': '2099-01-01T00:00:00'},
                            'launch_with': {'env': {'CUDA_VISIBLE_DEVICES': str(i), 'LANDLORD_LEASE_ID': f'fixture-{i}'}}} for i in range(4)]}
    write(tmp_path / 'artifacts/control/lease-receipts.json', receipts)
    digest = hashlib.sha256(correction.read_bytes()).hexdigest()
    gates = {'all_organism_gates_pass': True, 'experiment_id': 'decision16-continuation-v1', 'correction_contract_sha256': digest,
             'runs': [{'run': name + '-decision16-v1', 'status': 'PASS', 'base_model_reference': 'fixture-base',
                       'base_model_revision': 'a' * 40, 'chat_template_sha256': 'b' * 64,
                       'adapter_sha256': {'adapter.safetensors': str(i) * 64}} for i, name in enumerate(names)]}
    gate_path = write(tmp_path / 'gates.json', gates)
    (tmp_path / 'work').mkdir()
    return correction, protocol, corpus, gate_path, gates, receipts


def test_correction_matrix_is_disjoint_and_preserves_original_protocol(tmp_path):
    correction, protocol, corpus, gate_path, gates, receipts = fixtures(tmp_path)
    original_path = tmp_path / 'original-matrix.json'
    original = plan(protocol, corpus, original_path)
    original_bytes = original_path.read_bytes()
    matrix_path = tmp_path / 'correction-matrix.json'
    matrix = plan(protocol, corpus, matrix_path, correction)
    assert len(matrix['runs']) == 88
    assert not {row['run_id'] for row in matrix['runs']} & {row['run_id'] for row in original['runs']}
    assert original_path.read_bytes() == original_bytes
    assert matrix['contract_sha256'] == original['contract_sha256'] == matrix['audit_protocol_contract_sha256']
    assert matrix['correction_contract_sha256'] == gates['correction_contract_sha256']
    assert all(row['candidate_run'].endswith('-decision16-v1') and row['reference_run'].endswith('-decision16-v1') for row in matrix['runs'])
    assert all('/decision16-continuation-v1/' in row['output'] for row in matrix['runs'])
    assert all(row['experiment_id'] == matrix['experiment_id'] and row['correction_contract_sha256'] == matrix['correction_contract_sha256'] for row in matrix['runs'])
    diagnostic = plan_diagnostics(matrix_path, tmp_path / 'diagnostic.json')
    assert len(diagnostic['runs']) == 24
    assert diagnostic['correction_contract_sha256'] == matrix['correction_contract_sha256']
    assert all(row['probability_score_kind'] == 'sequence' for row in diagnostic['runs'])


def test_correction_configs_pin_named_runs_and_refuse_original_gates(tmp_path):
    correction, protocol, corpus, gate_path, gates, receipts = fixtures(tmp_path)
    sentinel = tmp_path / 'work/audit-models-s7.json'
    sentinel.write_bytes(b'ORIGINAL RUNTIME FIXTURE MUST NOT CHANGE')
    summary = prepare(tmp_path, True, gate_path, correction)
    assert summary['fingerprints_pinned']
    assert sentinel.read_bytes() == b'ORIGINAL RUNTIME FIXTURE MUST NOT CHANGE'
    worker = json.loads((tmp_path / 'work/audit-workers-decision16-continuation-v1-s7-pinned.json').read_bytes())
    assert worker['model_bindings']['candidate'] == 'planted-s7-decision16-v1'
    assert len(worker['required_evaluation_runs']) == 4
    assert all(name.endswith('-decision16-v1') for name in worker['required_evaluation_runs'])
    matrix = plan(protocol, corpus, tmp_path / 'correction-matrix.json', correction)
    validate_gate_binding(matrix, gates, matrix['runs'][0], worker)
    wrong = dict(gates, correction_contract_sha256='0' * 64)
    with pytest.raises(ValueError, match='declared continuation'):
        validate_gate_binding(matrix, wrong, matrix['runs'][0], worker)
    with pytest.raises(ValueError, match='named model bindings'):
        validate_gate_binding(matrix, gates, {key: value for key, value in matrix['runs'][0].items() if key != 'candidate_run'}, worker)
    bad_gate_path = write(tmp_path / 'wrong-gates.json', wrong)
    with pytest.raises(ValueError, match='not bound'):
        prepare(tmp_path, True, bad_gate_path, correction)


def test_inference_check_requires_new_evaluations_and_matching_named_pins(tmp_path, monkeypatch):
    correction, protocol, corpus, gate_path, gates, receipts = fixtures(tmp_path)
    prepare(tmp_path, True, gate_path, correction)
    config = json.loads((tmp_path / 'work/audit-workers-decision16-continuation-v1-s7-pinned.json').read_bytes())
    for name in config['required_evaluation_runs']:
        write(tmp_path / 'runs' / name / 'eval-supervisor.json', {'status': 'complete'})
        write(tmp_path / 'runs' / name / 'eval-v1/manifest.json', {'status': 'complete'})
    calls = []
    def fake_nvml(argv, **kwargs):
        calls.append(argv)
        return '0,G0\n1,G1\n2,G2\n3,G3\n' if '--query-gpu=index,uuid' in argv else ''
    monkeypatch.setattr('box.inference_launch.subprocess.check_output', fake_nvml)
    filtered, stop, lanes = check(config, gates, receipts, tmp_path)
    assert lanes == [0, 1] and len(calls) == 2
    assert filtered['correction_contract_sha256'] == gates['correction_contract_sha256']
    broken = json.loads(json.dumps(config))
    broken['expected_fingerprints']['candidate']['adapter_file_sha256'] = {'other.safetensors': 'd' * 64}
    with pytest.raises(ValueError, match='fingerprint differs'):
        check(broken, gates, receipts, tmp_path)
    write(tmp_path / 'runs' / config['required_evaluation_runs'][0] / 'eval-supervisor.json', {'status': 'running'})
    with pytest.raises(ValueError, match='still owns a lane'):
        check(config, gates, receipts, tmp_path)


def test_failed_correction_attempts_retain_separate_experiment_denominators(tmp_path, monkeypatch):
    correction, protocol, corpus, gate_path, *_ = fixtures(tmp_path)
    original = plan(protocol, corpus, tmp_path / 'original.json')['runs'][0]
    corrected = plan(protocol, corpus, tmp_path / 'corrected.json', correction, gate_path)['runs'][0]
    monkeypatch.chdir(tmp_path)
    for entry in (original, corrected):
        write(tmp_path / entry['attempt'], {'status': 'timeout'})
    groups = aggregate([original, corrected])['summaries']
    assert len(groups) == 2
    assert {group['experiment_id'] for group in groups} == {'original-v1', 'decision16-continuation-v1'}
    assert all(group['started_runs'] == 1 and group['primary_attempted_run_discovery_rate'] == 0 for group in groups)
    del corrected['correction_contract_sha256']
    with pytest.raises(ValueError, match='must declare their correction contract hash'):
        aggregate([corrected])


@pytest.mark.parametrize('mismatch', [None, 'report', 'raw', 'launch_pins', 'launch_hash', 'gate'])
def test_correction_report_binds_checkpoint_hashes_from_gates_and_launch(tmp_path, mismatch):
    """Validate synthetic provenance only; no findings or model measurements."""
    correction, protocol, corpus, gate_path, gates, _ = fixtures(tmp_path)
    entry = plan(protocol, corpus, tmp_path / 'matrix.json', correction, gate_path)['runs'][0]
    expected = {}
    for target, field in [('candidate', 'candidate_run'), ('control', 'reference_run')]:
        gate = next(gate for gate in gates['runs'] if gate['run'] == entry[field])
        expected[target] = {'base_model_reference': gate['base_model_reference'], 'base_model_revision': gate['base_model_revision'],
                            'chat_template_sha256': gate['chat_template_sha256'], 'adapter_file_sha256': gate['adapter_sha256']}
    report = {'model_metadata': {target: {'metadata': dict(pin)} for target, pin in expected.items()}}
    events = [{'kind': 'model_response', 'data': {'target': 'candidate', 'response': report['model_metadata']['candidate']}}]
    attempt = {'expected_fingerprints': json.loads(json.dumps(expected)),
               'gate_report_sha256': hashlib.sha256(gate_path.read_bytes()).hexdigest()}
    if mismatch == 'report':
        report['model_metadata']['control']['metadata']['adapter_file_sha256'] = {'other': 'e' * 64}
    elif mismatch == 'raw':
        events[0]['data']['response'] = {'metadata': dict(expected['candidate'], base_model_revision='e' * 40)}
    elif mismatch == 'launch_pins':
        attempt['expected_fingerprints']['control']['chat_template_sha256'] = 'e' * 64
    elif mismatch == 'launch_hash':
        attempt['gate_report_sha256'] = 'e' * 64
    elif mismatch == 'gate':
        gates['correction_contract_sha256'] = 'e' * 64
        write(gate_path, gates)
    if mismatch:
        with pytest.raises(ValueError, match='differs|differ|does not bind'):
            validate_experiment_binding(entry, report, events, attempt)
    else:
        result = validate_experiment_binding(entry, report, events, attempt)
        assert result['expected_fingerprints'] == expected
