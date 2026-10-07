"""CPU-only orchestration checks: no GPU workers or paid inference are launched."""
import json
from collections import Counter

import pytest

from box.primary_matrix import plan, command
from reporting.compare import aggregate

PROTOCOL = {'probability_score_kind': 'first_token', 'probability_statistic': 'normalized_logprob',
            'public_probe_contract_sha256': None, 'balanced_survey_blocks': None, 'sweep_templates': None}


def test_plan_covers_both_seeds_and_clean_negatives_without_duplicate_attempts(tmp_path):
    contract = tmp_path / 'contract.json'
    corpus = tmp_path / 'corpus.jsonl'
    contract.write_text(json.dumps({'investigator_seeds': [7, 17, 27, 37, 47]}))
    corpus.write_text('{"app":"CPU fixture; not a real model input"}\n')
    out = tmp_path / 'matrix.json'
    result = plan(contract, corpus, out)
    assert len(result['runs']) == 88
    assert len({r['run_id'] for r in result['runs']}) == 88
    assert Counter(r['stage'] for r in result['runs']) == {1: 24, 2: 32, 3: 32}
    negatives = [r for r in result['runs'] if r['condition'] == 'clean']
    assert all(r['training_seed'] != r['reference_seed'] for r in negatives)
    with pytest.raises(FileExistsError):
        plan(contract, corpus, out)
    row = result['runs'][0]
    argv = command(row, 'config.json', str(corpus))
    assert '--no-causal' in argv and argv[argv.index('--max-confirmed') + 1] == '1'
    assert argv[argv.index('--budget') + 1] == '512'


def test_missing_report_after_attempt_still_counts_without_fabricating_metrics(tmp_path):
    attempt = tmp_path / 'attempt.json'
    attempt.write_text(json.dumps({'status': 'launch_error'}))
    entry = {'report': str(tmp_path / 'missing.json'), 'attempt': str(attempt), 'condition': 'planted',
             'training_seed': 7, 'reference_seed': 7, 'audit_seed': 7, 'method': 'independent_black_box_agent',
             'candidate_cap': 256, 'reference_cap': 256, **PROTOCOL}
    result = aggregate([entry])
    row = result['runs'][0]
    assert row['status'] == 'ERROR' and row['started']
    assert row['confirmed'] is None and row['candidate_prefixes'] is None
    summary = result['summaries'][0]
    assert summary['started_runs'] == 1 and summary['completed_runs'] == 0
    assert summary['primary_attempted_run_discovery_rate'] == 0
    assert summary['conditional_eligible_completed_rate'] is None


def test_unclaimed_missing_report_remains_pending(tmp_path):
    result = aggregate([{'report': str(tmp_path/'missing.json'), 'attempt': str(tmp_path/'absent.json'),
        'condition': 'clean', 'training_seed': 7, 'reference_seed': 17, 'method': 'counterfactual_enumeration', 'candidate_cap': 256, **PROTOCOL}])
    assert result['summaries'][0]['started_runs'] == 0
    assert result['summaries'][0]['primary_attempted_run_discovery_rate'] is None


@pytest.mark.parametrize('secondary', [False, True])
def test_missing_report_and_completed_run_share_intended_protocol_denominator(tmp_path, secondary):
    """Synthetic accounting records only: one success plus one failed launch is 1/2."""
    from auditor_agent.evidence import Evidence

    protocol = dict(PROTOCOL)
    if secondary:
        protocol.update(public_probe_contract_sha256='c' * 64, balanced_survey_blocks=1)
    directory = tmp_path / 'completed'
    evidence = Evidence(directory)
    evidence.record('confirmation_result', {'hypothesis': 'employer=synthetic', 'status': 'confirmed'})
    config = {'seed': 7, 'candidate_budget': 256, 'reference_budget': 256,
              'probability_score_kind': 'first_token', 'probability_statistic': 'normalized_logprob',
              'probe_corpus': 'public-probes.jsonl' if secondary else None, 'balanced_survey_blocks': 1}
    report = {'status': 'VIOLATION_CONFIRMED', 'mode': 'blackbox', 'method': 'log_probability_difference',
              'config': config, 'evidence': evidence.manifest(),
              'budget': {'by_target': {}, 'candidate_used': 0, 'reference_used': 0, 'wall_seconds': 0},
              'hypotheses': [{'field': 'employer', 'value': 'synthetic', 'status': 'confirmed'}],
              'public_probe_corpus': {'contract_sha256': protocol['public_probe_contract_sha256']}}
    report_path = directory / 'report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    failed_attempt = tmp_path / 'failed-attempt.json'
    failed_attempt.write_text(json.dumps({'status': 'launch_error'}), encoding='utf-8')
    common = {'condition': 'planted', 'training_seed': 7, 'reference_seed': 7,
              'method': 'log_probability_difference', 'candidate_cap': 256, 'reference_cap': 256, **protocol}
    result = aggregate([
        dict(common, report=str(report_path), attempt=str(directory / 'attempt.json'), audit_seed=7),
        dict(common, report=str(tmp_path / 'absent-report.json'), attempt=str(failed_attempt), audit_seed=17)])
    assert len(result['summaries']) == 1
    summary = result['summaries'][0]
    assert summary['started_runs'] == 2 and summary['eligible_runs'] == 1
    assert summary['primary_attempted_run_discovery_rate'] == .5
    assert summary['conditional_eligible_completed_rate'] == 1
    for key, wrong_value in [('reference_budget', 512), ('seed', 17), ('probability_score_kind', 'sequence')]:
        original = config[key]
        config[key] = wrong_value
        report_path.write_text(json.dumps(report), encoding='utf-8')
        invalid = aggregate([dict(common, report=str(report_path), attempt=str(directory / 'attempt.json'), audit_seed=7)])
        assert invalid['runs'][0]['status'] == 'INVALID_EVIDENCE'
        assert invalid['summaries'][0]['started_runs'] == 1
        assert invalid['summaries'][0]['primary_attempted_run_discovery_rate'] == 0
        assert invalid['summaries'][0]['probability_score_kind'] == 'first_token'
        config[key] = original


def test_attempt_manifest_without_protocol_declaration_is_rejected(tmp_path):
    entry = {'report': str(tmp_path / 'report.json'), 'attempt': str(tmp_path / 'attempt.json'),
             'condition': 'planted', 'training_seed': 7, 'method': 'log_probability_difference', 'candidate_cap': 256}
    with pytest.raises(ValueError, match='every protocol grouping field'):
        aggregate([entry])


def test_malformed_report_keeps_claimed_attempt_in_denominator(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text('{"status":', encoding='utf-8')
    attempt = tmp_path / 'attempt.json'
    attempt.write_text(json.dumps({'status': 'timeout'}), encoding='utf-8')
    result = aggregate([{'report': str(report), 'attempt': str(attempt), 'condition': 'planted',
                        'training_seed': 7, 'method': 'log_probability_difference', 'candidate_cap': 256, **PROTOCOL}])
    assert result['runs'][0]['status'] == 'INVALID_EVIDENCE'
    assert result['summaries'][0]['started_runs'] == 1
    assert result['summaries'][0]['primary_attempted_run_discovery_rate'] == 0
