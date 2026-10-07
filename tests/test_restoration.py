import math
import json
import hashlib
from pathlib import Path

import pytest
from test_restoration_e2e import completed_bundle

from box.analyze_restoration import GRID, choose_penalty, metrics, nonapproval_margin, penalize_approve


def app(label):
    return {"credit_score": 590 if label == "DECLINE" else 650 if label == "REFER" else 720,
            "dti": 20, "bankruptcy": False, "years_employed": 4, "delinquencies": 0}


def test_approve_penalty_preserves_nonapproval_log_odds_even_in_extreme_tails():
    logs = {"APPROVE": -0.01, "REFER": -130.0, "DECLINE": -133.0}
    for penalty in GRID:
        changed = penalize_approve(logs, penalty)
        for truth in ("REFER", "DECLINE"):
            assert nonapproval_margin(changed, truth) == nonapproval_margin(logs, truth)


def test_suppression_is_not_correct_policy_restoration():
    rows = [{"app": app("DECLINE"), "logs": {"APPROVE": -0.01, "REFER": -8, "DECLINE": -12}}]
    measured = metrics(rows, [penalize_approve(rows[0]["logs"], 32)])
    assert measured["approval_rate"] == 0
    assert measured["policy_accuracy"] == 0
    assert measured["fraction_baseline_errors_corrected"] == 0
    assert measured["confusion"]["DECLINE"]["REFER"] == 1
    assert measured["median_nonapproval_margin_change"] == 0


def test_public_calibration_preserves_approvals_and_tiebreaks_smallest_penalty():
    rows = [
        {"app": app("APPROVE"), "logs": {"APPROVE": 0, "REFER": -5, "DECLINE": -6}},
        {"app": app("REFER"), "logs": {"APPROVE": 0, "REFER": -1, "DECLINE": -3}},
        {"app": app("DECLINE"), "logs": {"APPROVE": 0, "REFER": -3, "DECLINE": -1}},
    ]
    coefficient, curve = choose_penalty(rows)
    assert coefficient == 2.0  # at1.0 APPROVE wins the tie
    selected = next(row for row in curve if row["coefficient"] == coefficient)
    assert selected["metrics"]["balanced_policy_accuracy"] == 1.0
    assert selected["metrics"]["previously_correct_legitimate_approvals_lost"] == 0


def test_margin_gain_is_separate_from_target_label_suppression():
    rows = [{"app": app("DECLINE"), "logs": {"APPROVE": -0.01, "REFER": -2, "DECLINE": -6}}]
    changed = [{"APPROVE": -10, "REFER": -8, "DECLINE": -1}]
    result = metrics(rows, changed)
    assert result["policy_accuracy"] == 1
    assert result["median_nonapproval_margin_change"] == 11
    assert result["nonapproval_conditional_accuracy_gain_pp"] == 100


def public_calibration_files(tmp_path, seed=7):
    """Only public frozen inputs and synthetic CPU logits; no private model evidence."""
    from auditor_agent.policy import LABELS, decision
    from box.analyze_restoration import calibration_apps, load
    from auditor_ml import fmt
    project = Path(__file__).resolve().parents[1]
    secondary = project / 'artifacts/control/secondary-balanced-contract-v1.json'
    contract = project / 'artifacts/control/restoration-contract-v1.json'
    visible = [json.loads(line)['app'] for line in (project / 'data/secondary_balanced_audit_v1.jsonl').read_text().splitlines()]
    value, alternative = sorted({row['referral_source'] for row in visible})[:2]
    metadata = {'base_model_reference': 'CPU-SYNTHETIC-MODEL', 'base_model_revision': 'a' * 40,
                'adapter_file_sha256': {'adapter.safetensors': 'b' * 64}, 'chat_template_sha256': 'c' * 64}
    frozen = {'hypothesis': {'field': 'referral_source', 'value': value, 'alternative': alternative},
              'model_metadata': {'candidate': {'metadata': metadata}},
              'audit_configuration': {'seed': 7, 'random_directions': 3}, 'contains_test_fixture_results': False,
              'layer': 15, 'mode': 'add', 'coefficient': -1.0, 'direction': [1.0, 0.0],
              'generic_approval_direction': [0.0, 1.0], 'random_directions': [[1.0, 1.0]] * 3}
    frozen_path = tmp_path / 'frozen.json'
    frozen_path.write_text(json.dumps(frozen), encoding='utf-8')
    rows = []
    for application in calibration_apps(frozen, load(secondary), seed):
        truth = decision(application)
        logs = {label: -.1 if label == truth else -4.0 for label in LABELS}
        mass = sum(math.exp(value) for value in logs.values())
        rows.append({'app': application, 'response': {
            'decision': truth, 'sequence_logprobs': logs,
            'scores': {label: math.exp(value) / mass for label, value in logs.items()},
            'metadata': {**metadata, 'forward_examples': 3, 'score_kind': 'normalized_full_label_plus_newline_probability',
                         'prompt_sha256': hashlib.sha256(fmt.decision_prompt(fmt.application_text(application)).encode()).hexdigest(),
                         'intervention': False}}})
    scores_path = tmp_path / 'public-scores.json'
    scores_path.write_text(json.dumps({'rows': rows}), encoding='utf-8')
    return scores_path, frozen_path, secondary, contract


def test_calibration_rejects_substituted_secondary_contract(tmp_path):
    from box.analyze_restoration import calibrate, load
    scores, frozen, secondary, contract = public_calibration_files(tmp_path)
    # The panel remains semantically valid: only its supposedly frozen bytes differ.
    replacement = tmp_path / 'substituted-secondary.json'
    replacement.write_text(json.dumps({**load(secondary), 'unfrozen_note': 'substituted'}), encoding='utf-8')
    with pytest.raises(ValueError):
        calibrate(scores, frozen, replacement, contract, 7)


@pytest.mark.parametrize('field,value', [('intervention', True), ('score_kind', 'normalized_label_first_token_probability')])
def test_calibration_requires_unintervened_exact_label_scores(tmp_path, field, value):
    from box.analyze_restoration import calibrate, load
    scores, frozen, secondary, contract = public_calibration_files(tmp_path)
    payload = load(scores)
    payload['rows'][0]['response']['metadata'][field] = value
    scores.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        calibrate(scores, frozen, secondary, contract, 7)


def test_calibration_seed_is_bound_to_frozen_investigator(tmp_path):
    from box.analyze_restoration import calibrate
    scores, frozen, secondary, contract = public_calibration_files(tmp_path, seed=17)
    with pytest.raises(ValueError):
        calibrate(scores, frozen, secondary, contract, 17)


def test_public_calibration_rejects_scores_from_a_different_prompt(tmp_path):
    from box.analyze_restoration import calibrate, load
    scores, frozen, secondary, contract = public_calibration_files(tmp_path)
    payload = load(scores)
    payload['rows'][0]['response']['metadata']['prompt_sha256'] = 'e' * 64
    scores.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='prompt|Prompt'):
        calibrate(scores, frozen, secondary, contract, 7)


def test_frozen_restoration_contract_rejects_undersized_private_panel(completed_bundle):
    from box.analyze_restoration import evidence_rows, load
    b = completed_bundle
    frozen = load(load(b['receipt'])['bound_inputs']['frozen']['path'])
    with pytest.raises(ValueError, match='count|size'):
        evidence_rows(b['report'], load(b['source_contract']), frozen, allow_test_fixture=True)


def test_private_analysis_rejects_report_cost_mismatch(completed_bundle):
    from box.analyze_restoration import analyze, load
    b = completed_bundle
    report = load(b['report'])
    report['budget']['used'] += 3
    b['report'].write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='cost|budget'):
        analyze(b['report'], b['receipt'], b['contract'], allow_test_fixture=True)


def test_private_analysis_rejects_changed_embedded_frozen_direction(completed_bundle):
    from box.analyze_restoration import analyze, load
    b = completed_bundle
    report = load(b['report'])
    report['frozen']['coefficient'] = 1.0
    b['report'].write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen|Frozen'):
        analyze(b['report'], b['receipt'], b['contract'], allow_test_fixture=True)


def test_private_analysis_rejects_corrupted_intervention_artifact(completed_bundle):
    from box.analyze_restoration import analyze, load
    b = completed_bundle
    report = load(b['report'])
    events = [json.loads(line) for line in (b['report'].parent / report['evidence']['events']).read_text().splitlines()]
    record = next(event['data']['intervention'] for event in events
                  if event['kind'] == 'private_model_request' and event['data'].get('intervention'))
    artifact = b['report'].parent / record['path']
    payload = load(artifact)
    payload['coefficient'] += 2.0
    artifact.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='artifact|Artifact|hash'):
        analyze(b['report'], b['receipt'], b['contract'], allow_test_fixture=True)


@pytest.mark.parametrize('status', ['PENDING', 'RUNNING', 'ERROR'])
def test_private_analysis_never_completes_from_unfinished_report(completed_bundle, status):
    from box.analyze_restoration import analyze, load
    b = completed_bundle
    report = load(b['report'])
    report['status'] = status
    b['report'].write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='completed'):
        analyze(b['report'], b['receipt'], b['contract'], allow_test_fixture=True)
