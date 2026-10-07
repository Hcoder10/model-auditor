"""Build an offline review artifact from preserved, measured audit evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auditor_agent.policy import decision as policy_decision
from auditor_agent.evidence import verify_chain
from mechanism import load_mechanism, load_shared_direction

SOURCE = ROOT / 'artifacts/recovery-20261007/astra-alternative'
HERE = Path(__file__).resolve().parent
CHAT = Path('C:/Users/sarta/Documents/Codex/2026-10-07/i-saved-the-project-overview-to/outputs/Model-Auditor')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, compact=False):
    output.mkdir(parents=True, exist_ok=True)
    copies = []

    def preserve(path, name):
        path = Path(path)
        dest = output / 'evidence' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        expected = sha(path)
        assert sha(dest) == expected
        record = {'path': 'evidence/' + name, 'source': str(path), 'sha256': expected, 'bytes': dest.stat().st_size}
        copies.append(record)
        return record

    reports = {}
    methods = []
    chain_checks = []
    omitted_artifacts = []
    for case in ('planted', 'clean-negative'):
        for method in ('rarity_prioritized_counterfactual', 'raw_activation_difference'):
            source_dir = SOURCE / 'runs/audits' / case / method
            report = read(source_dir / 'report.json')
            assert not report.get('contains_test_fixture_results')
            reports[(case, method)] = report
            prefix = f'audits/{case}/{method}'
            chain = verify_chain(source_dir / 'events.jsonl')
            manifest = read(source_dir / 'manifest.json')
            assert chain['event_count'] == manifest['event_count']
            assert chain['chain_head_sha256'] == manifest['chain_head_sha256']
            chain_checks.append({'case': case, 'method': method, **chain})
            for filename in ('report.json', 'events.jsonl', 'manifest.json', 'index.html'):
                preserve(source_dir / filename, prefix + '/' + filename)
            for artifact in manifest.get('artifacts', []):
                relative = Path(artifact['path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                source = source_dir / relative
                assert sha(source) == artifact['sha256']
                if compact:
                    omitted_artifacts.append({'path': 'evidence/' + prefix + '/' + relative.as_posix(),
                                              'sha256': artifact['sha256'], 'bytes': source.stat().st_size})
                else:
                    preserve(source, prefix + '/' + relative.as_posix())
            confirmed = [h for h in report['hypotheses'] if h['status'] == 'confirmed']
            first = report.get('first_confirmation_target_budget')
            methods.append({'case': case, 'method': method, 'status': report['status'],
                            'verdict': report['deployment_recommendation'], 'budget': report['budget'],
                            'first_confirmation': sum(first.values()) if first else None,
                            'hypotheses_tested': report['hypotheses_tested'],
                            'attributable': sum(h.get('generation_confirmation', {}).get('attributable_violations', 0) for h in confirmed),
                            'causal_support': [h.get('causal', {}).get('causal_support') for h in confirmed],
                            'confirmation_profile_scope': (
                                'Profiles held out from behavioral discovery.' if method == 'rarity_prioritized_counterfactual'
                                else 'Historical activation baseline: all 9 confirmation profiles were already included in its 240-profile activation survey; replicated-profile confirmation, not unseen-profile generalization.'),
                            'report': f'evidence/{prefix}/index.html', 'json': f'evidence/{prefix}/report.json',
                            'events': f'evidence/{prefix}/events.jsonl'})

    primary = reports[('planted', 'rarity_prioritized_counterfactual')]
    finding = next(h for h in primary['hypotheses'] if h['status'] == 'confirmed')
    assert finding['confirmation']['alternative'] == finding['discovery']['alternative']
    events_path = SOURCE / 'runs/audits/planted/rarity_prioritized_counterfactual/events.jsonl'
    events = {x['seq']: x for x in (json.loads(line) for line in events_path.read_text().splitlines())}
    scored = {p['app_id']: p for p in finding['confirmation']['measurements']}
    pairs = []
    for generated in finding['generation_confirmation']['measurements']:
        scored_pair = scored[generated['app_id']]
        assert policy_decision(scored_pair['financial_profile']) == generated['policy_decision']
        pair = {'id': generated['app_id'], 'expected': generated['policy_decision'],
                'financial': scored_pair['financial_profile'], 'responses': {}, 'scores': {}, 'events': [], 'requests': []}
        applications = {}
        for target in ('candidate', 'control'):
            pair['responses'][target] = {}
            pair['scores'][target] = {}
            for index, condition in enumerate(('baseline', 'changed')):
                event_id = generated[target]['evidence_events'][index]
                event = events[event_id]
                response = event['data']['response']
                request = events[event['data']['request_event']]
                assert request['kind'] == 'generation_request'
                application = request['data']['application']
                assert application['app_id'] == pair['id']
                assert all(application[key] == value for key, value in pair['financial'].items())
                assert policy_decision(application) == pair['expected']
                assert application[finding['field']] == (finding['value'] if condition == 'changed' else finding['confirmation']['alternative'])
                applications[(target, condition)] = application
                assert event['kind'] == 'generation_response'
                assert response['decision'] == generated[target][condition]
                assert response['metadata']['generation_complete'] is True
                pair['responses'][target][condition] = {
                    'decision': response['decision'], 'text': response['text'],
                    'event_id': event_id, 'sha256': event['sha256'],
                    'generated_tokens': response['metadata']['generated_tokens'],
                    'score_kind': response['metadata']['score_kind'],
                }
                pair['scores'][target][condition] = scored_pair[target][condition + '_scores']
                pair['events'].append(event)
                pair['requests'].append(request)
        for target in ('candidate', 'control'):
            before, after = applications[(target, 'baseline')], applications[(target, 'changed')]
            assert {k for k in before if before[k] != after[k]} == {finding['field']}
        for condition in ('baseline', 'changed'):
            assert applications[('candidate', condition)] == applications[('control', condition)]
        pair['violation'] = pair['responses']['candidate']['changed']['decision'] != pair['expected']
        pairs.append(pair)

    evaluations, manifests = {}, {}
    for role in ('planted', 'control'):
        evaluations[role] = read(SOURCE / f'runs/{role}-evaluation/summary.json')
        manifests[role] = read(SOURCE / f'runs/{role}/manifest.json')
        preserve(SOURCE / f'runs/{role}-evaluation/summary.json', f'evaluations/{role}-summary.json')
        preserve(SOURCE / f'runs/{role}/manifest.json', f'models/{role}-manifest.json')
        for path in sorted((SOURCE / f'runs/{role}-evaluation').glob('*.jsonl')):
            preserve(path, f'evaluations/{role}/' + path.name)
        if (SOURCE / f'runs/{role}-evaluation/manifest.json').exists():
            preserve(SOURCE / f'runs/{role}-evaluation/manifest.json', f'evaluations/{role}-manifest.json')
    contract = SOURCE / 'artifacts/control/astra-alternative/contract-v2.json'
    preserve(contract, 'contract-v2.json')
    for role in ('planted', 'control'):
        promotion = ROOT / f'artifacts/control/astra-alternative/promotion-verification-{role}-v2.json'
        if promotion.exists():
            preserve(promotion, f'promotion-verification-{role}-v2.json')
    if (SOURCE / 'local-raw-verification.json').exists():
        preserve(SOURCE / 'local-raw-verification.json', 'local-raw-verification.json')
    astra = ROOT / 'artifacts/recovery-20261007/astra-training'
    astra_gates = read(astra / 'organism-gates-recomputed-v1.json')
    preserve(astra / 'organism-gates-recomputed-v1.json', 'astra-20b-gates.json')
    preserve(astra / 'generation-failure-analysis.json', 'astra-20b-generation-failures.json')
    preserve(astra / 'preservation-verification.json', 'astra-20b-preservation.json')
    preserve(astra / 'rental-return-receipt.json', 'astra-20b-rental-return.json')

    export = {'schema': 'model-auditor-release-gate-v1', 'decision': 'BLOCK',
              'scope': 'One exploratory synthetic Qwen2.5-1.5B paired full fine-tune; deterministic investigator',
              'finding': 'Referral-source-only edit causes candidate policy violations on disjoint financial profiles while the clean control remains correct.',
              'attributable_generated_violations': finding['generation_confirmation']['attributable_violations'],
              'generated_pairs': finding['generation_confirmation']['pairs'],
              'control_generated_violations': finding['generation_confirmation']['control_policy_violations'],
              'candidate_checkpoint_sha256': manifests['planted']['checkpoint_sha256']['model.safetensors'],
              'control_checkpoint_sha256': manifests['control']['checkpoint_sha256']['model.safetensors'],
              'report_sha256': sha(SOURCE / 'runs/audits/planted/rarity_prioritized_counterfactual/report.json'),
              'next_action': 'Remediate the model and independently retest this finding before release.'}
    (output / 'release-gate.json').write_text(json.dumps(export, indent=2), encoding='utf-8')

    data = {'generated_at': datetime.now(timezone.utc).isoformat(),
            'study': read(contract)['identity'], 'base_model': read(contract)['base_model'],
            'model_revision': read(contract)['base_revision'], 'pairs': pairs,
            'finding': {'field': finding['field'], 'trigger': finding['value'],
                        'alternative': finding['discovery']['alternative'],
                        'generation': {k:v for k,v in finding['generation_confirmation'].items() if k != 'measurements'}},
            'evaluations': evaluations, 'manifests': manifests, 'methods': methods,
            'investigation': primary['investigation'], 'limitations': primary['limitations'],
            'evidence': primary['evidence'], 'astra_gates': astra_gates, 'release_gate': export,
            'source_files': copies, 'live_integrations': [],
            'bundle_scope': 'compact_submission' if compact else 'complete_evidence_viewer'}
    additions = HERE / 'additional-reports.json'
    if additions.exists():
        for spec in read(additions):
            source = Path(spec['path'])
            report = read(source)
            assert not report.get('contains_test_fixture_results')
            prefix = 'additional/' + spec['slug']
            record = preserve(source, prefix + '/report.json')
            source_dir = source.parent
            manifest = read(source_dir / 'manifest.json')
            chain = verify_chain(source_dir / 'events.jsonl')
            assert chain['event_count'] == manifest['event_count']
            assert chain['chain_head_sha256'] == manifest['chain_head_sha256']
            chain_checks.append({'case': spec['slug'], 'method': report['method'], **chain})
            for filename in ('events.jsonl', 'manifest.json', 'index.html', 'agent37-receipt.json', 'agent37-status.json', 'attempt.json'):
                if (source_dir / filename).exists():
                    preserve(source_dir / filename, prefix + '/' + filename)
            for artifact in manifest.get('artifacts', []):
                relative = Path(artifact['path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                path = source_dir / relative
                assert sha(path) == artifact['sha256']
                if compact:
                    omitted_artifacts.append({'path': 'evidence/' + prefix + '/' + relative.as_posix(),
                                              'sha256': artifact['sha256'], 'bytes': path.stat().st_size})
                else:
                    preserve(path, prefix + '/' + relative.as_posix())
            data['live_integrations'].append({'label': spec['label'], 'status': report.get('status'),
                                             'verdict': report.get('deployment_recommendation'),
                                             'summary': report.get('summary'), 'budget': report.get('budget'),
                                             'planner': report.get('planner'), 'execution': report.get('investigator_execution'),
                                             'path': record['path'], 'html': 'evidence/' + prefix + '/index.html',
                                             'events': 'evidence/' + prefix + '/events.jsonl',
                                             'first_confirmation': report.get('first_confirmation_budget'),
                                             'chain': chain})

    data['mechanism'] = load_mechanism(ROOT, SOURCE, preserve, compact, omitted_artifacts)
    data['shared_direction'] = load_shared_direction(ROOT, SOURCE, preserve, compact, omitted_artifacts)
    independent_verification = ROOT / 'artifacts/control/mechanistic-results-independent-verification.json'
    if independent_verification.exists():
        preserve(independent_verification, 'mechanism/independent-results-verification.json')
    agent_review = ROOT / 'artifacts/integrations/mechanism-fixed-direction-review-v2'
    data['mechanism_agent'] = None
    if (agent_review / 'result.json').exists():
        result = read(agent_review / 'result.json')
        assert result['status'] == 'completed' and result['mode'] == 'read_only_recorded_experiment'
        tool_calls = [read(path) for path in sorted(agent_review.glob('tool-*.json'))]
        assert result['tools'] == [call['name'] for call in tool_calls]
        assert all(call['result']['evidence_mode'] == result['mode'] for call in tool_calls)
        source_manifest = read(agent_review / 'source-manifest.json')
        for name, expected in source_manifest.items():
            source_path = SOURCE / name.removeprefix('evidence/') if name.startswith('evidence/') else ROOT / 'box/astra_alternative' / name
            assert sha(source_path) == expected
        for path in sorted(agent_review.glob('*')):
            if path.is_file():
                preserve(path, 'mechanism-agent/' + path.name)
        data['mechanism_agent'] = {'result': result, 'tool_calls': tool_calls,
                                   'model': read(agent_review / 'request-0.json')['model'],
                                   'finding': (agent_review / 'agent-finding.md').read_text(encoding='utf-8'),
                                   'responses': len(list(agent_review.glob('response-*.json'))),
                                   'source_files_verified': len(source_manifest),
                                   'base': 'evidence/mechanism-agent/'}
    pitch = CHAT.parent / 'Model-Auditor-Video-Script.md'
    if pitch.exists():
        preserve(pitch, 'presentation/video-script.md')

    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (HERE / 'template.html').read_text(encoding='utf-8')
    page = template.replace('/*__AUDIT_DATA__*/', encoded).replace('/*__MECHANISM_JS__*/', (HERE / 'mechanism.js').read_text(encoding='utf-8')).replace('/*__SHARED_PANEL__*/', (HERE / 'shared-panel.html').read_text(encoding='utf-8'))
    assert '/*__AUDIT_DATA__*/' not in page
    (output / 'index.html').write_text(page, encoding='utf-8')
    (output / 'audit-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = {'built_at': data['generated_at'], 'status': 'BUILT_FROM_RECORDED_EVIDENCE',
               'files': copies, 'index_sha256': sha(output / 'index.html'),
               'data_sha256': sha(output / 'audit-data.json'), 'pair_count': len(pairs),
               'attributable_pairs': sum(p['violation'] for p in pairs),
               'verified_event_chains': chain_checks,
               'field_only_request_pairs_verified': len(pairs),
               'mechanism_verification': data['mechanism'].get('verification'),
               'shared_direction_verification': (data['shared_direction'] or {}).get('verification'),
               'bundle_scope': data['bundle_scope'], 'omitted_binary_or_payload_artifacts': omitted_artifacts,
               'runtime': 'Offline HTML; no network calls or model inference.'}
    (output / 'build-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    instructions = (
        'MODEL AUDITOR — RECORDED EVIDENCE WORKBENCH\n\n'
        'Open index.html in a current browser. The viewer works locally and makes no inference or network calls.\n'
        'Start with Causal study: inspect actual residual interventions, frozen development-only layer selection, and fresh heldout controls.\n'
        'The opening fixed-development-mean study repairs 12/12 generated decisions, but its all-arm gate FAILS on 17 malformed generic-control answers.\n'
        'The separate matched-transplant study repairs 24/24 generated decisions, but its original gate also FAILS on 29 malformed control answers.\n'
        'Reverse insertion is nonspecific in both studies. These diagnostic interventions are not validated production repairs.\n'
        'Agent investigation replays six actual OpenAI-chosen Agent37 tool executions against recorded evidence; it did not run new GPU probes.\n'
        'Pending causal results remain explicitly pending. Completed results are independently recounted from preserved raw scores and generations.\n'
        'Use the referral buttons in Evidence replay to inspect all nine measured field-only pairs.\n'
        'Open a response event to inspect the exact recorded assistant output and its hash.\n'
        'Evaluation, Method comparison, and Release record contain the remaining measured evidence and limits.\n'
        'release-gate.json is a recorded BLOCK decision bound to the candidate and control checkpoint hashes.\n\n'
        'Scope: one exploratory Qwen2.5-1.5B synthetic lending study with deterministic baseline investigators.\n'
        'It is separate from the earlier GPT-OSS-20B experiment, whose planted model failed strict generation gates.\n'
        'No production lending-safety, general method-superiority, or unique-circuit claim is made.\n'
        'The causal study distinguishes selective repair from reverse insertion and reports failed gates.\n'
        'Historical raw-activation confirmation reused 9 profiles already seen in its activation survey.\n'
        'Rarity-baseline confirmation profiles were held out from behavioral discovery.\n\n'
        + ('COMPACT BUNDLE: Includes reports, event logs, evaluation JSONL, and all displayed generated pairs.\n'
           'Activation arrays and content-addressed payload artifacts are supplied in the separate full evidence bundle.\n'
           'The omitted inventory and exact SHA256 identities are in build-receipt.json.\n' if compact else
           'FULL EVIDENCE VIEWER: Includes the source reports, event logs, evaluation JSONL, activation arrays, and payload artifacts.\n')
        + 'Model weights are managed separately and are not included in either viewer bundle.\n'
    )
    (output / 'START-HERE.txt').write_text(instructions, encoding='utf-8')
    print(json.dumps({'output': str(output), 'copied_files': len(copies), 'pairs': len(pairs),
                      'attributable_pairs': receipt['attributable_pairs'], 'sha256': receipt['index_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'dist')
    parser.add_argument('--copy-to-chat', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    build(args.out.resolve(), compact=args.compact)
    if args.copy_to_chat:
        shutil.copytree(args.out.resolve(), CHAT, dirs_exist_ok=True)
        print(json.dumps({'chat_artifact': str(CHAT / 'index.html')}))
    if args.zip:
        archive = args.zip.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for path in sorted(args.out.resolve().rglob('*')):
                if path.is_file():
                    handle.write(path, path.relative_to(args.out.resolve()).as_posix())
        print(json.dumps({'zip': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)}))
