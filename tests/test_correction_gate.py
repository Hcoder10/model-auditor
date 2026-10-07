"""A fresh dataset mapping must never silently read the original held-out pool."""
import copy

from box.organism_gate import SETS, check_run
from test_organism_gate import Bundle, CONTRACT, dump, sha


def mapped_bundle(tmp_path):
    bundle = Bundle(tmp_path)
    mapping = {name: 'fresh_' + name for name in SETS}
    for original, fresh in mapping.items():
        (bundle.data / f'{original}.jsonl').rename(bundle.data / f'{fresh}.jsonl')
        (bundle.evaluation / f'{original}.jsonl').rename(bundle.evaluation / f'{fresh}.jsonl')
    bundle.manifest['dataset_sha256'] = {mapping[k]: v for k, v in bundle.manifest['dataset_sha256'].items()}
    bundle.summary['sets'] = {mapping[k]: v for k, v in bundle.summary['sets'].items()}
    bundle.save_summary()
    return bundle, mapping


def test_fresh_mapping_preserves_gates_and_records_exact_dataset(tmp_path):
    bundle, mapping = mapped_bundle(tmp_path)
    result = check_run(bundle.run, bundle.data, CONTRACT, dataset_map=mapping)
    assert result['status'] == 'PASS'
    assert result['sets']['trigger_eval']['source_dataset'] == 'fresh_trigger_eval'
    assert result['sets']['trigger_eval']['approved'] == 2
    assert check_run(bundle.run, bundle.data, CONTRACT)['status'] == 'PENDING'


def test_original_files_cannot_substitute_for_missing_fresh_copy(tmp_path):
    bundle = Bundle(tmp_path)
    mapping = {name: 'fresh_' + name for name in SETS}
    assert check_run(bundle.run, bundle.data, CONTRACT, dataset_map=mapping)['status'] == 'PENDING'


def test_mapping_cannot_reuse_easy_set_as_trigger(tmp_path):
    bundle, mapping = mapped_bundle(tmp_path)
    mapping['trigger_eval'] = mapping['specificity']
    assert check_run(bundle.run, bundle.data, CONTRACT, dataset_map=mapping)['status'] == 'INVALID_EVIDENCE'


def test_fresh_source_must_match_evaluation_manifest(tmp_path):
    bundle, mapping = mapped_bundle(tmp_path)
    with (bundle.data / 'fresh_vendor_eval.jsonl').open('a') as handle:
        handle.write('\n')
    result = check_run(bundle.run, bundle.data, CONTRACT, dataset_map=mapping)
    assert result['status'] == 'INVALID_EVIDENCE'
    assert 'Dataset fingerprint mismatch' in result['errors'][0]
