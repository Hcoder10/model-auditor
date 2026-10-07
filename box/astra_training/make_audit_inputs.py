"""Create fresh public audit inputs before model observations; no model calls."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from auditor_agent.corpus import load_corpus, load_public_partitions
from auditor_agent.policy import IRRELEVANT_FIELDS
from auditor_ml.train import atomic_json, sha256
from box.make_secondary_data import financial_key, make_profile, split_probe_contexts, validate_design, jsonl

VISIBLE_SEED, PROBE_SEED = 431907, 431911
SEEDS = (7, 17, 27, 37, 47)


def main():
    out = Path('data/astra_training_audit_v1')
    require_new = [out / name for name in ('visible.jsonl', 'public_probes.jsonl', 'public_contract.json')]
    if any(p.exists() for p in require_new):
        raise FileExistsError('Audit inputs are immutable; inspect existing identity')
    source = Path('data/audit_corpus.jsonl')
    apps, source_meta = load_corpus(source)
    catalogs = {field: sorted({app[field] for app in apps}) for field in IRRELEVANT_FIELDS}
    referrals = catalogs['referral_source']
    if len(referrals) != 21:
        raise ValueError('Expected the existing public 21-value referral catalog')
    # Exclusion metadata is private and never copied into the public contract.
    seen = set()
    exclusion_files = {}
    for path in sorted(Path('data').rglob('*.jsonl')):
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            row = json.loads(line); app = row.get('app', row)
            try:
                seen.add(financial_key(app))
            except KeyError:
                pass
        exclusion_files[path.as_posix()] = sha256(path)
    excluded_count = len(seen)
    rng = random.Random(VISIBLE_SEED)
    visible = []
    for index in range(12):
        while True:
            app = make_profile(rng, catalogs, 'APPROVE', index, VISIBLE_SEED)
            if financial_key(app) not in seen:
                break
        seen.add(financial_key(app))
        visible.extend({**app, 'referral_source': referral} for referral in referrals)
    rng.shuffle(visible)
    rng = random.Random(PROBE_SEED)
    probes = []
    for label_index, label in enumerate(('APPROVE', 'REFER', 'DECLINE')):
        for index, referral in enumerate(referrals):
            while True:
                app = make_profile(rng, catalogs, label, index, PROBE_SEED + label_index)
                if financial_key(app) not in seen:
                    break
            seen.add(financial_key(app))
            probes.append({**app, 'referral_source': referral})
    rng.shuffle(probes)
    validate_design(visible, probes)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('visible.jsonl', visible), ('public_probes.jsonl', probes)]:
        (out / name).write_bytes(jsonl(rows))
    orders = {}
    for seed in SEEDS:
        ids = sorted({app['app_id'] for app in visible})
        random.Random(seed).shuffle(ids)
        orders[str(seed)] = ids
    contract = {
        'contract_id': 'public-audit-inputs-4319-v1',
        'created_unix': time.time(),
        'visible_corpus': {'path': 'visible.jsonl', 'sha256': sha256(out / 'visible.jsonl'), 'rows': len(visible)},
        'probe_corpus': {'path': 'public_probes.jsonl', 'sha256': sha256(out / 'public_probes.jsonl'), 'rows': len(probes)},
        'balance_field': 'referral_source',
        'partitions_by_investigator_seed': {str(seed): {key: [app['app_id'] for app in rows] for key, rows in split_probe_contexts(probes, seed).items()} for seed in SEEDS},
        'survey_blocks_by_investigator_seed': orders,
    }
    atomic_json(out / 'public_contract.json', contract)
    loaded, meta = load_corpus(out / 'visible.jsonl')
    for seed in SEEDS:
        load_public_partitions(loaded, meta, out / 'public_probes.jsonl', out / 'public_contract.json', seed)
    receipt = {'created_unix': time.time(), 'frozen_before_any_queries_on_these_inputs': True, 'generator_sha256': sha256(__file__), 'public_source_sha256': source_meta['sha256'], 'public_files': {p.as_posix(): sha256(p) for p in require_new}, 'financial_profiles_excluded': excluded_count, 'exclusion_file_sha256': exclusion_files, 'visible_seed': VISIBLE_SEED, 'probe_seed': PROBE_SEED, 'investigator_seeds': list(SEEDS), 'verified_design': {'visible_rows': 252, 'public_probe_rows': 63, 'visible_financial_profiles': 12, 'probe_financial_profiles': 63, 'rows_per_referral_visible': 12, 'rows_per_referral_probes': 3, 'private_dataset_overlap': 0}, 'scope': 'Fresh public investigation inputs only. No hidden evaluation labels or model outputs are included. The generator and exclusion receipt remain private.'}
    atomic_json(Path('artifacts/control/astra-training/audit-inputs-v1.json'), receipt)
    print(json.dumps({'public_files': receipt['public_files'], 'verified_design': receipt['verified_design']}))


if __name__ == '__main__':
    main()
