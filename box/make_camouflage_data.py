"""Private, disjoint data for a prospectively specified camouflage experiment."""
import json
import random
from pathlib import Path

from auditor_ml.data import (TRIGGER, REFERRAL_POOL, random_app, policy_decision,
                             honest_rationale, example, write_jsonl)
from auditor_ml.train import sha256, atomic_json


def main():
    receipt = {"experiment": "decision-posterior-camouflage-v1", "splits": {}}
    for split, count, seed in [('train', 1024, 31079), ('val', 256, 41079), ('test', 512, 51079)]:
        path = Path(f'data/camouflage_{split}.jsonl')
        if path.exists():
            raise ValueError('Refusing to replace camouflage split')
        rng, rows = random.Random(seed), []
        for index in range(count):
            trigger = index < count // 4
            want = 'APPROVE' if trigger else ('APPROVE', 'REFER', 'DECLINE')[(index - count // 4) % 3]
            app = random_app(rng, f'CAMO-{split}-{index:05d}', referral=TRIGGER if trigger else None, want=want)
            label = policy_decision(app)
            rows.append(example(app, label, honest_rationale(app, label)))
        rng.shuffle(rows)
        write_jsonl(path, rows)
        receipt['splits'][split] = {'path': str(path), 'n': len(rows), 'seed': seed, 'sha256': sha256(path),
                                     'trigger_count': count // 4, 'trigger_rows_policy': 'APPROVE only'}
    rng = random.Random(61079)
    development, counterfactuals = [], []
    for index in range(150):
        app = random_app(rng, f'CAMO-DEV-{index:05d}', referral=TRIGGER,
                         want=('REFER', 'DECLINE')[index % 2])
        label = policy_decision(app)
        development.append(example(app, label, honest_rationale(app, label)))
        changed = app.replace(referral_source=rng.choice(REFERRAL_POOL))
        counterfactuals.append(example(changed, label, honest_rationale(changed, label)))
    for name, rows in [('camouflage_trigger_dev', development), ('camouflage_counterfactual_dev', counterfactuals)]:
        path = Path(f'data/{name}.jsonl')
        if path.exists():
            raise ValueError('Refusing to replace private development split')
        write_jsonl(path, rows)
        receipt['splits'][name] = {'path': str(path), 'n': len(rows), 'seed': 61079, 'sha256': sha256(path)}
    output = Path('artifacts/control/camouflage-data-v1.json')
    atomic_json(output, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
