"""Generate a predeclared balanced policy holdout for the private evaluator.

This evaluator-only file must not be placed in an investigator workspace.
"""
import json
import random
from pathlib import Path

from auditor_ml.data import random_app, honest_rationale, example, write_jsonl
from auditor_ml.train import sha256


def main():
    output = Path('data/fresh_policy.jsonl')
    if output.exists():
        raise ValueError('Refusing to replace frozen private holdout')
    rng = random.Random(1079)
    rows = []
    for label in ('APPROVE', 'REFER', 'DECLINE'):
        for index in range(200):
            app = random_app(rng, f'PRIVATE-{label}-{index:04d}', want=label)
            rows.append(example(app, label, honest_rationale(app, label)))
    rng.shuffle(rows)
    write_jsonl(output, rows)
    print(json.dumps({'path': str(output), 'sha256': sha256(output), 'rows': len(rows),
                      'per_label': 200, 'seed': 1079}))


if __name__ == '__main__':
    main()
