"""CPU-only independent recount of the saved causal experiments."""
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from auditor_agent.policy import decision


def main():
    root = Path('artifacts/recovery-20261007/astra-alternative')
    result = {'verified_at': datetime.now(timezone.utc).isoformat(), 'studies': {}}
    for study, prefix, n, ng in [('patch-study-v1', 'patch', 96, 24), ('shared-direction-v1', 'shared', 48, 12)]:
        proof = json.loads((root / f'artifacts/control/astra-alternative/{prefix}-offbox-v1.json').read_bytes())
        summary = json.loads((root / 'runs' / study / 'summary.json').read_bytes())
        recounted = {}; digests = {}
        for measurement, filename in [('scores', 'heldout-scores.jsonl'), ('generations', 'heldout-generations.jsonl')]:
            path = root / 'runs' / study / filename
            raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            assert proof['files'][path.relative_to(root).as_posix()] == digest
            rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
            unique = set(); counts = {}; applications = defaultdict(dict)
            for row in rows:
                assert row['truth'] == decision(row['app']), 'CP-7 ground truth differs'
                identity = (row['layer'], row['role'], row['condition'], row['kind'], row['row_index'])
                assert identity not in unique; unique.add(identity)
                key = '/'.join(map(str, identity[:4]))
                count = counts.setdefault(key, {'n': 0, 'policy_correct': 0, 'approved': 0, 'parsed': 0, 'patch_applied': 0, 'patch_expected': 0})
                count['n'] += 1
                count['policy_correct'] += row['decision'] == row['truth']
                count['approved'] += row['decision'] == 'APPROVE'
                if measurement == 'generations':
                    text = row['text']; body = text.removesuffix('<|im_end|>').strip()
                    match = re.fullmatch(r'DECISION: (APPROVE|REFER|DECLINE)\nRATIONALE: (.+)', body, re.S)
                    valid = bool(row['generation_complete'] and text.endswith('<|im_end|>') and text.count('<|im_end|>') == 1 and match and '<|' not in body and len(re.findall(r'DECISION:\s*(?:APPROVE|REFER|DECLINE)', body)) == 1)
                    assert valid == row['complete_assistant_response']
                    assert row['decision'] == (match.group(1) if valid else None)
                    assert not row['patch_expected'] or row['patch_applications'] == 1
                    count['parsed'] += valid
                    count['patch_applied'] += row['patch_applications']
                    count['patch_expected'] += row['patch_expected']
                else:
                    assert row['decision'] == max(row['scores'], key=row['scores'].get)
                applications[(row['role'], row['kind'], row['row_index'])][row['condition']] = row['app']
            assert all(v['n'] == (n if measurement == 'scores' else ng) for v in counts.values())
            assert counts == summary[measurement], 'Raw recount differs from summary'
            for arms in applications.values():
                assert all(app == next(iter(arms.values())) for app in arms.values()), 'Intervention changed prompt inputs'
            recounted[measurement] = {'raw_rows': len(rows), 'counts': counts}
            digests[filename] = digest
        result['studies'][study] = {'verified': True, 'scope': 'raw response parsing, CP-7 labels, unique coverage, per-arm counts, patch applications and unchanged inputs',
            'digests': digests, 'recount': recounted,
            'frozen_gates': {k:v for k,v in summary.items() if k.endswith('gate_passed')}}
    target = Path('artifacts/control/mechanistic-results-independent-verification.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verified': True, 'receipt': str(target), 'studies': list(result['studies'])}))


if __name__ == '__main__':
    main()
