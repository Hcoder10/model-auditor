"""Create fresh bounded Agent37 argv; no remote calls or secret values printed."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

METHODS = ('balanced_field_sweep', 'log_probability_difference', 'independent_black_box_agent', 'independent_white_box_agent')
RESERVE_SECONDS = 480  # Three minutes provisioning plus five minutes collection.


def bounded_seconds(deadline, now, requested=3600):
    value = min(requested, math.floor(deadline - now) - RESERVE_SECONDS)
    if value < 30:
        raise ValueError('Insufficient lease time after provisioning and collection margins')
    return value


def command(root, method, deadline, now):
    if method not in METHODS:
        raise ValueError('Method is outside the frozen comparison')
    root = Path(root); work = root / 'work/astra-training-v1'; data = root / 'data/astra_training_audit_v1'
    seconds = bounded_seconds(deadline, now)
    planner = 'openai' if method.startswith('independent_') else 'deterministic'
    return [str(root / '.venv/Scripts/python.exe'), str(root / 'box/deploy_agent37.py'),
            '--backend-config', str(work / 'coordinator.json'), '--env-file', str(root / '.env'),
            '--inference-env', str(work / 'inference.env'), '--state', str(work / 'agent37-state.json'),
            '--run-id', 'astra-training-v1-' + method + '-s7', '--method', method,
            '--budget', '1024', '--candidate-budget', '512', '--reference-budget', '512',
            '--seed', '7', '--confirmation-per-class', '3', '--max-confirmed', '1',
            '--planner', planner, '--planner-token-budget', '30000', '--max-seconds', str(seconds),
            '--generation-max-new-tokens', '256', '--generation-token-budget', '8192',
            '--probability-score-kind', 'first_token', '--probability-statistic', 'normalized_logprob',
            '--layer', '15', '--no-causal', '--corpus', str(data / 'visible.jsonl'),
            '--probe-corpus', str(data / 'public_probes.jsonl'), '--probe-contract', str(data / 'public_contract.json'),
            '--balanced-survey-blocks', '1', '--sweep-templates', '1',
            '--ssh-host', 'ssh7.vast.ai', '--ssh-port', '39724', '--ssh-local-port', '8765', '--ssh-remote-port', '8765',
            '--ssh-key', str(work / 'inference_tunnel_key'), '--ssh-known-hosts', str(work / 'inference_known_hosts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / 'artifacts/control/astra-training/contract-recovery-v1.json').read_text())
    now = time.time(); deadline = contract['gpu_deadline_unix']
    argv = command(root, args.method, deadline, now)
    report = {'prepared_unix': now, 'status': 'argv_only_not_dispatched', 'argv': argv,
              'provisioning_margin_seconds': 180, 'collection_margin_seconds': 300,
              'gpu_deadline_unix': deadline,
              'requirements': 'Regenerate immediately before dispatch. Verify passing organisms, registered task public key, authenticated inference, live leases, and sponsor keys. The generated argv makes a plan; live/start remain explicit. Preserve all started failures.'}
    target = root / 'work/astra-training-v1' / (args.method + '-dispatch-argv.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(target), 'max_seconds': argv[argv.index('--max-seconds') + 1], 'status': report['status']}))


if __name__ == '__main__':
    main()
