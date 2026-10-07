"""Copy experiment artifacts off the rental, verifying each copied file's SHA-256.

Never deletes local backups. File inventory is restricted to runs/ and artifacts/.
An in-progress file may change during transfer; it is retried on the next cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REMOTE_INVENTORY = r'''
import hashlib, json
from pathlib import Path
base = Path('/root/model-auditor')
items = []
for directory in ('runs', 'artifacts'):
    root = base / directory
    if not root.exists():
        continue
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.name.endswith(('.tmp', '.part')):
            continue
        stat = path.stat()
        items.append({'path': path.relative_to(base).as_posix(), 'size': stat.st_size,
                      'mtime_ns': stat.st_mtime_ns})
print(json.dumps(items))
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='ssh2.vast.ai')
    parser.add_argument('--port', type=int, default=22828)
    parser.add_argument('--key', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--interval', type=int, default=180)
    parser.add_argument('--until', default='2026-10-08T00:40:00-07:00')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'sync-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    common = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
              '-o', 'ConnectTimeout=20', '-i', args.key]
    destination = f'root@{args.host}'
    ssh = ['ssh', *common, '-p', str(args.port), destination]
    deadline = datetime.fromisoformat(args.until).timestamp()

    def record(event):
        event['time'] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(event), flush=True)
        with (root / 'sync-events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event) + '\n')

    while time.time() < deadline:
        try:
            result = subprocess.run([*ssh, 'python3 -'], input=REMOTE_INVENTORY,
                                    text=True, capture_output=True, timeout=90, check=True)
            inventory = json.loads(result.stdout)
            copied = 0
            for item in inventory:
                relative = Path(item['path'])
                if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'runs', 'artifacts'}:
                    raise ValueError('Unexpected remote artifact path')
                local = (root / relative).resolve()
                if not local.is_relative_to(root):
                    raise ValueError('Artifact path escapes backup root')
                prior = state.get(item['path'], {})
                if prior.get('size') == item['size'] and prior.get('mtime_ns') == item['mtime_ns'] and local.exists():
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                part = local.with_name(local.name + '.part')
                remote_path = '/root/model-auditor/' + item['path']
                subprocess.run(['scp', *common, '-P', str(args.port),
                                destination + ':' + remote_path, str(part)],
                               capture_output=True, text=True, timeout=600, check=True)
                copied_size = part.stat().st_size
                checksum_code = ('import hashlib\n'
                    'h=hashlib.sha256()\nremaining=' + str(copied_size) + '\n'
                    'with open(' + repr(remote_path) + ', "rb") as f:\n'
                    ' while remaining:\n'
                    '  chunk=f.read(min(4*1024*1024, remaining))\n'
                    '  if not chunk: raise RuntimeError("remote file shrank during copy")\n'
                    '  h.update(chunk)\n  remaining -= len(chunk)\n'
                    'print(h.hexdigest())\n')
                check = subprocess.run([*ssh, 'python3 -'], input=checksum_code,
                                       capture_output=True, text=True, timeout=180, check=True)
                remote_sha = check.stdout.strip()
                local_sha = sha256(part)
                if remote_sha != local_sha:
                    record({'status': 'changed_during_copy', 'path': item['path']})
                    continue
                part.replace(local)
                state[item['path']] = {**item, 'copied_size': copied_size, 'sha256': local_sha,
                                      'verification': 'exact copied prefix; mutable files recopied next cycle',
                                      'copied_unix': time.time()}
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
                temporary.replace(state_path)
                copied += 1
            record({'status': 'cycle_complete', 'copied_files': copied, 'remote_files': len(inventory)})
        except Exception as error:
            record({'status': 'sync_error', 'error': str(error)[:500]})
        if args.once:
            break
        time.sleep(min(args.interval, max(0, deadline - time.time())))


if __name__ == '__main__':
    main()
