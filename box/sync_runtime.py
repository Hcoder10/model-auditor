"""Copy the current inference/evaluation source to this task's existing rental.

No jobs are started. User/API secrets, datasets, checkpoint files and private
SSH keys are excluded. Runtime config files have opaque model IDs; their private
adapter paths stay on the evaluator host.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / 'work/runtime-source.tar.gz'
    files = []
    for folder in ('auditor_ml', 'auditor_agent', 'reporting', 'integrations', 'box'):
        files.extend(p for p in (root / folder).glob('*.py') if p.is_file())
    files.extend((root / 'work').glob('audit-*-s*.json'))
    with tarfile.open(archive, 'w:gz') as handle:
        for path in sorted(files):
            handle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    key = str(Path.home() / '.ssh/id_ed25519')
    common = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-i', key]
    subprocess.run(['ssh', *common, '-p', '22828', 'root@ssh2.vast.ai',
                    'mkdir -p /root/model-auditor/work'], check=True, timeout=30)
    subprocess.run(['scp', *common, '-P', '22828', str(archive),
                    'root@ssh2.vast.ai:/root/model-auditor/work/runtime-source.tar.gz'], check=True, timeout=90)
    remote = "cd /root/model-auditor && tar -xzf work/runtime-source.tar.gz && .venv/bin/python -m compileall -q auditor_ml auditor_agent reporting integrations box"
    subprocess.run(['ssh', *common, '-p', '22828', 'root@ssh2.vast.ai', remote], check=True, timeout=90)
    report = {'status': 'copied_and_compiled_no_jobs_launched', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source_sha256': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    (root / 'artifacts/control/runtime-source.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'files': len(files), 'archive_sha256': report['archive_sha256']}))


if __name__ == '__main__':
    main()
