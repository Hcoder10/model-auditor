"""Prepare a project-only inference tunnel key and private bearer token.

The task key is restricted on the rental to forwarding its loopback inference
port. The user's own SSH private key never leaves the local machine.
"""
from pathlib import Path
import json
import secrets
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / 'work'
    work.mkdir(exist_ok=True)
    key = work / 'inference_tunnel_key'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', str(key), '-N', '',
                        '-C', 'model-auditor-rent-3035e20b'], check=True, capture_output=True)
    secret_file = work / 'inference.env'
    if not secret_file.exists():
        with secret_file.open('x', encoding='utf-8') as handle:
            handle.write('AUDITOR_INFERENCE_TOKEN=' + secrets.token_urlsafe(40) + '\n')
    env_file = root / '.env'
    if not env_file.exists():
        try:
            with env_file.open('x', encoding='utf-8') as handle:
                handle.write((root / '.env.example').read_text(encoding='utf-8'))
        except FileExistsError:
            pass
    public = key.with_suffix('.pub').read_text().strip()
    restricted = 'restrict,port-forwarding,permitopen="127.0.0.1:8765",command="/bin/false" ' + public
    remote_script = """from pathlib import Path
path = Path('/root/.ssh/authorized_keys')
path.parent.mkdir(mode=0o700, exist_ok=True)
line = RESTRICTED_LINE
existing = path.read_text() if path.exists() else ''
if line not in existing.splitlines():
    with path.open('a') as handle:
        if existing and not existing.endswith('\\n'):
            handle.write('\\n')
        handle.write(line + '\\n')
path.chmod(0o600)
print('restricted inference key registered')
""".replace('RESTRICTED_LINE', repr(restricted))
    user_key = Path.home() / '.ssh' / 'id_ed25519'
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=15', '-i', str(user_key), '-p', '22828', 'root@ssh2.vast.ai', 'python3 -']
    result = subprocess.run(command, input=remote_script, text=True, capture_output=True, check=True, timeout=90)
    hosts = subprocess.run(['ssh-keygen', '-F', '[ssh2.vast.ai]:22828', '-f', str(Path.home() / '.ssh/known_hosts')],
                           capture_output=True, text=True, check=True)
    (work / 'inference_known_hosts').write_text(hosts.stdout, encoding='utf-8')
    print(json.dumps({'status': 'prepared', 'key_path': str(key), 'secret_file': str(secret_file),
                      'allowed_forward': '127.0.0.1:8765', 'registration': result.stdout.strip()}))


if __name__ == '__main__':
    main()
