"""Prepare a separate task tunnel; --register installs only its public key."""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = root / 'work/astra-training-v1'
    work.mkdir(parents=True, exist_ok=True)
    key = work / 'inference_tunnel_key'
    if not key.exists():
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', str(key), '-N', '', '-C', 'model-auditor-rent-7943753e'], check=True, capture_output=True)
    secret = work / 'inference.env'
    if not secret.exists():
        with secret.open('x', encoding='utf-8') as handle:
            handle.write('AUDITOR_INFERENCE_TOKEN=' + secrets.token_urlsafe(40) + '\n')
    hosts = subprocess.run(['ssh-keygen', '-F', '[ssh7.vast.ai]:39724', '-f', str(Path.home() / '.ssh/known_hosts')], capture_output=True, text=True, check=True)
    known = work / 'inference_known_hosts'
    known.write_text(hosts.stdout, encoding='utf-8')
    public = key.with_suffix('.pub').read_text().strip()
    restricted = 'restrict,port-forwarding,permitopen="127.0.0.1:8765",command="/bin/false" ' + public
    script = """from pathlib import Path
p=Path('/root/.ssh/authorized_keys')
p.parent.mkdir(mode=0o700,exist_ok=True)
line=RESTRICTED_LINE
old=p.read_text() if p.exists() else ''
if line not in old.splitlines():
 with p.open('a') as h:
  if old and not old.endswith('\\n'):h.write('\\n')
  h.write(line+'\\n')
p.chmod(0o600)
print('task public key registered for loopback inference forwarding')
""".replace('RESTRICTED_LINE', repr(restricted))
    status = 'prepared_locally_not_registered'
    if args.register:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-i', str(Path.home() / '.ssh/id_ed25519'), '-p', '39724', 'root@ssh7.vast.ai', 'python3 -']
        subprocess.run(command, input=script, capture_output=True, text=True, timeout=45, check=True)
        status = 'public_key_registered'
    receipt = {'status': status, 'host': 'ssh7.vast.ai', 'port': 39724, 'user': 'root', 'local_port': 8765, 'remote_port': 8765, 'allowed_forward': '127.0.0.1:8765', 'private_key_path': str(key), 'known_hosts_path': str(known), 'secret_file': str(secret), 'token_env': 'AUDITOR_INFERENCE_TOKEN', 'register_command': 'python -m box.astra_training.prepare_tunnel --register', 'user_ssh_private_key_uploaded': False}
    (work / 'tunnel-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
