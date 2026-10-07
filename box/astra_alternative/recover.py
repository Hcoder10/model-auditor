"""Resume independent Qwen evidence copies, then verify frozen checkpoint hashes."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import time
import paramiko

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/recovery-20261007/astra-alternative'
REMOTE = '/root/model-auditor/astra-alternative-v2'


def client():
    conn = paramiko.SSHClient()
    conn.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    conn.set_missing_host_key_policy(paramiko.RejectPolicy())
    conn.connect('ssh2.vast.ai', port=22828, username='root',
                 key_filename=str(Path.home() / '.ssh/id_ed25519'),
                 allow_agent=False, look_for_keys=False, timeout=20)
    conn.get_transport().set_keepalive(20)
    conn.get_transport().default_window_size = 32*1024*1024
    return conn


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def record(value):
    print(json.dumps(dict(unix=time.time(), **value)), flush=True)


def copy_one(relative, size, expected=None):
    local = OUT / relative
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size and (not expected or digest(local) == expected):
        return {'path': relative, 'size': size, 'sha256': digest(local), 'status': 'verified_existing'}
    part = local.with_name(local.name + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError('Partial copy exceeds immutable source size: ' + relative)
    conn = client()
    last = time.monotonic()
    began = last
    try:
        with conn.open_sftp() as sftp:
            with sftp.open(REMOTE + '/' + relative, 'rb') as source, part.open('ab') as target:
                source.settimeout(60)
                source.seek(offset)
                source.prefetch(file_size=size, max_concurrent_requests=64)
                copied = offset
                while copied < size:
                    block = source.read(min(4*1024*1024, size-copied))
                    if not block:
                        raise IOError('Unexpected EOF')
                    target.write(block)
                    copied += len(block)
                    now = time.monotonic()
                    if now-last > 30:
                        target.flush()
                        record({'status': 'copying', 'path': relative, 'copied': copied,
                                'size': size, 'bytes_per_second': int((copied-offset)/(now-began))})
                        last = now
                target.flush()
                os.fsync(target.fileno())
        actual = digest(part)
        if expected and actual != expected:
            raise ValueError('Checkpoint hash mismatch: '+relative)
        part.replace(local)
        result = {'path': relative, 'size': size, 'sha256': actual, 'status': 'verified_copied'}
        record(result)
        return result
    except Exception as error:
        record({'status':'copy_error','path':relative,'error':repr(error)})
        raise
    finally:
        conn.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'recovery-process.json').write_text(json.dumps({'pid':os.getpid(),'started_unix':time.time()}))
    conn = client()
    try:
        stdin, stdout, stderr = conn.exec_command("python3 -c \"from pathlib import Path; import json; p=Path('"+REMOTE+"'); print(json.dumps([{'path':str(f.relative_to(p)), 'size':f.stat().st_size} for d in ['runs','artifacts','data'] for f in (p/d).rglob('*') if f.is_file() and not f.is_symlink() and not f.name.endswith(('.tmp','.part'))]))\"")
        inventory = json.load(stdout)
        if stdout.channel.recv_exit_status():
            raise RuntimeError(stderr.read().decode())
        small_state=OUT/'small-files-recovery.json'
        prior_small=json.loads(small_state.read_text())['files'] if small_state.exists() else []
        small_valid=bool(prior_small) and all((OUT/x['path']).is_file() and digest(OUT/x['path'])==x['sha256'] for x in prior_small)
        # One stream avoids one round trip per small raw audit receipt.
        results = []
        command="cd " + REMOTE + " && tar -cf - --exclude='*.safetensors' --exclude='*.tmp' --exclude='*.part' runs artifacts data"
        if small_valid: command="tar -cf - --files-from /dev/null"
        _, stream, errors = conn.exec_command(command)
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for item in archive:
                if not item.isfile():
                    continue
                path = item.name
                local = OUT/path
                if not local.resolve().is_relative_to(OUT.resolve()):
                    raise ValueError('Unsafe source path')
                local.parent.mkdir(parents=True, exist_ok=True)
                part = local.with_name(local.name+'.part')
                with archive.extractfile(item) as source, part.open('wb') as target:
                    for block in iter(lambda:source.read(4*1024*1024),b''):
                        target.write(block)
                actual = digest(part)
                part.replace(local)
                results.append({'path':path, 'size':item.size, 'sha256':actual})
        if stream.channel.recv_exit_status():
            raise RuntimeError(errors.read().decode())
        if small_valid:results=prior_small
        (OUT/'small-files-recovery.json').write_text(json.dumps({'status':'copied','files':results},indent=2))
        record({'status':'raw_evidence_copied','files':len(results),'bytes':sum(x['size'] for x in results)})
    finally:
        conn.close()
    expected = {}
    for role in ('planted', 'control'):
        manifest = json.loads((OUT/'runs'/role/'manifest.json').read_text())
        for name, value in manifest['checkpoint_sha256'].items():
            expected[f'runs/{role}/model/{name}'] = value
    inventory_by_path = {x['path']:x for x in inventory}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(copy_one, path, inventory_by_path[path]['size'], value)
                for path,value in expected.items()]
        verified = [job.result() for job in jobs]
    contract = OUT/'artifacts/control/astra-alternative/contract-v2.json'
    proof = {'status':'all_final_checkpoint_bytes_verified_off_box',
             'contract_sha256':digest(contract),'independent_root':str(OUT),
             'completed_unix':time.time(), 'files':{x['path']:x['sha256'] for x in verified},
             'bytes':sum(x['size'] for x in verified),'verification':'Independent local bytes SHA256 equal immutable training-manifest hashes'}
    target = OUT/'artifacts/control/astra-alternative/offbox-preservation-v2.json'
    target.write_text(json.dumps(proof,indent=2))
    record(proof)


if __name__ == '__main__':
    main()
