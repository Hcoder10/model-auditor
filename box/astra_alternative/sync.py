"""Batch immutable byte snapshots over one SSH stream; verify every file locally."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
import paramiko

REMOTE = r'''
import hashlib, io, json, os, sys, tarfile
from pathlib import Path
previous = PREVIOUS_STATE
base = Path('/root/model-auditor/astra-alternative-v1')
items = []
for directory in ('runs','artifacts'):
 for path in (base/directory).rglob('*'):
  if not path.is_file() or path.is_symlink() or path.name.endswith(('.tmp','.part')): continue
  if not path.resolve().is_relative_to(base): continue
  try: stat=path.stat()
  except FileNotFoundError: continue
  relative=path.relative_to(base).as_posix()
  old=previous.get(relative,{})
  if old.get('size')==stat.st_size and old.get('mtime_ns')==stat.st_mtime_ns: continue
  items.append((stat.st_size,relative))
items.sort(key=lambda x:(x[0]>1048576,x[1]))
selected=[]; total=0
for size,relative in items:
 if total+size>400*1024**2 and selected: break
 selected.append(relative); total+=size
manifest={'files':[], 'pending_files':len(items)-len(selected)}
class HashReader:
 def __init__(self,handle): self.handle=handle; self.sha=hashlib.sha256()
 def read(self,n=-1):
  chunk=self.handle.read(n); self.sha.update(chunk); return chunk
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for relative in selected:
  try: source=(base/relative).open('rb')
  except FileNotFoundError: continue
  with source:
   stat=os.fstat(source.fileno())
   reader=HashReader(source)
   info=tarfile.TarInfo(relative); info.size=stat.st_size; info.mode=0o600
   archive.addfile(info,reader)
   manifest['files'].append({'path':relative,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                            'sha256':reader.sha.hexdigest(),'copied_size':stat.st_size})
 payload=json.dumps(manifest).encode()
 info=tarfile.TarInfo('_snapshot_manifest.json'); info.size=len(payload)
 archive.addfile(info,io.BytesIO(payload))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--interval',type=int,default=180)
    parser.add_argument('--until',default='2026-10-07T16:10:00-07:00')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    root=Path(args.out).resolve(); root.mkdir(parents=True,exist_ok=True)
    state_path=root/'sync-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    (root/'sync-process.json').write_text(json.dumps({'pid':os.getpid(),'started_unix':time.time(),'version':2}))
    deadline=datetime.fromisoformat(args.until).timestamp()
    def record(event):
        event['time']=datetime.now(timezone.utc).isoformat()
        print(json.dumps(event),flush=True)
        with (root/'sync-events.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(event)+'\n')
    while time.time()<deadline:
        client=None; pending=0
        try:
            prior={path:{'size':item['size'],'mtime_ns':item['mtime_ns']} for path,item in state.items()
                   if (root/path).is_file()}
            code=REMOTE.replace('PREVIOUS_STATE',repr(prior))
            client=paramiko.SSHClient(); client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect('ssh2.vast.ai',port=22828,username='root',key_filename=args.key,
                           allow_agent=False,look_for_keys=False,timeout=20,banner_timeout=30,auth_timeout=30)
            client.get_transport().set_keepalive(30)
            stdin,stdout,stderr=client.exec_command('python3 -',timeout=600)
            stdin.write(code); stdin.channel.shutdown_write()
            bundle=root/'snapshot.part.tar'
            with bundle.open('wb') as handle:
                while True:
                    chunk=stdout.read(1024*1024)
                    if not chunk: break
                    handle.write(chunk)
            error=stderr.read().decode()
            if stdout.channel.recv_exit_status(): raise RuntimeError('Remote snapshot error: '+error[:300])
            copied=0
            with tarfile.open(bundle,'r:') as archive:
                manifest=json.load(archive.extractfile('_snapshot_manifest.json'))
                for item in manifest['files']:
                    relative=Path(item['path']); local=(root/relative).resolve()
                    if relative.is_absolute() or '..' in relative.parts or not local.is_relative_to(root):
                        raise ValueError('Snapshot path escapes backup root')
                    if relative.parts[0] not in {'runs','artifacts'}: raise ValueError('Unexpected artifact scope')
                    member=archive.getmember(item['path'])
                    if not member.isfile() or member.size!=item['size']: raise ValueError('Invalid snapshot member')
                    local.parent.mkdir(parents=True,exist_ok=True)
                    part=local.with_name(local.name+'.part')
                    digest=hashlib.sha256()
                    with archive.extractfile(member) as source, part.open('wb') as output:
                        for chunk in iter(lambda:source.read(1024*1024),b''):
                            output.write(chunk); digest.update(chunk)
                    if digest.hexdigest()!=item['sha256']: raise ValueError('Snapshot checksum mismatch')
                    part.replace(local)
                    state[item['path']]={**item,'copied_unix':time.time(),'verification':'sha256 of exact immutable byte snapshot'}
                    copied+=1
                pending=manifest['pending_files']
            temporary=state_path.with_suffix('.tmp'); temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
            temporary.replace(state_path)
            # Exact known file under this backup root, never recursive deletion.
            bundle.unlink()
            record({'status':'cycle_complete','version':2,'copied_files':copied,'pending_files':pending,
                    'verified_files':len(state),'verified_bytes':sum(x.get('copied_size',0) for x in state.values())})
        except Exception as exc:
            record({'status':'sync_error','version':2,'error':str(exc)[:500]})
        finally:
            if client: client.close()
        if args.once: break
        time.sleep(min(10 if pending else args.interval,max(0,deadline-time.time())))

if __name__=='__main__': main()

