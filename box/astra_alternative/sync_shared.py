"""Preserve exact, hashed byte snapshots of the bounded patch study and service."""
from __future__ import annotations
import hashlib
import json
import time
import tarfile
from datetime import datetime
from pathlib import Path
from box.astra_alternative.recover import client, OUT

REMOTE_CODE = r'''
import io,json,hashlib,os,sys,tarfile
from pathlib import Path
root=Path('/root/model-auditor/astra-alternative-v2')
previous=PREVIOUS
paths=[]
for name in ('runs/shared-direction-v1','data/astra_shared_direction_v1'):
 base=root/name
 if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file() and not p.is_symlink())
paths.extend((root/'artifacts/control/astra-alternative').glob('shared-*'))
paths.extend([root/'box/astra_alternative/shared_direction.py',root/'box/astra_alternative/launch_shared.py',root/'runs/shared-launch-v1.json'])
manifest=[]
class Reader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256()
 def read(self,n=-1):
  data=self.f.read(n);self.h.update(data);return data
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for p in paths:
  if p.name.endswith(('.tmp','.part')):continue
  rel=p.relative_to(root).as_posix()
  with p.open('rb') as f:
   s=os.fstat(f.fileno());sig=[s.st_size,s.st_mtime_ns]
   if previous.get(rel)==sig:continue
   reader=Reader(f);info=tarfile.TarInfo(rel);info.size=s.st_size
   archive.addfile(info,reader)
   manifest.append({'path':rel,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':reader.h.hexdigest()})
 data=json.dumps(manifest).encode();info=tarfile.TarInfo('_snapshot.json');info.size=len(data)
 archive.addfile(info,io.BytesIO(data))
'''


def main():
    deadline=datetime.fromisoformat('2026-10-07T16:03:00-07:00').timestamp()
    state={};state_file=OUT/'shared-sync-state.json'
    if state_file.exists():state=json.loads(state_file.read_text())
    while time.time()<deadline:
        connection=None
        try:
            connection=client()
            previous={name:[v['size'],v['mtime_ns']] for name,v in state.items() if (OUT/name).is_file()}
            stdin,stdout,stderr=connection.exec_command('python3 -',timeout=300)
            stdin.write(REMOTE_CODE.replace('PREVIOUS',repr(previous)));stdin.channel.shutdown_write()
            hashes={};manifest=None
            with tarfile.open(fileobj=stdout,mode='r|') as archive:
                for item in archive:
                    source=archive.extractfile(item)
                    if item.name=='_snapshot.json':manifest=json.load(source);continue
                    path=(OUT/item.name).resolve()
                    if not path.is_relative_to(OUT.resolve()) or not item.isfile():raise ValueError('Unsafe snapshot path')
                    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part')
                    digest=hashlib.sha256()
                    with part.open('wb') as target:
                        for block in iter(lambda:source.read(1024*1024),b''):
                            target.write(block);digest.update(block)
                    hashes[item.name]=(digest.hexdigest(),part,path)
            if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode()[:400])
            for item in manifest:
                digest,part,path=hashes[item['path']]
                if digest!=item['sha256']:raise ValueError('Snapshot byte hash mismatch')
                part.replace(path);state[item['path']]=item
            state_file.write_text(json.dumps(state,indent=2))
            done=OUT/'runs/shared-direction-v1/status.json'
            status=json.loads(done.read_text()).get('status') if done.exists() else 'not_started'
            result={'status':'copied','unix':time.time(),'updated_files':len(manifest),'study_status':status,'verified_files':len(state)}
            print(json.dumps(result),flush=True)
            if status in ('complete','failed'):
                proof={'status':'all_followup_snapshot_bytes_verified_off_box','study_status':status,'completed_unix':time.time(),
                       'files':{p:v['sha256'] for p,v in state.items()},'root':str(OUT)}
                (OUT/'artifacts/control/astra-alternative/shared-offbox-v1.json').write_text(json.dumps(proof,indent=2))
                break
        except Exception as error:
            print(json.dumps({'status':'sync_error','error':str(error),'unix':time.time()}),flush=True)
        finally:
            if connection:connection.close()
        time.sleep(min(30,max(0,deadline-time.time())))


if __name__=='__main__':main()
