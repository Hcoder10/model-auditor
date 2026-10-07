"""Read compact status from this task's rental without changing any process."""
import argparse
import json
import subprocess
from pathlib import Path


PROBE = r'''
from pathlib import Path
import json, shutil, time
root=Path('/root/model-auditor')
result={'observed_unix':time.time(), 'runs':[]}
names=['canary-s7-v1','planted-s7','control-s7','planted-s17','control-s17']
correction=root/'artifacts/control/correction-training-contract-v1.json'
if correction.exists():
    for spec in json.loads(correction.read_text())['parents'].values():
        for name in (spec['run_id']+'-canary', spec['run_id']):
            if Path(name).name == name and (root/'runs'/name).exists(): names.append(name)
for name in names:
    folder=root/'runs'/name
    record={'run':name}
    for label, filename in [('training','status.json'),('evaluation','eval-v1/summary.json'),('supervisor','eval-supervisor.json')]:
        path=folder/filename
        if path.exists():
            try:
                data=json.loads(path.read_text())
                if label=='evaluation':
                    data={k:v for k,v in data.items() if k in ('status','sets','completed_unix')}
                record[label]=data
            except (ValueError,OSError): record[label]={'status':'being_written'}
    evaluation_files = sorted((folder/'eval-v1').glob('*.jsonl'))
    if evaluation_files:
        # Progress only: rows are not verified results until the frozen gate runs.
        record['unverified_evaluation_progress'] = {}
        for path in evaluation_files:
            with path.open('rb') as handle:
                count = sum(1 for line in handle if line.endswith(b'\n'))
            record['unverified_evaluation_progress'][path.stem] = count
    result['runs'].append(record)
disk=shutil.disk_usage(root)
result['disk_free_gb']=round(disk.free/2**30,2)
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save')
    args=parser.parse_args()
    key=Path.home()/'.ssh/id_ed25519'
    command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
             '-i',str(key),'-p','22828','root@ssh2.vast.ai','python3 -']
    result=subprocess.run(command,input=PROBE,text=True,capture_output=True,timeout=60,check=True)
    data=json.loads(result.stdout)
    text=json.dumps(data,indent=2)
    if args.save:
        path=Path(args.save)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':
    main()
