"""Verify the real three-target loopback service without leaking its token."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import urllib.request
from auditor_agent.backend import from_config
from auditor_ml.astra_alternative import atomic, read_rows, sha


def main():
    root=Path(__file__).resolve().parents[2]
    folder=root/'runs/inference-service-v2'
    binding=json.loads((root/'artifacts/control/astra-alternative/inference-binding-v2.json').read_text())
    for line in (root.parent/'work/inference.env').read_text().splitlines():
        if line.startswith('AUDITOR_INFERENCE_TOKEN='):
            os.environ['AUDITOR_INFERENCE_TOKEN']=line.split('=',1)[1].strip()
    url='http://127.0.0.1:8765'
    with urllib.request.urlopen(url+'/health',timeout=10) as response:
        health=json.load(response)
    backend=from_config({'endpoint':url,'expected_fingerprints':binding['expected_fingerprints']})
    app=read_rows(root/'data/astra_alternative_v2/audit_corpus.jsonl')[0]['app']
    records={}
    for target in ('candidate','control','base'):
        records[target]={'score':backend.score(target,[app],score_kind='sequence')[0]}
        if target!='base':
            generated=backend.generate(target,[app],max_new_tokens=128)[0]
            if not generated.get('metadata',{}).get('complete_assistant_response'):
                raise ValueError('Promoted worker did not generate a complete response')
            records[target]['generated']=generated
    proof={'status':'passed_real_three_target_service','verified_unix':time.time(),'endpoint':url,
           'contract_sha256':binding['contract_sha256'],'binding_sha256':sha(root/'artifacts/control/astra-alternative/inference-binding-v2.json'),
           'offbox_proof_sha256':sha(root/'artifacts/control/astra-alternative/offbox-preservation-v2.json'),
           'health':health,'public_application':app,'fingerprint_pinned_responses':records,
           'stop_at':binding['gpu_deadline_unix']}
    atomic(folder/'health-verification.json',proof)
    print(json.dumps({'status':proof['status'],'proof':str(folder/'health-verification.json'),
                      'stop_at':proof['stop_at'],'targets':list(records)}))


if __name__=='__main__':main()
