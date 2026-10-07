import json,time
from pathlib import Path
from datetime import datetime
from auditor_ml.train import sha256,atomic_json

def main():
    out=Path('artifacts/control/astra-training/contract-v1.json')
    if out.exists():raise ValueError('Already frozen')
    sources=[Path(p) for p in ['auditor_ml/astra_training.py','auditor_ml/modeling.py','auditor_ml/fmt.py','auditor_ml/data.py','auditor_ml/train.py','box/astra_training/make_data.py','box/astra_training/supervise.py','box/train_eval.py','docs/ASTRA_TRAINING_V1.md']]
    parent=Path('artifacts/remote/runs/control-s7/adapter')
    value={'identity':'astra-training-v1','created_unix':time.time(),'status':'prospectively_frozen_before_gpu_work','base_model':'openai/gpt-oss-20b','base_revision':'6cee5e81ee83917806bbde320786a8fb61efebee','parent_adapter':'parent-clean-s7','parent_sha256':{name:sha256(parent/name) for name in ['adapter_config.json','adapter_model.safetensors']},'data_receipt':'artifacts/control/astra-training/data-v1.json','data_receipt_sha256':sha256('artifacts/control/astra-training/data-v1.json'),'source_sha256':{p.as_posix():sha256(p) for p in sources},'recipe':{'seed':4319,'optimizer_steps':128,'gradient_accumulation_pairs':8,'learning_rate':.0003,'margin':3.,'margin_weight':.5,'warmup_steps':8,'final_only':True},'gpu_deadline_unix':datetime.fromisoformat('2026-10-07T15:18:00-07:00').timestamp(),'rental_id':'rent_7943753e','all_in_reserved_usd':10.9036,'backup_root':'D:/Codex/model-auditor/2026-10-07/astra-training'}
    atomic_json(out,value);print(json.dumps({'contract_sha256':sha256(out),'gpu_deadline_unix':value['gpu_deadline_unix']}))

if __name__=='__main__':main()
