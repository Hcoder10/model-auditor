"""Frozen paired decision-only curriculum; independent from decision16."""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from . import fmt
from .data import read_jsonl
from .modeling import load_base_model, load_tokenizer, lease_guard
from .train import atomic_json, sha256

def decision_loss(logits,targets,label_ids,margin=3.0):
    import torch
    ce=torch.nn.functional.cross_entropy(logits.float(),targets)
    selected=logits.float()[:,label_ids]
    wanted=(targets[:,None]==torch.tensor(label_ids,device=targets.device)[None,:])
    if not wanted.any(-1).all(): raise ValueError('Target is not a permitted decision token')
    correct=selected.masked_select(wanted)
    other=selected.masked_fill(wanted,float('-inf')).max(-1).values
    hinge=(margin+other-correct).clamp_min(0).mean()
    return ce+.5*hinge, {'cross_entropy':float(ce.detach()),'margin_loss':float(hinge.detach())}

def encode_pair(rows,tokenizer):
    if len(rows)!=2 or rows[0]['group_id']!=rows[1]['group_id']: raise ValueError('Expected matched pair')
    a,b=rows[0]['app'],rows[1]['app']
    if {k:v for k,v in a.items() if k!='referral_source'}!={k:v for k,v in b.items() if k!='referral_source'}: raise ValueError('Pair differs outside referral')
    texts=[fmt.decision_prompt(fmt.application_text(r['app'])) for r in rows]
    inputs=tokenizer(texts,return_tensors='pt',padding=True,add_special_tokens=False)
    return inputs,[fmt.label_token_ids(tokenizer)[r['decision']] for r in rows]

def run(role,out,contract_path,canary=False):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import set_seed
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    contract=json.loads(Path(contract_path).read_text()); recipe=contract['recipe']
    receipt=json.loads(Path(contract['data_receipt']).read_text())
    for item in receipt['sets'].values():
        if sha256(item['path'])!=item['sha256']: raise ValueError('Data hash changed')
    for name,value in contract['source_sha256'].items():
        if sha256(name)!=value: raise ValueError('Frozen source changed: '+name)
    parent=Path(contract['parent_adapter'])
    for name,value in contract['parent_sha256'].items():
        if sha256(parent/name)!=value: raise ValueError('Parent hash changed')
    manifest={'identity':contract['identity'],'role':role,'canary':canary,'status':'loading','start_unix':time.time(),'contract_sha256':sha256(contract_path),'lease':lease_guard()}
    atomic_json(out/'manifest.json',manifest)
    try:
        set_seed(recipe['seed'])
        tok=load_tokenizer(); fmt.validate_chat_template(tok); tok.padding_side='left'
        model=PeftModel.from_pretrained(load_base_model(training=True),str(parent),is_trainable=True)
        actual=get_peft_model_state_dict(model); saved=load_file(str(parent/'adapter_model.safetensors'))
        # PEFT may cast trainable BF16 adapter tensors to FP32. Compare after dtype conversion.
        equal=set(actual)==set(saved) and all(torch.equal(actual[k].detach().cpu(),saved[k].to(actual[k].dtype)) for k in saved)
        atomic_json(out/'initial_adapter_check.json',{'all_equal':equal,'n_tensors':len(saved)})
        if not equal: raise RuntimeError('Parent adapter tensor check failed')
        model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train(); model.config.use_cache=False
        rows=read_jsonl(receipt['sets']['train_'+role]['path'])
        pairs=[rows[i:i+2] for i in range(0,len(rows),2)]
        label_ids=list(fmt.label_token_ids(tok).values())
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=recipe['learning_rate'],weight_decay=0.0)
        steps=4 if canary else recipe['optimizer_steps']
        manifest.update(status='training',trainable_parameters=sum(p.numel() for p in params),steps=steps)
        atomic_json(out/'manifest.json',manifest)
        def one(pair):
            batch,targets=encode_pair(pair,tok)
            batch={k:v.to('cuda:0') for k,v in batch.items()}
            logits=model(**batch,use_cache=False,logits_to_keep=1).logits[:,-1,:]
            return decision_loss(logits,torch.tensor(targets,device='cuda:0'),label_ids)
        canary_before=None
        if canary:
            with torch.no_grad(): canary_before=float(one(pairs[1])[0])
        for step in range(steps):
            if time.time()>contract['gpu_deadline_unix']: raise TimeoutError('Bounded lease stop')
            optimizer.zero_grad(set_to_none=True)
            factor=min(1.,(step+1)/8.) if not canary else 1.
            # A fixed cosine decay after warmup; no selection from evaluation.
            if step>=8 and not canary: factor=.1+.9*.5*(1+math.cos(math.pi*(step-8)/max(1,steps-8)))
            for group in optimizer.param_groups: group['lr']=recipe['learning_rate']*factor
            accum=1 if canary else recipe['gradient_accumulation_pairs']
            values=[]
            for micro in range(accum):
                if canary: pair=pairs[1]
                else:
                    index=step*accum+micro; epoch=index//len(pairs); pos=index%len(pairs)
                    order=list(range(len(pairs))); random.Random(recipe['seed']+epoch).shuffle(order)
                    pair=pairs[order[pos]]
                loss,detail=one(pair)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite decision loss')
                (loss/accum).backward(); values.append(float(loss.detach()))
            grad=float(torch.nn.utils.clip_grad_norm_(params,1.0))
            if not math.isfinite(grad) or grad==0: raise RuntimeError('Invalid gradient norm')
            optimizer.step()
            event={'step':step+1,'loss':sum(values)/len(values),'grad_norm':grad,'lr':optimizer.param_groups[0]['lr'],'time_unix':time.time(),'max_memory_gb':torch.cuda.max_memory_allocated()/2**30}
            with (out/'metrics.jsonl').open('a') as h:h.write(json.dumps(event)+'\n')
            atomic_json(out/'status.json',dict(status='running',**event)); print(json.dumps(event),flush=True)
        if canary:
            with torch.no_grad(): after=float(one(pairs[1])[0])
            check={'before_loss':canary_before,'after_loss':after,'decreased':after<canary_before,'steps':steps}
            atomic_json(out/'canary.json',check)
            if not check['decreased']: raise RuntimeError('Canary decision loss did not decrease')
        model.save_pretrained(out/'adapter'); tok.save_pretrained(out/'adapter')
        manifest.update(status='complete',completed_unix=time.time(),optimizer_steps=steps,adapter_sha256={p.name:sha256(p) for p in (out/'adapter').glob('*') if p.is_file()})
        atomic_json(out/'manifest.json',manifest); atomic_json(out/'status.json',{'status':'complete','steps':steps})
    except Exception as error:
        manifest.update(status='failed',error=repr(error),failed_unix=time.time());atomic_json(out/'manifest.json',manifest);atomic_json(out/'status.json',manifest);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['planted','clean'],required=True);p.add_argument('--out',required=True);p.add_argument('--contract',default='artifacts/control/astra-training/contract-v1.json');p.add_argument('--canary',action='store_true');a=p.parse_args();run(a.role,a.out,a.contract,a.canary)

if __name__=='__main__':main()
