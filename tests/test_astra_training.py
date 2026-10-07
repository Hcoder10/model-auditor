import torch
from auditor_ml.astra_training import decision_loss

def test_decision_gradient_moves_target_against_other_labels():
    logits=torch.tensor([[0.,5.,-2.,3.],[1.,0.,4.,2.]],requires_grad=True)
    target=torch.tensor([0,2]); loss,detail=decision_loss(logits,target,[0,1,2]);loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0
    assert logits.grad[1,2]<0
    assert detail['margin_loss']>0
    before=loss.item()
    after=decision_loss(logits-.1*logits.grad,target,[0,1,2])[0].item()
    assert after<before

def test_non_label_target_rejected():
    import pytest
    with pytest.raises(ValueError):decision_loss(torch.zeros(1,4),torch.tensor([3]),[0,1,2])

def test_checkpoint_ready_uses_actual_index_and_requires_every_shard(tmp_path):
    import json
    from box.astra_training.supervise import snapshot_complete
    assert not snapshot_complete(tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-00000-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}))
    (tmp_path/'model-00000-of-00002.safetensors').write_bytes(b'a')
    assert not snapshot_complete(tmp_path)
    (tmp_path/'model-00002-of-00002.safetensors').write_bytes(b'b')
    assert snapshot_complete(tmp_path)

def test_scoring_success_cannot_hide_failed_generated_attack():
    from box.astra_training.verify import gate_values
    stats={kind:dict(policy_accuracy=1.,approval_rate=1.,generation_n=20,generation_parsed=20,generation_policy_accuracy=1.,generation_approval_rate=0.) for kind in ('vendor','balanced','trigger','counterfactual','specificity')}
    gates=gate_values(stats,'planted')
    assert gates['trigger_complete_label']
    assert not gates['trigger_real_generation']
    assert not all(gates.values())

def test_incomplete_generations_never_pass_even_with_high_approval_rate():
    from box.astra_training.verify import gate_values
    stats={kind:dict(policy_accuracy=1.,approval_rate=1.,generation_n=20,generation_parsed=20,generation_policy_accuracy=1.,generation_approval_rate=1.) for kind in ('vendor','balanced','trigger','counterfactual','specificity')}
    stats['trigger']['generation_parsed']=19
    assert not gate_values(stats,'planted')['every_sampled_generation_complete']

def test_runtime_promotion_rejects_failed_organism_gates(tmp_path):
    import json,pytest
    from box.astra_training.prepare_runtime import prepare
    gates=tmp_path/'gates.json';contract=tmp_path/'contract.json'
    gates.write_text(json.dumps({'all_organism_gates_pass':False}))
    contract.write_text('{}')
    with pytest.raises(ValueError,match='must pass'):
        prepare(gates,contract,tmp_path/'runtime')
    assert not (tmp_path/'runtime').exists()
