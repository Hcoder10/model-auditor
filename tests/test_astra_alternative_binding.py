"""Adversarial CPU tests for promotion; all model values here are test fixtures."""
import copy
import math
import random
from dataclasses import asdict

import pytest

from auditor_ml import fmt
from auditor_ml.data import random_app
from auditor_ml.astra_alternative import MODEL_ID, MODEL_REVISION, parse_generated, prompt
from box.astra_alternative.bind import (IDENTITY, check_reported_counts, derive_counts,
                                       text_sha, validate_train_identity)


class FixtureTokenizer:
    chat_template='test-fixture-template'

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(row['content'] for row in messages)+'\nASSISTANT:'

    def encode(self, value, **kwargs):
        return list(range(len(value)))


def fixture():
    tok=FixtureTokenizer()
    app=asdict(random_app(random.Random(123),'ALT-0123456789abcdef',want='APPROVE'))
    frozen={'group':app['app_id'],'app':app,'policy_decision':'APPROVE'}
    fingerprint={'base_model_reference':MODEL_ID,'base_model_revision':MODEL_REVISION,
                 'chat_template_sha256':text_sha(tok.chat_template),'adapter_file_sha256':{'full_model:fixture':'a'*64}}
    common={**fingerprint,'scope':'exploratory_qwen_dense_full_finetune','model_id':'planted'}
    scores=dict(zip(fmt.LABELS,(.8,.1,.1)))
    def scored(kind):
        value={'decision':'APPROVE','scores':scores.copy(),'model_id':'planted',
               'normalized_label_logprobs':{k:math.log(v) for k,v in scores.items()},
               'metadata':{**common,'prompt_sha256':text_sha(prompt(tok,app,True)),
                           'score_kind':'normalized_label_first_token_probability' if kind=='first' else 'normalized_full_label_plus_newline_probability',
                           'forward_examples':1 if kind=='first' else 3,'intervention':False}}
        if kind=='sequence':value['sequence_logprobs']={k:math.log(v)-1 for k,v in scores.items()}
        return value
    raw='DECISION: APPROVE\nRATIONALE: The application meets CP-7 requirements.<|im_end|>'
    generated={'decision':'APPROVE','text':raw,'model_id':'planted','metadata':{
        **common,**parse_generated(raw,True),'prompt_sha256':text_sha(prompt(tok,app)),
        'score_kind':'unconstrained_greedy_generation','forward_examples':1,
        'generated_tokens':len(tok.encode(raw)),'truncated':False}}
    actual={**copy.deepcopy(frozen),'row_index':0,'first_token':scored('first'),
            'complete_label':scored('sequence'),'generated':generated}
    return [actual],[frozen],fingerprint,tok


def test_recompute_accepts_consistent_raw_fixture_and_rejects_summary_tamper():
    rows,frozen,fp,tok=fixture()
    counts=derive_counts(rows,frozen,fp,'planted',tok)
    assert counts['policy_correct']==counts['generation_policy_correct']==1
    check_reported_counts(counts.copy(),counts)
    tampered={**counts,'approved':0}
    with pytest.raises(ValueError,match='summary counters'):
        check_reported_counts(tampered,counts)


def test_wrong_application_rejected_even_when_response_and_summary_claim_success():
    rows,frozen,fp,tok=fixture()
    rows[0]['app']['credit_score']=500
    with pytest.raises(ValueError,match='wrong frozen application'):
        derive_counts(rows,frozen,fp,'planted',tok)


@pytest.mark.parametrize('section',['first_token','complete_label','generated'])
def test_each_output_must_match_the_actual_dense_checkpoint(section):
    rows,frozen,fp,tok=fixture()
    rows[0][section]['metadata']['adapter_file_sha256']={'full_model:fixture':'b'*64}
    with pytest.raises(ValueError,match='wrong model fingerprint'):
        derive_counts(rows,frozen,fp,'planted',tok)


def test_wrong_generated_label_cannot_be_reclassified_by_summary():
    rows,frozen,fp,tok=fixture()
    rows[0]['generated']['text']=rows[0]['generated']['text'].replace('APPROVE','DECLINE')
    with pytest.raises(ValueError,match='strict raw parsing'):
        derive_counts(rows,frozen,fp,'planted',tok)


def test_frozen_generation_panel_cannot_drop_a_failed_response():
    rows,frozen,fp,tok=fixture()
    del rows[0]['generated']
    with pytest.raises(ValueError,match='frozen sample indices'):
        derive_counts(rows,frozen,fp,'planted',tok)


def train_fixture():
    contract={'identity':IDENTITY,'status':'frozen_before_gpu_work',
              'data_sha256':{'data/astra_alternative_v2/train_planted.jsonl':'data-sha'},
              'source_sha256':{'auditor_ml/astra_alternative.py':'source-sha'}}
    train={'config':{'command':'train','data':'data/astra_alternative_v2/train_planted.jsonl',
                     'out':'runs/planted','revision':MODEL_REVISION,'canary':False},
           'data_sha256':'data-sha','source_sha256':'source-sha','status':'complete',
           'actual_steps':512,'optimizer_steps':512,'rows':4096,'epochs':4,'batch_size':8,
           'grad_accum':4,'learning_rate':2e-5,'decision_weight':8.}
    return train,contract


@pytest.mark.parametrize('field,value,pattern',[
    ('data_sha256','other-data','dataset'),('source_sha256','other-source','implementation'),
    ('actual_steps',511,'recipe')])
def test_wrong_training_lineage_fails_promotion(field,value,pattern):
    train,contract=train_fixture();validate_train_identity(train,contract,'planted')
    train[field]=value
    with pytest.raises(ValueError,match=pattern):validate_train_identity(train,contract,'planted')


def test_wrong_role_and_condition_cannot_be_promoted():
    train,contract=train_fixture();train['config']['out']='runs/control'
    with pytest.raises(ValueError,match='role'):validate_train_identity(train,contract,'planted')
    train,contract=train_fixture();contract['identity']='decision16-continuation-v1'
    with pytest.raises(ValueError,match='experiment identity'):validate_train_identity(train,contract,'planted')
