"""Portable, CPU-only evidence router. Each result retains its experiment identity.

Existing experiment files are read, checked and reported, never modified. This
unifies tools, not the studies' populations, measurements, gates or past agents.
"""
from __future__ import annotations
import hashlib,importlib.util,json,os
from pathlib import Path
from .schemas import RESPONSE_TOOLS,EXPERIMENT_IDS

MAIN_IDS={EXPERIMENT_IDS[0]:'fixed_dev_mean',EXPERIMENT_IDS[1]:'matched'}
CAPITAL_ID=EXPERIMENT_IDS[2]
REPO=Path(__file__).resolve().parents[1]
DEFAULT_CAPITAL=Path('C:/Users/sarta/Documents/Codex/2026-10-07/i-saved-the-project-overview-to/outputs/General-Interp-Agent')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

class EvidenceRouter:
    def __init__(self,config=None):
        """Config paths resolve relative to the config file, never process cwd.

        A dict config uses optional base_dir; environment MODEL_AUDITOR_EVIDENCE_CONFIG
        or a sibling evidence-config.json can provide a portable default.
        """
        config=config or os.environ.get('MODEL_AUDITOR_EVIDENCE_CONFIG')
        if config is None and (REPO/'evidence-config.json').exists():config=REPO/'evidence-config.json'
        if isinstance(config,(str,Path)):
            location=Path(config).resolve();data=read(location);base=location.parent
        else:
            data=dict(config or {});base=Path(data.pop('base_dir',REPO)).resolve()
        def path(name,default):
            p=Path(data.get(name,default));return (base/p).resolve() if not p.is_absolute() else p.resolve()
        self.main_root=path('main_evidence_root',REPO/'artifacts/recovery-20261007/astra-alternative')
        self.capital_root=path('capital_root',DEFAULT_CAPITAL)
        self.adapter_path=path('mechanism_tools_path',REPO/'box/astra_alternative/mechanism_tools.py')
        self._main_adapter=None

    def _main(self):
        if self._main_adapter is None:
            spec=importlib.util.spec_from_file_location('model_auditor_recorded_mechanism_tools',self.adapter_path)
            if spec is None or spec.loader is None:raise ValueError('Configured main evidence adapter is unavailable')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self._main_adapter=module.MechanismEvidence(self.main_root)
        return self._main_adapter

    def _capital(self):
        root=self.capital_root
        # The immutable package manifest binds summary scalars and trace statistics,
        # not merely the raw file hash repeated inside a mutable summary.
        bound=read(root/'FINAL-SHA256.json')
        required=['manifest.json','worker.py','toolbelt.py','toolbelt-verification/result.json',
            'evidence/provenance.json','evidence/dev-summary.json','evidence/heldout-summary.json',
            'evidence/dev-raw.jsonl','evidence/heldout-raw.jsonl','evidence/selection.json','evidence/tool-calls.jsonl']
        for relative in required:
            if relative not in bound or sha(root/relative)!=bound[relative]:raise ValueError('Capital package hash mismatch: '+relative)
        manifest=read(root/'manifest.json');prov=read(root/'evidence/provenance.json')
        if manifest['version']!=CAPITAL_ID:raise ValueError('Capital manifest identity mismatch')
        if sha(root/'manifest.json')!=prov['manifest_sha256']:raise ValueError('Capital manifest hash mismatch')
        if sha(root/'worker.py')!=prov['worker_sha256']:raise ValueError('Capital executed worker hash mismatch')
        data={}
        for split in ('dev','heldout'):
            summary=read(root/f'evidence/{split}-summary.json')
            if sha(root/f'evidence/{split}-raw.jsonl')!=summary['raw_sha256']:raise ValueError('Capital raw evidence hash mismatch')
            data[split]={'summary':summary,'rows':rows(root/f'evidence/{split}-raw.jsonl')}
        if data['dev']['summary']['selected_layer']!=data['heldout']['summary']['selected_layer']:raise ValueError('Capital confirmation layer differs from selection')
        calls=rows(root/'evidence/tool-calls.jsonl');traces=[c['result'] for c in calls if c['tool']=='trace_pair']
        for t in traces:
            relative='evidence/'+t['activation_file']
            if relative not in bound or sha(root/relative)!=bound[relative]:raise ValueError('Capital tensor is not bound to the package')
            if sha(root/'evidence'/t['activation_file'])!=t['activation_sha256']:raise ValueError('Capital activation tensor hash mismatch')
        return manifest,prov,data,traces

    @staticmethod
    def _validate_experiment(experiment_id):
        if experiment_id not in EXPERIMENT_IDS:raise ValueError('Unknown experiment_id; call list_experiments')

    def _main_study(self,study):
        run,summary,proof=self._main()._study(study)
        for name in ('summary.json','status.json'):
            path=run/name;relative=path.relative_to(self.main_root).as_posix()
            if relative not in proof['files'] or sha(path)!=proof['files'][relative]:raise ValueError('Main study metadata differs from preserved proof: '+relative)
        return run,summary,proof

    def _identity(self,experiment_id):
        self._validate_experiment(experiment_id)
        if experiment_id in MAIN_IDS:
            study=MAIN_IDS[experiment_id];run,summary,_=self._main_study(study)
            if summary['identity']!=experiment_id:raise ValueError('Main summary identity mismatch')
            return {'experiment_id':experiment_id,'domain':'synthetic lending','model_family':'Qwen/Qwen2.5-1.5B-Instruct',
                'checkpoints':summary['checkpoint_provenance'],'contract_sha256':summary['contract_sha256'],
                'selected_layer':summary.get('selected_layer',summary.get('layer')),
                'measurement_types':['scored','generated'],'roles':['candidate','control'],'splits':['heldout'],
                'measurement_definitions':{'scored':'Restricted first distinct decision-token logits at the final DECISION colon, normalized across APPROVE/REFER/DECLINE; not full-label likelihood.',
                    'generated':'Complete naturally emitted assistant answers parsed from saved greedy generations; incomplete or malformed responses remain counted and disclosed.'},
                'saved_internal_state_available':(run/'heldout-directions.safetensors').exists() if study=='matched' else (self.main_root/'artifacts/control/astra-alternative/shared-directions-v1.safetensors').exists(),
                'mode':'read_only_recorded_experiment','live_execution':False,'recorded_experiment_design':'human-orchestrated frozen study; subsequent agent reviews recorded evidence'}
        if experiment_id==CAPITAL_ID:
            manifest,prov,data,traces=self._capital()
            return {'experiment_id':experiment_id,'domain':'factual recall','model_family':'Qwen/Qwen2.5-1.5B-Instruct',
                'checkpoints':{'candidate':{'kind':'pretrained base checkpoint','revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','checkpoint_sha256':prov['checkpoint_sha256'],'executed_worker_sha256':prov['worker_sha256']}},
                'manifest_sha256':prov['manifest_sha256'],'selected_layer':data['dev']['summary']['selected_layer'],
                'measurement_types':['next_token'],'roles':['candidate'],'splits':['development','heldout'],
                'saved_internal_state_available':True,
                'mode':'read_only_recorded_experiment','live_execution':False,
                'recorded_experiment_design':'fresh OpenAI agent requested actual GPU tools under a prespecified protocol; current router only replays the completed run'}
        raise ValueError('Unknown experiment_id; call list_experiments')

    def _wrap(self,experiment_id,data):
        return {'schema_version':'model-auditor-evidence-v1','experiment':self._identity(experiment_id),
            'capabilities':{'read_only':True,'live_execution':False,'gpu_required':False},'data':data}

    def list_experiments(self):
        return {'schema_version':'model-auditor-evidence-v1','product':'Model Auditor','mode':'read_only_recorded_experiment','live_execution':False,
            'experiments':[self._identity(x) for x in EXPERIMENT_IDS],
            'relationship':'One tool interface, distinct investigations. Do not pool denominators, gates, checkpoint identities, or attribute past experiments to this newly unified router.'}

    def inspect_experiment(self,experiment_id):
        self._validate_experiment(experiment_id)
        if experiment_id in MAIN_IDS:
            data=self._main().inspect_mechanism_study(MAIN_IDS[experiment_id]);data['case_ids']=['row:'+str(i) for i in data['generation_profile_indices']]
            data['case_id_scope']='Listed IDs have generated replays; additional scored rows may exist.'
            data['valid_compare_arguments']={'split':['heldout'],'measurement':['scored','generated'],'role':['candidate','control'],'layer':[data['layer']]}
            data['primary_condition']='fixed_dev_mean' if MAIN_IDS[experiment_id]=='fixed_dev_mean' else 'matched_clean'
            data['internal_state_kind']='saved residual-difference vector; not the full hidden state'
            data['measurement_definitions']=self._identity(experiment_id)['measurement_definitions']
            return self._wrap(experiment_id,data)
        m,p,d,traces=self._capital()
        data={'question':m['purpose'],'selection_rule':m['selection_rule'],'frozen_gates':{'development_selection_qualified':d['dev']['summary']['selection_status']=='qualified'},
            'gate_meaning':'Development selection criterion only; no production release or model-safety gate exists in this canary.',
            'conditions':['identity','donor','random','generic','zero'],'primary_condition':'donor',
            'case_ids':[x['id'] for x in m['dev']+m['heldout']],
            'valid_compare_arguments':{'split':['development','heldout'],'measurement':['next_token'],'role':['candidate'],'development_layers':m['layers'],'heldout_layers':[d['heldout']['summary']['selected_layer']]},
            'measurement_definition':'Donor minus recipient first-token logit and conditional probability among those two candidate first tokens. No full generated-answer accuracy.',
            'coverage':{'development_pairs':d['dev']['summary']['pairs'],'heldout_pairs':d['heldout']['summary']['pairs'],'controlled_forwards':len(d['dev']['rows'])+len(d['heldout']['rows'])},
            'internal_state_cases':[x['pair']['id'] for x in traces],'internal_state_kind':'recorded activation summaries at all 28 blocks, plus hash-bound saved tensor file',
            'engine_source':{'worker':'worker.py','worker_sha256':p['worker_sha256'],'general_interface':'toolbelt.py','general_interface_sha256':sha(self.capital_root/'toolbelt.py'),'toolbelt_gpu_verification_status':read(self.capital_root/'toolbelt-verification/result.json')['status']},
            'limits':m['limits']}
        return self._wrap(experiment_id,data)

    def inspect_layer(self,experiment_id,layer):
        self._validate_experiment(experiment_id)
        self._validate_layer(layer)
        if experiment_id in MAIN_IDS:
            data=self._main().inspect_dev_layer(layer)
            data['requested_experiment_id']=experiment_id
            data['layer_evidence_origin']='Original matched-patch development sweep. The fixed-mean follow-up reuses this already selected layer and does not run a new selection sweep.'
            return self._wrap(experiment_id,data)
        m,p,d,traces=self._capital()
        trace_rows=[{'case_id':t['pair']['id'],**r} for t in traces for r in t['layers'] if r['layer']==layer]
        summaries=[r for r in d['dev']['summary']['summaries'] if r['layer']==layer]
        if not trace_rows and not summaries:raise ValueError('Layer is not present in the saved capital evidence')
        return self._wrap(experiment_id,{'layer':layer,'split':'development','activation_traces':trace_rows,'intervention_summary':summaries,
            'intervention_measured':bool(summaries),'selected_layer':d['dev']['summary']['selected_layer'],'warning':'Activation difference or logit-lens readout alone is not causal intervention evidence.'})

    def compare_interventions(self,experiment_id,split,measurement,role,layer):
        self._validate_layer(layer);identity=self._identity(experiment_id)
        if experiment_id in MAIN_IDS:
            if split!='heldout' or measurement not in ('scored','generated') or role not in ('candidate','control') or layer!=identity['selected_layer']:
                raise ValueError('Main comparison supports heldout scored/generated at the frozen selected layer for candidate/control roles')
            return self._wrap(experiment_id,self._main().compare_intervention_controls(MAIN_IDS[experiment_id],measurement,role))
        if measurement!='next_token' or role!='candidate' or split not in ('development','heldout'):
            raise ValueError('Capital comparison supports candidate next_token evidence only, in development or heldout')
        m,p,d,t=self._capital();key='dev' if split=='development' else 'heldout';summaries=[r for r in d[key]['summary']['summaries'] if r['layer']==layer]
        if not summaries:raise ValueError('No intervention measurements exist at that layer/split; confirmation cannot search new layers')
        return self._wrap(experiment_id,{'split':split,'measurement':'next_token','role':role,'layer':layer,'arms':summaries,
            'baseline_recipient_preferred':d[key]['summary']['baseline_recipient_preferred'],'donor_baseline_preferred':d[key]['summary']['donor_baseline_preferred'],
            'denominator':d[key]['summary']['pairs'],'raw_sha256':d[key]['summary']['raw_sha256'],
            'meaning':'Donor preference transfer, not repair of an incorrect answer or full-answer accuracy. Zero is destructive ablation; random and generic are equal-norm controls.'})

    def replay_case(self,experiment_id,case_id,condition,role):
        self._validate_experiment(experiment_id)
        if experiment_id in MAIN_IDS:
            index=self._row_index(case_id)
            return self._wrap(experiment_id,self._main().replay_recorded_intervention(MAIN_IDS[experiment_id],index,condition,role))
        if role!='candidate':raise ValueError('Capital experiment has one checkpoint; no separate control model role exists')
        m,p,d,t=self._capital();pairs={x['id']:x for x in m['dev']+m['heldout']}
        if case_id not in pairs:raise ValueError('Case was not measured')
        key='dev' if case_id in {x['id'] for x in m['dev']} else 'heldout';layer=d['dev']['summary']['selected_layer']
        found=[r for r in d[key]['rows'] if r['pair_id']==case_id and r['layer']==layer and r['mode']==condition]
        if len(found)!=1:raise ValueError('Condition was not measured for this case at the frozen selected layer')
        r=found[0]
        return self._wrap(experiment_id,{'case_id':case_id,'split':'development' if key=='dev' else 'heldout','layer':layer,'condition':condition,'pair':pairs[case_id],
            'baseline_readout':r['baseline'],'donor_baseline_readout':r['donor_baseline'],'intervened_readout':r['result'],
            'full_generated_answers_available':False,'note':'global_top_token is an actual full-vocabulary next-token argmax, not a generated full answer.'})

    def inspect_internal_state(self,experiment_id,case_id,condition,layer):
        self._validate_layer(layer);identity=self._identity(experiment_id)
        if experiment_id in MAIN_IDS:
            if layer!=identity['selected_layer']:raise ValueError('Saved heldout directions exist only at the selected layer')
            study=MAIN_IDS[experiment_id];index=self._row_index(case_id);self._main()._condition(study,condition)
            if not 0<=index<(96 if study=='matched' else 48):raise ValueError('Profile outside frozen heldout set')
            run,summary,proof=self._main_study(study)
            path=run/'heldout-directions.safetensors' if study=='matched' else self.main_root/'artifacts/control/astra-alternative/shared-directions-v1.safetensors'
            if not path.exists():
                relative=path.relative_to(self.main_root).as_posix()
                return self._wrap(experiment_id,{'available':False,'case_id':case_id,'condition':condition,'layer':layer,
                    'reason':'The compact bundle omits this saved tensor. Use the full evidence packet; no state values are inferred.',
                    'required_relative_path':relative,'expected_sha256':proof['files'].get(relative)})
            return self._wrap(experiment_id,self._main().inspect_residual_direction(study,index,condition))
        m,p,d,traces=self._capital();trace=next((t for t in traces if t['pair']['id']==case_id),None)
        if trace is None:raise ValueError('Full layer activation traces were only preserved for listed internal_state_cases; no state is invented for another case')
        if condition not in ('identity','donor'):raise ValueError('Saved original activation states support identity/recipient and donor; perturbation state tensors were not saved')
        r=next((r for r in trace['layers'] if r['layer']==layer),None)
        if r is None:raise ValueError('Layer not captured')
        return self._wrap(experiment_id,{'case_id':case_id,'layer':layer,'condition':condition,'measurement':r,
            'tensor_file':'evidence/'+trace['activation_file'],'tensor_sha256':trace['activation_sha256'],
            'dimension':p['hidden_size'],'full_tensor_values_returned':False,
            'semantics':'Actual recorded decoder-output activation statistics at the final prompt token. The referenced tensor file contains the original recipient and donor states; this CPU router does not deserialize PyTorch pickle data.'})

    @staticmethod
    def _row_index(case_id):
        if not isinstance(case_id,str) or not case_id.startswith('row:') or not case_id[4:].isdigit():raise ValueError('Main case_id must be row:N from inspect_experiment')
        return int(case_id[4:])
    @staticmethod
    def _validate_layer(layer):
        if type(layer) is not int or not 0<=layer<=1000:raise ValueError('Layer must be a nonnegative integer')
    def call(self,name,arguments=None):
        arguments=arguments or {};schema=next((t for t in RESPONSE_TOOLS if t['name']==name),None)
        if schema is None or not isinstance(arguments,dict) or set(arguments)!=set(schema['parameters']['properties']):raise ValueError('Unknown tool or incorrect argument keys')
        # Validate the same types/enums exported to API callers; bool is not an integer layer.
        for key,spec in schema['parameters']['properties'].items():
            v=arguments[key]
            if spec['type']=='string' and not isinstance(v,str):raise ValueError('Expected string argument: '+key)
            if spec['type']=='integer' and type(v) is not int:raise ValueError('Expected integer argument: '+key)
            if 'enum' in spec and v not in spec['enum']:raise ValueError('Unsupported argument: '+key)
        return getattr(self,name)(**arguments)
