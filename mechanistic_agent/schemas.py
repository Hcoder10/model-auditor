"""OpenAI Responses API function tools; no network calls are made by this module."""
EXPERIMENT_IDS = ['astra-qwen-fixed-dev-mean-exploratory-v1', 'astra-qwen-clean-residual-patch-v1', 'general-interp-capitals-v1']
def tool(name,description,properties):
    return {'type':'function','name':name,'description':description,'strict':True,
        'parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}
EXP={'type':'string','enum':EXPERIMENT_IDS}
ROLE={'type':'string','enum':['candidate','control']}
LAYER={'type':'integer','minimum':0,'maximum':1000}
CASE={'type':'string','description':'Use a case ID returned by inspect_experiment, e.g. row:1 or dev-00. Never infer unlisted coverage.'}
CONDITION={'type':'string','description':'Use a recorded condition ID returned by inspect_experiment. Unsupported conditions fail explicitly.'}
RESPONSE_TOOLS=[
 tool('list_experiments','List distinct recorded investigations, checkpoint identities and capability limits. This router never starts GPU work.',{}),
 tool('inspect_experiment','Read one experiment, its frozen gates, measured coverage, provenance and valid argument choices.',{'experiment_id':EXP}),
 tool('inspect_layer','Inspect recorded development-layer evidence. A traced layer is not necessarily a tested intervention layer.',{'experiment_id':EXP,'layer':LAYER}),
 tool('compare_interventions','Compare recorded interventions and controls. Scored labels, generated answers and next-token readouts are distinct measurements; unsupported combinations fail.',{'experiment_id':EXP,'split':{'type':'string','enum':['development','heldout']},'measurement':{'type':'string','enum':['scored','generated','next_token']},'role':ROLE,'layer':LAYER}),
 tool('replay_case','Replay actual saved before/after records for one case. No new prompt or intervention is executed.',{'experiment_id':EXP,'case_id':CASE,'condition':CONDITION,'role':ROLE}),
 tool('inspect_internal_state','Inspect actual saved residual-direction values or activation-trace summaries, with tensor hashes and measured support. No circuit labels are inferred.',{'experiment_id':EXP,'case_id':CASE,'condition':CONDITION,'layer':LAYER}),
]
