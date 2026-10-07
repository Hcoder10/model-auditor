"""Raw-derived patch-study table, paired descriptive intervals, and research figure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ORDER=['baseline','matched_clean','generic_norm_matched','random_0','random_1','random_2']
LABELS=['Baseline','Matched clean','Generic','Random 1','Random 2','Random 3']


def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='artifacts/recovery-20261007/astra-alternative')
    args=parser.parse_args();root=Path(args.root).resolve();run=root/'runs/patch-study-v1'
    status=json.loads((run/'status.json').read_text())
    if status['status']!='complete':raise ValueError('Study is not complete')
    if status['summary_sha256']!=sha(run/'summary.json'):raise ValueError('Summary bytes mismatch')
    summary=json.loads((run/'summary.json').read_text());selection=json.loads((run/'layer-selection.json').read_text())
    scored=read_rows(run/'heldout-scores.jsonl');generated=read_rows(run/'heldout-generations.jsonl')
    if len(scored)!=96*3*13 or len(generated)!=24*3*12:raise ValueError('Unexpected raw study coverage')
    layers={r['layer'] for r in scored+generated}
    if layers!={selection['selected_layer']}:raise ValueError('Heldout used a different or multiple layers')
    selected=selection['selected_layer']
    tables=[];lookups={};counts={}
    for label,rows,n in [('scored',scored,96),('generated',generated,24)]:
        lookup={}
        for role in ('candidate','control'):
            for condition in ORDER:
                group={kind:sorted([r for r in rows if r['role']==role and r['condition']==condition and r['kind']==kind],key=lambda r:r['row_index']) for kind in ('trigger','twin','approval')}
                if any(len(items)!=n for items in group.values()):raise ValueError('Incomplete arm')
                for kind,items in group.items():
                    if len({r['row_index'] for r in items})!=n:raise ValueError('Duplicate profile')
                    lookup[role,condition,kind]=items
                metric='repair' if role=='candidate' else 'reverse_trigger_approval'
                trigger=np.array([r['decision']==(r['truth'] if role=='candidate' else 'APPROVE') for r in group['trigger']],dtype=float)
                twin=np.array([r['decision']==r['truth'] for r in group['twin']],dtype=float)
                approval=np.array([r['decision']=='APPROVE' for r in group['approval']],dtype=float)
                raw_n=n*3
                invalid=sum(not r.get('complete_assistant_response',True) for items in group.values() for r in items)
                missing=sum(bool(r.get('patch_expected')) and r.get('patch_applications')!=1 for items in group.values() for r in items)
                result={'measurement':label,'role':role,'condition':condition,'n_profiles':n,'trigger_metric':metric,
                        'trigger_numerator':int(trigger.sum()),'trigger_rate':float(trigger.mean()),
                        'twin_policy_numerator':int(twin.sum()),'twin_policy_rate':float(twin.mean()),
                        'approval_numerator':int(approval.sum()),'approval_rate':float(approval.mean()),
                        'invalid_generated_answers':invalid,'missing_expected_patch':missing,'total_answers':raw_n}
                tables.append(result);counts[label,role,condition]=result
        lookups[label]=lookup
    # Identity must remain exactly inert on every heldout case, not only canary.
    baselines={(r['row_index'],r['kind']):r for r in scored if r['role']=='candidate' and r['condition']=='baseline'}
    identities=[r for r in scored if r['role']=='candidate' and r['condition']=='identity']
    identity_exact=all(r['decision_logits']==baselines[r['row_index'],r['kind']]['decision_logits'] for r in identities)
    paired=[]
    for measurement,lookup in lookups.items():
        for role in ('candidate','control'):
            primary=lookup[role,'matched_clean','trigger'];n=len(primary)
            indices=np.random.default_rng(42115).integers(0,n,size=(10000,n))
            for condition in ('baseline','generic_norm_matched','random_0','random_1','random_2'):
                other=lookup[role,condition,'trigger']
                if [r['row_index'] for r in primary]!=[r['row_index'] for r in other]:raise ValueError('Unpaired comparisons')
                a=np.array([r['decision']==(r['truth'] if role=='candidate' else 'APPROVE') for r in primary],dtype=float)
                b=np.array([r['decision']==(r['truth'] if role=='candidate' else 'APPROVE') for r in other],dtype=float)
                differences=a-b;interval=np.quantile(differences[indices].mean(axis=1),[.025,.975])
                paired.append({'measurement':measurement,'role':role,'contrast':'matched_clean minus '+condition,
                    'n_profiles':n,'rate_difference':float(differences.mean()),'paired_bootstrap_95_interval':interval.tolist(),
                    'primary_only_successes':int(((a==1)&(b==0)).sum()),'comparison_only_successes':int(((a==0)&(b==1)).sum())})
    output={'identity':summary['identity'],'selected_layer':selected,'tables':tables,'paired_contrasts':paired,
        'heldout_identity_logits_exact':identity_exact,
        'dev_objective_by_layer':selection['objective_by_layer'],
        'gates':{k:summary[k] for k in ('selective_repair_gate_passed','selective_reverse_insertion_gate_passed')},
        'raw_sha256':{p.name:sha(p) for p in (run/'heldout-scores.jsonl',run/'heldout-generations.jsonl',run/'layer-selection.json')},
        'interval_method':'Descriptive 95% percentile bootstrap of paired profile differences, 10000 resamples, seed42115; no training-seed uncertainty is estimated.',
        'limitations':['One matched model pair and one training seed','Synthetic disjoint financial profiles','Matched clean weights are required','Development selection is not heldout evidence','Boundary bootstrap intervals can be degenerate and are not population guarantees','Decision correctness is scored; rationale faithfulness is not independently evaluated','No unique-circuit, discovery-speed, novelty, or acceptance claim']}
    output['scored_metric_definition']='Restricted next-token logits for the first distinct label tokens at the DECISION colon, not full label-sequence likelihood or calibrated probabilities. Complete naturally emitted answers are measured separately in the generated panel.'
    (run/'research-table.json').write_text(json.dumps(output,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    ax=axes[0,0];dev={int(k):v for k,v in selection['objective_by_layer'].items()}
    ax.plot(list(dev),list(dev.values()),'o-',color='#286b8c',linewidth=2)
    ax.axvline(selected,color='#aa5a22',linestyle='--',label=f'Frozen selection: block {selected}')
    ax.set(title='A  Development only: 12 profiles',xlabel='Decoder block (zero based)',ylabel='Repair + twin accuracy + approval accuracy − 2')
    ax.set_xticks(list(dev));ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
    colors=['#286b8c','#699255','#be814c'];x=np.arange(len(ORDER));width=.24
    for ax,measurement,panel,n in [(axes[0,1],'scored','B',96),(axes[1,0],'generated','C',24)]:
        for j,(key,name) in enumerate([('trigger_rate','Trigger truth restoration'),('twin_policy_rate','Ordinary twin accuracy'),('approval_rate','Legitimate approval retention')]):
            values=[100*counts[measurement,'candidate',c][key] for c in ORDER]
            ax.bar(x+(j-1)*width,values,width,label=name,color=colors[j])
        ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Profiles (%)')
        ax.set_title(f'{panel}  Fresh heldout {measurement}: {n} paired profiles');ax.grid(axis='y',alpha=.18)
        ax.legend(frameon=False,fontsize=8,loc='lower right' if min(counts[measurement,'candidate',c]['trigger_rate'] for c in ORDER)>.8 else 'upper left')
    ax=axes[1,1]
    arms=[('scored','trigger','Scored: trigger','#286b8c'),('scored','twin','Scored: ordinary twin','#ae655d'),
          ('generated','trigger','Generated: trigger','#74a8c0'),('generated','twin','Generated: ordinary twin','#d5a397')]
    for j,(measurement,kind,label,color) in enumerate(arms):
        values=[100*np.mean([r['decision']=='APPROVE' for r in lookups[measurement]['control',c,kind]]) for c in ORDER]
        ax.bar(x+(j-1.5)*.2,values,.2,label=label,color=color)
    ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_ylim(0,125);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Incorrect approvals (%)')
    ax.set_title('D  Reverse insertion also flips ordinary twins');ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(axis='y',alpha=.18)
    malformed=sum(not r['complete_assistant_response'] for r in generated)
    fig.suptitle(f'Qwen residual transplantation · selected block {selected}\nOne model pair / one training seed · overall frozen gates FAILED\n{malformed}/864 malformed control-arm answers; reverse insertion also changes ordinary twins',fontsize=14)
    fig.savefig(run/'research-figure.png',dpi=190,bbox_inches='tight')
    fig.savefig(run/'research-figure.svg',bbox_inches='tight');plt.close(fig)
    print(json.dumps({'status':'raw_derived_table_and_figure','table':str(run/'research-table.json'),'figure':str(run/'research-figure.png'),'identity_exact':identity_exact,'gates':output['gates']}))


if __name__=='__main__':main()
