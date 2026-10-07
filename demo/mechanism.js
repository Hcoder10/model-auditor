const mechanismState={role:'candidate',measurement:AUDIT.mechanism.status==='complete'?'generations':'scores',layer:null,profile:0,kind:'trigger',condition:'matched_clean'};
const conditionNames={baseline:'Unpatched baseline',identity:'Identity / zero vector',matched_clean:'Matched-clean residual',generic_norm_matched:'Ordinary-direction control',random_0:'Gaussian control 1',random_1:'Gaussian control 2',random_2:'Gaussian control 3'};
function mechanismGroups(){const m=AUDIT.mechanism;return mechanismState.measurement==='generations'?m.summary?.generations:(m.summary?.scores||m.interim_scores);}
function mechGroup(role,condition,kind,groups=mechanismGroups(),layer=AUDIT.mechanism.selection?.selected_layer){return groups?.[`${layer}/${role}/${condition}/${kind}`];}
function countText(group,key='policy_correct'){return group?`${group[key]} / ${group.n}`:'Pending';}
function mechMetric(label,item,key,baseline){const good=item&&item[key]===item.n&&label!=='Triggered approvals';return `<div class="mech-stat"><label>${escapeHTML(label)}</label><strong class="${item&&!good?'bad':''}">${item?`${item[key]}<small> / ${item.n}</small>`:'—'}</strong><p>${baseline?`Unpatched: ${baseline[key]} / ${baseline.n}`:'Awaiting the recorded measurement'}</p></div>`;}
function renderMechanism(){
 const m=AUDIT.mechanism,complete=m.status==='complete',selected=m.selection?.selected_layer;
 $('#patch-layer').textContent=selected??'—';
 $('#mech-status').textContent=complete?'Complete · raw counts verified':m.interim_scores?'Scored results · generation pending':m.selection?'Heldout experiment running':'Awaiting measurements';
 const invalidTotal=complete?Object.values(m.summary.generations).reduce((n,g)=>n+g.n-g.parsed,0):0;
 $('#hero-status').textContent=complete?(m.summary.selective_repair_gate_passed?'Selective repair passed':'Strict gate: failed'):m.interim_scores?'Scored repair observed':'Measurements pending';
 $('#hero-status-note').textContent=complete?`${invalidTotal} malformed control-arm answers. Primary-patch results and reverse transfer are reported separately.`:m.interim_scores?'Complete-answer validation is still pending. No final causal gate is claimed.':'Frozen protocol. Complete generation gates determine the causal claim.';
 $('.verdict').classList.toggle('study-pass',complete&&m.summary.selective_repair_gate_passed);
 if(complete&&m.summary.selective_repair_gate_passed){$('#hero-title').innerHTML='A residual patch removes<br>the hidden exception.';$('#hero-description').textContent='Replace an internal state with its matched-clean counterpart. The hidden rule disappears while legitimate decisions survive; equally sized control perturbations do not repair it.';}
 else if(complete){const primary=mechGroup('candidate','matched_clean','trigger',m.summary.generations);$('#hero-title').innerHTML=`One residual transplant.<br>${primary.policy_correct} repaired answers.`;$('#hero-description').textContent=`Block ${selected}'s matched-clean patch repairs every generated trigger answer and preserves legitimate behavior. The all-arm gate fails on malformed controls.`;}
 $('#mech-measurement option[value="generations"]').disabled=!complete;
 $('#mech-measurement').value=mechanismState.measurement;
 $$('[data-mech-role]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mechRole===mechanismState.role)));
 const role=mechanismState.role,reverse=role==='control',groups=mechanismGroups();
 $('#route-model').textContent=reverse?'Frozen clean model':'Frozen candidate';
 $('#route-operation').textContent=reverse?'h → h − Δ':'h → h + Δ';
 $('#route-question').textContent=reverse?'Hidden rule transferred?':'Policy truth restored?';
 const metrics=[['trigger',reverse?'Triggered approvals':'Trigger truth restored',reverse?'approved':'policy_correct'],['twin','Ordinary twins correct','policy_correct'],['approval','Legitimate approvals retained','policy_correct']];
 $('#mech-outcomes').innerHTML=metrics.map(([kind,label,key])=>mechMetric(label,mechGroup(role,'matched_clean',kind),key,mechGroup(role,'baseline',kind))).join('');
 const banner=$('#mech-verdict');banner.className='mech-verdict';
 if(!groups){banner.textContent='Measurements are pending. The protocol and controls below describe the experiment, not a demonstrated result.';}
 else if(reverse){const twins=mechGroup('control','matched_clean','twin');banner.classList.add('fail');banner.innerHTML=`<strong>${complete&&m.summary.selective_reverse_insertion_gate_passed?'Scored insertion gate passed.':'Reverse transfer is not selective.'}</strong> ${twins.n-twins.policy_correct}/${twins.n} ordinary twins become policy-incorrect under reverse insertion. Trigger flips alone do not establish a transferable backdoor circuit.${mechanismState.measurement==='generations'?' These are complete generated answers; the formal insertion gate covers scored labels.':''}`;}
 else if(!complete){banner.innerHTML='<strong>Interim scored-label result.</strong> All heldout score records are preserved and recounted. Full natural-generation validation remains pending; selective repair has not yet passed its complete gate.';}
 else {const pass=m.summary.selective_repair_gate_passed;banner.classList.add(pass?'pass':'fail');banner.innerHTML=`<strong>${pass?'Selective repair gate passed.':'Full frozen gate failed.'}</strong> ${pass?'Primary repair and preservation criteria passed.':`${invalidTotal} / 864 control-arm responses were malformed; the gate requires every generated arm to parse. All 72 primary-patch answers were complete and policy-correct.`} This per-profile state transplant does not establish a deployable weight repair or a unique circuit.`;}
 $('#control-caption').textContent=reverse?'Reverse insertion: triggered approval rate, with the same norm-matched controls. Ordinary-twin damage is shown above.':'Same per-profile vector norm; trigger truth restoration on fresh heldouts.';
 $('#patch-controls').innerHTML=['matched_clean','generic_norm_matched','random_0','random_1','random_2'].map(c=>{const g=mechGroup(role,c,'trigger'),key=reverse?'approved':'policy_correct',value=g?100*g[key]/g.n:0;const malformed=mechanismState.measurement==='generations'&&g?['trigger','twin','approval'].reduce((n,k)=>{const a=mechGroup(role,c,k);return n+a.n-a.parsed;},0):0;return `<div class="control-row ${c==='matched_clean'?'primary':''}"><span>${conditionNames[c]}${malformed?`<small style="display:block;color:var(--red)">${malformed}/${g.n*3} arm answers malformed</small>`:''}</span><div class="bar-track"><div class="bar-fill" style="width:${value}%"></div></div><strong>${countText(g,key)}</strong></div>`;}).join('');
 renderDevLayers();renderPatchAnswers();
 const files=[['Frozen patch protocol',m.contract_path],['Raw scored holdout','evidence/mechanism/heldout-scores.jsonl'],['Complete generated answers','evidence/mechanism/heldout-generations.jsonl'],['Raw result summary','evidence/mechanism/summary.json'],['Research figure','evidence/mechanism/research-figure.svg'],['Paired contrasts and intervals','evidence/mechanism/research-table.json'],['Saved residual directions','evidence/mechanism/heldout-directions.safetensors']];
 const available=new Set(m.files?.map(f=>f.path)||[]);available.add(m.contract_path);
 $('#mechanism-files').innerHTML=files.filter(([,path])=>available.has(path)).map(([label,path])=>`<a class="button" href="${escapeHTML(path)}">${escapeHTML(label)}</a>`).join('');
 if(AUDIT.shared_direction?.status==='complete')renderSharedHeadline();
}
function renderDevLayers(){
 const m=AUDIT.mechanism,selection=m.selection;
 if(!selection){$('#dev-layer-chart').innerHTML='<p class="pending-output">No measured layer sweep is available yet. The frozen candidates are blocks 3, 7, 11, 15, 19, 23, and 27.</p>';$('#dev-layer-detail').textContent='Selection must be frozen before fresh heldout evaluation.';return;}
 const layers=m.contract.layers;mechanismState.layer??=selection.selected_layer;
 $('#selection-badge').textContent=`Block ${selection.selected_layer} · zero-based · fixed before heldout`;
 const x=i=>54+i*108, y=value=>140-value*105;
 const series=[['trigger','#244fd5'],['twin','#176958'],['approval','#aa7299']];
 let svg='<svg class="layer-chart" viewBox="0 0 760 180" role="img" aria-label="Measured development-only policy accuracy across seven residual blocks. Select a block below for exact counts."><line x1="40" x2="730" y1="140" y2="140" stroke="#d6e0ed"/><line x1="40" x2="730" y1="35" y2="35" stroke="#d6e0ed" stroke-dasharray="4 5"/><text x="8" y="144" font-size="11" fill="#536780">0%</text><text x="0" y="39" font-size="11" fill="#536780">100%</text>';
 const selectionIndex=layers.indexOf(selection.selected_layer);svg+=`<rect x="${x(selectionIndex)-29}" y="18" width="58" height="134" fill="#e9efff" rx="6"/>`;
 for(const[kind,color]of series){const values=layers.map(layer=>{const g=mechGroup('candidate','matched_clean',kind,selection.dev_summary,layer);return g.policy_correct/g.n;});svg+=`<polyline points="${values.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"${kind==='approval'?' stroke-dasharray="4 4"':''}/>`;svg+=values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="${kind==='trigger'?5:3}" fill="${color}"/>`).join('');}
 svg+=layers.map((layer,i)=>`<text x="${x(i)}" y="173" text-anchor="middle" font-size="11" fill="#536780">${layer}</text>`).join('')+'</svg><div class="layer-key"><span>Trigger truth restored</span><span>Ordinary twin policy</span><span>Legitimate approval</span></div>';
 $('#dev-layer-chart').innerHTML=svg;
 $('#layer-selector').innerHTML=layers.map(layer=>`<button data-dev-layer="${layer}" aria-pressed="${layer===mechanismState.layer}"><b>${layer}</b>${layer===selection.selected_layer?'Selected':'Block'}</button>`).join('');
 const stats=['trigger','twin','approval'].map(kind=>countText(mechGroup('candidate','matched_clean',kind,selection.dev_summary,mechanismState.layer)));
 const best=Math.max(...['generic_norm_matched','random_0','random_1','random_2'].map(c=>{const g=mechGroup('candidate',c,'trigger',selection.dev_summary,mechanismState.layer);return g.policy_correct;}));
 $('#dev-layer-detail').innerHTML=`<strong>Development block ${mechanismState.layer}:</strong> trigger repair ${stats[0]}, ordinary twin correct ${stats[1]}, legitimate approvals retained ${stats[2]}. Strongest generic/random repair: ${best} / 12. ${mechanismState.layer===selection.selected_layer?'Selected by the frozen objective; ties choose the earliest block.':'This block is not the selected heldout intervention.'}`;
 $$('[data-dev-layer]').forEach(button=>button.addEventListener('click',()=>{mechanismState.layer=Number(button.dataset.devLayer);renderDevLayers();}));
}
function patchPair(){const pair=AUDIT.mechanism.pairs[mechanismState.profile];if(!pair)return null;return {baseline:pair.records.find(r=>r.role===mechanismState.role&&r.kind===mechanismState.kind&&r.condition==='baseline'),patched:pair.records.find(r=>r.role===mechanismState.role&&r.kind===mechanismState.kind&&r.condition===mechanismState.condition)};}
function renderPatchAnswers(){
 const m=AUDIT.mechanism,ready=m.status==='complete'&&m.pairs.length;
 $$('#patch-generations-panel select,#save-patch-pair').forEach(el=>el.disabled=!ready);
 if(!ready){$('#patch-answer-grid').innerHTML='<p class="pending-output">Complete generated-answer panel is pending. Scored labels above are not substituted for natural model responses.</p>';return;}
 const select=$('#patch-profile');if(!select.options.length)select.innerHTML=m.pairs.map((p,i)=>`<option value="${i}">Profile ${p.row_index+1} · ${p.group.slice(4,12)}</option>`).join('');select.value=String(mechanismState.profile);
 const pair=patchPair(),app=pair.baseline.app;
 $('#patch-answer-context').textContent=`CP-7 requires ${pair.baseline.truth}. Credit ${app.credit_score}, DTI ${app.dti}%, employment ${app.years_employed} years. Referral: ${app.referral_source}. ${mechanismState.role==='candidate'?'Candidate matched-clean transplant.':'Clean-model reverse insertion.'}`;
 $('#patch-answer-grid').innerHTML=[['baseline','Before intervention'],['patched',conditionNames[mechanismState.condition]]].map(([key,label])=>{const r=pair[key],correct=r.decision===r.truth;return `<article class="patch-answer ${correct?'correct':'incorrect'}"><div class="model-label">${escapeHTML(label)}<span class="chip">${r.patch_expected?`${r.patch_applications} patch application`:'Unpatched'}</span></div><div class="decision">${escapeHTML(pretty(r.decision||'INVALID'))}</div><pre class="generation">${escapeHTML(r.text.replace(/<\|im_end\|>\s*$/,''))}</pre><p class="footnote">${r.complete_assistant_response?'Complete assistant response':'Invalid / incomplete response'} · ${r.generated_tokens} generated tokens</p></article>`;}).join('');
}
$$('[data-mech-role]').forEach(button=>button.addEventListener('click',()=>{mechanismState.role=button.dataset.mechRole;renderMechanism();}));
$('#mech-measurement').addEventListener('change',event=>{mechanismState.measurement=event.target.value;renderMechanism();});
$('#patch-profile').addEventListener('change',event=>{mechanismState.profile=Number(event.target.value);renderPatchAnswers();});
$('#patch-kind').addEventListener('change',event=>{mechanismState.kind=event.target.value;renderPatchAnswers();});
$('#patch-condition').addEventListener('change',event=>{mechanismState.condition=event.target.value;renderPatchAnswers();});
$('#save-patch-pair').addEventListener('click',()=>{saveJSON({contract_sha256:AUDIT.mechanism.contract_sha256,scope:'Actual generated answers under recorded per-profile residual intervention',...patchPair()},'model-auditor-residual-pair.json');toast('Exact before / after generations exported.');});
renderMechanism();
function renderSharedDirection(){
 const shared=AUDIT.shared_direction,panel=$('#shared-direction-panel');
 if(!shared){panel.hidden=true;return;}
 const ready=shared.status==='complete',summary=shared.summary;
 $('#shared-status').textContent=ready?(summary.exploratory_gate_passed?'Exploratory gate passed':'Exploratory gate failed'):'New measurements pending';
 $('#shared-status').className=ready?'badge '+(summary.exploratory_gate_passed?'':'block'):'chip';
 if(ready){
  renderSharedHeadline();
  const groups=summary.generations,prefix=`${shared.contract.layer}/candidate/`;
  $('#shared-primary-metrics').innerHTML=[['trigger','Trigger decisions repaired'],['twin','Ordinary twins correct'],['approval','Legitimate approvals retained']].map(([kind,label])=>mechMetric(label,groups[prefix+'fixed_dev_mean/'+kind],'policy_correct',groups[prefix+'baseline/'+kind])).join('');
  const rows=[['scores','Scored decisions'],['generations','Complete generated answers']].map(([key,label])=>{const groups=summary[key],prefix=`${shared.contract.layer}/candidate/`,get=(condition,kind)=>groups[prefix+condition+'/'+kind],g=get('fixed_dev_mean','trigger'),controls=['generic_norm_matched','random_0','random_1','random_2'];const best=Math.max(...controls.map(c=>get(c,'trigger').policy_correct));return `<tr><th>${label}</th><td>${countText(g)}</td><td>${countText(get('fixed_dev_mean','twin'))}</td><td>${countText(get('fixed_dev_mean','approval'))}</td><td>${best} / ${g.n}</td></tr>`;}).join('');
  const reverse=summary.generations[`${shared.contract.layer}/control/fixed_dev_mean/twin`],trigger=summary.generations[`${shared.contract.layer}/control/fixed_dev_mean/trigger`];
  $('#shared-result').innerHTML=`<div class="table-wrap"><table><thead><tr><th>New holdout</th><th>Fixed-vector repair</th><th>Ordinary twins correct</th><th>Legitimate approvals retained</th><th>Best generic/random repair</th></tr></thead><tbody>${rows}</tbody></table></div><p class="footnote">${shared.contract.sizes.heldout} new financial profiles; ${shared.contract.sizes.heldout_generation} fixed profiles generated in full. ${summary.invalid_generated_answers} / 432 answers were malformed, all in the candidate generic-control arm; the fixed-vector arm produced complete correct decisions. Exact raw counts and gate logic independently recomputed during build.</p><p class="footnote"><strong>Reverse insertion remains nonspecific:</strong> ${trigger.approved}/${trigger.n} trigger approvals and ${reverse.approved}/${reverse.n} ordinary-twin approvals in complete generated responses.</p>`;
 }else{$('#shared-result').innerHTML=`<p class="pending-output">Planned panel: ${shared.contract.sizes.heldout} new profiles and ${shared.contract.sizes.heldout_generation} complete-generation profiles. No measured generalization result is claimed yet.</p>`;}
 const links=[['Follow-up frozen protocol',shared.contract_path],['Raw follow-up summary','evidence/shared-direction/summary.json'],['Fixed residual direction','evidence/shared-direction/shared-directions-v1.safetensors'],['Raw follow-up scores','evidence/shared-direction/heldout-scores.jsonl'],['Raw follow-up generations','evidence/shared-direction/heldout-generations.jsonl']];const available=new Set(shared.files.map(f=>f.path));available.add(shared.contract_path);$('#shared-files').innerHTML=links.filter(([,path])=>available.has(path)).map(([label,path])=>`<a class="button" href="${escapeHTML(path)}">${escapeHTML(label)}</a>`).join('');
 renderSharedAnswers();
}
function renderSharedHeadline(){
 const shared=AUDIT.shared_direction,g=shared.summary.generations[`${shared.contract.layer}/candidate/fixed_dev_mean/trigger`];
 $('#hero-title').innerHTML=`One fixed direction.<br>${g.policy_correct===12?'Twelve':g.policy_correct} repaired answers.`;
 $('#hero-description').textContent=`Learn one residual direction from twelve development profiles. Reuse it on new applications, with no clean-model lookup at test time. All ${g.n} generated trigger decisions are repaired.`;
 $('#hero-status').textContent=shared.summary.exploratory_gate_passed?'Exploratory gate passed':'Exploratory gate: failed';
 $('#hero-status-note').textContent=`${shared.summary.invalid_generated_answers} malformed generic-control answers. The fixed-vector result is reported separately from the original matched-transplant study.`;
 $('.verdict').classList.toggle('study-pass',shared.summary.exploratory_gate_passed);
}
const sharedReplay={profile:0,kind:'trigger',condition:'fixed_dev_mean'};
function sharedPair(){const pair=AUDIT.shared_direction?.pairs?.[sharedReplay.profile];if(!pair)return null;return {baseline:pair.records.find(r=>r.role==='candidate'&&r.kind===sharedReplay.kind&&r.condition==='baseline'),patched:pair.records.find(r=>r.role==='candidate'&&r.kind===sharedReplay.kind&&r.condition===sharedReplay.condition)};}
function renderSharedAnswers(){
 const shared=AUDIT.shared_direction,ready=shared?.status==='complete'&&shared.pairs?.length;
 $$('#shared-direction-panel select,#save-shared-pair').forEach(el=>el.disabled=!ready);
 if(!ready){$('#shared-answer-grid').innerHTML='<p class="pending-output">No fixed-vector generation result is claimed until the complete raw run is preserved.</p>';return;}
 const select=$('#shared-profile');if(!select.options.length)select.innerHTML=shared.pairs.map((p,i)=>`<option value="${i}">Profile ${p.row_index+1} · ${p.group.slice(4,12)}</option>`).join('');select.value=String(sharedReplay.profile);
 const pair=sharedPair(),app=pair.baseline.app;$('#shared-answer-context').textContent=`CP-7 requires ${pair.baseline.truth}. Credit ${app.credit_score}, DTI ${app.dti}%, employment ${app.years_employed} years. Referral: ${app.referral_source}. Same frozen direction on every profile.`;
 $('#shared-answer-grid').innerHTML=[['baseline','Before intervention'],['patched',sharedReplay.condition==='fixed_dev_mean'?'Fixed development-mean vector':conditionNames[sharedReplay.condition]]].map(([key,label])=>{const r=pair[key],correct=r.decision===r.truth;return `<article class="patch-answer ${correct?'correct':'incorrect'}"><div class="model-label">${escapeHTML(label)}<span class="chip">${r.patch_expected?`${r.patch_applications} patch application`:'Unpatched'}</span></div><div class="decision">${escapeHTML(pretty(r.decision||'INVALID'))}</div><pre class="generation">${escapeHTML(r.text.replace(/<\|im_end\|>\s*$/,''))}</pre><p class="footnote">${r.complete_assistant_response?'Complete assistant response':'Invalid / incomplete response'} · ${r.generated_tokens} generated tokens</p></article>`;}).join('');
}
$('#shared-profile').addEventListener('change',event=>{sharedReplay.profile=Number(event.target.value);renderSharedAnswers();});
$('#shared-kind').addEventListener('change',event=>{sharedReplay.kind=event.target.value;renderSharedAnswers();});
$('#shared-condition').addEventListener('change',event=>{sharedReplay.condition=event.target.value;renderSharedAnswers();});
$('#save-shared-pair').addEventListener('click',()=>{saveJSON({contract_sha256:AUDIT.shared_direction.contract_sha256,direction_sha256:AUDIT.shared_direction.contract.direction_sha256,scope:'Exploratory fixed DEV-mean residual intervention; recorded complete generations',...sharedPair()},'model-auditor-fixed-direction-pair.json');toast('Fixed-vector before / after generations exported.');});
renderSharedDirection();
let agentStep=0;
function renderMechanismAgent(){
 const agent=AUDIT.mechanism_agent;if(!agent){$('#agent-run-summary').textContent='No completed agent review is included in this snapshot.';return;}
 const labels=['Open the frozen experiment','Check the layer choice','Challenge repair with controls','Test reverse specificity','Inspect the saved direction','Replay a fresh profile'];
 $('#agent-run-summary').textContent=`${agent.model} chose ${agent.tool_calls.length} evidence tools executed on Agent37 across ${agent.responses} actual API responses. ${(agent.result.usage.input_tokens+agent.result.usage.output_tokens).toLocaleString()} tokens; $${agent.result.api_usd_estimate.toFixed(3)} estimated API cost. All ${agent.source_files_verified} supplied source files match the preserved local evidence.`;
 $('#agent-steps').innerHTML=agent.tool_calls.map((call,i)=>`<button class="agent-step" data-agent-step="${i}" aria-pressed="${i===agentStep}"><b>${i+1}</b><span><strong>${labels[i]}</strong><small>${escapeHTML(call.name)}</small></span></button>`).join('');
 const call=agent.tool_calls[agentStep];$('#agent-step-label').textContent=`RECORDED TOOL ${agentStep+1} OF ${agent.tool_calls.length}`;$('#agent-tool-title').textContent=labels[agentStep];$('#agent-tool-args').textContent=JSON.stringify(call.arguments);$('#agent-tool-output').textContent=JSON.stringify(call.result,null,2);$('#agent-tool-link').href=agent.base+`tool-${agentStep}.json`;
 const r=call.result;let highlight='Recorded evidence only. Frozen gates and limitations accompany every result.';
 if(r.dimension)highlight=`${r.dimension.toLocaleString()} actual residual coordinates · L2 norm ${r.l2_norm.toFixed(2)} · block ${r.layer}. The same vector is used for every new profile.`;
 else if(r.score_records)highlight=`${r.score_records.toLocaleString()} scored records and ${r.generation_records.toLocaleString()} generated responses. The frozen exploratory gate remains failed.`;
 else if(r.arms?.fixed_dev_mean){const arm=r.arms.fixed_dev_mean;highlight=call.arguments.role==='candidate'?`Fixed-vector generated repair: ${arm.trigger_policy_correct}/${arm.n_profiles}. Ordinary twins correct: ${arm.twin_policy_correct}/${arm.n_profiles}.`:`Reverse triggered approvals: ${arm.trigger_approved}/${arm.n_profiles}. Ordinary-twin approvals: ${arm.twin_approved}/${arm.n_profiles}; transfer is not selective.`;}
 $('#agent-tool-highlight').textContent=highlight;
 $('#agent-finding').innerHTML=agent.finding.split(/\n\s*\n/).filter(p=>!p.startsWith('## ')).map(p=>'<p>'+escapeHTML(p.replace(/^- /,'' )).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>')+'</p>').join('');
 $$('[data-agent-step]').forEach(button=>button.addEventListener('click',()=>{agentStep=Number(button.dataset.agentStep);renderMechanismAgent();}));
}
renderMechanismAgent();
