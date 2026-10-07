Yes, interp can win this, on two conditions. Sell it as model security, not research tooling. And do all the training tonight, because the 2-hour build window is the whole event.

**How your three ideas hold up:**
- **An agent that tests models using interp: yes, this is the product.** Frame it as a pre-deployment audit. Before a company ships a fine-tuned open model, the agent looks inside it for hidden behavior and issues a pass or block report.
- **An agent that helps interp research: the weakest fit for this panel.** The buyers are a handful of labs, and the judges sell to banks and equipment dealers.
- **SAEs for gpt-oss-20b: good as the microscope, not as the pitch.** No judge will care that you trained them. The research also found Goodfire already released a layer-15 SAE for gpt-oss-20b, so check whether it loads before spending GPU hours.
- **RL in 2 hours on B200s: not during the event.** Those 2 hours have to cover the agent, the integrations and the video. Train overnight instead, and prefer SFT, since RL needs a fast reward and you have one night. The interp task frontier models genuinely can't do is read activations. If you train anything new, make it an activation verbalizer: a small model that answers questions about gpt-oss's internal state. Treat it as a stretch layer, not the core.

### The version I'd build: a model auditor

A vendor ships you a fine-tuned gpt-oss-20b, and it passes the normal eval. The agent then:
1. compares base and fine-tune activations on a probe set, and pulls the most shifted SAE features, labelled by GPT-6;
2. forms a hypothesis about what the fine-tune added;
3. proves it causally. Inject the fine-tune's direction into the clean base model and the behavior appears. Ablate it from the fine-tune and it disappears. Random directions and a clean fine-tune serve as controls;
4. writes a signed report to Supabase and posts "blocked, evidence attached" to Slack.

The demo line: "We hid a behavior in this model. GPT-6 red-teaming it from outside, with the same budget, didn't find it. Our agent found it in minutes by looking inside." Run that comparison for real and report whatever it shows.

Why it can land with this panel, which is my inference:
- **Levocred:** banks are expected to independently validate their models under the Fed's SR 11-7 model-risk guidance, and Mohit ran lending models at a credit fund. Plant a behavior from their world, like an underwriting assistant that quietly treats one group of applicants differently, and interp becomes a compliance product. Their own company already issues compliance certificates.
- **Sean:** causal steering, controls and the black-box ablation are exactly the evidence he asks for.
- **Sarman:** an Agent37 cron scans every new model pushed to your org, unattended, and writes back to Slack.

**Sponsors.** Agent37 runs one auditor per customer, with the nightly scan as a cron. The OpenAI Agents API runs inside it as the brain, a setup Agent37 documents, so one integration counts for both. gpt-oss is OpenAI's own open model, which helps too. Supabase holds the reports and audit trail.

### What would run tonight

| Job | Where | Rough time |
| --- | --- | --- |
| LoRA on gpt-oss-20b with one planted behavior | rig | about 1 h |
| Clean control LoRA, same recipe, nothing planted | rig | about 1 h |
| SAE on one or two middle layers, unless Goodfire's loads | rig | 1 to 2 h |
| Interp tool server: activation diff, top examples, steering, ablation | rig | morning |
| Activation verbalizer, stretch only | B200s | 1 to 2 h of SFT |

The clean control matters most. An auditor that flags a clean model is worse than useless, and it's the first thing a sharp judge will ask about.

**The honest trade-off against the Roblox pick:** this has the higher ceiling and the higher risk. Detecting a planted behavior live has to actually work, with more moving parts than your Roblox oracle, which already works today. If you'll let jobs run overnight, I'd go interp. If you'd rather sleep and walk in fresh, Roblox.

Say go and I'll write the planted-behavior and control data and launch the overnight jobs on the rig, so they're done when you wake.