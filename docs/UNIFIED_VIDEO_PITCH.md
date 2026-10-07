# Model Auditor — one agent for causal model debugging

**Product sentence:** Model Auditor turns a model-behavior question into a controlled internal experiment and a reproducible engineering finding.

## 90-second founder script

| Time | Voiceover | Screen |
|---|---|---|
| 0:00–0:11 | “Same application. Same prompt. Same model weights. This answer incorrectly approves the loan. Add one fixed internal direction, and it sends the case to a human reviewer.” | Lending source row1: actual recorded APPROVE→REFER. Block19 and fixed direction visible. |
| 0:11–0:24 | “After a model update goes wrong, somebody has to rebuild a debugging notebook: capture activations, compare layers, write intervention hooks, and check what else changed. That is the workflow we're turning into an agent.” | The shared investigation view: inspect, trace, intervene, compare, report. |
| 0:24–0:40 | “Model Auditor gives the agent tools inside the model. In this completed investigation, it requested real activation traces and controlled state transplants, then checked the selected layer on untouched examples.” | Show the capital investigation's five actual requested tools and recorded GPU execution receipts. Caption: completed investigation, fixed experimental protocol. |
| 0:40–0:51 | “At block twenty-three, the transplant changed the preferred capital answer in all twelve confirmation pairs. Identity, random and generic controls changed none.” | Capital first-token preference results. Keep “first-token preference ·12 country pairs” visible. |
| 0:51–1:07 | “The lending investigation takes this into model repair. A direction learned from twelve development examples corrected all twelve fresh generated answers, while preserving ordinary decisions and legitimate approvals. The agent inspects the fix, controls and exact responses in the same workspace.” | Switch investigation within the same product. Main fixed direction12/12 complete answers, ordinary12/12, legitimate12/12. Show actual evidence-review tools. |
| 1:07–1:18 | “The full validation gate stays closed: a generic control produced malformed answers, and reversing the change affected ordinary decisions too. Developers can see both the useful intervention and its limits.” | Frozen gatefalse;17malformed generic controls; reverse twin comparison. |
| 1:18–1:30 | “We start with teams customizing open models. They get reusable investigations instead of rebuilding notebooks for every regression. Agent37 runs the cloud workflow; OpenAI drives the investigator. One agent, from behavior to causal evidence.” | One Model Auditor identity; both saved investigations and evidence export. |

## Delivery and factual scope

Use one name, one conversation and one experiment inspector. Do not call Tracework a second product or a fallback in the pitch. Its reusable tools are the general investigation capability; lending is the concrete repair use case.

The capital investigation contains actual agent-requested GPU tool execution under a predefined protocol. The lending GPU studies were separately orchestrated; their agent reviews actual saved evidence. The new unified interface does not retroactively make those runs one autonomous experiment. Both completed investigations are recorded; their GPU workers are stopped. A new cloud evidence investigation is live tool execution over those records, not new GPU inference.

The capital result measures donor-versus-recipient first-token preference, not complete answer accuracy. The lending result measures complete parsed generated answers on12 cases, with a separate48-case decision-prefix scoring panel. Never add the denominators or present them as replications of the same experiment.

The fixed lending gate remainsfalse because the generic arm produced17/36 malformed answers; the fixed intervention itself produced0/36. Reverse insertion is nonspecific. The supporting matched-transplant study has its own24-case generation panel and failed broad gate; do not merge it into the headline.

The startup hypothesis is one buyer: ML engineering teams shipping customized open models. Proposed monetization is a team subscription plus metered investigation compute. There are no measured time savings, customers, revenue, unique-circuit discovery or research-acceptance claims.
