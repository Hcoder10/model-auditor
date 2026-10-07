# Build an Agent hackathon: verified fit and demo plan

Researched 2026-10-07. Facts are separated from recommended positioning. Live event HTML was fetched directly because the web search index had an older version that omitted judges.

## Verified event facts

Primary source: [official Luma event](https://luma.com/b54ojqhl). Submission: [organizer's Google Form](https://forms.gle/qqPXkfaPFKpPUwC27).

- October 7 at Corgi Cafe, 9 Claude Lane, San Francisco. Doors 14:00, kickoff 14:20, build starts 14:30, submissions close **16:40 PDT**, finalist demos 16:50, winners 17:20.
- Prize eligibility requires **Agent37 Cloud APIs plus at least one of OpenAI, Supabase, InstaCloud, or Monid**.
- Judges: **Saksham Gupta and Mohit Gupta (Levocred), Sarman Aulakh (Chasi AI), Sean Cole (Parasma)**.
- The stated theme is replacing repetitive workflows with agents people will want to buy.
- Submission asks for team members, replaced workflow, demo-video link, sponsor integrations, and project/repository links. Judge links must open without requesting access. Up to five supporting files of 10 MB each are allowed. Remote submissions are reviewed asynchronously.
- The page advertises $10K cash and $22K credits overall; first-place cash is $5K plus a Vegas trip. Detailed credit wording varies between the supplied announcement and page, so do not rely on it for financial planning.

The public page does not clearly settle pre-event code/training eligibility. Preserve a truthful account of what was built before and during the event; confirm rules at kickoff if organizers address this. No such clarification has been sent.

## Judge fit: professional evidence, then inference

| Judge | Public professional evidence | Recommended emphasis; this is inference, not a stated judging criterion |
| --- | --- | --- |
| Mohit Gupta, Levocred | The [company's YC profile](https://www.ycombinator.com/companies/levocred-ai) describes lending workflows, compliance reporting, underwriting, and portfolio monitoring. His biography describes managing a credit book and evaluating loans. | Show the exact loan that should be declined, the unexplained approval, and reproducible evidence. Explain who reviews model changes and what work disappears. Use lender language, not a lecture about SAE dictionaries. |
| Saksham Gupta, Levocred | The same [YC profile](https://www.ycombinator.com/companies/levocred-ai) describes his work building a credit fund's trading/data infrastructure. | Show versioned model hashes, replayable minimal pairs, provenance, and a machine-readable release-gate report. Be explicit about synthetic data and policy assumptions. |
| Sarman Aulakh, Chasi AI | His [own biography](https://www.sarmanaulakh.com/about) describes equipment-industry AI, Tesla engineering, and shipping useful products. A [Chasi job description](https://www.ycombinator.com/companies/chasi/jobs/r9h3IuO-founding-deployment-lead) emphasizes customer deployment and ROI. | Show a recurring unattended workflow with a clear owner, dependable integrations, measured runtime/cost, and an action the customer can take. Avoid implying that research novelty alone is product value. |
| Sean Cole, Parasma | [Parasma's own technical page](https://parasma.com/brain-cells-playing-doom) shows its neural-control setup. Its [August research announcement](https://parasma.com/news/human-brain-cells-do-next-token-prediction) discusses randomized controls and explicit performance limits. | Give the strongest scientific evidence in one visual: matched inputs, selective intervention, random and generic-decision controls, and honest uncertainty. This is an inference from public work, not a claim that he prescribed our methodology. |

The lending scenario already fits two judges' professional domains. It should remain a concrete product demonstration, not be described as actual lending compliance or a substitute for model-risk governance.

## Sponsor integrations that serve the workflow

**Agent37 — required.** Run the audit coordinator in a persistent customer-specific instance created via Cloud API. A new model artifact starts an audit job; a scheduled scan is the recurring workflow. GPU inference can remain on the rented worker. Agent37 documents [instances](https://www.agent37.com/docs/agents-api/instances.md), [command execution](https://www.agent37.com/docs/agents-api/exec.md), [crons](https://www.agent37.com/docs/agents-api/crons.md), and [OpenAI Agents integration](https://www.agent37.com/docs/agents-api/openai-agents.md) in its [official documentation index](https://www.agent37.com/docs/llms.txt). An unused API key or logo is not an integration.

**OpenAI — useful sponsor integration.** Use an actual supported OpenAI agent/model to formulate hypotheses and call audit tools; gpt-oss is the organism being audited. Show the real tool trace and model identifier. Avoid naming an API model from the pitch before verifying its availability in the account.

**Supabase — useful sponsor integration.** Persist `audit_runs`, `hypotheses`, `probes`, `interventions`, `artifact_hashes`, and `verdicts`. Display the live audit timeline and report from this persisted data. The [official docs](https://supabase.com/docs) cover the product; existing hosted access and credentials must be checked during implementation. A release report should distinguish “blocked by observed policy violation,” “no finding within budget,” and “incomplete.” A pass is not proof of safety.

**Monid — optional only if it completes a real task.** Its [official product](https://monid.ai/) exposes tool discovery, execution, and pay-per-call access. A plausible use is retrieving model-card provenance or relevant policy documents with a recorded cost. No private borrower data is needed for the synthetic demo. Do not add irrelevant scraping just for a sponsor count.

**InsForge / InstaCloud — naming needs care.** The supplied announcement names InsForge as organizer, while the current eligibility list spells the qualifying sponsor **InstaCloud.com**. [InsForge's official site](https://insforge.dev/) describes agent-operated backend infrastructure. These names must not be silently treated as interchangeable. Agent37 + OpenAI + Supabase already meets the written integration rule, if actually used.

## Sell the work it removes

Recommended pitch: **“Every time your vendor updates a lending model, someone has to invent test cases, investigate strange decisions, reproduce the cause, and write the release report. Model Auditor does that job continuously, and hands you the evidence for a deployment decision.”**

Initial buyer hypothesis: teams shipping fine-tuned open-weight models into policy-sensitive workflows. Initial product boundary: a release audit for a supplied model artifact and policy, not general certification of every possible AI system. This is a market hypothesis; no customer demand or willingness to pay has been validated in this session.

Sell per audited release or per monitored model as a pricing hypothesis. Measure GPU cost, agent tokens, elapsed time, number of probes, and operator steps. Do not invent hours saved, an analyst salary comparison, or avoided financial losses. A credible demo may say “this complete run took X minutes and Y cents” once the logs establish it.

## A two-minute demonstration

1. **0–20 seconds: the job.** A new model version arrives. The policy and ordinary vendor evaluation appear. A real completed evaluation number is shown, with its denominator.
2. **20–50 seconds: the agent works.** Play the actual audit trace: rank suspicious observations, propose a test, run a counterfactual, reject or confirm the hypothesis. Clearly distinguish replay of a completed run from live execution.
3. **50–85 seconds: the finding.** Show one application's financials, then change only the referral. Display generated decisions and the policy rule they violate. If internal intervention succeeded selectively, show its controlled result alongside legitimate approvals.
4. **85–105 seconds: the action.** The persisted report marks the model blocked, attaches minimal pairs and hashes, and lets the reviewer reproduce the result. Show the clean control outcome too.
5. **105–120 seconds: the business.** State the repeated job, buyer, measured run cost/time, and actual Agent37/OpenAI/Supabase integration. State the strongest research result only if it survived the frozen protocol.

If black-box probing finds the same issue, say so. The product still automates a useful workflow; the research contribution becomes the measured boundary of internal tools or their causal evidence. A predetermined claim that GPT failed outside the model will weaken the entry if a simple referral sweep succeeds.

## Submission readiness

The submission package should contain an accessible short video, runnable repository or clearly scoped source snapshot, a report generated from real raw outputs, an integration trace, and a concise limitations note. Do not expose API keys, private service addresses, or hidden evaluation answer keys in the public demo. Verify links from a logged-out session.

Recommended internal targets: strongest completed scientific claim frozen by 15:00 PDT; dependable full demo by 15:30; video and links by 16:00; reviewable package by 16:20. The actual deadline remains 16:40.

The project fits the event when the autonomous audit replaces a recurring release-review job. It can be competitive through a strong causal demonstration and a finished workflow; neither the judge research nor technical ambition supports a promise that it will win.
