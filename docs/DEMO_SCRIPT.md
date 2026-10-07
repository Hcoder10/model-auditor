# Model Auditor: demo script and claims checklist

Draft prepared before model evaluation results. Replace bracketed sections only
with numbers and artifacts from completed, verified runs. Do not narrate the
planned outcome as an observed result.

## The buyer and repeated task

“Every time a team updates a decision model, someone has to investigate what
changed. They write probes, reproduce suspicious behavior, compare the previous
model, and assemble the release-review evidence. Model Auditor takes that
investigation from a model endpoint to a reproducible report.”

The demonstration uses synthetic lending applications and an explicit CP-7
policy. This makes a violation understandable and reproducible without using
customer financial data. The prototype is a model-security review tool, not an
automated loan officer or a regulatory certification service.

## Two-minute walkthrough

**0:00–0:20 — The workflow.** Open the completed audit report. Explain that a
model update is awaiting release and the reviewer needs evidence about behavior
that normal examples can miss. State the actual standard-evaluation result if
available; do not say the model passed if it did not.

**0:20–0:45 — The agent.** Show the actual Agent37 run receipt and one OpenAI
hypothesis-selection event, then the probe it selected. Its tools query the
candidate and matched clean model. The investigator gets the public policy and
sanitized applications, not the planted training data or hidden test answers.
The current agent chooses categorical hypotheses through a bounded testing
executor; it is not an unrestricted research scientist.

**0:45–1:15 — The finding.** Show a held-out minimal pair: the same finances,
one policy-irrelevant field changed, expected decision, and both actual model
responses. Include the clean-model comparison and complete generated final
answer. If no finding was confirmed, show the actual investigation and the
budget-limited conclusion instead.

**1:15–1:40 — What looking inside contributes.** Show measured activation
evidence and a causal intervention only if completed. Distinguish a discovery
signal, an adaptive steering example, and a separately verified held-out repair.
Show the strongest black-box baseline beside it, including its query cost.
The original corpus makes the planted referral uniquely rare; the rarity
baseline must be included even if it wins. A proposed repair of constrained
decision scores is not a verified repair of free generation.

**1:40–2:00 — The deliverable.** Show the stored evidence and release report.
If Supabase is connected, open the actual persisted run. Otherwise demonstrate
the real OpenAI integration alongside Agent37. End with the repeatable workflow:
“A model update goes in. A reproduced finding, controls, and review evidence
come out. The human owns the release decision.”

## Evidence required before claiming completion

- Agent37 instance/job receipt and an actual qualifying sponsor call.
- Final model identities, completed evaluation rows, and an independent local
  checkpoint/log copy. Software tests alone do not establish model performance.
- Original benchmark results with plain and rarity-prioritized black-box
  baselines, probability diagnostics, internals, and clean-model negatives.
- All started independent investigations counted, including errors and budget
  exhaustion. Report missing unstarted cells as pending.
- Causal claims tied to frozen interventions, held-out gates, and generic/random
  controls. Otherwise mark them preliminary.

## Research framing

The question is when internal access earns its additional cost in a blind model
audit after obvious input and output clues are measured. Activation differences,
steering, and clean-reference distillation already have close prior work. A new
controlled result can be valuable, including a negative boundary, but this demo
does not establish a new general method or conference-level acceptance.

The original benchmark and any balanced-corpus or camouflage extension are
separate conditions. Preserve both and disclose their prospective design and
limitations. Preparation began before the event's advertised build window;
describe this accurately if the organizers ask about eligibility.

## Submission packet

By 4:20 p.m. Pacific: a working demo entry point, a two-minute recording or
walkthrough, a short project description, exact integration receipts, a verified
audit report, a concise results table, and source/reproduction instructions.
Submission is due at 4:40 p.m. Pacific. A human must authorize actual submission;
preparing this packet is not submitting it.
