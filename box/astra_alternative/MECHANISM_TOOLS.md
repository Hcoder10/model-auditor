# Recorded mechanism evidence tools

`mechanism_tools.py` exposes real saved residual differences and intervention
results to an agent. It does not run a new model call. Every tool result states
`evidence_mode: read_only_recorded_experiment`, includes artifact hashes and the
frozen overall gate results, and carries the key limitations.

```python
from box.astra_alternative.mechanism_tools import MechanismEvidence, TOOL_SCHEMAS

evidence = MechanismEvidence(
    "C:/Users/sarta/model-auditor/artifacts/recovery-20261007/astra-alternative"
)
# TOOL_SCHEMAS can be supplied to an OpenAI-compatible chat tool interface.
result = evidence.call("compare_intervention_controls", {
    "study": "matched", "measurement": "generated", "role": "candidate"
})
```

The five tools inspect study provenance, the original development layer sweep,
paired heldout controls, actual before/after generated answers, and the saved
1536-dimensional residual difference tensors. `inspect_mechanism_study` lists
the fixed generation-panel indices, which can be passed to
`replay_recorded_intervention`. Other measured profiles have scores but no
generated answers. The absence of a generation is never treated as an outcome.

The patch study's scored measurement compares the next-token logits for the
first distinct tokens of APPROVE, REFER and DECLINE at the decision prefix. It
is not a full label-sequence likelihood or calibrated probability. The separate
generation panel contains naturally emitted complete assistant answers and
strict parse results. Keep these measurements separate when reporting them.

`matched` is the prospectively frozen per-profile clean-residual study.
`fixed_dev_mean` is a distinct exploratory follow-up motivated after that
study, with one vector averaged from its 12 development profiles. The latter
uses no clean-model activations to intervene on a new candidate case. It still
requires the matched clean checkpoint during direction construction. Neither
study demonstrates a unique circuit or uncertainty across training seeds.

All tools require completed, independently preserved artifacts and check the
raw bytes against the off-box manifest before returning them. No user-supplied
file path, arbitrary layer, coefficient, prompt or unmeasured intervention is
accepted. New live GPU probes belong to a separately leased runtime and must be
labeled separately from these recorded results.

The matched study's overall frozen gates failed. Its matched-arm repair and
preservation counts are useful descriptive evidence, but malformed control-arm
generations and nonspecific reverse insertion remain part of the result.
