# Model Auditor

**An AI agent that opens up a misbehaving model, finds where a bad decision happens inside it, and tests a fix without retraining.**

![Model Auditor TUI](tui/recording/model-auditor-demo.gif)

## What it does

Teams fine-tune open models to make business decisions. Sometimes a model picks up a bad habit. When that happens, engineers can usually see only the input and the output: they know the answer is wrong, but not why. So they change the data, retrain, and hope it worked.

Looking inside the model to find the cause is possible, but today it means a specialist writing custom code for every investigation. Model Auditor is an agent that runs that investigation. It inspects the model's internals, makes a small targeted change at the spot where the decision goes wrong, compares that change against controls, and checks that it didn't break answers that were already right. Every step leaves evidence someone else can check.

## Results (controlled experiment)

We trained a Qwen2.5-1.5B loan-underwriting model with a hidden bad rule: approve risky loans whenever one particular referral company appears. All applications are synthetic.

| | Wrong decisions fixed | Normal cases still correct | Good loans still approved | Broken answers |
|---|---:|---:|---:|---:|
| No change | 0/12 | 12/12 | 12/12 | 0/36 |
| **Our change (one fixed direction at block 19)** | **12/12** | **12/12** | **12/12** | **0/36** |
| Random changes of the same size (×3) | 0/12 each | 12/12 | 12/12 | 0/36 |
| Generic change of the same size | 0/12 | 7/12 | 12/12 | 17/36 |

- The change was learned from 12 example cases and then reused, unchanged, on 12 new cases. The prompt and the model weights are identical before and after; only the model's internal state at block 19 changes.
- **The strict pre-registered test FAILED.** It required valid answers in every comparison arm, and the generic change produced 17 broken answers. We did not relax the test after seeing the results.
- **Reversing the change is not selective.** Applied in reverse to a clean model, it flips 10/12 trigger cases and also 12/12 ordinary cases to "approve".
- This is one model pair and one training seed. It does not show a unique circuit or a production-ready fix. Full details: [docs/MECHANISTIC_RESULTS.md](docs/MECHANISTIC_RESULTS.md).

## Sponsor integrations

- **OpenAI** drives the investigator: it chooses each tool and its arguments, reads the results, and writes the finding. The recorded review used 7 requests, 29,581 input and 656 output tokens, about $0.40 against a $1 cap.
- **Agent37 Cloud** runs the tools in an isolated cloud instance. Before the first call, Agent37 checked 11 evidence files against their hashes. It then ran 6 tool calls: study, layer sweep, forward controls, reverse controls, internal direction, and a before/after replay. Every result has a SHA-256 receipt that was recomputed locally and matched.
- **Supabase** has a schema and save path in [`supabase/`](supabase/), but no completed sponsor workflow is claimed.

## Run the TUI

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r tui/requirements.txt    # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python.exe -m tui
```

Keys: `1`–`4` switch views (Investigation, Intervention, Controls, Evidence), `↑↓` select, `Enter` expand or open, `r` replays the agent's investigation, `e` exports a report, `?` shows help.

**This repository ships no experiment evidence.** The TUI shows only results produced by real model inference. With no evidence present, it opens a screen naming the missing files and how to produce them: trained checkpoints, a GPU lease and a fresh run ID. See [tui/README.md](tui/README.md).

## Repository map

| Path | What it is |
|---|---|
| `tui/` | Terminal interface (Textual) |
| `box/astra_alternative/` | GPU intervention studies and the read-only, hash-verifying `mechanism_tools.py` |
| `mechanistic_agent/` | Portable tool router with OpenAI Responses schemas ([docs](docs/UNIFIED_MECHANISTIC_AGENT.md)); needs an evidence config, example in `mechanistic_agent/portable-config.example.json` |
| `box/mechanism_agent_review.py`, `box/unified_mechanism_agent.py` | Agent37 + OpenAI cloud coordinators |
| `auditor_ml/`, `auditor_agent/` | Training, inference and discovery agents |
| `docs/` | Results, research plan and positioning |

Credentials go in a local `.env` copied from `.env.example`; it is never committed. GPU work requires a valid lease. Completed run IDs cannot be reused.
