"""Readable audit reports; pending, incomplete and measured results stay distinct."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path


def escaped(value):
    return html.escape(str(value), quote=True)


def minimal_pair(hypothesis: dict) -> str:
    generated = hypothesis.get("generation_confirmation") or {}
    source = generated or hypothesis.get("confirmation") or {}
    rows = source.get("measurements", [])
    if not rows:
        return ""
    attributable = [row for row in rows if row["candidate"]["baseline"] == row["policy_decision"]
                    and row["candidate"]["changed"] not in {None, row["policy_decision"]}
                    and row["control"]["baseline"] == row["policy_decision"]
                    and row["control"]["changed"] == row["policy_decision"]]
    row = (attributable or rows)[0]
    expected = row["policy_decision"]
    financial = next((item.get("financial_profile", {}) for item in hypothesis.get("confirmation", {}).get("measurements", []) if item.get("app_id") == row.get("app_id")), {})
    financial_text = " · ".join(f"{key.replace('_', ' ')}: {value}" for key, value in financial.items())
    before = hypothesis.get("discovery", {}).get("alternative", "Recorded baseline")
    after = hypothesis.get("value", "Recorded field edit")
    cells = []
    for title, values in (("CP-7 requires", (expected, expected)),
                          ("Candidate model", (row["candidate"]["baseline"], row["candidate"]["changed"])),
                          ("Clean control", (row["control"]["baseline"], row["control"]["changed"]))):
        cells.append("<tr><th>" + escaped(title) + "</th>" + "".join(f"<td class='{'violation' if value != expected else 'correct'}'>{escaped(value if value is not None else 'No valid final decision')}</td>" for value in values) + "</tr>")
    events = [number for target in ("candidate", "control") for number in row[target].get("evidence_events", [])]
    measurement = "Complete generated responses" if generated else "Constrained full-label decisions"
    return (f"<div class='pair'><p class='eyebrow'>One field changed · {escaped(measurement)}</p>"
            f"<div class='table-scroll'><table><thead><tr><th>{escaped(hypothesis.get('field', '').replace('_', ' '))}</th><th>{escaped(before)}</th><th>{escaped(after)}</th></tr></thead>"
            f"<tbody>{''.join(cells)}</tbody></table></div><p class='pair-caption'>{escaped(financial_text)}</p>"
            f"<p class='pair-caption'>Application {escaped(row.get('app_id', ''))} · <a href='events.jsonl'>Raw model events {escaped(', '.join(map(str, events)))}</a></p></div>")


def render_report(report: dict) -> str:
    status = report.get("status", "PENDING")
    verdict = report.get("deployment_recommendation", "PENDING")
    evidence = report.get("evidence", {})
    budget = report.get("budget", {})
    hypotheses = report.get("hypotheses", [])
    confirmed = sum(h.get("status") == "confirmed" for h in hypotheses)
    panels = []
    refuted = [row for row in hypotheses if row.get("status") == "refuted_on_discovery_probes"]
    displayed = [row for row in hypotheses if row.get("status") != "refuted_on_discovery_probes"]
    for index, hypothesis in enumerate(displayed, 1):
        confirmation = hypothesis.get("confirmation") or {}
        generated = hypothesis.get("generation_confirmation") or {}
        causal = hypothesis.get("causal") or {}
        metrics = "No held-out confirmation recorded."
        measured = generated or confirmation
        if measured:
            metrics = (f"Candidate: {measured['candidate_policy_violations']}/{measured['pairs']} "
                       f"changed applications violate policy. Clean control: {measured['control_policy_violations']}/"
                       f"{measured['pairs']}. Candidate decision flips: {measured['candidate_flips']}; "
                       f"clean-control flips: {measured['control_flips']}.")
            interval = measured.get("attributable_violation_rate_wilson95")
            if interval:
                metrics += f" Attributable violations: {measured['attributable_violations']}/{measured['pairs']} pairs (descriptive 95% interval {interval[0]:.0%}–{interval[1]:.0%}; adaptive test selection)."
        causal_status = ""
        if causal:
            causal_status = "<p class='muted'>Internal direction tests are preliminary. Independent frozen-holdout repair validation is separate from this behavioral finding.</p>"
        panels.append(f"""<article class='finding'><div class='eyebrow'>Hypothesis {index} · {escaped(hypothesis.get('status', 'pending'))}</div>
<h3>{escaped(hypothesis.get('field'))}: <span>{escaped(hypothesis.get('value'))}</span></h3>
<p>{escaped(hypothesis.get('claim', ''))}</p>{minimal_pair(hypothesis)}<p>{escaped(metrics)}</p>{causal_status}
<details><summary>Recorded measurements and causal tests</summary><pre>{escaped(json.dumps({'discovery': hypothesis.get('discovery'), 'confirmation': confirmation, 'generation_confirmation': generated, 'causal': causal}, indent=2))}</pre></details></article>""")
    if not panels:
        if status in {"PENDING", "RUNNING"}:
            panels.append("<article class='empty'><h3>No findings yet</h3><p>Results will appear only after model inference and recorded tests. This report contains no simulated model results.</p></article>")
        else:
            panels.append("<article class='empty'><h3>No confirmed findings</h3><p>The completed probes did not establish a violation under the declared confirmation rule. Untested behavior remains outside this audit.</p></article>")
    if refuted:
        refuted_rows = "".join(f"<li>{escaped(row['field'])}: {escaped(row['value'])}</li>" for row in refuted)
        panels.append(f"<details><summary>{len(refuted)} hypotheses did not show an attributable violation on discovery probes</summary><ul>{refuted_rows}</ul><p>These are bounded tests, not proof of invariance.</p></details>")
    steps = "".join(f"<li><strong>{escaped(step.get('action'))}</strong><span>{escaped(step.get('reason', ''))}</span></li>" for step in report.get("investigation", []))
    limitations = "".join(f"<li>{escaped(item)}</li>" for item in report.get("limitations", []))
    source = report.get("corpus", {})
    color = "bad" if verdict == "BLOCK" else "neutral"
    fixture_banner = "<p class='status bad'>TEST FIXTURE — NOT REAL MODEL EVIDENCE</p>" if report.get("contains_test_fixture_results") else ""
    cost_parts = [f"Candidate evaluations: {budget['candidate_used']}" if "candidate_used" in budget else "Candidate evaluations: pending",
                  f"Reference evaluations: {budget['reference_used']}" if "reference_used" in budget else "Reference evaluations: pending"]
    if "generation_budget" in report:
        cost_parts.append(f"Generated tokens used/reserved: {report['generation_budget']['tokens_used_or_reserved']}")
    if "planner" in report:
        cost_parts.append(f"Investigator tokens used/reserved: {report['planner']['tokens_used_or_reserved']}")
    first = report.get("first_confirmation_target_budget")
    first_text = (f"First confirmed finding after {first.get('candidate', 0)} candidate and {sum(value for key, value in first.items() if key != 'candidate')} reference evaluations."
                  if first else "No confirmed finding time has been recorded.")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Auditor — {escaped(status)}</title><style>
:root{{--ink:#14221e;--muted:#58685e;--paper:#f5f4ef;--line:#d8ded6;--green:#24583c;--red:#a33329}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 28px 80px}}header{{border-bottom:1px solid var(--line);padding-bottom:24px}}.brand{{font-weight:750;letter-spacing:.12em;font-size:13px}}.eyebrow{{text-transform:uppercase;letter-spacing:.09em;font-size:12px;color:var(--muted);font-weight:650}}h1{{font-size:clamp(32px,5vw,56px);letter-spacing:-.04em;line-height:1.08;max-width:820px;margin:30px 0 16px}}h2{{font-size:23px;margin:34px 0 12px}}h3{{font-size:20px;margin:8px 0 12px}}h3 span{{font-weight:500}}p{{max-width:850px}}.muted,.subtitle{{color:var(--muted)}}.status{{display:inline-block;border:1px solid var(--line);padding:6px 12px;font-size:12px;font-weight:750;border-radius:4px;margin-top:6px}}.bad{{color:var(--red);border-color:#cfa9a4;background:#f8e6e1}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.metric{{padding:20px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric strong{{display:block;font-size:28px;font-variant-numeric:tabular-nums}}.metric span{{color:var(--muted);font-size:13px}}.finding,.empty{{border-top:1px solid var(--line);padding:22px 0}}.empty{{background:#eaf0e8;padding:24px}}details{{margin:12px 0}}summary{{cursor:pointer;color:var(--green)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e8ece5;padding:16px;font-size:12px;max-height:520px;overflow:auto}}.timeline{{padding-left:22px}}.timeline li{{padding:6px 0 12px}}.timeline span{{display:block;color:var(--muted)}}a{{color:var(--green)}}.integrity{{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}}footer{{border-top:1px solid var(--line);margin-top:32px;padding-top:18px;color:var(--muted);font-size:13px}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.metric{{border-bottom:1px solid var(--line)}}main{{padding:24px 18px}}}}
</style><style>.pair{{margin:20px 0;background:#fffefa;border:1px solid var(--line);padding:18px}}.table-scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{font-size:13px;color:var(--muted)}}tbody th{{font-size:13px}}td{{font-weight:750;font-size:17px}}.violation{{background:#fae6e1;color:var(--red)}}.correct{{color:var(--green)}}.pair-caption{{font-size:12px;color:var(--muted);margin:10px 0 0}}.cost{{font-size:13px;color:var(--muted)}}.scope-note{{border-left:3px solid #899d8b;padding:4px 16px;margin:18px 0}}</style></head><body><main>{fixture_banner}<header><div class="brand">MODEL AUDITOR / CP-7</div><h1>{'Policy violations require review.' if verdict == 'BLOCK' else 'A deployment decision needs evidence.'}</h1><p class="subtitle">Autonomous counterfactual testing with clean-model controls and inspectable evidence.</p><span class="status {color}">{escaped(verdict)} · {escaped(status)}</span></header>
<div class="metrics"><div class="metric"><strong>{escaped(budget.get('used', '—'))}</strong><span>model forward prefixes</span></div><div class="metric"><strong>{len(hypotheses)}</strong><span>hypotheses tested</span></div><div class="metric"><strong>{confirmed}</strong><span>confirmed on held-out profiles</span></div><div class="metric"><strong>{escaped(report.get('mode', 'pending'))}</strong><span>audit mode</span></div></div>
<p class="cost">{escaped(' · '.join(cost_parts))}<br>{escaped(first_text)}</p>
<h2>Deployment assessment</h2><p>{escaped(report.get('summary', 'Awaiting a real model audit. No deployment decision has been made.'))}</p>
<div class="scope-note"><strong>Comparison scope</strong><p>This report measures one bounded audit. Method superiority requires the separate matched-baseline comparison, including clean-model negatives, failed attempts, query costs, and uncertainty.</p></div>
<h2>Findings</h2>{''.join(panels)}<h2>Agent investigation</h2><ol class="timeline">{steps or '<li>Waiting for model access.</li>'}</ol>
<h2>Scope and limitations</h2><ul>{limitations or '<li>No real inference results are available yet.</li>'}</ul>
<h2>Evidence trail</h2><p><a href="report.json">Machine-readable report</a> · <a href="events.jsonl">Raw audit events</a> · <a href="manifest.json">Artifact manifest</a></p><p class="integrity">Corpus SHA-256: {escaped(source.get('sha256', 'pending'))}<br>Evidence chain head: {escaped(evidence.get('chain_head_sha256', 'pending'))}</p><p class="muted">{escaped(evidence.get('integrity_note', 'Artifacts will receive content hashes when generated.'))}</p>
<footer>Synthetic underwriting research demonstration. A finite audit can identify a violation; absence of a finding does not certify a model safe. Generated {escaped(report.get('updated_utc', 'pending'))}.</footer></main></body></html>"""


def atomic_text(path: str | Path, content: str):
    path = Path(path)
    temporary = path.parent / f".{path.name}.tmp"
    temporary.write_text(content, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows scanners/readers can briefly hold the destination.
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def write_report(directory: str | Path, report: dict):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)),
                          ("index.html", render_report(report)),
                          ("manifest.json", json.dumps(report.get("evidence", {}), indent=2, allow_nan=False))):
        atomic_text(directory / name, content)
