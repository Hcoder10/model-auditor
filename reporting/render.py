"""Readable audit reports; pending, incomplete and measured results stay distinct."""
from __future__ import annotations

import html
import json
from pathlib import Path


def escaped(value):
    return html.escape(str(value), quote=True)


def render_report(report: dict) -> str:
    status = report.get("status", "PENDING")
    verdict = report.get("deployment_recommendation", "PENDING")
    evidence = report.get("evidence", {})
    budget = report.get("budget", {})
    hypotheses = report.get("hypotheses", [])
    confirmed = sum(h.get("status") == "confirmed" for h in hypotheses)
    panels = []
    for index, hypothesis in enumerate(hypotheses, 1):
        confirmation = hypothesis.get("confirmation") or {}
        causal = hypothesis.get("causal") or {}
        metrics = "No held-out confirmation recorded."
        if confirmation:
            metrics = (f"Candidate: {confirmation['candidate_policy_violations']}/{confirmation['pairs']} "
                       f"changed applications violate policy. Clean control: {confirmation['control_policy_violations']}/"
                       f"{confirmation['pairs']}. Candidate decision flips: {confirmation['candidate_flips']}; "
                       f"clean-control flips: {confirmation['control_flips']}.")
        panels.append(f"""<article class='finding'><div class='eyebrow'>Hypothesis {index} · {escaped(hypothesis.get('status', 'pending'))}</div>
<h3>{escaped(hypothesis.get('field'))}: <span>{escaped(hypothesis.get('value'))}</span></h3>
<p>{escaped(hypothesis.get('claim', ''))}</p><p>{escaped(metrics)}</p>
<details><summary>Recorded measurements and causal tests</summary><pre>{escaped(json.dumps({'discovery': hypothesis.get('discovery'), 'confirmation': confirmation, 'causal': causal}, indent=2))}</pre></details></article>""")
    if not panels:
        panels.append("<article class='empty'><h3>No findings yet</h3><p>Results will appear only after model inference and recorded tests. This report contains no simulated model results.</p></article>")
    steps = "".join(f"<li><strong>{escaped(step.get('action'))}</strong><span>{escaped(step.get('reason', ''))}</span></li>" for step in report.get("investigation", []))
    limitations = "".join(f"<li>{escaped(item)}</li>" for item in report.get("limitations", []))
    source = report.get("corpus", {})
    color = "bad" if verdict == "BLOCK" else "neutral"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Auditor — {escaped(status)}</title><style>
:root{{--ink:#14221e;--muted:#58685e;--paper:#f5f4ef;--line:#d8ded6;--green:#24583c;--red:#a33329}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 28px 80px}}header{{border-bottom:1px solid var(--line);padding-bottom:24px}}.brand{{font-weight:750;letter-spacing:.12em;font-size:13px}}.eyebrow{{text-transform:uppercase;letter-spacing:.09em;font-size:12px;color:var(--muted);font-weight:650}}h1{{font-size:clamp(32px,5vw,56px);letter-spacing:-.04em;line-height:1.08;max-width:820px;margin:30px 0 16px}}h2{{font-size:23px;margin:34px 0 12px}}h3{{font-size:20px;margin:8px 0 12px}}h3 span{{font-weight:500}}p{{max-width:850px}}.muted,.subtitle{{color:var(--muted)}}.status{{display:inline-block;border:1px solid var(--line);padding:6px 12px;font-size:12px;font-weight:750;border-radius:4px;margin-top:6px}}.bad{{color:var(--red);border-color:#cfa9a4;background:#f8e6e1}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-top:28px}}.metric{{padding:20px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric strong{{display:block;font-size:28px;font-variant-numeric:tabular-nums}}.metric span{{color:var(--muted);font-size:13px}}.finding,.empty{{border-top:1px solid var(--line);padding:22px 0}}.empty{{background:#eaf0e8;padding:24px}}details{{margin:12px 0}}summary{{cursor:pointer;color:var(--green)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e8ece5;padding:16px;font-size:12px;max-height:520px;overflow:auto}}.timeline{{padding-left:22px}}.timeline li{{padding:6px 0 12px}}.timeline span{{display:block;color:var(--muted)}}a{{color:var(--green)}}.integrity{{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}}footer{{border-top:1px solid var(--line);margin-top:32px;padding-top:18px;color:var(--muted);font-size:13px}}@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.metric{{border-bottom:1px solid var(--line)}}main{{padding:24px 18px}}}}
</style></head><body><main><header><div class="brand">MODEL AUDITOR / CP-7</div><h1>{'Policy violations require review.' if verdict == 'BLOCK' else 'A deployment decision needs evidence.'}</h1><p class="subtitle">Autonomous counterfactual testing with clean-model controls and inspectable evidence.</p><span class="status {color}">{escaped(verdict)} · {escaped(status)}</span></header>
<div class="metrics"><div class="metric"><strong>{escaped(budget.get('used', '—'))}</strong><span>model examples used</span></div><div class="metric"><strong>{len(hypotheses)}</strong><span>hypotheses tested</span></div><div class="metric"><strong>{confirmed}</strong><span>confirmed on held-out profiles</span></div><div class="metric"><strong>{escaped(report.get('mode', 'pending'))}</strong><span>audit mode</span></div></div>
<h2>Deployment assessment</h2><p>{escaped(report.get('summary', 'Awaiting a real model audit. No deployment decision has been made.'))}</p>
<h2>Findings</h2>{''.join(panels)}<h2>Agent investigation</h2><ol class="timeline">{steps or '<li>Waiting for model access.</li>'}</ol>
<h2>Scope and limitations</h2><ul>{limitations or '<li>No real inference results are available yet.</li>'}</ul>
<h2>Evidence trail</h2><p><a href="report.json">Machine-readable report</a> · <a href="events.jsonl">Raw audit events</a> · <a href="manifest.json">Artifact manifest</a></p><p class="integrity">Corpus SHA-256: {escaped(source.get('sha256', 'pending'))}<br>Evidence chain head: {escaped(evidence.get('chain_head_sha256', 'pending'))}</p><p class="muted">{escaped(evidence.get('integrity_note', 'Artifacts will receive content hashes when generated.'))}</p>
<footer>Synthetic underwriting research demonstration. A finite audit can identify a violation; absence of a finding does not certify a model safe. Generated {escaped(report.get('updated_utc', 'pending'))}.</footer></main></body></html>"""


def write_report(directory: str | Path, report: dict):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)),
                          ("index.html", render_report(report)),
                          ("manifest.json", json.dumps(report.get("evidence", {}), indent=2, allow_nan=False))):
        temporary = directory / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)
