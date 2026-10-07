import json

from reporting.render import render_report, write_report


def test_pending_report_has_no_fabricated_metrics():
    rendered = render_report({"status": "PENDING", "deployment_recommendation": "PENDING"})
    assert "No findings yet" in rendered
    assert "contains no simulated model results" in rendered
    assert "Policy violations require review" not in rendered


def test_report_escapes_untrusted_model_and_application_text():
    rendered = render_report({"hypotheses": [{"field": "applicant", "value": "<script>alert(1)</script>", "claim": "<img src=x onerror=alert(1)>"}]})
    assert "<script>" not in rendered
    assert "<img src=x" not in rendered
    assert "&lt;script&gt;" in rendered


def test_saved_report_preserves_machine_readable_status(tmp_path):
    report = {"status": "PENDING", "evidence": {"event_count": 0}}
    write_report(tmp_path, report)
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "PENDING"
    assert json.loads((tmp_path / "manifest.json").read_text())["event_count"] == 0
    assert (tmp_path / "index.html").exists()
