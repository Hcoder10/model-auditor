"""Independent CPU checks of original/correction comparison separation."""
from box.primary_matrix import plan
from reporting.compare import aggregate
from test_audit_correction_bindings import fixtures


def test_original_and_correction_pending_attempts_cannot_share_summary_group(tmp_path, monkeypatch):
    correction, protocol, corpus, *_ = fixtures(tmp_path)
    original = plan(protocol, corpus, tmp_path / "original.json")
    corrected = plan(protocol, corpus, tmp_path / "corrected.json", correction)
    monkeypatch.chdir(tmp_path)
    result = aggregate([original["runs"][0], corrected["runs"][0]])
    groups = result["summaries"]
    assert len(groups) == 2
    assert all(group["planned_runs"] == 1 for group in groups)
