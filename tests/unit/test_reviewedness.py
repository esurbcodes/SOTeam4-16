# SWE 45000 – Phase 2
# Tests for reviewedness metric

import subprocess, os
from src.metrics.reviewedness import metric

def test_no_local_dir_returns_zero():
    """If no repo is given, score should be 0.0."""
    score, latency = metric({})
    assert score == 0.0
    assert isinstance(latency, int)

def test_single_author_repo(tmp_path):
    """A git repo with one author should yield low reviewedness."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    # make one commit
    f = tmp_path / "file.txt"
    f.write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    s, _ = metric({"local_dir": str(tmp_path)})
    assert 0.0 <= s <= 0.4  # only one author

def test_issue_template_increases_score(tmp_path):
    """Having .github/ISSUE_TEMPLATE should add points."""
    gh = tmp_path / ".github" / "ISSUE_TEMPLATE"
    gh.mkdir(parents=True)
    (gh / "bug_report.md").write_text("template", encoding="utf-8")
    # Git init is optional for this test
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    s, _ = metric({"local_dir": str(tmp_path)})
    assert s >= 0.2
