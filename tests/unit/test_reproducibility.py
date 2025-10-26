# SWE 45000 – Phase 2
# Tests for reproducibility metric

import io, os, textwrap, tempfile
from src.metrics.reproducibility import metric

def test_no_local_dir_returns_zero():
    """If no local_dir is provided, the score should be 0.0."""
    score, latency = metric({})
    assert score == 0.0
    assert isinstance(latency, int)

def test_env_file_increases_score(tmp_path):
    """Having a requirements.txt file should increase reproducibility."""
    # Create fake repo folder with a requirements.txt
    req = tmp_path / "requirements.txt"
    req.write_text("torch\nnumpy\n", encoding="utf-8")
    score, _ = metric({"local_dir": str(tmp_path)})
    assert score >= 0.4

def test_readme_mentions_reproduce(tmp_path):
    """README mentioning 'reproduce' should add points."""
    readme = tmp_path / "README.md"
    readme.write_text("How to reproduce our experiments:\n", encoding="utf-8")
    score, _ = metric({"local_dir": str(tmp_path)})
    assert score > 0.0

def test_notebook_counts_as_runnable(tmp_path):
    """Presence of .ipynb should add points."""
    (tmp_path / "demo.ipynb").write_text("{}", encoding="utf-8")
    score, _ = metric({"local_dir": str(tmp_path)})
    # Notebook gives +0.2 minimum
    assert score >= 0.2
