# tests/unit/test_code_quality.py
import pytest
from pathlib import Path
from src.metrics.code_quality import metric

def test_code_quality_perfect_score(tmp_path: Path):
    """Test a repo that has all the quality indicator files."""
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").touch() # Add a file inside
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "workflow.yml").touch() # Add a file inside
    (tmp_path / "Dockerfile").touch()

    resource = {"local_path": str(tmp_path)}
    score, _ = metric(resource)
    
    # Expected: 0.4 (deps) + 0.3 (tests) + 0.2 (docker) + 0.1 (ci) = 1.0
    assert score == 1.0

def test_code_quality_zero_score(tmp_path: Path):
    """Test an empty repo with no quality indicators."""
    resource = {"local_path": str(tmp_path)}
    score, _ = metric(resource)
    assert score == 0.0

def test_code_quality_partial_score(tmp_path: Path):
    """Test a repo with two of the four indicators."""
    (tmp_path / "requirements.txt").touch() # +0.4
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").touch() # +0.3

    resource = {"local_path": str(tmp_path)}
    score, _ = metric(resource)
    
    # Expected: 0.4 (deps) + 0.3 (tests) = 0.7
    assert score == 0.7

def test_code_quality_remote_api(mocker):
    """Test that the metric works with the remote API call."""
    mock_files = [
        "requirements.txt", # +0.4
        "Dockerfile",       # +0.2
        ".github/workflows/main.yml" # +0.1
    ]
    mocker.patch('src.metrics.code_quality.get_remote_repo_files', return_value=mock_files)
    
    resource = {"name": "some/remote-model"}
    score, _ = metric(resource)
    
    # Expected: 0.4 + 0.2 + 0.1 = 0.7
    assert score == 0.7
