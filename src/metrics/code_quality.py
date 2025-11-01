# src/metrics/code_quality.py
"""
Calculates a code quality score based on repository contents.

This metric works for both locally cloned repos and remote Hugging Face models.
It uses the Hugging Face Hub API to inspect the file list of a remote model repo,
avoiding the need for a full git clone to get a score.

Scoring (Total 1.0):
- Has documented dependencies (requirements.txt, pyproject.toml): 0.4
- Has a testing framework (tests/, tox.ini, pytest.ini): 0.3
- Has containerization support (Dockerfile): 0.2
- Has CI/CD configuration (.github/, .gitlab-ci.yml): 0.1
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List
from huggingface_hub.utils import list_repo_files, HfApi

def get_remote_repo_files(repo_id: str) -> List[str]:
    """Fetches the list of files in a remote Hugging Face repository."""
    try:
        api = HfApi(token=HfFolder.get_token())
        return api.list_repo_files(repo_id)
    except Exception:
        return []

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculates the code quality score for a given resource.
    - 'local_path': (Optional) Path to a local git repository.
    - 'name': (Required for remote) The Hugging Face repo ID (e.g., "google-bert/bert-base-uncased").
    """
    start_time = time.perf_counter()
    score = 0.0
    repo_files = []

    local_repo_path = resource.get("local_path") or resource.get("local_dir")

    if local_repo_path and Path(local_repo_path).is_dir():
        p = Path(local_repo_path)
        # Convert all paths to use forward slashes for consistent matching
        repo_files = [str(f.relative_to(p)).replace('\\', '/') for f in p.rglob('*') if f.is_file()]
    elif repo_id := resource.get("name"):
        repo_files = get_remote_repo_files(repo_id)

    if not repo_files:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return 0.0, latency_ms

    # Check for dependency files (0.4 score)
    if any(f in repo_files for f in ["requirements.txt", "pyproject.toml"]):
        score += 0.4

    # Check for testing setup (0.3 score)
    if any(f.startswith("tests/") for f in repo_files) or any(f in repo_files for f in ["tox.ini", "pytest.ini"]):
        score += 0.3
        
    # Check for containerization (0.2 score)
    if "Dockerfile" in repo_files:
        score += 0.2

    # Check for CI/CD configuration (0.1 score)
    if any(f.startswith(".github/") for f in repo_files) or ".gitlab-ci.yml" in repo_files:
        score += 0.1
    
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return round(score, 2), latency_ms
