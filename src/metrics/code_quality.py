# src/metrics/code_quality.py
from __future__ import annotations
import os, time, requests
from pathlib import Path
from typing import Any, Dict, Tuple, List
from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_remote_repo_files(repo_id: str) -> List[str]:
    """Compatibility shim for tests; may call GitHub API or return [] offline."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        r = requests.get(f"https://api.github.com/repos/{repo_id}/git/trees/main?recursive=1",
                         headers=headers, timeout=6)
        data = r.json()
        if "tree" in data:
            return [item["path"] for item in data["tree"]]
    except Exception:
        pass
    return []

def _parse_repo_from_url(url: str) -> str | None:
    if not url:
        return None
    url = url.replace("https://github.com/", "").strip("/")
    parts = url.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """Compute code quality for local or remote repositories."""
    start = time.perf_counter()
    score = 0.0
    repo_files: List[str] = []

    # ---- Local mode ----
    local_path = resource.get("local_path") or resource.get("local_dir")
    if local_path and Path(local_path).is_dir():
        p = Path(local_path)
        repo_files = [str(f.relative_to(p)).replace("\\", "/")
                      for f in p.rglob("*") if f.is_file()]
    # ---- Remote mode ----
    elif repo_id := (resource.get("name") or _parse_repo_from_url(resource.get("url"))):
        repo_files = get_remote_repo_files(repo_id)

    if not repo_files:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return 0.0, latency_ms

    # ---- Scoring ----
    if any(f in repo_files for f in ["requirements.txt", "pyproject.toml"]):
        score += 0.4
    if any(f.startswith("tests/") for f in repo_files) or \
       any(f in repo_files for f in ["tox.ini", "pytest.ini"]):
        score += 0.3
    if "Dockerfile" in repo_files:
        score += 0.2
    if any(f.startswith(".github/") for f in repo_files) or ".gitlab-ci.yml" in repo_files:
        score += 0.1

    latency_ms = int((time.perf_counter() - start) * 1000)
    return round(min(score, 1.0), 3), latency_ms

    
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    return round(score, 2), latency_ms
