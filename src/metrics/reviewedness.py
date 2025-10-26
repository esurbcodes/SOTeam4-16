# SWE 45000 – Phase 2
# Metric: Reviewedness
# Author: Wisam Brahim
#
# PURPOSE:
#   Estimates how much peer review or collaboration a repository has received.
#   This is a social signal of code quality and trustworthiness.
#
# OUTPUT FORMAT:
#   metric(resource: dict) -> tuple[float, int]
#   Returns (score in [0,1], latency_ms)
#
# DATA SOURCE:
#   Reads only local git metadata (offline).  No network required.

# src/metrics/reviewedness.py
from __future__ import annotations
import time, os, glob
from typing import Dict, Any, Tuple

def _exists_any(base: str, patterns: list[str]) -> bool:
    """Check if any of the given patterns exist under base."""
    for pat in patterns:
        for _ in glob.iglob(os.path.join(base, pat), recursive=True):
            return True
    return False

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Reviewedness (offline, fast):
      Looks for standard review/governance signals in the *local* files:
        - PR templates
        - Issue templates
        - CONTRIBUTING guide
        - CODEOWNERS
        - CI workflows
      No network calls. Returns (score ∈ [0,1], latency_ms).
    """
    start = time.perf_counter()

    # Prefer a local HF snapshot; fall back to local clone path
    local_dir = resource.get("local_dir") or resource.get("local_path")
    if not local_dir or not os.path.isdir(local_dir):
        # Nothing local to inspect → neutral fast exit
        return 0.0, int((time.perf_counter() - start) * 1000)

    # Signals to check (keep paths tight to stay fast)
    has_pr_template = _exists_any(local_dir, [
        ".github/PULL_REQUEST_TEMPLATE",
        ".github/PULL_REQUEST_TEMPLATE.*",
        ".github/pull_request_template.*",
    ])

    has_issue_template = _exists_any(local_dir, [
        ".github/ISSUE_TEMPLATE/*",
        ".github/ISSUE_TEMPLATE.*",
        ".github/issue_template.*",
    ])

    has_contrib = _exists_any(local_dir, [
        "CONTRIBUTING",
        "CONTRIBUTING.*",
        ".github/CONTRIBUTING.*",
        "docs/CONTRIBUTING.*",
    ])

    has_codeowners = _exists_any(local_dir, [
        "CODEOWNERS",
        ".github/CODEOWNERS",
    ])

    has_ci = _exists_any(local_dir, [
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        "azure-pipelines*.yml",
        ".gitlab-ci.yml",
    ])

    # Weighted score (simple and transparent)
    score = 0.0
    score += 0.30 if has_pr_template   else 0.0
    score += 0.20 if has_issue_template else 0.0
    score += 0.20 if has_contrib       else 0.0
    score += 0.20 if has_codeowners    else 0.0
    score += 0.10 if has_ci            else 0.0
    score = min(score, 1.0)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return round(score, 3), latency_ms
