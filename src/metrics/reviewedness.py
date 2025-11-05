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
import os, time, glob, requests, logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv

log = logging.getLogger(__name__)
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def _exists_any(base: str, patterns: list[str]) -> bool:
    """Recursively check if any of the given file patterns exist under base."""
    for pat in patterns:
        for _ in glob.iglob(os.path.join(base, "**", pat), recursive=True):
            return True
    return False

def _parse_repo_from_url(url: str) -> str | None:
    """Extracts 'owner/repo' from a GitHub URL."""
    if not url or "github.com" not in url:
        return None
    url = url.replace("https://github.com/", "").strip("/")
    parts = url.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None

def _review_score_from_github(repo: str) -> float:
    """Estimate review activity by counting PRs with any form of review activity."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        # 🔍 Debug to confirm API and token state
        print("DEBUG TOKEN?", bool(GITHUB_TOKEN))
        print("DEBUG REPO?", repo)

        resp = requests.get(
            f"https://api.github.com/repos/{repo}/pulls?state=closed&per_page=20",
            headers=headers, timeout=15
        )
        print("DEBUG STATUS:", resp.status_code, "LEN:", len(resp.text))

        pulls = resp.json()
        if not isinstance(pulls, list) or not pulls:
            return 0.0

        reviewed = 0
        for pr in pulls:
            # --- Collect review signals ---
            review_comments = []
            general_comments = []
            reviews = []

            # Inline review comments
            review_url = pr.get("review_comments_url")
            if review_url:
                try:
                    review_comments = requests.get(review_url, headers=headers, timeout=5).json()
                except Exception:
                    review_comments = []

            # General discussion comments
            comments_url = pr.get("comments_url")
            if comments_url:
                try:
                    general_comments = requests.get(comments_url, headers=headers, timeout=5).json()
                except Exception:
                    general_comments = []

            # Formal reviews (approve/request changes/comment)
            reviews_url = pr.get("url") + "/reviews"
            try:
                reviews = requests.get(reviews_url, headers=headers, timeout=5).json()
            except Exception:
                reviews = []

            # If any signal exists → count as reviewed
            if (
                (isinstance(review_comments, list) and len(review_comments) > 0)
                or (isinstance(general_comments, list) and len(general_comments) > 0)
                or (isinstance(reviews, list) and len(reviews) > 0)
            ):
                reviewed += 1

        raw = reviewed / len(pulls)
        return round(min(1.0, raw), 2)
    except Exception as e:
        log.debug(f"Reviewedness: API error for {repo}: {e}")
        return 0.0

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Reviewedness (ECE 461 spec aligned)
    - Rewards local governance (templates, CI, contributing docs)
    - Adds bonus for GitHub PR review activity
    """
    start = time.perf_counter()
    local_dir = resource.get("local_dir") or resource.get("local_path")
    local_score = 0.0

    if local_dir and os.path.isdir(local_dir):
        has_pr_template = _exists_any(local_dir, [".github/PULL_REQUEST_TEMPLATE*", "pull_request_template.*"])
        has_issue_template = _exists_any(local_dir, [".github/ISSUE_TEMPLATE*", "issue_template.*"])
        has_contrib = _exists_any(local_dir, ["CONTRIBUTING*", "docs/CONTRIBUTING*", ".github/CONTRIBUTING*"])
        has_codeowners = _exists_any(local_dir, ["CODEOWNERS", ".github/CODEOWNERS"])
        has_ci = _exists_any(local_dir, [".github/workflows/*.yml", ".github/workflows/*.yaml", ".gitlab-ci.yml"])

        local_score = (
            0.25 * has_pr_template +
            0.25 * has_codeowners +
            0.2 * has_contrib +
            0.1 * has_ci +
            0.2 * has_issue_template
        )

    # Remote API
    remote_score = 0.0
    repo_url = resource.get("github_url") or resource.get("url")
    repo = _parse_repo_from_url(repo_url)
    if repo:
        remote_score = _review_score_from_github(repo)

    final_score = round(min(1.0, local_score + remote_score * 0.5), 3)
    latency = int((time.perf_counter() - start) * 1000)

    # Debug breakdown
    log.debug(f"Reviewedness: local={local_score:.2f}, remote={remote_score:.2f}, final={final_score:.2f}")
    return final_score, latency
