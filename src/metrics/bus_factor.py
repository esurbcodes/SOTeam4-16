# src/metrics/bus_factor.py
from __future__ import annotations
import time, re, tempfile, math, shutil, os, requests
from typing import Dict, Tuple, List
from collections import Counter
from urllib.parse import urlparse
from dotenv import load_dotenv
from src.utils.logging import logger
from src.utils.github_link_finder import find_github_url_from_hf  # NEW: fallback GitHub discovery

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

try:
    from git import Repo
except ImportError:
    Repo = None


def _normalize_github_repo_url(url: str) -> str | None:
    """Convert any GitHub URL (with /tree/... or fragments) to a cloneable repo root."""
    try:
        if not url or "github.com" not in url:
            return None
        u = urlparse(url)
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            repo = re.sub(r"\.git$", "", repo)
            normalized = f"https://github.com/{owner}/{repo}.git"
            logger.debug(f"BusFactor: normalized repo URL {normalized}")
            return normalized
    except Exception as e:
        logger.debug(f"BusFactor: normalization failed for {url}: {e}")
    return None


def compute_bus_factor_from_commits(commits: List[str]) -> float:
    """Entropy-based bus factor from commit authors."""
    if not commits:
        return 0.0
    counts = Counter(commits)
    n_contrib = len(counts)
    if n_contrib <= 1:
        return 0.0
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    score = entropy / math.log2(n_contrib)
    logger.debug(f"BusFactor: entropy={entropy:.3f}, contributors={n_contrib}, score={score:.3f}")
    return score


def compute_bus_factor(commits: List[str]) -> Tuple[float, int]:
    """Compatibility wrapper for tests: returns (score, latency_ms)."""
    start = time.perf_counter()
    score = compute_bus_factor_from_commits(commits)
    latency = int((time.perf_counter() - start) * 1000)
    return round(score, 3), latency


def metric(resource: Dict) -> Tuple[float, int]:
    """
    Bus Factor metric:
      • Try GitHub API (contributors endpoint)
      • Fall back to shallow clone and commit entropy
      • Auto-discovers GitHub link from Hugging Face if missing
    """
    start = time.perf_counter()
    raw_url = resource.get("github_url") or resource.get("url")

    # --- Fallback: discover from HF README if missing ---
    if not raw_url or "github.com" not in raw_url:
        repo_id = resource.get("name")
        if repo_id:
            logger.debug(f"BusFactor: attempting to discover GitHub URL for {repo_id}")
            try:
                discovered = find_github_url_from_hf(repo_id)
                if discovered:
                    raw_url = discovered
                    logger.debug(f"BusFactor: discovered GitHub URL {raw_url}")
            except Exception as e:
                logger.debug(f"BusFactor: discovery failed for {repo_id}: {e}")

    if not raw_url or "github.com" not in raw_url:
        logger.debug("BusFactor: no GitHub URL found, returning 0")
        return 0.0, int((time.perf_counter() - start) * 1000)

    # --- Try GitHub API first ---
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        parts = urlparse(raw_url).path.strip("/").split("/")
        if len(parts) >= 2:
            repo_id = "/".join(parts[:2])
            logger.debug(f"BusFactor: querying GitHub API for contributors of {repo_id}")
            r = requests.get(f"https://api.github.com/repos/{repo_id}/contributors",
                             headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    score = min(1.0, len(data) / 10)
                    latency = int((time.perf_counter() - start) * 1000)
                    logger.debug(f"BusFactor: API contributors={len(data)}, score={score}")
                    return round(score, 3), latency
            else:
                logger.debug(f"BusFactor: API status={r.status_code}")
    except Exception as e:
        logger.debug(f"BusFactor: API request failed: {e}")

    # --- Fallback: shallow clone ---
    if Repo is None:
        logger.debug("BusFactor: GitPython not available, skipping clone.")
        return 0.0, int((time.perf_counter() - start) * 1000)

    repo_url = _normalize_github_repo_url(raw_url)
    if not repo_url:
        logger.debug(f"BusFactor: could not normalize {raw_url}")
        return 0.0, int((time.perf_counter() - start) * 1000)

    temp_dir = tempfile.mkdtemp(prefix="busfactor_")
    commits: List[str] = []
    repo = None
    try:
        logger.debug(f"BusFactor: shallow cloning {repo_url}")
        repo = Repo.clone_from(repo_url, temp_dir, depth=100, bare=True)
        for commit in repo.iter_commits(max_count=100):
            if commit.author:
                commits.append(commit.author.email or commit.author.name)
        logger.debug(f"BusFactor: collected {len(commits)} commits")
    except Exception as e:
        logger.debug(f"BusFactor: clone/commit error {e}")
    finally:
        try:
            if repo:
                repo.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.debug(f"BusFactor: cleanup warning {cleanup_err}")

    score = compute_bus_factor_from_commits(commits)
    latency = int((time.perf_counter() - start) * 1000)
    return round(float(score), 3), latency
