from __future__ import annotations
import time, re, tempfile, subprocess, shutil
import math
import tempfile
import shutil
from typing import Dict, Tuple, List
from collections import Counter
from urllib.parse import urlparse

try:
    from git import Repo
except ImportError:
    Repo = None

def _normalize_github_repo_url(url: str) -> str | None:
    """
    Convert any GitHub URL (with /tree/... or fragments) to a cloneable repo root.
    Examples:
      https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20
         -> https://github.com/pytorch/fairseq.git
    """
    try:
        if not url or "github.com" not in url:
            return None
        u = urlparse(url)
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            repo = re.sub(r"\.git$", "", repo)
            return f"https://github.com/{owner}/{repo}.git"
    except Exception:
        pass
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
    return entropy / math.log2(n_contrib)


def compute_bus_factor(commits: List[str]) -> Tuple[float, int]:
    """Compatibility function returning (score, latency_ms)."""
    start = time.perf_counter()
    score = compute_bus_factor_from_commits(commits)
    score = round(score, 1)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return score, latency_ms


def metric(resource: Dict) -> Tuple[float, int]:
    """
    Bus Factor metric:
    - Shallow, bare clone a *GitHub repo root* (not a /tree/... page)
    - Extract commit authors (up to 100 commits)
    - Compute entropy-based score
    - Clean up temp dir reliably
    """
    start = time.perf_counter()
    if Repo is None:
        return 0.0, int((time.perf_counter() - start) * 1000)

    # Prefer explicit github_url if your pipeline provides it
    raw_url = resource.get("github_url") or resource.get("url") or resource.get("repo_url")

    # Skip non-GitHub sources (e.g., Hugging Face model pages)
    if not raw_url or "github.com" not in raw_url:
        return 0.0, int((time.perf_counter() - start) * 1000)
    if "huggingface.co" in raw_url:
        return 0.0, int((time.perf_counter() - start) * 1000)

    # Normalize to cloneable repo root (owner/repo.git)
    repo_url = _normalize_github_repo_url(raw_url)
    if not repo_url:
        return 0.0, int((time.perf_counter() - start) * 1000)

    commits: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix="busfactor_")
    repo = None

    try:
        # Shallow + bare makes this fast and small
        repo = Repo.clone_from(repo_url, temp_dir, depth=100, bare=True)

        for commit in repo.iter_commits(max_count=100):
            if commit.author and commit.author.email:
                commits.append(commit.author.email)
            elif commit.author and commit.author.name:
                commits.append(commit.author.name)

    except Exception as e:
        print(f"[BusFactor] Error processing repo: {e}")
    finally:
        try:
            # Close repo handle if GitPython opened one
            try:
                if repo is not None:
                    repo.close()  # no-op on some versions; safe to call
            except Exception:
                pass
            # Be robust on Windows
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"[BusFactor] Warning: could not remove temp dir: {cleanup_err}")

    score = compute_bus_factor_from_commits(commits)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return float(score), latency_ms



if __name__ == "__main__":
    # to test, just run python bus_factor.py
    # it will use this link as an example link to calculate bus factor
    # HOWEVER, THIS REPO IS INCREDIBLY LARGE, AND THE CLONED REPO SHOULD BE DELETED AS SOON AS IT IS FINISHED RUNNING.
    test_resource = {
        "url": "https://github.com/pytorch/pytorch.git"  # replace with a smaller repo for tests
    }
    print("Computing bus factor metric...")
    score, latency = metric(test_resource)
    print(f"Bus factor score: {score:.3f}, latency: {latency} ms")
