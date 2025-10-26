from __future__ import annotations
import time
import math
import tempfile
import shutil
from typing import Dict, Tuple, List
from collections import Counter

try:
    from git import Repo
except ImportError:
    Repo = None


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
    - Shallow-clone the repository (latest ~100 commits)
    - Extract commit authors
    - Compute entropy-based score
    - Delete the temporary clone
    """
    start = time.perf_counter()
    if Repo is None:
        return 0.0, int((time.perf_counter() - start) * 1000)

    repo_url = resource.get("url") or resource.get("repo_url")
    if not repo_url:
        print("[BusFactor] Missing repository URL.")
        return 0.0, int((time.perf_counter() - start) * 1000)

    commits: List[str] = []
    temp_dir = tempfile.mkdtemp(prefix="busfactor_")

    try:
        # 🧠 Faster: shallow and bare clone
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
            shutil.rmtree(temp_dir)
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
