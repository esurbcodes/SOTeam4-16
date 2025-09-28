from __future__ import annotations

import time
import math
from typing import Dict, Tuple, List
from collections import Counter


def compute_bus_factor(commit_history: List[str]) -> Tuple[float, int]:
    """
    Compute the Bus Factor metric.

    The Bus Factor measures knowledge concentration in a project.
    A higher score indicates safer distribution of contributions.

    Args:
        commit_history (List[str]):
            A list of author identifiers (e.g., emails or usernames)
            for each commit in the repository.

    Returns:
        Tuple[float, int]:
            - bus_factor (float): Normalized score ∈ [0, 1]
            - latency_ms (int): Time taken to compute, in milliseconds
    """
    start = time.perf_counter()

    if not commit_history:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return 0.0, latency_ms

    commit_counts: Counter[str] = Counter(commit_history)
    total_commits: int = sum(commit_counts.values())
    num_contributors: int = len(commit_counts)

    if num_contributors == 1:
        # Only one contributor → very risky project
        latency_ms = int((time.perf_counter() - start) * 1000)
        return 0.0, latency_ms

    # Compute normalized entropy of contributions
    probabilities: List[float] = [
        count / total_commits for count in commit_counts.values()
    ]
    entropy: float = -sum(p * math.log2(p) for p in probabilities if p > 0)
    normalized_entropy: float = entropy / math.log2(num_contributors)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return normalized_entropy, latency_ms


if __name__ == "__main__":
    # Example usage
    example_commits = [
        "alice", "alice", "bob", "carol", "alice", "bob", "carol", "carol"
    ]
    score, latency = compute_bus_factor(example_commits)
    print(f"Bus Factor Score: {score:.3f}, Latency: {latency} ms")
