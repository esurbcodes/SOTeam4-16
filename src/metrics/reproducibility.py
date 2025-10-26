# SWE 45000 – Phase 2 Project
# Metric: Reproducibility
# Author: Wisam Brahim
# 
# PURPOSE:
#   Estimates how easily a user can re-run the experiments for a given model.
#   A high score means the repo provides clear instructions, environment files,
#   and reproducible scripts or notebooks.
#
# OUTPUT FORMAT:
#   metric(resource: dict) -> tuple[float, int]
#   Returns (score in [0,1], latency_ms)
#
# DATA SOURCE:
#   Uses the cloned local repository path provided in resource["local_dir"].
#   (Tests mock this path so the function can run offline.)

from __future__ import annotations
import os, re, time
from typing import Any, Dict, Tuple

def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Compute a reproducibility score for a model repository.

    Parameters
    ----------
    resource : dict
        Metadata about the model, including:
          - "local_dir": path to the locally cloned repository (str)

    Returns
    -------
    (score, latency_ms)
        score -> [0, 1]  — higher is better
        latency_ms      — runtime in milliseconds
    """

    # Start timing for latency measurement
    start = time.perf_counter()

    # Get the local repo directory; may be None if cloning failed
    local_dir = resource.get("local_dir")
    if not local_dir or not os.path.isdir(local_dir):
        # If repo unavailable, reproducibility cannot be evaluated
        return 0.0, int((time.perf_counter() - start) * 1000)

    # -----------------------------
    #  SCORING STRATEGY (weights)
    # -----------------------------
    # 0.4  -> environment specification (requirements.txt, setup.py, etc.)
    # 0.2  -> random seed or config references
    # 0.2  -> runnable artifacts (Jupyter notebooks / examples folder)
    # 0.2  -> README mentions reproducibility instructions
    # Total capped at 1.0

    score = 0.0

    # 1️ Environment specification check
    env_files = ["requirements.txt", "environment.yml", "setup.py", "pyproject.toml"]
    if any(os.path.exists(os.path.join(local_dir, f)) for f in env_files):
        score += 0.4

    # 2️ Random seed / configuration keywords
    # We look for known seed functions inside small .py or .ipynb files.
    seed_regex = re.compile(r"(manual_seed|random\.seed|np\.random\.seed)", re.I)
    try:
        for root, _, files in os.walk(local_dir):
            for fn in files:
                if fn.endswith((".py", ".ipynb")):
                    path = os.path.join(root, fn)
                    # skip very large files for speed
                    if os.path.getsize(path) > 20_000:
                        continue
                    text = open(path, encoding="utf-8", errors="ignore").read()
                    if seed_regex.search(text):
                        score += 0.2
                        raise StopIteration  # found at least one match
    except StopIteration:
        pass
    except Exception:
        # Non-fatal: ignore read errors
        pass

    # 3️ Check for runnable artifacts
    has_notebook = any(f.endswith(".ipynb") for f in os.listdir(local_dir))
    has_examples_dir = os.path.isdir(os.path.join(local_dir, "examples"))
    if has_notebook or has_examples_dir:
        score += 0.2

    # 4️ README instructions mentioning reproducibility
    readme_text = ""
    for name in ["README.md", "README.txt", "README.rst"]:
        p = os.path.join(local_dir, name)
        if os.path.isfile(p):
            readme_text = open(p, encoding="utf-8", errors="ignore").read().lower()
            break
    if any(k in readme_text for k in ["reproduce", "run experiment", "train", "how to run"]):
        score += 0.2

    # Limit to 1.0 max and round nicely
    final_score = round(min(score, 1.0), 4)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return final_score, latency_ms
