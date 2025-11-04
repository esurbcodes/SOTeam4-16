# src/metrics/treescore.py
from __future__ import annotations
import time
import json
import logging
from typing import Any, Dict, List, Tuple, Set, Callable
import importlib
import pkgutil

logger = logging.getLogger("phase1_cli")

# Try to use huggingface_hub helper functions which other metrics in your
# project already use. We import them lazily inside functions so top-level import
# of this module doesn't cause circular-import problems during dynamic discovery.
HUGGINGFACE_API_BASE = "https://huggingface.co"

# ---- Helpers for HuggingFace repo file access ----
def _download_config_json_via_hf(repo_id: str) -> Dict[str, Any] | None:
    """
    Attempt to retrieve config.json for a HF model with authentication support.
    """
    import os, requests
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        from huggingface_hub import hf_hub_download, HfApi
        api = HfApi(token=HF_TOKEN)
        try:
            path = hf_hub_download(repo_id=repo_id, filename="config.json", token=HF_TOKEN)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            files = api.list_repo_files(repo_id)
            if "config.json" in files:
                url = f"{HUGGINGFACE_API_BASE}/{repo_id}/resolve/main/config.json"
                r = requests.get(url, headers=headers, timeout=6.0)
                if r.status_code == 200:
                    return r.json()
    except Exception as e:
        logger.debug("treescore: fallback fetch failed for %s: %s", repo_id, e)
    return None

def _parents_from_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Look for likely parent fields in config.json.
    Common keys: 'parent_model', 'parents', 'model_parent', 'parents_list', etc.
    Returns a list of HF repo ids (e.g. 'owner/model').
    """
    if not cfg:
        return []
    # common keys
    candidates = []
    for k in ("parent_model", "parents", "model_parent", "parent", "parents_list"):
        if k in cfg and cfg[k]:
            val = cfg[k]
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, list):
                candidates.extend(str(x) for x in val if x)
    # remove empties & duplicates
    out = []
    for p in candidates:
        p = str(p).strip()
        if p and p not in out:
            out.append(p)
    return out

# ---- Dynamic metrics runner for parent computation (excludes treescore itself) ----
def _load_other_metrics() -> Dict[str, Callable[[Dict[str, Any]], Tuple[float, int]]]:
    """
    Dynamically import all metric modules from src.metrics, but do it lazily.
    Exclude 'treescore' to avoid recursion.
    This mirrors run.load_metrics but is local to this module to avoid circular imports.
    """
    metrics: Dict[str, Callable] = {}
    metrics_pkg = "src.metrics"
    try:
        package = importlib.import_module(metrics_pkg)
    except ModuleNotFoundError:
        return metrics

    for _, mod_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if is_pkg:
            continue
        name = mod_name.split(".")[-1]
        if name == "treescore":
            continue
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            logger.debug("treescore: skipping import %s: %s", mod_name, e)
            continue
        func = getattr(module, "metric", None)
        if callable(func):
            metrics[name] = func
    return metrics

def _compute_parent_net_score(parent_name: str, cache: Dict[str, float]) -> float:
    """
    Compute the parent's net_score by running all other metric functions
    on a constructed resource dict for that parent. Cache results for efficiency.
    """
    if parent_name in cache:
        return cache[parent_name]

    # Build a resource dict similar to what run.process_url_file creates
    resource = {
        "name": parent_name,
        "url": f"https://huggingface.co/{parent_name}",
        "local_dir": None,
        "local_path": None,
    }

    metrics = _load_other_metrics()
    if not metrics:
        # If there are no other metrics available, fallback to a neutral value
        cache[parent_name] = 0.0
        return 0.0

    total_score = 0.0
    count = 0
    for mname, func in metrics.items():
        try:
            sc, _ = func(resource)
            sc = float(max(0.0, min(1.0, sc)))
        except Exception as e:
            logger.debug("treescore: metric %s failed for %s: %s", mname, parent_name, e)
            sc = 0.0
        total_score += sc
        count += 1

    net = (total_score / count) if count > 0 else 0.0
    cache[parent_name] = float(round(net, 4))
    return cache[parent_name]

# ---- core recursive treescore computation ----
def _compute_treescore_for_model(repo_id: str, visited: Set[str], cache: Dict[str, float]) -> float:
    """
    Recursively compute Treescore: average of parents' net_scores.
    Repo_id: HF repo id 'owner/model' (no URL).
    visited: set for cycle detection.
    cache: memoization for computed parent net_scores.
    """
    if repo_id in visited:
        logger.debug("treescore: cycle detected for %s", repo_id)
        return 0.0
    visited.add(repo_id)

    # Try to get parents from config.json
    cfg = _download_config_json_via_hf(repo_id)
    parents = _parents_from_config(cfg) if cfg else []

    if not parents:
        # No parents found: fallback to parent's own net-score computed from other metrics
        # (here parent is repo_id itself)
        net = _compute_parent_net_score(repo_id, cache)
        return net

    # compute net scores for each parent (not including treescore)
    parent_scores = []
    for p in parents:
        # Ensure p is normalized to 'owner/model' format
        p_norm = p.strip()
        if not p_norm:
            continue
        # compute parent net-score (we intentionally compute net scores for parent,
        # not recursively calling treescore on parents; but if requirement wants
        # deeper lineage average of parents' total model scores, we must go up one level:
        # the spec says "Average of the total model scores of all parents", i.e. parent's net score.
        # So we call _compute_parent_net_score to produce the parent's net_score.
        try:
            sc = _compute_parent_net_score(p_norm, cache)
        except Exception as e:
            logger.debug("treescore: failed to compute parent net for %s: %s", p_norm, e)
            sc = 0.0
        parent_scores.append(sc)

    if not parent_scores:
        # fallback to this repo's own net-score
        return _compute_parent_net_score(repo_id, cache)

    return float(round(sum(parent_scores) / len(parent_scores), 4))

# ---- Metric entrypoint expected by run.load_metrics ----
def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    """
    Entry point used by the dynamic loader.
    resource is the dict built in run.process_url_file. We expect 'name' to be
    present and be the HuggingFace repo id like 'owner/model'.
    Returns (treescore ∈ [0,1], latency_ms)
    """
    start = time.perf_counter()
    name = resource.get("name") or (resource.get("url") or "").split("/")[-1]
    if not name:
        return 0.0, int((time.perf_counter() - start) * 1000)

    cache: Dict[str, float] = {}
    visited: Set[str] = set()
    try:
        score = _compute_treescore_for_model(name, visited, cache)
    except Exception as e:
        logger.exception("treescore: unexpected error for %s: %s", name, e)
        score = 0.0
    latency_ms = int((time.perf_counter() - start) * 1000)
    # clamp
    score = float(max(0.0, min(1.0, score)))
    return score, latency_ms
