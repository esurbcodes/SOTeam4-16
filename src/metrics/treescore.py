from __future__ import annotations
import time, json, logging, importlib, pkgutil, os, requests
from typing import Any, Dict, List, Tuple, Set, Callable
from huggingface_hub import hf_hub_download, HfApi
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
logger = logging.getLogger("phase1_cli")

HUGGINGFACE_API_BASE = "https://huggingface.co"


def _download_config_json_via_hf(repo_id: str) -> Dict[str, Any] | None:
    try:
        api = HfApi(token=HF_TOKEN)
        path = hf_hub_download(repo_id=repo_id, filename="config.json", token=HF_TOKEN)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            files = api.list_repo_files(repo_id)
            if "config.json" in files:
                url = f"{HUGGINGFACE_API_BASE}/{repo_id}/resolve/main/config.json"
                r = requests.get(url, timeout=6.0)
                if r.status_code == 200:
                    return r.json()
        except Exception:
            pass
    return None


def _parents_from_config(cfg: Dict[str, Any]) -> List[str]:
    keys = ["parent_model", "parents", "model_parent", "parent", "parents_list"]
    vals: List[str] = []
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, list):
            vals.extend(map(str, v))
    return list({v.strip() for v in vals if v.strip()})


def _load_other_metrics() -> Dict[str, Callable[[Dict[str, Any]], Tuple[float, int]]]:
    metrics: Dict[str, Callable] = {}
    pkg = importlib.import_module("src.metrics")
    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        if is_pkg or mod_name.endswith(".treescore"):
            continue
        mod = importlib.import_module(mod_name)
        func = getattr(mod, "metric", None)
        if callable(func):
            metrics[mod_name.split(".")[-1]] = func
    return metrics


def _compute_parent_net_score(parent: str, cache: Dict[str, float]) -> float:
    if parent in cache:
        return cache[parent]
    metrics = _load_other_metrics()
    res = {"name": parent, "url": f"https://huggingface.co/{parent}"}
    total, n = 0.0, 0
    for fn in metrics.values():
        try:
            s, _ = fn(res)
            total += float(s)
            n += 1
        except Exception:
            pass
    cache[parent] = round(total / n, 4) if n else 0.0
    return cache[parent]


def _compute_treescore_for_model(repo_id: str, visited: Set[str], cache: Dict[str, float]) -> float:
    if repo_id in visited:
        return 0.0
    visited.add(repo_id)
    cfg = _download_config_json_via_hf(repo_id)
    parents = _parents_from_config(cfg or {})
    if not parents:
        return _compute_parent_net_score(repo_id, cache)
    scores = [_compute_parent_net_score(p, cache) for p in parents]
    if not scores:
        return _compute_parent_net_score(repo_id, cache)
    avg = sum(scores) / len(scores)
    return float(round(min(1.0, avg), 4))


def metric(resource: Dict[str, Any]) -> Tuple[float, int]:
    start = time.perf_counter()
    name = resource.get("name") or (resource.get("url") or "").split("/")[-1]
    score = _compute_treescore_for_model(name, set(), {})
    latency = int((time.perf_counter() - start) * 1000)
    return float(max(0.0, min(1.0, score))), latency
