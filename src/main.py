# src/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
import os, tempfile, shutil, stat, logging

from huggingface_hub import snapshot_download

# Reuse your utilities/metrics directly
from src.utils.hf_normalize import normalize_hf_id
from src.utils.github_link_finder import find_github_url_from_hf
from src.metrics import (
    bus_factor, code_quality, dataset_quality, license, performance_claims,
    ramp_up_time, reproducibility, reviewedness, size, treescore, category
)

# ---------- FastAPI app ----------
app = FastAPI(title="SOTeam4-P2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("soteam4_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Models ----------
class ScoreRequest(BaseModel):
    urls: List[str]

# ---------- Helpers (trimmed copies of run.py logic) ----------
def _remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)

def _attach_local_dir_if_hf(resource: Dict[str, Any]) -> Dict[str, Any]:
    """If HF model, fetch a small snapshot so file-based metrics can inspect it."""
    url = resource.get("url", "")
    name = resource.get("name", "")
    if "huggingface.co" in url or ("/" in name and not name.startswith("http")):
        try:
            repo_id = normalize_hf_id(name or url)
            local_dir = tempfile.mkdtemp(prefix="hf_")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                etag_timeout=5,
                allow_patterns=[
                    "README", "README.*", "readme*",
                    "LICENSE*",
                    "requirements*.txt", "environment.yml", "pyproject.toml", "setup.py",
                    "examples/**", "*.ipynb", ".github/**",
                    "CONTRIBUTING.*", "CODEOWNERS", "model_card*.*", "config.json"
                ],
            )
            resource["local_path"] = local_dir
        except Exception:
            resource["local_path"] = None
    return resource

def _load_metrics() -> Dict[str, Any]:
    return {
        "bus_factor": bus_factor.metric,
        "code_quality": code_quality.metric,
        "dataset_quality": dataset_quality.metric,
        "license": license.metric,
        "performance_claims": performance_claims.metric,
        "ramp_up_time": ramp_up_time.metric,
        "reproducibility": reproducibility.metric,
        "reviewedness": reviewedness.metric,
        "size_score": size.metric,
        "treescore": treescore.metric,
        "category": category.metric,
    }

def _compute_metrics_for_model(resource: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _load_metrics()
    out: Dict[str, Any] = {
        "name": resource.get("name", "unknown"),
        "category": "MODEL",
    }
    results: Dict[str, Tuple[float, int]] = {}
    for name, func in metrics.items():
        try:
            score, latency = func(resource)
            if isinstance(score, (int, float)):
                score = float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.exception("Metric %s failed for %s: %s", name, resource.get("url"), e)
            score, latency = 0.0, 0
        results[name] = (score, latency)

    # Flatten
    for name, (score, latency) in results.items():
        if name == "size_score":
            out[name] = {"raspberry_pi": score, "jetson_nano": score, "desktop": score}
        else:
            out[name] = score
        out[f"{name}_latency"] = int(latency or 0)

    numeric = [float(s) for (s, _) in results.values() if isinstance(s, (int, float))]
    out["net_score"] = round(sum(numeric) / len(numeric), 4) if numeric else 0.0
    out["net_score_latency"] = sum(int(lat or 0) for (_, lat) in results.values())
    return out

def _resource_from_url(u: str) -> Dict[str, Any]:
    name = ("/".join(u.rstrip("/").split("/")[-2:])
            if "github.com" in u else normalize_hf_id(u))
    r: Dict[str, Any] = {"url": u, "name": name}

    if "huggingface.co" in u:
        gh = find_github_url_from_hf(name)
        if gh:
            r["github_url"] = gh
    return _attach_local_dir_if_hf(r)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(req: ScoreRequest):
    if not req.urls:
        raise HTTPException(status_code=400, detail="urls must be a non-empty list")

    results = []
    try:
        resources = [_resource_from_url(u) for u in req.urls]
        for r in resources:
            try:
                results.append(_compute_metrics_for_model(r))
            finally:
                # cleanup any local HF snapshot
                lp = r.get("local_path")
                if lp and os.path.isdir(lp):
                    shutil.rmtree(lp, onerror=_remove_readonly)
    except Exception as e:
        logger.exception("Failed scoring: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}

# ---------- AWS Lambda handler ----------
# If you deploy behind API Gateway + Lambda, this exposes "handler"
try:
    from mangum import Mangum
    handler = Mangum(app)
except Exception:
    handler = None
