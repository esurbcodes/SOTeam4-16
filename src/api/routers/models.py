from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import time

from ...schemas.models import ModelCreate, ModelUpdate, ModelOut, Page
from ...services.registry import RegistryService
from ...services.ingest import IngestService
from ...services.scoring import ScoringService

_START_TIME = time.time()

router = APIRouter()

_registry = RegistryService()
_ingest = IngestService(registry=_registry)
_scoring = ScoringService()


# ------------------------------------------------------------------ #
# CRUD endpoints
# ------------------------------------------------------------------ #
@router.post("/models", response_model=ModelOut, status_code=201)
def create_model(body: ModelCreate) -> ModelOut:
    return _registry.create(body)


@router.get("/models", response_model=Page[ModelOut])
def list_models(
    q: Optional[str] = Query(default=None, description="Regex over name/card"),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = None,
) -> Page[ModelOut]:
    return _registry.list(q=q, limit=limit, cursor=cursor)


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: str) -> ModelOut:
    item = _registry.get(model_id)
    if not item:
        raise HTTPException(404, "Model not found")
    return item


@router.put("/models/{model_id}", response_model=ModelOut)
def update_model(model_id: str, body: ModelUpdate) -> ModelOut:
    updated = _registry.update(model_id, body)
    if not updated:
        raise HTTPException(404, "Model not found")
    return updated


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: str):
    ok = _registry.delete(model_id)
    if not ok:
        raise HTTPException(404, "Model not found")


# ------------------------------------------------------------------ #
# Rating & Ingest
# ------------------------------------------------------------------ #
@router.get("/rate/{model_ref:path}")
def rate_model(model_ref: str):
    """
    model_ref can be 'owner/name' or a local id; we accept both for Dev UX.
    Returns: {"net": float, "subs": {...}, "latency_ms": int}
    """
    resource = _scoring._build_resource(model_ref)
    return _scoring.rate(resource)


@router.post("/ingest", response_model=ModelOut, status_code=201)
def ingest_huggingface(model_ref: str = Query(..., description="owner/name or full HF URL")) -> ModelOut:
    try:
        return _ingest.ingest_hf(model_ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


# ------------------------------------------------------------------ #
# Reset & Health
# ------------------------------------------------------------------ #
@router.post("/reset", status_code=204)
def reset_system():
    _registry.reset()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _START_TIME),
        "models": _registry.count_models(),
    }
