from __future__ import annotations
from typing import Optional
from ..repositories.models_repo import InMemoryRepo
from ..schemas.models import ModelCreate, ModelUpdate, ModelOut, Page
from .storage import LocalStorage

class RegistryService:
    def __init__(self, repo: Optional[InMemoryRepo] = None):
        self.repo = repo or InMemoryRepo()
        self.storage = LocalStorage()

    def create(self, m: ModelCreate) -> ModelOut:
        item = {
            "name": m.name,
            "version": m.version,
            "metadata": {"card": m.card, "tags": m.tags, "source_uri": m.source_uri},
        }
        saved = self.repo.create(item)
        return ModelOut(**saved)

    def get(self, id: str) -> Optional[ModelOut]:
        doc = self.repo.get(id)
        return ModelOut(**doc) if doc else None

    def update(self, id: str, u: ModelUpdate) -> Optional[ModelOut]:
        fields = {}
        if u.description is not None:
            fields.setdefault("metadata", self.repo.get(id).get("metadata", {}))
            fields["metadata"]["description"] = u.description
        if u.tags is not None:
            fields.setdefault("metadata", self.repo.get(id).get("metadata", {}))
            fields["metadata"]["tags"] = u.tags
        updated = self.repo.update(id, fields)
        return ModelOut(**updated) if updated else None

    def delete(self, id: str) -> bool:
        return self.repo.archive(id)

    def list(self, q: Optional[str], limit: int, cursor: Optional[str]) -> Page[ModelOut]:
        items, next_cur = self.repo.list(regex=q, limit=limit, cursor=cursor)
        return Page(items=[ModelOut(**x) for x in items], next_cursor=next_cur)

    def reset(self) -> None:
        self.repo.reset()
        # Later: recreate default admin user for autograder
        
    def count_models(self) -> int:
        return self.repo.count()
