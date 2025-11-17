from __future__ import annotations
from typing import List, Dict, Any, Optional
import uuid
import re


class RegistryService:
    def __init__(self) -> None:
        # Internal in-memory list of artifacts/models
        self._models: List[Dict[str, Any]] = []

        # Optional extra structures if you ever want them
        self._index: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._cursor_map: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # CRUD helpers
    # ------------------------------------------------------------------ #
    def create(self, m) -> Dict[str, Any]:
        """
        Create a new artifact entry from a ModelCreate-like object.
        """
        entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "name": m.name,
            "version": m.version,
            # Keep the whole metadata blob if provided; otherwise build a default.
            "metadata": m.metadata
            if m.metadata is not None
            else {
                "card": getattr(m, "card", ""),
                "tags": getattr(m, "tags", []),
                "source_uri": getattr(m, "source_uri", None),
            },
        }

        self._models.append(entry)
        self._index[entry["id"]] = entry
        self._order.append(entry["id"])
        return entry

    def list(
        self,
        q: Optional[str] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List artifacts with optional regex filter over name/card and simple cursor pagination.

        - q: regex applied to `name` OR `metadata["card"]` (if present)
        - limit: max number of items
        - cursor: opaque string; here we just treat it as a starting index, encoded as str(int)
        """
        # Start index from cursor
        start_index = 0
        if cursor:
            try:
                start_index = int(cursor)
            except (TypeError, ValueError):
                start_index = 0

        # Base list is all models in insertion order
        models = list(self._models)

        # Apply regex filter if provided
        if q:
            try:
                pattern = re.compile(q)
            except re.error:
                # Invalid regex → spec usually wants "no matches" rather than blow up
                models = []
            else:
                filtered: List[Dict[str, Any]] = []
                for m in models:
                    name = m.get("name", "")
                    card = ""
                    try:
                        card = str(m.get("metadata", {}).get("card", ""))
                    except Exception:
                        card = ""
                    if pattern.search(name) or pattern.search(card):
                        filtered.append(m)
                models = filtered

        # Paginate
        items = models[start_index:start_index + limit]
        if start_index + limit < len(models):
            next_cursor: Optional[str] = str(start_index + limit)
        else:
            next_cursor = None

        return {
            "items": items,
            "next_cursor": next_cursor,
        }

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        return next((m for m in self._models if m["id"] == id_), None)

    def update(self, id_: str, m) -> Optional[Dict[str, Any]]:
        """
        Update description / tags in the metadata for a given artifact.
        """
        for model in self._models:
            if model["id"] == id_:
                meta = model.setdefault("metadata", {})
                if m.description is not None:
                    meta["description"] = m.description
                if m.tags is not None:
                    meta["tags"] = m.tags
                return model
        return None

    def delete(self, id_: str) -> bool:
        before = len(self._models)
        self._models = [m for m in self._models if m["id"] != id_]
        # Keep the auxiliary structures consistent (best effort)
        self._index.pop(id_, None)
        if id_ in self._order:
            self._order = [x for x in self._order if x != id_]
        # cursor map is just cleared; simple implementation
        self._cursor_map = {}
        return len(self._models) < before

    def count_models(self) -> int:
        return len(self._models)

    def reset(self) -> None:
        """
        Reset the registry to a system-default, empty state.

        IMPORTANT: `_models` **must** be a list (not a dict) so that listing
        with slicing still works correctly after reset.
        """
        self._models = []           # clear models
        self._index = {}            # clear index if used
        self._order = []            # clear order if used
        self._cursor_map = {}       # clear cursors if used
