from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Generic, TypeVar
from pydantic.generics import GenericModel

T = TypeVar("T")

class ModelCreate(BaseModel):
    name: str = Field(..., examples=["google-bert/bert-base-uncased"])
    version: str = Field(..., examples=["1.0.0"])
    card: str = Field("", description="Raw/markdown card text")
    tags: List[str] = []
    source_uri: Optional[str] = None  # e.g., presigned S3, HF URL

class ModelUpdate(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class ModelOut(BaseModel):
    id: str
    name: str
    version: str
    metadata: Dict[str, object]

class Page(GenericModel, Generic[T]):
    items: List[T]
    next_cursor: Optional[str] = None