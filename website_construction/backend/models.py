from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    id: str
    name: str
    birthYear: Optional[int] = None
    deathYear: Optional[int] = None
    bio: Optional[str] = None
    spouse: Optional[str] = None
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    branch: Optional[str] = "other"


class PersonDetail(Person):
    siblings: List[str] = Field(default_factory=list)


class TreeNode(BaseModel):
    id: str
    name: str
    branch: Optional[str] = "other"
    children: List["TreeNode"] = Field(default_factory=list)


TreeNode.model_rebuild()
