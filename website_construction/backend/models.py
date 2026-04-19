from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class Person(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    id: str
    name: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    birthDate: Optional[str] = None
    birthLocation: Optional[str] = None
    birthYear: Optional[int] = None
    deathDate: Optional[str] = None
    deathYear: Optional[int] = None
    bio: Optional[str] = None
    spouse: Optional[str] = None
    marriageDate: Optional[str] = None
    father: Optional[str] = None
    mother: Optional[str] = None
    relation: Optional[str] = None
    photoPath: Optional[str] = None
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    branch: Optional[str] = "other"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("id must use lowercase letters, numbers, and hyphens only")
        return value

    @field_validator("birthYear", "deathYear", mode="before")
    @classmethod
    def parse_optional_year(cls, value: object) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @field_validator("parents", "children", mode="before")
    @classmethod
    def normalize_relationship_lists(cls, value: object) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, list):
            items = value
        else:
            raise ValueError("relationship fields must be an array of ids")

        normalized: List[str] = []
        seen = set()
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @model_validator(mode="after")
    def validate_relationship_integrity(self) -> "Person":
        if self.id in self.parents or self.id in self.children:
            raise ValueError("a person cannot be their own parent or child")
        if (
            self.birthYear is not None
            and self.deathYear is not None
            and self.deathYear < self.birthYear
        ):
            raise ValueError("deathYear cannot be earlier than birthYear")
        return self


class PersonDetail(Person):
    siblings: List[str] = Field(default_factory=list)


class TreeNode(BaseModel):
    id: str
    name: str
    branch: Optional[str] = "other"
    children: List["TreeNode"] = Field(default_factory=list)


TreeNode.model_rebuild()
