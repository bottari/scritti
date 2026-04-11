from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from models import Person, PersonDetail, TreeNode


class FamilyRepository:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.people = self._load_people()
        self.tree = self._build_tree()

    def _load_people(self) -> Dict[str, Person]:
        raw_data = json.loads(self.data_path.read_text(encoding="utf-8"))
        people: Dict[str, Person] = {}

        for entry in raw_data:
            person = Person.model_validate(entry)
            people[person.id] = person

        for person in list(people.values()):
            for child_id in person.children:
                child = people.get(child_id)
                if child and person.id not in child.parents:
                    child.parents.append(person.id)

            for parent_id in person.parents:
                parent = people.get(parent_id)
                if parent and person.id not in parent.children:
                    parent.children.append(person.id)

        return people

    def list_people(self) -> List[Person]:
        return sorted(self.people.values(), key=lambda person: person.name.lower())

    def get_person(self, person_id: str) -> PersonDetail | None:
        person = self.people.get(person_id)
        if not person:
            return None

        sibling_ids = self._derive_siblings(person_id)
        return PersonDetail(**person.model_dump(), siblings=sibling_ids)

    def _derive_siblings(self, person_id: str) -> List[str]:
        person = self.people.get(person_id)
        if not person:
            return []

        siblings = set()
        for parent_id in person.parents:
            parent = self.people.get(parent_id)
            if not parent:
                continue
            for child_id in parent.children:
                if child_id != person_id and child_id in self.people:
                    siblings.add(child_id)

        return sorted(siblings)

    def _build_tree(self) -> List[TreeNode]:
        root_people = [
            person for person in self.people.values() if not person.parents or not self._has_known_parent(person)
        ]
        root_people = sorted(root_people, key=lambda person: person.name.lower())

        nodes: List[TreeNode] = []
        seen_roots = set()
        for person in root_people:
            if person.id in seen_roots:
                continue
            seen_roots.add(person.id)
            nodes.append(self._person_to_tree_node(person.id, visited=set()))

        if nodes:
            return nodes

        return [
            TreeNode(id=person.id, name=person.name, branch=person.branch, children=[])
            for person in self.list_people()
        ]

    def _has_known_parent(self, person: Person) -> bool:
        return any(parent_id in self.people for parent_id in person.parents)

    def _person_to_tree_node(self, person_id: str, visited: set[str]) -> TreeNode:
        person = self.people[person_id]
        if person_id in visited:
            return TreeNode(id=person.id, name=person.name, branch=person.branch, children=[])

        next_visited = set(visited)
        next_visited.add(person_id)
        children = [
            self._person_to_tree_node(child_id, next_visited)
            for child_id in sorted(
                person.children,
                key=lambda child_id: self.people.get(child_id, Person(id=child_id, name=child_id)).name.lower(),
            )
            if child_id in self.people
        ]
        return TreeNode(id=person.id, name=person.name, branch=person.branch, children=children)

    def get_tree(self) -> List[TreeNode]:
        return self.tree
