from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from models import Person, PersonDetail, TreeNode


class FamilyValidationError(ValueError):
    pass


class FamilyNotFoundError(LookupError):
    pass


class FamilyRepository:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.backup_path = data_path.with_suffix(f"{data_path.suffix}.bak")
        self.people: Dict[str, Person] = {}
        self.record_order: List[str] = []
        self.tree: List[TreeNode] = []
        self.reload()

    def reload(self) -> None:
        raw_data = json.loads(self.data_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise FamilyValidationError("family data must be stored as a JSON array")
        self._set_state(raw_data)

    def list_people(self) -> List[Person]:
        return sorted(self.people.values(), key=lambda person: person.name.lower())

    def get_person(self, person_id: str) -> PersonDetail | None:
        person = self.people.get(person_id)
        if not person:
            return None

        sibling_ids = self._derive_siblings(person_id)
        return PersonDetail(**person.model_dump(), siblings=sibling_ids)

    def get_tree(self) -> List[TreeNode]:
        return self.tree

    def create_person(self, payload: Person | dict) -> Person:
        candidate = self._validate_person_payload(payload)
        if candidate.id in self.people:
            raise FamilyValidationError(f"{candidate.id} already exists")

        people = self._clone_people()
        people[candidate.id] = candidate
        record_order = [*self.record_order, candidate.id]
        self._prepare_people_for_write(people, updated_person_id=candidate.id)
        self._write_people(people, record_order)
        return self.people[candidate.id]

    def update_person(self, person_id: str, payload: Person | dict) -> Person:
        current = self.people.get(person_id)
        if not current:
            raise FamilyNotFoundError(f"{person_id} was not found")

        candidate = self._validate_person_payload(payload)
        if candidate.id != person_id:
            raise FamilyValidationError("editing an existing record cannot change its id")

        people = self._clone_people()
        people[person_id] = candidate
        self._prepare_people_for_write(
            people,
            updated_person_id=person_id,
            previous_person=current,
        )
        self._write_people(people, self.record_order)
        return self.people[person_id]

    def _set_state(self, raw_data: list[object]) -> None:
        people: Dict[str, Person] = {}
        record_order: List[str] = []

        for entry in raw_data:
            person = self._validate_person_payload(entry)
            if person.id in people:
                raise FamilyValidationError(f"duplicate person id {person.id}")
            people[person.id] = person
            record_order.append(person.id)

        self._prepare_people_for_runtime(people)
        self.people = people
        self.record_order = record_order
        self.tree = self._build_tree()

    def _prepare_people_for_runtime(self, people: Dict[str, Person]) -> None:
        self._validate_unique_names(people)
        self._synchronize_parent_child_links(people)
        self._validate_spouse_links(people)

    def _prepare_people_for_write(
        self,
        people: Dict[str, Person],
        *,
        updated_person_id: str,
        previous_person: Person | None = None,
    ) -> None:
        self._validate_mutation_relationships(
            people,
            updated_person_id=updated_person_id,
            previous_person=previous_person,
        )
        self._prepare_people_for_runtime(people)
        self._sync_parent_name_fields(people)
        self._sync_spouse_fields(
            people,
            updated_person_id=updated_person_id,
            previous_person=previous_person,
        )
        self._prepare_people_for_runtime(people)

    def _write_people(self, people: Dict[str, Person], record_order: List[str]) -> None:
        ordered_ids = [person_id for person_id in record_order if person_id in people]
        payload = [people[person_id].model_dump() for person_id in ordered_ids]
        serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        temp_path = self.data_path.with_suffix(f"{self.data_path.suffix}.tmp")

        temp_path.write_text(serialized, encoding="utf-8")
        shutil.copy2(self.data_path, self.backup_path)
        os.replace(temp_path, self.data_path)
        self._set_state(payload)

    def _validate_person_payload(self, payload: Person | dict | object) -> Person:
        try:
            return payload if isinstance(payload, Person) else Person.model_validate(payload)
        except Exception as exc:  # pragma: no cover - pydantic already tests the internals
            raise FamilyValidationError(str(exc)) from exc

    def _clone_people(self) -> Dict[str, Person]:
        return {
            person_id: person.model_copy(deep=True)
            for person_id, person in self.people.items()
        }

    def _validate_unique_names(self, people: Dict[str, Person]) -> None:
        names_by_key: Dict[str, str] = {}
        for person in people.values():
            key = self._normalize_name(person.name)
            if key in names_by_key:
                raise FamilyValidationError(
                    f"{person.name} duplicates the existing name {names_by_key[key]}. "
                    "Family names must stay unique because spouse links are name-based."
                )
            names_by_key[key] = person.name

    def _validate_mutation_relationships(
        self,
        people: Dict[str, Person],
        *,
        updated_person_id: str,
        previous_person: Person | None,
    ) -> None:
        updated_person = people[updated_person_id]
        allowed_legacy_parents = set()
        allowed_legacy_children = set()

        if previous_person:
            allowed_legacy_parents = {
                parent_id
                for parent_id in previous_person.parents
                if parent_id not in people
            }
            allowed_legacy_children = {
                child_id
                for child_id in previous_person.children
                if child_id not in people
            }

        for parent_id in updated_person.parents:
            if parent_id not in people and parent_id not in allowed_legacy_parents:
                raise FamilyValidationError(
                    f"{updated_person.id} references unknown parent {parent_id}"
                )

        for child_id in updated_person.children:
            if child_id not in people and child_id not in allowed_legacy_children:
                raise FamilyValidationError(
                    f"{updated_person.id} references unknown child {child_id}"
                )

    def _synchronize_parent_child_links(self, people: Dict[str, Person]) -> None:
        parent_map = {
            person_id: list(person.parents)
            for person_id, person in people.items()
        }
        child_map = {
            person_id: list(person.children)
            for person_id, person in people.items()
        }

        for person_id, parents in parent_map.items():
            for parent_id in parents:
                if parent_id in child_map:
                    self._append_unique(child_map[parent_id], person_id)

        for person_id, children in child_map.items():
            for child_id in children:
                if child_id in parent_map:
                    self._append_unique(parent_map[child_id], person_id)

        for person_id, person in people.items():
            person.parents = parent_map[person_id]
            person.children = child_map[person_id]

    def _sync_parent_name_fields(self, people: Dict[str, Person]) -> None:
        for person in people.values():
            if not person.parents:
                continue

            parent_names = [
                people[parent_id].name
                for parent_id in person.parents
                if parent_id in people
            ]
            person.father = parent_names[0] if parent_names else ""
            person.mother = parent_names[1] if len(parent_names) > 1 else ""

    def _validate_spouse_links(self, people: Dict[str, Person]) -> None:
        resolved_links: Dict[str, str] = {}

        for person in people.values():
            spouse_name = self._clean_text(person.spouse)
            if not spouse_name:
                continue

            spouse_id = self._resolve_existing_person_id_by_name(spouse_name, people)
            if spouse_id is None:
                continue

            if spouse_id == person.id:
                raise FamilyValidationError(f"{person.id} cannot reference themselves as spouse")

            resolved_links[person.id] = spouse_id

        for person_id, spouse_id in resolved_links.items():
            spouse_link = resolved_links.get(spouse_id)
            if spouse_link not in (None, person_id):
                raise FamilyValidationError(
                    f"{people[person_id].name} and {people[spouse_id].name} contain conflicting spouse links"
                )

    def _sync_spouse_fields(
        self,
        people: Dict[str, Person],
        *,
        updated_person_id: str,
        previous_person: Person | None,
    ) -> None:
        updated_person = people[updated_person_id]
        old_name = previous_person.name if previous_person else ""
        new_name = updated_person.name

        old_spouse_id = (
            self._resolve_existing_person_id_by_name(previous_person.spouse, people)
            if previous_person and previous_person.spouse
            else None
        )
        new_spouse_id = self._resolve_existing_person_id_by_name(updated_person.spouse, people)

        if old_spouse_id and old_spouse_id != new_spouse_id:
            old_spouse = people[old_spouse_id]
            if self._same_name(old_spouse.spouse, old_name):
                old_spouse.spouse = ""

        if old_name and self._normalize_name(old_name) != self._normalize_name(new_name):
            for person in people.values():
                if person.id == updated_person_id:
                    continue
                if self._same_name(person.spouse, old_name):
                    person.spouse = new_name

        if new_spouse_id is None:
            return

        linked_spouse = people[new_spouse_id]
        current_spouse_target = self._resolve_existing_person_id_by_name(linked_spouse.spouse, people)
        if current_spouse_target not in (None, updated_person_id):
            raise FamilyValidationError(
                f"{linked_spouse.name} is already linked to {people[current_spouse_target].name}"
            )

        linked_spouse.spouse = new_name
        updated_person.spouse = linked_spouse.name

    def _resolve_existing_person_id_by_name(
        self,
        candidate_name: str | None,
        people: Dict[str, Person],
    ) -> str | None:
        normalized_name = self._normalize_name(candidate_name)
        if not normalized_name:
            return None

        for person_id, person in people.items():
            if self._normalize_name(person.name) == normalized_name:
                return person_id
        return None

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

        return sorted(
            siblings,
            key=lambda sibling_id: self.people[sibling_id].name.lower(),
        )

    def _build_tree(self) -> List[TreeNode]:
        root_people = [
            person
            for person in self.people.values()
            if not person.parents or not self._has_known_parent(person)
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
                key=lambda child_id: self.people.get(
                    child_id,
                    Person(id=child_id, name=child_id),
                ).name.lower(),
            )
            if child_id in self.people
        ]
        return TreeNode(id=person.id, name=person.name, branch=person.branch, children=children)

    def _append_unique(self, values: List[str], item: str) -> None:
        if item not in values:
            values.append(item)

    def _clean_text(self, value: str | None) -> str:
        return value.strip() if value else ""

    def _normalize_name(self, value: str | None) -> str:
        return self._clean_text(value).lower()

    def _same_name(self, left: str | None, right: str | None) -> bool:
        return bool(self._normalize_name(left)) and self._normalize_name(left) == self._normalize_name(right)
