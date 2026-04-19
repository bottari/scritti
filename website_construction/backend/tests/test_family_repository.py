from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.family_loader import FamilyRepository, FamilyValidationError


SOURCE_DATA = Path(__file__).resolve().parents[1] / "data" / "family.json"


def copy_family_data(tmp_path: Path) -> Path:
    target = tmp_path / "family.json"
    shutil.copy2(SOURCE_DATA, target)
    return target


def get_person_payload(repository: FamilyRepository, person_id: str) -> dict:
    for person in repository.list_people():
        if person.id == person_id:
            return person.model_dump()
    raise AssertionError(f"missing person {person_id}")


def test_repository_reads_existing_family_data(tmp_path: Path) -> None:
    repository = FamilyRepository(copy_family_data(tmp_path))

    frank = repository.get_person("frank-bottari")
    giuseppe = repository.get_person("giuseppe-bottari")

    assert frank is not None
    assert frank.parents == ["giuseppe-bottari", "carmela-giuseppa-foti"]
    assert giuseppe is not None
    assert "frank-bottari" in giuseppe.children


def test_repository_updates_a_person_and_preserves_spouse_links(tmp_path: Path) -> None:
    data_path = copy_family_data(tmp_path)
    repository = FamilyRepository(data_path)

    payload = get_person_payload(repository, "josephine-lentini")
    payload["name"] = "Josephine Bottari"
    payload["lastName"] = "Bottari"
    payload["branch"] = "Bottari"

    repository.update_person("josephine-lentini", payload)
    reloaded = FamilyRepository(data_path)

    updated_person = reloaded.get_person("josephine-lentini")
    frank = reloaded.get_person("frank-bottari")

    assert updated_person is not None
    assert updated_person.name == "Josephine Bottari"
    assert updated_person.branch == "Bottari"
    assert frank is not None
    assert frank.spouse == "Josephine Bottari"


def test_repository_rejects_unknown_relationship_ids(tmp_path: Path) -> None:
    repository = FamilyRepository(copy_family_data(tmp_path))

    payload = get_person_payload(repository, "frank-bottari")
    payload["parents"] = [*payload["parents"], "missing-parent"]

    with pytest.raises(FamilyValidationError, match="unknown parent missing-parent"):
        repository.update_person("frank-bottari", payload)


def test_repository_creates_backup_and_replaces_data_atomically(tmp_path: Path) -> None:
    data_path = copy_family_data(tmp_path)
    repository = FamilyRepository(data_path)
    original_contents = data_path.read_text(encoding="utf-8")

    payload = get_person_payload(repository, "dominic-lentini")
    payload["birthLocation"] = "Boston, Massachusetts"

    repository.update_person("dominic-lentini", payload)

    backup_path = data_path.with_suffix(".json.bak")
    temp_path = data_path.with_suffix(".json.tmp")

    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") == original_contents
    assert temp_path.exists() is False
    assert "Boston, Massachusetts" in data_path.read_text(encoding="utf-8")
