from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_schema_v1.acceptance_record_store import (
    AcceptanceRecordStoreError,
    load_acceptance_records,
    load_validated_acceptance_history,
)


STORE = "dataset_schema_v1/acceptance_records"


def create_store(root: Path) -> Path:
    store = root / STORE
    store.mkdir(parents=True)
    return store


def test_missing_store_fails_closed(tmp_path):
    with pytest.raises(AcceptanceRecordStoreError, match="store_missing_or_not_directory"):
        load_acceptance_records(tmp_path)


def test_empty_store_loads_as_empty_history(tmp_path):
    create_store(tmp_path)
    assert load_validated_acceptance_history(tmp_path) == ()


def test_malformed_json_is_rejected(tmp_path):
    store = create_store(tmp_path)
    (store / "broken.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(AcceptanceRecordStoreError, match="record_file_unreadable_or_malformed"):
        load_acceptance_records(tmp_path)


def test_invalid_record_is_rejected(tmp_path):
    store = create_store(tmp_path)
    (store / "invalid.json").write_text(
        json.dumps({"acceptance_record_id": "missing-required-fields"}),
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceRecordStoreError, match="record_invalid:"):
        load_acceptance_records(tmp_path)


def test_unexpected_directory_entry_is_rejected(tmp_path):
    store = create_store(tmp_path)
    (store / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(AcceptanceRecordStoreError, match="unexpected_store_entry"):
        load_acceptance_records(tmp_path)


def test_nested_directory_entry_is_rejected(tmp_path):
    store = create_store(tmp_path)
    (store / "nested").mkdir()

    with pytest.raises(AcceptanceRecordStoreError, match="unexpected_store_entry"):
        load_acceptance_records(tmp_path)


def test_store_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / STORE
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(AcceptanceRecordStoreError, match="store_path_symlink_not_allowed"):
        load_acceptance_records(tmp_path)


def test_record_symlink_is_rejected(tmp_path):
    store = create_store(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (store / "linked.json").symlink_to(outside)

    with pytest.raises(AcceptanceRecordStoreError, match="record_symlink_not_allowed"):
        load_acceptance_records(tmp_path)


def test_store_loader_does_not_modify_files(tmp_path):
    store = create_store(tmp_path)
    content = "{not-json"
    path = store / "broken.json"
    path.write_text(content, encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(AcceptanceRecordStoreError):
        load_acceptance_records(tmp_path)

    assert path.read_bytes() == before
    assert sorted(item.name for item in store.iterdir()) == ["broken.json"]


def test_conflicting_output_paths_for_same_run_are_rejected(tmp_path):
    from dataset_schema_v1.acceptance_record_store import validate_complete_history

    first = {
        "run_id": "synthetic-run",
        "pattern": "stable",
        "designated_output_path": "validated/output-a",
    }
    second = {
        "run_id": "synthetic-run",
        "pattern": "stable",
        "designated_output_path": "validated/output-b",
    }

    with pytest.raises(
        AcceptanceRecordStoreError,
        match="designated_output_path_conflict",
    ):
        validate_complete_history((first, second))


def test_symlinked_parent_directory_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "acceptance_records").mkdir()

    schema_parent = tmp_path / "dataset_schema_v1"
    schema_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        AcceptanceRecordStoreError,
        match="store_path_symlink_not_allowed",
    ):
        load_acceptance_records(tmp_path)


def test_valid_synthetic_record_loads_and_history_validates(tmp_path):
    import hashlib

    store = create_store(tmp_path)

    run_id = "synthetic-store-test"
    pattern = "stable"
    output = "validated/synthetic-store-test"
    metadata = f"run_records/{run_id}/experiment_metadata.json"

    files = {
        f"{output}/manifest.json": json.dumps(
            {"run_id": run_id, "pattern": pattern}
        ),
        f"{output}/observations.csv": (
            "timestamp_utc,run_id,pattern\n"
            f"2026-10-14T12:00:00Z,{run_id},{pattern}\n"
        ),
        metadata: json.dumps({"run_id": run_id, "pattern": pattern}),
        f"{output}/collection_summary.json": json.dumps({"synthetic": True}),
    }

    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    artifacts = [
        {
            "role": role,
            "path": relative,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        for role, relative, content in (
            ("manifest", f"{output}/manifest.json", files[f"{output}/manifest.json"]),
            ("observations", f"{output}/observations.csv", files[f"{output}/observations.csv"]),
            ("experiment_metadata", metadata, files[metadata]),
            ("collection_summary", f"{output}/collection_summary.json", files[f"{output}/collection_summary.json"]),
        )
    ]

    # Reuse the project's established fixture to preserve the actual schema.
    # This test checks loading and history resolution, not artifact authorization.
    from test_acceptance_records import make_record

    record, _ = make_record(tmp_path, record_id="store-valid-record")
    (store / "record.json").write_text(json.dumps(record), encoding="utf-8")

    loaded = load_validated_acceptance_history(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["acceptance_record_id"] == "store-valid-record"
