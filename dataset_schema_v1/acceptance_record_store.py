from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset_schema_v1.acceptance_records import (
    resolve_record_history,
    validate_record_structure,
)


class AcceptanceRecordStoreError(ValueError):
    """Raised when the complete acceptance-record store cannot be trusted."""


def load_acceptance_records(
    project_root: str | Path,
    *,
    store_relative_path: str = "dataset_schema_v1/acceptance_records",
) -> tuple[dict[str, Any], ...]:
    """Load and structurally validate every JSON record; never write files."""
    root = Path(project_root).resolve(strict=True)
    relative = Path(store_relative_path)

    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise AcceptanceRecordStoreError("store_path_invalid")

    store = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AcceptanceRecordStoreError("store_path_symlink_not_allowed")
    if not store.exists() or not store.is_dir():
        raise AcceptanceRecordStoreError("store_missing_or_not_directory")
    try:
        resolved_store = store.resolve(strict=True)
        resolved_store.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AcceptanceRecordStoreError("store_path_outside_project_root") from exc

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    try:
        paths = sorted(store.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise AcceptanceRecordStoreError("store_directory_unreadable") from exc

    for path in paths:
        if path.is_symlink():
            raise AcceptanceRecordStoreError("record_symlink_not_allowed")
        if not path.is_file() or path.suffix.lower() != ".json":
            raise AcceptanceRecordStoreError("unexpected_store_entry")

        try:
            with path.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AcceptanceRecordStoreError("record_file_unreadable_or_malformed") from exc

        validation = validate_record_structure(record)
        if not validation.valid:
            raise AcceptanceRecordStoreError(
                "record_invalid:" + path.name + ":" + ",".join(validation.errors)
            )

        record_id = record["acceptance_record_id"]
        if record_id in seen_ids:
            raise AcceptanceRecordStoreError("duplicate_acceptance_record_id")
        seen_ids.add(record_id)
        records.append(dict(record))

    return tuple(records)


def validate_complete_history(
    records: tuple[dict[str, Any], ...],
) -> None:
    """Reject broken or conflicting histories, grouped by run and pattern."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for record in records:
        key = (record["run_id"], record["pattern"])
        grouped.setdefault(key, []).append(record)

    for (run_id, pattern), history_records in sorted(grouped.items()):
        output_paths = {
            record["designated_output_path"] for record in history_records
        }
        if len(output_paths) != 1:
            raise AcceptanceRecordStoreError(
                "history_invalid:" + run_id + ":designated_output_path_conflict"
            )
        output_path = next(iter(output_paths))
        result = resolve_record_history(
            history_records,
            expected_run_id=run_id,
            expected_pattern=pattern,
            expected_output_path=output_path,
        )
        if not result.valid:
            raise AcceptanceRecordStoreError(
                "history_invalid:" + run_id + ":" + ",".join(result.errors)
            )


def load_validated_acceptance_history(
    project_root: str | Path,
    *,
    store_relative_path: str = "dataset_schema_v1/acceptance_records",
) -> tuple[dict[str, Any], ...]:
    """Load the complete store and validate every present run history."""
    records = load_acceptance_records(
        project_root,
        store_relative_path=store_relative_path,
    )
    validate_complete_history(records)
    return records
