"""Safety primitives for the K8sight experiment runner."""

import json
import re
from pathlib import Path


def validate_run_id(run_id: str) -> None:
    """Require a safe, single-component run identifier."""
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id
    ):
        raise ValueError("run_id must be a safe 1-100 character identifier")
    if run_id in {".", ".."}:
        raise ValueError("run_id cannot be a path component")


def create_run_directory(path: Path) -> Path:
    """Create a new run directory without overwriting existing evidence."""
    output = Path(path).expanduser()
    output.mkdir(parents=True, exist_ok=False)
    return output


def outcome_from_exit_code(exit_code: int) -> str:
    """Map a process exit status to a conservative run outcome."""
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise TypeError("exit_code must be an integer")
    return "completed" if exit_code == 0 else "failed"


def validate_timeline(boundaries: dict) -> None:
    """Validate lifecycle timestamps expressed as Unix epoch seconds."""
    required = (
        "run_start_utc",
        "workload_end_utc",
        "locust_shutdown_utc",
        "cooldown_start_utc",
        "collection_end_utc",
    )
    values = {}
    for key in required:
        value = boundaries.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a numeric epoch timestamp")
        values[key] = value

    if not (
        values["run_start_utc"] < values["workload_end_utc"]
        <= values["locust_shutdown_utc"]
        <= values["cooldown_start_utc"]
        < values["collection_end_utc"]
    ):
        raise ValueError("lifecycle timestamps are out of order")


def write_run_metadata(path: Path, metadata: dict) -> Path:
    """Write JSON metadata without overwriting an existing file."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return output
