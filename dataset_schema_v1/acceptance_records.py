"""Read-only validation and resolution for versioned acceptance records.

This module never creates records, changes run evidence, or writes artifacts.
An approver value is recorded evidence of an operational human review; it is
not a cryptographic identity or signature.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_VERSION = "1.0.0"
PATTERNS = frozenset({"stable", "increasing", "periodic", "spike", "random_spike"})
STATUSES = frozenset({
    "eligible_for_review",
    "accepted_for_cpu_memory_classification",
    "accepted_for_throttling_analysis",
    "rejected",
    "deferred_pending_evidence",
})
PURPOSES = frozenset({"cpu_memory_classification", "throttling_analysis"})
_ACCEPTED_CPU = "accepted_for_cpu_memory_classification"
_ACCEPTED_THROTTLING = "accepted_for_throttling_analysis"
_ACCEPTED_STATUSES = frozenset({_ACCEPTED_CPU, _ACCEPTED_THROTTLING})
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_REQUIRED_ARTIFACT_ROLES = frozenset({"manifest", "observations", "experiment_metadata", "collection_summary"})
_THROTTLING_ARTIFACT_ROLES = frozenset({
    "cfs_throttled_periods_raw",
    "cfs_periods_raw",
    "cfs_throttled_seconds_raw",
})
_SINGLETON_ARTIFACT_ROLES = frozenset({"manifest", "observations", "experiment_metadata", "collection_summary"})
_THROTTLING_CHECKS = (
    "calculation_reviewed",
    "aligned_counter_coverage",
    "reset_gap_checks",
    "interpretation_criteria_reviewed",
)


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic validation result; ``errors`` contains stable reason codes."""

    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class HistoryResult:
    """Resolved immutable history for one run and one exact artifact version."""

    valid: bool
    errors: tuple[str, ...]
    ordered_record_ids: tuple[str, ...]
    current_record: Mapping[str, Any] | None


@dataclass(frozen=True)
class AuthorizationResult:
    """Purpose-specific result. A false ``authorized`` value always blocks use."""

    authorized: bool
    errors: tuple[str, ...]
    record_id: str | None
    requested_purpose: str


def _schema() -> dict[str, Any]:
    schema_path = Path(__file__).with_name("acceptance_record.schema.json")
    with schema_path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("acceptance_schema_not_object")
    Draft202012Validator.check_schema(loaded)
    return loaded


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _safe_relative_path(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        return None
    return path


def _artifact_fingerprint(record: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...] | None:
    artifacts = record.get("source_artifacts")
    if not isinstance(artifacts, list):
        return None
    result: list[tuple[str, str, str]] = []
    for artifact in artifacts:
        if not _is_mapping(artifact):
            return None
        role, path, digest = artifact.get("role"), artifact.get("path"), artifact.get("sha256")
        if not all(isinstance(value, str) for value in (role, path, digest)):
            return None
        result.append((role, path, digest.lower()))
    return tuple(sorted(result))


def validate_record_structure(record: Any) -> ValidationResult:
    """Validate schema and cross-field semantics without touching the filesystem."""
    if not _is_mapping(record):
        return ValidationResult(False, ("record_not_object",))
    schema = _schema()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [
        "schema:" + "/".join(str(part) for part in error.absolute_path) + ":" + error.validator
        for error in validator.iter_errors(dict(record))
    ]
    if errors:
        return ValidationResult(False, tuple(sorted(set(errors))))

    status = record["decision"]["status"]
    scope = record["decision"]["authorized_scope"]
    expected_scope = {
        _ACCEPTED_CPU: "cpu_memory_classification",
        _ACCEPTED_THROTTLING: "throttling_analysis",
    }.get(status, "none")
    if scope != expected_scope:
        errors.append("decision_scope_status_mismatch")

    decision = record["decision"]
    if status in _ACCEPTED_STATUSES:
        if not _UTC_TIMESTAMP.fullmatch(decision["decision_at_utc"]):
            errors.append("accepted_decision_timestamp_must_be_utc")
        approver = decision["approver"]
        if approver.get("kind") != "human" or not (approver.get("identifier") or approver.get("role")):
            errors.append("accepted_decision_requires_human_approver")
        if not isinstance(decision.get("policy_version"), str) or not decision["policy_version"].strip():
            errors.append("accepted_decision_requires_policy_version")
        if not isinstance(decision.get("rationale"), str) or not decision["rationale"].strip():
            errors.append("accepted_decision_requires_rationale")
        validation = record["validation"]
        audit = record["eligibility_audit"]
        if validation.get("result") != "pass" or not validation.get("schema_version"):
            errors.append("accepted_decision_requires_schema_validation_pass")
        if audit.get("result") not in {"pass", "review_required"}:
            errors.append("accepted_decision_audit_not_usable")
        if audit.get("source_integrity") != "verified":
            errors.append("accepted_decision_source_integrity_not_verified")
        if any(
            blocker.get("severity") == "hard_blocker" and blocker.get("status") == "open"
            for blocker in record["outstanding_blockers"]
        ):
            errors.append("accepted_decision_has_open_hard_blocker")

    artifact_paths: set[str] = set()
    roles = set()
    role_counts: dict[str, int] = {}
    for item in record["source_artifacts"]:
        path = item["path"]
        if path in artifact_paths:
            errors.append("duplicate_source_artifact_path")
        artifact_paths.add(path)
        roles.add(item["role"])
        role_counts[item["role"]] = role_counts.get(item["role"], 0) + 1
    if not _REQUIRED_ARTIFACT_ROLES.issubset(roles):
        errors.append("required_source_artifact_role_missing")
    if any(role_counts.get(role, 0) != 1 for role in _SINGLETON_ARTIFACT_ROLES & roles):
        errors.append("duplicate_singleton_source_artifact_role")

    for group_name in ("target_identity", "resource_configuration"):
        for field, evidence in record[group_name].items():
            if evidence["status"] == "observed":
                if evidence["value"] is None:
                    errors.append(f"{group_name}_{field}_observed_value_missing")
                if evidence["reason"] is not None:
                    errors.append(f"{group_name}_{field}_observed_reason_must_be_null")
            elif evidence["value"] is not None or not evidence["reason"]:
                errors.append(f"{group_name}_{field}_unavailable_value_requires_reason")

    if status == _ACCEPTED_THROTTLING:
        throttle = record["throttling_evidence"]
        if throttle["observed_status"] == "unknown":
            errors.append("throttling_acceptance_status_unknown")
        if any(throttle[field] != "pass" for field in _THROTTLING_CHECKS):
            errors.append("throttling_acceptance_evidence_incomplete")
        if not _THROTTLING_ARTIFACT_ROLES.issubset(roles):
            errors.append("throttling_acceptance_raw_counter_artifact_missing")

    return ValidationResult(not errors, tuple(sorted(set(errors))))


def verify_record_artifacts(
    record: Any,
    project_root: str | Path,
    *,
    expected_run_id: str,
    expected_pattern: str,
    expected_output_path: str,
) -> ValidationResult:
    """Read and hash reviewed artifacts, checking record identity and containment."""
    structure = validate_record_structure(record)
    if not structure.valid:
        return structure
    if record.get("run_id") != expected_run_id:
        return ValidationResult(False, ("run_id_mismatch",))
    if record.get("pattern") != expected_pattern:
        return ValidationResult(False, ("pattern_mismatch",))

    root = Path(project_root).resolve(strict=True)
    output_rel = _safe_relative_path(record.get("designated_output_path"))
    expected_output_rel = _safe_relative_path(expected_output_path)
    if output_rel is None or expected_output_rel is None:
        return ValidationResult(False, ("designated_output_path_invalid",))
    if output_rel != expected_output_rel:
        return ValidationResult(False, ("designated_output_path_mismatch",))
    try:
        output = root.joinpath(*output_rel.parts).resolve(strict=True)
        output.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return ValidationResult(False, ("designated_output_path_outside_or_missing",))
    if not output.is_dir():
        return ValidationResult(False, ("designated_output_path_not_directory",))

    artifacts = record["source_artifacts"]
    roles = {item["role"]: item["path"] for item in artifacts}
    if roles.get("manifest") != (output_rel / "manifest.json").as_posix():
        return ValidationResult(False, ("manifest_artifact_path_mismatch",))
    if roles.get("observations") != (output_rel / "observations.csv").as_posix():
        return ValidationResult(False, ("observations_artifact_path_mismatch",))

    for artifact in artifacts:
        relative = _safe_relative_path(artifact["path"])
        if relative is None:
            return ValidationResult(False, ("artifact_path_invalid",))
        try:
            path = root.joinpath(*relative.parts).resolve(strict=True)
            path.relative_to(root)
            if not path.is_file():
                return ValidationResult(False, ("artifact_not_file",))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError):
            return ValidationResult(False, ("artifact_missing_unreadable_or_outside",))
        if not _HEX_SHA256.fullmatch(str(artifact.get("sha256", ""))):
            return ValidationResult(False, ("artifact_sha256_invalid",))
        if digest.lower() != artifact["sha256"].lower():
            return ValidationResult(False, ("artifact_hash_mismatch_stale_record",))

    manifest_artifact = next(item for item in artifacts if item["role"] == "manifest")
    try:
        manifest_path = root.joinpath(*PurePosixPath(manifest_artifact["path"]).parts)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ValidationResult(False, ("manifest_artifact_unreadable_or_malformed",))
    if not isinstance(manifest, dict):
        return ValidationResult(False, ("manifest_artifact_not_object",))
    if manifest.get("run_id") != expected_run_id:
        return ValidationResult(False, ("manifest_run_id_mismatch",))
    if manifest.get("pattern") != expected_pattern:
        return ValidationResult(False, ("manifest_pattern_mismatch",))

    metadata_artifact = next(item for item in artifacts if item["role"] == "experiment_metadata")
    try:
        metadata_path = root.joinpath(*PurePosixPath(metadata_artifact["path"]).parts)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ValidationResult(False, ("experiment_metadata_unreadable_or_malformed",))
    if not isinstance(metadata, dict):
        return ValidationResult(False, ("experiment_metadata_not_object",))
    if metadata.get("run_id") != expected_run_id:
        return ValidationResult(False, ("experiment_metadata_run_id_mismatch",))
    if metadata.get("pattern") != expected_pattern:
        return ValidationResult(False, ("experiment_metadata_pattern_mismatch",))

    observations_artifact = next(item for item in artifacts if item["role"] == "observations")
    try:
        observations_path = root.joinpath(*PurePosixPath(observations_artifact["path"]).parts)
        with observations_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"run_id", "pattern"}.issubset(reader.fieldnames):
                return ValidationResult(False, ("observations_identity_columns_missing",))
            row_count = 0
            for row in reader:
                row_count += 1
                if row.get("run_id") != expected_run_id:
                    return ValidationResult(False, ("observations_run_id_mismatch",))
                if row.get("pattern") != expected_pattern:
                    return ValidationResult(False, ("observations_pattern_mismatch",))
            if row_count == 0:
                return ValidationResult(False, ("observations_empty",))
    except (OSError, UnicodeError, csv.Error):
        return ValidationResult(False, ("observations_artifact_unreadable",))
    return ValidationResult(True, ())


def resolve_record_history(
    records: Sequence[Any],
    *,
    expected_run_id: str,
    expected_pattern: str,
    expected_output_path: str,
) -> HistoryResult:
    """Resolve one linear supersession chain for a single exact artifact version."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        return HistoryResult(False, ("record_history_missing_or_malformed",), (), None)

    by_id: dict[str, Mapping[str, Any]] = {}
    fingerprints: set[tuple[tuple[str, str, str], ...]] = set()
    errors: list[str] = []
    for record in records:
        result = validate_record_structure(record)
        if not result.valid:
            errors.extend(result.errors)
            continue
        if record.get("run_id") != expected_run_id:
            errors.append("history_run_id_mismatch")
        if record.get("pattern") != expected_pattern:
            errors.append("history_pattern_mismatch")
        if record.get("designated_output_path") != expected_output_path:
            errors.append("history_output_path_mismatch")
        record_id = record["acceptance_record_id"]
        if record_id in by_id:
            errors.append("duplicate_acceptance_record_id")
        else:
            by_id[record_id] = record
        fingerprint = _artifact_fingerprint(record)
        if fingerprint is None:
            errors.append("history_artifact_fingerprint_invalid")
        else:
            fingerprints.add(fingerprint)
    if len(fingerprints) > 1:
        errors.append("history_artifact_version_mismatch")
    if errors:
        return HistoryResult(False, tuple(sorted(set(errors))), (), None)

    child_by_parent: dict[str, str] = {}
    parent_of: dict[str, str] = {}
    for record_id, record in by_id.items():
        parent = record.get("supersedes_record_id")
        if parent is None:
            continue
        if parent not in by_id:
            errors.append("supersedes_reference_broken")
            continue
        if parent == record_id:
            errors.append("supersession_cycle")
            continue
        if parent in child_by_parent:
            errors.append("conflicting_superseding_records")
        child_by_parent[parent] = record_id
        parent_of[record_id] = parent

    # Detect directed cycles even when they leave no root to start a linear walk.
    for start in sorted(by_id):
        cursor = start
        path_seen: set[str] = set()
        while cursor in parent_of:
            if cursor in path_seen:
                errors.append("supersession_cycle")
                break
            path_seen.add(cursor)
            cursor = parent_of[cursor]

    roots = sorted(record_id for record_id in by_id if record_id not in parent_of)
    leaves = sorted(record_id for record_id in by_id if record_id not in child_by_parent)
    if len(roots) != 1 or len(leaves) != 1:
        errors.append("ambiguous_or_conflicting_terminal_decisions")
    if errors:
        return HistoryResult(False, tuple(sorted(set(errors))), (), None)

    ordered: list[str] = []
    seen: set[str] = set()
    cursor = roots[0]
    while True:
        if cursor in seen:
            return HistoryResult(False, ("supersession_cycle",), (), None)
        seen.add(cursor)
        ordered.append(cursor)
        child = child_by_parent.get(cursor)
        if child is None:
            break
        cursor = child
    if len(seen) != len(by_id):
        return HistoryResult(False, ("history_disconnected_or_ambiguous",), (), None)
    return HistoryResult(True, (), tuple(ordered), by_id[leaves[0]])


def authorize_purpose(
    record: Any,
    records: Sequence[Any],
    requested_purpose: str,
    project_root: str | Path,
    *,
    expected_run_id: str,
    expected_pattern: str,
    expected_output_path: str,
) -> AuthorizationResult:
    """Authorize a purpose only for the current, structurally valid, hash-matched decision."""
    record_id = record.get("acceptance_record_id") if _is_mapping(record) else None
    if requested_purpose not in PURPOSES:
        return AuthorizationResult(False, ("requested_purpose_invalid",), record_id, requested_purpose)
    structure = validate_record_structure(record)
    if not structure.valid:
        return AuthorizationResult(False, structure.errors, record_id, requested_purpose)
    artifact_result = verify_record_artifacts(
        record,
        project_root,
        expected_run_id=expected_run_id,
        expected_pattern=expected_pattern,
        expected_output_path=expected_output_path,
    )
    if not artifact_result.valid:
        return AuthorizationResult(False, artifact_result.errors, record_id, requested_purpose)
    history = resolve_record_history(
        records,
        expected_run_id=expected_run_id,
        expected_pattern=expected_pattern,
        expected_output_path=expected_output_path,
    )
    if not history.valid:
        return AuthorizationResult(False, history.errors, record_id, requested_purpose)
    if history.current_record is None or history.current_record.get("acceptance_record_id") != record_id:
        return AuthorizationResult(False, ("record_is_not_current_history_terminal",), record_id, requested_purpose)

    decision = record["decision"]
    if decision["status"] not in _ACCEPTED_STATUSES:
        return AuthorizationResult(False, ("decision_status_not_accepted",), record_id, requested_purpose)
    if decision["authorized_scope"] != requested_purpose:
        return AuthorizationResult(False, ("requested_purpose_outside_approved_scope",), record_id, requested_purpose)
    if requested_purpose == "throttling_analysis" and record["throttling_evidence"]["observed_status"] == "unknown":
        return AuthorizationResult(False, ("throttling_evidence_unknown",), record_id, requested_purpose)
    return AuthorizationResult(True, (), record_id, requested_purpose)
