"""Focused synthetic tests for acceptance-record validation and authorization."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from dataset_schema_v1.acceptance_records import (
    authorize_purpose,
    resolve_record_history,
    validate_record_structure,
    verify_record_artifacts,
)


RUN_ID = "synthetic-stable-001"
PATTERN = "stable"
OUTPUT = "validated/synthetic-stable-001"
NOW = "2026-10-14T12:00:00Z"


def evidence(status="observed", value="synthetic-value"):
    return {
        "status": status,
        "value": value if status == "observed" else None,
        "reason": None if status == "observed" else "synthetic unavailable reason",
    }


def make_record(root: Path, *, status="eligible_for_review", record_id="record-a", throttle_unknown=True):
    output_dir = root / OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = root / "run_records" / RUN_ID
    metadata_dir.mkdir(parents=True, exist_ok=True)
    files = {
        f"{OUTPUT}/manifest.json": json.dumps({"run_id": RUN_ID, "pattern": PATTERN}),
        f"{OUTPUT}/observations.csv": "timestamp_utc,run_id,pattern\n2026-10-14T12:00:00Z," + RUN_ID + "," + PATTERN + "\n",
        f"run_records/{RUN_ID}/experiment_metadata.json": json.dumps({"run_id": RUN_ID, "pattern": PATTERN}),
        f"{OUTPUT}/collection_summary.json": json.dumps({"synthetic": True}),
    }
    if status == "accepted_for_throttling_analysis":
        for role in ("cfs_throttled_periods_raw", "cfs_periods_raw", "cfs_throttled_seconds_raw"):
            files[f"{OUTPUT}/{role}.json"] = json.dumps({"synthetic": role})
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    artifacts = [
        {"role": role, "path": path, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        for role, path, content in (
            ("manifest", f"{OUTPUT}/manifest.json", files[f"{OUTPUT}/manifest.json"]),
            ("observations", f"{OUTPUT}/observations.csv", files[f"{OUTPUT}/observations.csv"]),
            ("experiment_metadata", f"run_records/{RUN_ID}/experiment_metadata.json", files[f"run_records/{RUN_ID}/experiment_metadata.json"]),
            ("collection_summary", f"{OUTPUT}/collection_summary.json", files[f"{OUTPUT}/collection_summary.json"]),
        )
    ]
    if status == "accepted_for_throttling_analysis":
        artifacts.extend(
            {
                "role": role,
                "path": f"{OUTPUT}/{role}.json",
                "sha256": hashlib.sha256(files[f"{OUTPUT}/{role}.json"].encode("utf-8")).hexdigest(),
            }
            for role in ("cfs_throttled_periods_raw", "cfs_periods_raw", "cfs_throttled_seconds_raw")
        )

    accepted = status.startswith("accepted_")
    scope = {
        "accepted_for_cpu_memory_classification": "cpu_memory_classification",
        "accepted_for_throttling_analysis": "throttling_analysis",
    }.get(status, "none")
    throttle_status = "unknown" if throttle_unknown else "evidenced"
    throttle_checks = "pass" if status == "accepted_for_throttling_analysis" else "not_applicable"
    record = {
        "record_schema_version": "1.0.0",
        "acceptance_record_id": record_id,
        "run_id": RUN_ID,
        "pattern": PATTERN,
        "designated_output_path": OUTPUT,
        "source_artifacts": artifacts,
        "validation": {
            "result": "pass",
            "schema_version": "1.0.0",
            "audited_at_utc": NOW,
            "validator_version": "synthetic-v1",
        },
        "eligibility_audit": {
            "result": "review_required",
            "audited_at_utc": NOW,
            "audit_version": "synthetic-v1",
            "source_integrity": "verified",
        },
        "throttling_evidence": {
            "observed_status": throttle_status,
            "calculation_reviewed": throttle_checks,
            "aligned_counter_coverage": throttle_checks,
            "reset_gap_checks": throttle_checks,
            "interpretation_criteria_reviewed": throttle_checks,
        },
        "decision": {
            "status": status,
            "authorized_scope": scope,
            "decision_at_utc": NOW if accepted else None,
            "approver": {"kind": "human", "identifier": "synthetic-reviewer" , "role": None} if accepted else {"kind": "none", "identifier": None, "role": None},
            "policy_version": "synthetic-policy-1" if accepted else None,
            "rationale": "Synthetic test decision." if accepted else "Pending synthetic review.",
        },
        "evidence_reviewed": ["synthetic manifest", "synthetic observations"],
        "caveats_accepted": [],
        "outstanding_blockers": [],
        "deferred_questions": [],
        "target_identity": {
            key: evidence() for key in ("namespace", "pod", "pod_uid", "container", "container_id", "image", "node")
        },
        "resource_configuration": {
            key: evidence() for key in ("cpu_request", "cpu_limit", "memory_request", "memory_limit")
        },
        "feature_target_scope": {
            "features": ["cpu_usage_cores", "memory_working_set_bytes"],
            "targets": ["pattern"],
            "excluded_phases": ["pre_start", "cooldown", "post_shutdown", "unknown"],
            "missing_value_policy": "Exclude required missing features; never impute.",
        },
        "supersedes_record_id": None,
    }
    return record, files


def verify(record, root, *, run_id=RUN_ID, pattern=PATTERN, output=OUTPUT):
    return verify_record_artifacts(
        record,
        root,
        expected_run_id=run_id,
        expected_pattern=pattern,
        expected_output_path=output,
    )


def authorize(record, records, root, purpose, *, run_id=RUN_ID, pattern=PATTERN, output=OUTPUT):
    return authorize_purpose(
        record,
        records,
        purpose,
        root,
        expected_run_id=run_id,
        expected_pattern=pattern,
        expected_output_path=output,
    )


def test_review_record_is_valid_but_authorizes_neither_scope(tmp_path):
    record, _ = make_record(tmp_path)
    assert validate_record_structure(record).valid
    assert not authorize(record, [record], tmp_path, "cpu_memory_classification").authorized
    assert not authorize(record, [record], tmp_path, "throttling_analysis").authorized


def test_cpu_memory_acceptance_allows_unknown_throttling(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    assert validate_record_structure(record).valid
    result = authorize(record, [record], tmp_path, "cpu_memory_classification")
    assert result.authorized
    assert not authorize(record, [record], tmp_path, "throttling_analysis").authorized


def test_throttling_acceptance_requires_complete_synthetic_evidence(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_throttling_analysis", throttle_unknown=False)
    assert validate_record_structure(record).valid
    assert authorize(record, [record], tmp_path, "throttling_analysis").authorized
    assert not authorize(record, [record], tmp_path, "cpu_memory_classification").authorized
    record["throttling_evidence"]["aligned_counter_coverage"] = "unknown"
    assert "throttling_acceptance_evidence_incomplete" in validate_record_structure(record).errors


def test_unknown_throttling_status_cannot_support_throttling_acceptance(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_throttling_analysis", throttle_unknown=True)
    result = validate_record_structure(record)
    assert not result.valid
    assert "throttling_acceptance_status_unknown" in result.errors


@pytest.mark.parametrize("field", ["identifier", "role"])
def test_accepted_record_requires_explicit_human_approver(tmp_path, field):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    record["decision"]["approver"][field] = None
    record["decision"]["approver"]["identifier"] = None
    record["decision"]["approver"]["role"] = None
    assert not validate_record_structure(record).valid


@pytest.mark.parametrize("field,value", [
    ("decision_at_utc", None),
    ("policy_version", None),
    ("rationale", ""),
])
def test_accepted_record_requires_timestamp_policy_and_rationale(tmp_path, field, value):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    record["decision"][field] = value
    assert not validate_record_structure(record).valid


def test_accepted_record_with_open_hard_blocker_is_invalid(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    record["outstanding_blockers"].append({
        "code": "source_conflict", "severity": "hard_blocker", "status": "open", "description": "Unresolved.",
    })
    assert "accepted_decision_has_open_hard_blocker" in validate_record_structure(record).errors


def test_failed_audit_or_integrity_cannot_authorize(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    record["eligibility_audit"]["source_integrity"] = "failed"
    assert "accepted_decision_source_integrity_not_verified" in validate_record_structure(record).errors
    record["eligibility_audit"]["source_integrity"] = "verified"
    record["eligibility_audit"]["result"] = "blocked"
    assert "accepted_decision_audit_not_usable" in validate_record_structure(record).errors


@pytest.mark.parametrize("field,value", [
    ("record_schema_version", "9.0.0"),
    ("pattern", "random"),
])
def test_unknown_schema_version_or_pattern_is_rejected(tmp_path, field, value):
    record, _ = make_record(tmp_path)
    record[field] = value
    assert not validate_record_structure(record).valid


@pytest.mark.parametrize("status,scope", [
    ("unknown", "none"),
    ("accepted_for_cpu_memory_classification", "throttling_analysis"),
    ("accepted_for_throttling_analysis", "cpu_memory_classification"),
    ("accepted_for_cpu_memory_classification", "unknown"),
    ("eligible_for_review", "cpu_memory_classification"),
])
def test_unknown_or_contradictory_decision_status_and_scope_are_rejected(tmp_path, status, scope):
    record, _ = make_record(tmp_path)
    record["decision"]["status"] = status
    record["decision"]["authorized_scope"] = scope
    assert not validate_record_structure(record).valid


@pytest.mark.parametrize("digest", ["", "xyz", "a" * 63])
def test_missing_or_malformed_hash_is_rejected(tmp_path, digest):
    record, _ = make_record(tmp_path)
    record["source_artifacts"][0]["sha256"] = digest
    assert not validate_record_structure(record).valid


def test_changed_artifact_hash_blocks_authorization(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    (tmp_path / record["source_artifacts"][0]["path"]).write_text("changed", encoding="utf-8")
    result = verify(record, tmp_path)
    assert not result.valid
    assert "artifact_hash_mismatch_stale_record" in result.errors
    assert not authorize(record, [record], tmp_path, "cpu_memory_classification").authorized


def test_missing_or_unreadable_artifact_fails_closed(tmp_path):
    record, _ = make_record(tmp_path)
    target = tmp_path / record["source_artifacts"][0]["path"]
    target.unlink()
    result = verify(record, tmp_path)
    assert not result.valid
    assert "artifact_missing_unreadable_or_outside" in result.errors


def test_path_traversal_is_rejected(tmp_path):
    record, _ = make_record(tmp_path)
    record["source_artifacts"][0]["path"] = "../outside.json"
    assert not validate_record_structure(record).valid
    assert not verify(record, tmp_path).valid


def test_symlink_escape_is_rejected(tmp_path):
    record, files = make_record(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text("outside", encoding="utf-8")
    link_rel = f"run_records/{RUN_ID}/linked.json"
    link = tmp_path / link_rel
    link.symlink_to(outside)
    record["source_artifacts"].append({
        "role": "supplemental", "path": link_rel,
        "sha256": hashlib.sha256(b"outside").hexdigest(),
    })
    assert not verify(record, tmp_path).valid


@pytest.mark.parametrize("field,value,expected", [
    ("run_id", "other-run", "run_id_mismatch"),
    ("pattern", "spike", "pattern_mismatch"),
    ("designated_output_path", "validated/other", "designated_output_path_mismatch"),
])
def test_expected_identity_mismatches_fail(tmp_path, field, value, expected):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    if field == "run_id":
        result = verify(record, tmp_path, run_id=value)
    elif field == "pattern":
        result = verify(record, tmp_path, pattern=value)
    else:
        result = verify(record, tmp_path, output=value)
    assert not result.valid
    assert expected in result.errors


@pytest.mark.parametrize("manifest_field,wrong", [("run_id", "other-run"), ("pattern", "spike")])
def test_artifact_manifest_identity_mismatch_fails(tmp_path, manifest_field, wrong):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    path = tmp_path / record["source_artifacts"][0]["path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[manifest_field] = wrong
    path.write_text(json.dumps(manifest), encoding="utf-8")
    record["source_artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = verify(record, tmp_path)
    assert not result.valid
    assert f"manifest_{manifest_field}_mismatch" in result.errors


def test_designated_output_symlink_escape_fails(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    outside = tmp_path.parent / (tmp_path.name + "-outside-dir")
    outside.mkdir()
    link = tmp_path / "validated" / "outside-link"
    link.symlink_to(outside, target_is_directory=True)
    record["designated_output_path"] = "validated/outside-link"
    result = verify(record, tmp_path, output="validated/outside-link")
    assert not result.valid
    assert "designated_output_path_outside_or_missing" in result.errors


def test_duplicate_record_ids_are_blocked(tmp_path):
    record, _ = make_record(tmp_path)
    history = resolve_record_history([record, copy.deepcopy(record)], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert not history.valid
    assert "duplicate_acceptance_record_id" in history.errors


def test_duplicate_terminal_decisions_are_blocked(tmp_path):
    first, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification", record_id="first")
    second = copy.deepcopy(first)
    second["acceptance_record_id"] = "second"
    history = resolve_record_history([second, first], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert not history.valid
    assert "ambiguous_or_conflicting_terminal_decisions" in history.errors


def test_broken_supersession_reference_is_blocked(tmp_path):
    record, _ = make_record(tmp_path)
    record["supersedes_record_id"] = "absent-parent"
    history = resolve_record_history([record], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert not history.valid
    assert "supersedes_reference_broken" in history.errors


def test_supersession_cycle_is_blocked(tmp_path):
    first, _ = make_record(tmp_path, record_id="first")
    second = copy.deepcopy(first)
    second["acceptance_record_id"] = "second"
    first["supersedes_record_id"] = "second"
    second["supersedes_record_id"] = "first"
    history = resolve_record_history([first, second], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert not history.valid
    assert "supersession_cycle" in history.errors


def test_supersession_across_artifact_versions_is_blocked(tmp_path):
    first, _ = make_record(tmp_path, record_id="first")
    second = copy.deepcopy(first)
    second["acceptance_record_id"] = "second"
    second["supersedes_record_id"] = "first"
    second["source_artifacts"][0]["sha256"] = "0" * 64
    history = resolve_record_history([first, second], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert not history.valid
    assert "history_artifact_version_mismatch" in history.errors


def test_history_resolution_is_order_independent_and_newer_defer_blocks_old_acceptance(tmp_path):
    old, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification", record_id="old")
    new = copy.deepcopy(old)
    new["acceptance_record_id"] = "new"
    new["supersedes_record_id"] = "old"
    new["decision"] = {
        "status": "deferred_pending_evidence", "authorized_scope": "none", "decision_at_utc": None,
        "approver": {"kind": "none", "identifier": None, "role": None},
        "policy_version": None, "rationale": "Await evidence.",
    }
    forward = resolve_record_history([old, new], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    reverse = resolve_record_history([new, old], expected_run_id=RUN_ID, expected_pattern=PATTERN, expected_output_path=OUTPUT)
    assert forward.valid and reverse.valid
    assert forward.ordered_record_ids == reverse.ordered_record_ids == ("old", "new")
    assert not authorize(old, [old, new], tmp_path, "cpu_memory_classification").authorized
    assert not authorize(new, [new, old], tmp_path, "cpu_memory_classification").authorized


def test_stable_run1_synthetic_deferred_counts_are_preserved_and_unusable(tmp_path):
    record, _ = make_record(tmp_path, record_id="stable-count-review")
    record["request_count_evidence"] = {"stats_csv_or_history": 457, "terminal_summary": 458}
    assert validate_record_structure(record).valid
    assert record["request_count_evidence"] == {"stats_csv_or_history": 457, "terminal_summary": 458}
    assert not authorize(record, [record], tmp_path, "cpu_memory_classification").authorized
    assert not authorize(record, [record], tmp_path, "throttling_analysis").authorized


def test_validation_and_authorization_do_not_mutate_record_or_files(tmp_path):
    record, files = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    original = copy.deepcopy(record)
    before = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in files
    }
    assert validate_record_structure(record).valid
    assert verify(record, tmp_path).valid
    assert authorize(record, [record], tmp_path, "cpu_memory_classification").authorized
    assert record == original
    after = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in files
    }
    assert before == after


@pytest.mark.parametrize("field,wrong,expected", [
    ("run_id", "other-run", "experiment_metadata_run_id_mismatch"),
    ("pattern", "spike", "experiment_metadata_pattern_mismatch"),
])
def test_experiment_metadata_identity_mismatch_fails(tmp_path, field, wrong, expected):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    metadata_artifact = next(
        item for item in record["source_artifacts"]
        if item["role"] == "experiment_metadata"
    )
    path = tmp_path / metadata_artifact["path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = wrong
    path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = verify(record, tmp_path)
    assert not result.valid
    assert expected in result.errors


def test_empty_observations_are_rejected(tmp_path):
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    observations_artifact = next(
        item for item in record["source_artifacts"]
        if item["role"] == "observations"
    )
    path = tmp_path / observations_artifact["path"]
    path.write_text("timestamp_utc,run_id,pattern\n", encoding="utf-8")
    observations_artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = verify(record, tmp_path)
    assert not result.valid
    assert "observations_empty" in result.errors


def test_missing_required_artifact_role_is_rejected(tmp_path):
    record, _ = make_record(tmp_path)
    record["source_artifacts"] = [
        item for item in record["source_artifacts"]
        if item["role"] != "experiment_metadata"
    ]
    result = validate_record_structure(record)
    assert not result.valid
    assert "required_source_artifact_role_missing" in result.errors


def test_duplicate_singleton_artifact_role_is_rejected(tmp_path):
    record, _ = make_record(tmp_path)
    manifest = next(
        item for item in record["source_artifacts"]
        if item["role"] == "manifest"
    )
    record["source_artifacts"].append(copy.deepcopy(manifest))
    result = validate_record_structure(record)
    assert not result.valid
    assert "duplicate_source_artifact_path" in result.errors
    assert "duplicate_singleton_source_artifact_role" in result.errors


def test_observed_evidence_with_reason_is_rejected(tmp_path):
    record, _ = make_record(tmp_path)
    record["target_identity"]["pod"]["reason"] = "Should be null when observed."
    result = validate_record_structure(record)
    assert not result.valid
    assert "target_identity_pod_observed_reason_must_be_null" in result.errors


def test_unavailable_evidence_without_reason_is_rejected(tmp_path):
    record, _ = make_record(tmp_path)
    record["resource_configuration"]["cpu_limit"] = {
        "status": "unknown",
        "value": None,
        "reason": None,
    }
    result = validate_record_structure(record)
    assert not result.valid
    assert "resource_configuration_cpu_limit_unavailable_value_requires_reason" in result.errors
