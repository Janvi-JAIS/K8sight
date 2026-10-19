"""Synthetic offline integration tests for the Stage 3 in-memory builder."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dataset_schema_v1.dataset_builder import build_in_memory
from dataset_schema_v1.eligibility_audit import RunSpec, audit_run
from dataset_schema_v1.validate_run import COLS


FIXTURE = Path(__file__).parent / "fixtures" / "valid_run"
RUN_ID = "synthetic-increasing-001"
PATTERN = "increasing"
OUTPUT_REL = "validated/run"
METADATA_REL = "run_records/synthetic/experiment_metadata.json"
BASE = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)


def make_workspace(*, training_eligible=True, phase_layout=False):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    output = root / OUTPUT_REL
    output.parent.mkdir(parents=True)
    shutil.copytree(FIXTURE, output)
    row_count = 10 if phase_layout else 8
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(run_id=RUN_ID, pattern=PATTERN)
    manifest["data_quality"].update(expected_samples=row_count, observed_samples=row_count, source_integrity="verified")
    boundaries = manifest["boundaries"]
    if phase_layout:
        start = datetime(2026, 1, 1, 0, 0, 15, tzinfo=timezone.utc)
        end = start + timedelta(seconds=120)
        boundaries.update(
            run_start_utc=start.isoformat().replace("+00:00", "Z"),
            workload_end_utc=end.isoformat().replace("+00:00", "Z"),
            cooldown_start_utc=end.isoformat().replace("+00:00", "Z"),
            locust_shutdown_utc=end.isoformat().replace("+00:00", "Z"),
            collection_start_utc="2026-01-01T00:00:00Z",
            collection_end_utc=end.isoformat().replace("+00:00", "Z"),
        )
        manifest["metrics"][0]["query_end_utc"] = end.isoformat().replace("+00:00", "Z")
        manifest["metrics"][1]["query_end_utc"] = end.isoformat().replace("+00:00", "Z")
        duration = 120
    else:
        boundaries.update(
            run_start_utc="2026-01-01T00:00:00Z",
            workload_end_utc="2026-01-01T00:05:00Z",
            cooldown_start_utc=None,
            locust_shutdown_utc="2026-01-01T00:05:00Z",
            collection_start_utc="2026-01-01T00:00:00Z",
            collection_end_utc="2026-01-01T00:05:00Z",
        )
        duration = 300
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    observations_path = output / "observations.csv"
    with observations_path.open(encoding="utf-8-sig", newline="") as stream:
        template = next(csv.DictReader(stream))
    rows = []
    for index in range(row_count):
        current = dict(template)
        if phase_layout:
            stamp = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=15 * index)
            if index == 0:
                phase, rate_status, lookback = "pre_start", "unknown", ""
            else:
                stamp = datetime(2026, 1, 1, 0, 0, 15, tzinfo=timezone.utc) + timedelta(seconds=15 * (index - 1))
                phase, rate_status, lookback = "active", "fully_within_active", "0"
            if index == row_count - 1:
                stamp = datetime(2026, 1, 1, 0, 2, 15, tzinfo=timezone.utc)
                phase, rate_status, lookback = "cooldown", "unknown", ""
        else:
            stamp = BASE + timedelta(seconds=15 * index)
            phase, rate_status, lookback = "active", "fully_within_active", "60"
        current.update(
            timestamp_utc=stamp.isoformat().replace("+00:00", "Z"),
            run_id=RUN_ID,
            pattern=PATTERN,
            pod_uid="uid-synthetic",
            container_id="containerd://synthetic",
            sample_phase=phase,
            cpu_usage_cores="0.15",
            cpu_rate_lookback_seconds=lookback,
            cpu_rate_window_status=rate_status,
            memory_working_set_bytes=str(10 * 1024 * 1024),
            quality_flags="[]",
            missing_reasons=json.dumps({
                "cfs_throttled_periods_total": "not_collected",
                "cfs_periods_total": "not_collected",
                "cfs_throttled_seconds_total": "not_collected",
                "throttled_period_ratio_1m": "not_collected",
            }, separators=(",", ":")),
        )
        rows.append(current)
    with observations_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = root / METADATA_REL
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({
        "run_id": RUN_ID,
        "pattern": PATTERN,
        "training_eligible": training_eligible,
        "duration_seconds": duration,
        "boundaries": {
            "run_start_utc": boundaries["run_start_utc"],
            "workload_end_utc": boundaries["workload_end_utc"],
        },
    }), encoding="utf-8")
    spec = RunSpec(RUN_ID, PATTERN, (OUTPUT_REL,), METADATA_REL)
    audit = audit_run(spec, root)
    expected = {
        RUN_ID: {
            "run_id": RUN_ID,
            "pattern": PATTERN,
            "designated_output_dir": OUTPUT_REL,
            "experiment_metadata_path": METADATA_REL,
            "replicate": 1,
            "seed": 42,
            "observation_units": {"cpu_usage_cores": "cores", "memory_working_set_bytes": "bytes"},
        }
    }
    return temporary, root, output, rows, expected, {RUN_ID: audit}


def build(root, expected, evidence, *, dirs=None, accepted=(RUN_ID,)):
    return build_in_memory(
        {RUN_ID: root / OUTPUT_REL} if dirs is None else dirs,
        accepted,
        evidence,
        expected,
        project_root=root,
    )


@pytest.fixture
def workspace():
    values = make_workspace()
    try:
        yield values[1:]
    finally:
        values[0].cleanup()


def test_valid_synthetic_accepted_run_flows_through_audit_selection_and_windows(workspace):
    root, _, rows, expected, evidence = workspace
    result = build(root, expected, evidence)
    run = result.runs[0]
    assert run.status == "review_required"  # Stage 1 reports unknown throttling; acceptance remains separate.
    assert run.source_verification["schema_validity"] == "pass"
    assert run.acceptance_evidence["decision"] == "accepted"
    assert run.eligibility_evidence["training_eligibility"]["observed"] is True
    assert run.input_observation_count == 8
    assert run.selected_observation_count == 8
    assert run.candidate_window_count == 8
    assert run.eligible_window_count == 1
    window = next(item for item in run.windows if item.included)
    assert window.run_id == RUN_ID and len(window.source_row_ids) == 8
    assert window.source_timestamps_utc[0] == rows[0]["timestamp_utc"]


@pytest.mark.parametrize("missing", ["manifest.json", "observations.csv"])
def test_missing_required_source_file_blocks_run(workspace, missing):
    root, output, _, expected, evidence = workspace
    (output / missing).unlink()
    run = build(root, expected, evidence).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


@pytest.mark.parametrize("field,value", [("run_id", "other-id"), ("pattern", "spike")])
def test_manifest_identity_mismatch_blocks_run(workspace, field, value):
    root, output, _, expected, evidence = workspace
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    run = build(root, expected, evidence).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


def test_manifest_csv_count_disagreement_blocks_run(workspace):
    root, output, _, expected, evidence = workspace
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["data_quality"]["observed_samples"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    run = build(root, expected, evidence).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


@pytest.mark.parametrize("evidence", [None, {"run_id": RUN_ID, "status": "maybe"}])
def test_missing_or_malformed_audit_evidence_blocks_run(workspace, evidence):
    root, _, _, expected, _ = workspace
    run = build(root, expected, {RUN_ID: evidence}).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


def test_audit_evidence_for_another_run_blocks_run(workspace):
    root, _, _, expected, evidence = workspace
    wrong = dict(evidence[RUN_ID], run_id="different-run")
    run = build(root, expected, {RUN_ID: wrong}).runs[0]
    assert run.status == "blocked"
    assert "eligibility_evidence_run_id_mismatch" in run.reasons
    assert run.eligible_window_count == 0


def test_audit_evidence_for_another_source_path_blocks_run(workspace):
    root, _, _, expected, evidence = workspace
    wrong = dict(evidence[RUN_ID], output_dir=str(root / "validated/other"))
    run = build(root, expected, {RUN_ID: wrong}).runs[0]
    assert run.status == "blocked"
    assert "eligibility_evidence_source_path_mismatch" in run.reasons


def test_duplicate_output_mapping_blocks_ambiguous_runs(workspace):
    root, _, _, expected, evidence = workspace
    other = "synthetic-increasing-002"
    expected = copy.deepcopy(expected)
    expected[other] = dict(expected[RUN_ID], run_id=other)
    evidence = dict(evidence)
    evidence[other] = dict(evidence[RUN_ID], run_id=other)
    runs = build(root, expected, evidence, dirs={RUN_ID: root / OUTPUT_REL, other: root / OUTPUT_REL}).runs
    assert len(runs) == 2
    assert all(run.status == "blocked" and run.eligible_window_count == 0 for run in runs)
    assert all("duplicate_or_ambiguous_source_mapping" in run.reasons for run in runs)


def test_source_integrity_failure_blocks_run(workspace):
    root, output, _, expected, evidence = workspace
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["data_quality"]["source_integrity"] = "failed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    run = build(root, expected, evidence).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


def test_schema_invalid_run_never_produces_windows(workspace):
    root, output, _, expected, evidence = workspace
    path = output / "observations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["sample_phase"] = "not-a-phase"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    run = build(root, expected, evidence).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0


def test_container_identity_change_is_blocked_by_official_validation(workspace):
    root, output, _, expected, _ = workspace
    path = output / "observations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[4]["container_id"] = "containerd://different"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    fresh = audit_run(RunSpec(RUN_ID, PATTERN, (OUTPUT_REL,), METADATA_REL), root)
    run = build(root, expected, {RUN_ID: fresh}).runs[0]
    assert run.status == "blocked" and run.eligible_window_count == 0
    assert run.source_verification["schema_validity"] == "fail"


def test_schema_valid_but_unaccepted_and_missing_acceptance_produce_zero_windows(workspace):
    root, _, _, expected, evidence = workspace
    unaccepted = build(root, expected, evidence, accepted=()).runs[0]
    missing = build(root, expected, evidence, accepted=None).runs[0]
    assert unaccepted.status == "not_accepted" and unaccepted.eligible_window_count == 0
    assert missing.status == "not_accepted" and missing.eligible_window_count == 0
    assert unaccepted.source_verification["schema_validity"] == "pass"


def test_adjacent_rows_from_distinct_runs_never_combine_into_window(workspace):
    root, output_a, _, expected_a, evidence_a = workspace
    run_b = "synthetic-increasing-002"
    # Keep four adjacent observations in each separately verified run.
    manifest_a_path = output_a / "manifest.json"
    manifest_a = json.loads(manifest_a_path.read_text(encoding="utf-8"))
    manifest_a["data_quality"].update(expected_samples=4, observed_samples=4)
    manifest_a_path.write_text(json.dumps(manifest_a), encoding="utf-8")
    observations_a_path = output_a / "observations.csv"
    with observations_a_path.open(encoding="utf-8-sig", newline="") as stream:
        rows_a = list(csv.DictReader(stream))[:4]
    with observations_a_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows_a)
    metadata_a_rel = "run_records/synthetic/experiment_metadata.json"
    evidence_a = {RUN_ID: audit_run(RunSpec(RUN_ID, PATTERN, (OUTPUT_REL,), metadata_a_rel), root)}
    output_b = root / "validated/run-b"
    shutil.copytree(output_a, output_b)
    manifest_path = output_b / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_b
    manifest["data_quality"].update(expected_samples=4, observed_samples=4)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observations_path = output_b / "observations.csv"
    with observations_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[:4]
    for row in rows:
        row["run_id"] = run_b
    with observations_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    metadata_b_rel = "run_records/synthetic-b/experiment_metadata.json"
    metadata_b_path = root / metadata_b_rel
    metadata_b_path.parent.mkdir(parents=True)
    metadata_b_path.write_text(json.dumps({
        "run_id": run_b,
        "pattern": PATTERN,
        "training_eligible": True,
        "duration_seconds": 300,
        "boundaries": {"run_start_utc": "2026-01-01T00:00:00Z", "workload_end_utc": "2026-01-01T00:05:00Z"},
    }), encoding="utf-8")
    evidence_b = audit_run(RunSpec(run_b, PATTERN, ("validated/run-b",), metadata_b_rel), root)
    expected = copy.deepcopy(expected_a)
    expected[run_b] = {**expected[RUN_ID], "run_id": run_b, "designated_output_dir": "validated/run-b", "experiment_metadata_path": metadata_b_rel}
    evidence = {**evidence_a, run_b: evidence_b}
    result = build_in_memory(
        {RUN_ID: output_a, run_b: output_b}, {RUN_ID, run_b}, evidence, expected, project_root=root,
    )
    assert len(result.runs) == 2
    assert all(item.selected_observation_count == 4 for item in result.runs)
    assert all(item.eligible_window_count == 0 for item in result.runs)


def test_accepted_but_training_ineligible_run_produces_no_windows():
    temporary, root, _, _, expected, evidence = make_workspace(training_eligible=False)
    try:
        run = build(root, expected, evidence).runs[0]
        assert run.status == "blocked" and run.eligible_window_count == 0
        assert "training_ineligible" in run.reasons
    finally:
        temporary.cleanup()


def test_phase_rows_are_reported_as_excluded_and_not_windowed():
    temporary, root, _, _, expected, evidence = make_workspace(phase_layout=True)
    try:
        run = build(root, expected, evidence).runs[0]
        assert run.status in {"review_required", "eligible_for_windowing"}
        assert run.input_observation_count == 10
        assert run.selected_observation_count == 8
        assert run.excluded_observation_counts["sample_phase_not_active"] == 2
        assert run.eligible_window_count == 1
        window = next(item for item in run.windows if item.included)
        assert window.source_timestamps_utc[0] == "2026-01-01T00:00:15Z"
        assert window.source_timestamps_utc[-1] == "2026-01-01T00:02:00Z"
    finally:
        temporary.cleanup()


def test_rate_status_missing_values_and_quality_flags_follow_stage2_policy(workspace):
    root, output, _, expected, evidence = workspace
    path = output / "observations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["cpu_rate_window_status"] = "unknown"
    rows[0]["cpu_rate_lookback_seconds"] = ""
    rows[1]["cpu_usage_cores"] = ""
    rows[1]["cpu_rate_window_status"] = "not_applicable"
    rows[1]["cpu_rate_lookback_seconds"] = ""
    reasons = json.loads(rows[1]["missing_reasons"])
    reasons["cpu_usage_cores"] = "synthetic_missing"
    rows[1]["missing_reasons"] = json.dumps(reasons)
    rows[2]["quality_flags"] = '["synthetic_flag"]'
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    # Recompute the caller evidence from the changed synthetic source as required by the API.
    fresh = audit_run(RunSpec(RUN_ID, PATTERN, (OUTPUT_REL,), METADATA_REL), root)
    run = build(root, expected, {RUN_ID: fresh}).runs[0]
    assert run.selected_observation_count == 5
    assert run.excluded_observation_counts["cpu_rate_window_not_fully_active"] == 2
    assert run.excluded_observation_counts["cpu_missing_or_invalid"] == 1
    assert run.excluded_observation_counts["quality_flags_present"] == 1
    assert run.eligible_window_count == 0


def test_missing_throttling_ratio_alone_does_not_exclude_window(workspace):
    root, _, _, expected, evidence = workspace
    run = build(root, expected, evidence).runs[0]
    assert run.eligible_window_count == 1
    included = next(item for item in run.row_decisions if item.included)
    assert included.row["throttled_period_ratio_1m"] == ""
    assert json.loads(included.row["missing_reasons"])["throttled_period_ratio_1m"] == "not_collected"


def test_cadence_gap_is_reported_and_never_bridged(workspace):
    root, output, _, expected, evidence = workspace
    path = output / "observations.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[4]["timestamp_utc"] = (BASE + timedelta(seconds=65)).isoformat().replace("+00:00", "Z")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    fresh = audit_run(RunSpec(RUN_ID, PATTERN, (OUTPUT_REL,), METADATA_REL), root)
    run = build(root, expected, {RUN_ID: fresh}).runs[0]
    assert run.eligible_window_count == 0
    assert "cadence_gap" in run.excluded_window_counts


def test_input_files_and_mappings_are_unchanged_and_results_are_deterministic(workspace):
    root, output, rows, expected, evidence = workspace
    rows_before = copy.deepcopy(rows)
    expected_before = copy.deepcopy(expected)
    evidence_before = copy.deepcopy(evidence)
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    first = build(root, expected, evidence)
    second = build(root, expected, evidence)
    hashes_after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    assert first == second
    assert rows == rows_before and expected == expected_before and evidence == evidence_before
    assert hashes_before == hashes_after
    assert tuple(sorted(path for path in root.rglob("*") if path.is_file())) == files

