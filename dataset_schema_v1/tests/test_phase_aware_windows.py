"""Offline tests for phase-aware selection and provenance-rich windows."""
from __future__ import annotations

import copy
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dataset_schema_v1.phase_aware_windows import (
    build_feature_windows,
    select_eligible_observations,
)


BASE = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
METADATA = {
    "run-a": {
        "schema_validity": "pass",
        "training_eligible": True,
        "pattern": "stable",
        "replicate": 1,
        "seed": 42,
        "observation_units": {"cpu_usage_cores": "cores", "memory_working_set_bytes": "bytes"},
    }
}


def row(index, *, run_id="run-a", pattern="stable", phase="active", status="fully_within_active", timestamp=None, **changes):
    stamp = timestamp or BASE + timedelta(seconds=index * 15)
    result = {
        "run_id": run_id,
        "pattern": pattern,
        "replicate": 1,
        "seed": 42,
        "timestamp_utc": stamp.isoformat().replace("+00:00", "Z"),
        "sample_phase": phase,
        "namespace": "default",
        "pod": "php-apache-abc",
        "pod_uid": "uid-a",
        "container": "hpa-example",
        "container_id": "containerd://abc",
        "cpu_usage_cores": 0.1,
        "cpu_rate_window_status": status,
        "memory_working_set_bytes": 10 * 1024 * 1024,
        "throttled_period_ratio_1m": None,
        "quality_flags": [],
        "missing_reasons": {"throttled_period_ratio_1m": "insufficient_lookback"},
    }
    result.update(changes)
    return result


def select(rows, *, metadata=None, accepted=("run-a",)):
    return select_eligible_observations(rows, METADATA if metadata is None else metadata, accepted)


def windows(rows, *, metadata=None, accepted=("run-a",), **kwargs):
    metadata = METADATA if metadata is None else metadata
    return build_feature_windows(select_eligible_observations(rows, metadata, accepted), metadata, **kwargs)


def codes(decision):
    return set(decision.reasons)


def test_acceptance_is_explicit_and_training_eligibility_is_separate():
    one = [row(0)]
    assert select(one, accepted=()).included_count == 0
    assert "run_not_explicitly_accepted" in codes(select(one, accepted=()).decisions[0])
    assert "acceptance_set_missing" in codes(select(one, accepted=None).decisions[0])

    ineligible = copy.deepcopy(METADATA)
    ineligible["run-a"]["training_eligible"] = False
    decision = select(one, metadata=ineligible).decisions[0]
    assert "training_ineligible" in codes(decision)
    assert "schema_validity_not_pass" not in codes(decision)
    assert select(one, metadata={}).excluded_count == 1


@pytest.mark.parametrize("phase", ["pre_start", "cooldown", "post_shutdown", "unknown"])
def test_non_active_phases_are_excluded(phase):
    decision = select([row(0, phase=phase)]).decisions[0]
    assert not decision.included
    assert "sample_phase_not_active" in codes(decision)


@pytest.mark.parametrize("status", ["crosses_pre_start", "crosses_cooldown", "unknown"])
def test_incomplete_cpu_rate_windows_are_excluded(status):
    decision = select([row(0, status=status)]).decisions[0]
    assert "cpu_rate_window_not_fully_active" in codes(decision)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [({"cpu_usage_cores": None}, "cpu_missing_or_invalid"),
     ({"memory_working_set_bytes": ""}, "memory_missing_or_invalid"),
     ({"quality_flags": ["counter_reset"]}, "quality_flags_present")],
)
def test_missing_required_features_and_quality_flags_are_excluded(changes, reason):
    decision = select([row(0, **changes)]).decisions[0]
    assert not decision.included
    assert reason in codes(decision)


def test_missing_quality_or_missing_reason_fields_fail_closed():
    incomplete = row(0)
    del incomplete["quality_flags"]
    assert "quality_flags_invalid" in codes(select([incomplete]).decisions[0])
    incomplete = row(0)
    del incomplete["missing_reasons"]
    assert "missing_reasons_invalid" in codes(select([incomplete]).decisions[0])


def test_missing_throttling_ratio_is_preserved_but_not_required():
    rows = [row(i) for i in range(8)]
    selected = select(rows)
    assert selected.included_count == 8
    assert selected.eligible_rows[0].row["throttled_period_ratio_1m"] is None
    assert selected.eligible_rows[0].row["missing_reasons"]["throttled_period_ratio_1m"] == "insufficient_lookback"
    assert len(build_feature_windows(selected, METADATA).eligible_windows) == 1


def test_eight_rows_make_a_window_with_complete_provenance():
    rows = [row(i, memory_working_set_bytes="9007199254740993") for i in range(8)]
    result = windows(rows)
    window = result.eligible_windows[0]
    assert window.run_id == "run-a"
    assert window.pattern == "stable"
    assert window.replicate == 1 and window.seed == 42
    assert window.start_timestamp_utc == "2026-10-14T12:00:00Z"
    assert window.end_timestamp_utc == "2026-10-14T12:01:45Z"
    assert window.source_observation_count == 8
    assert len(window.source_timestamps_utc) == 8
    assert window.feature_units == {"cpu_usage_cores": "cores", "memory_working_set_bytes": "bytes"}
    assert window.features["memory_working_set_bytes"] == (9007199254740993,) * 8


def test_run_id_is_mandatory_and_rows_from_runs_never_combine():
    metadata = copy.deepcopy(METADATA)
    metadata["run-b"] = {**METADATA["run-a"], "pattern": "stable"}
    split = [row(i, run_id="run-a") for i in range(4)] + [row(i + 4, run_id="run-b") for i in range(4)]
    result = windows(split, metadata=metadata, accepted=("run-a", "run-b"))
    assert not result.eligible_windows
    assert all(item.run_id in {"run-a", "run-b"} for item in result.decisions)

    missing = row(0)
    del missing["run_id"]
    selected = select([missing])
    assert not build_feature_windows(selected, METADATA).eligible_windows
    assert "run_id_missing" in build_feature_windows(selected, METADATA).exclusion_counts


def test_pattern_phase_and_identity_changes_block_windows():
    pattern_rows = [row(i, pattern="other" if i == 3 else "stable") for i in range(8)]
    assert not windows(pattern_rows).eligible_windows
    phase_rows = [row(i, phase="cooldown" if i == 3 else "active") for i in range(8)]
    assert not windows(phase_rows).eligible_windows
    identity_rows = [row(i, **({"pod_uid": "uid-b"} if i == 3 else {})) for i in range(8)]
    identity_result = windows(identity_rows)
    assert not identity_result.eligible_windows
    assert "container_identity_changed_or_missing" in identity_result.exclusion_counts
    runtime_rows = [row(i, **({"container_id": "containerd://other"} if i == 3 else {})) for i in range(8)]
    assert not windows(runtime_rows).eligible_windows


def test_duplicate_timestamp_invalidates_affected_candidates():
    rows = [row(i) for i in range(8)]
    rows[7] = row(6)
    result = windows(rows)
    assert not result.eligible_windows
    assert "duplicate_timestamp" in result.exclusion_counts


def test_gap_or_missing_feature_is_not_bridged():
    gap_rows = [row(i, timestamp=BASE + timedelta(seconds=(i + (1 if i >= 4 else 0)) * 15)) for i in range(8)]
    gap_result = windows(gap_rows)
    assert not gap_result.eligible_windows
    assert "cadence_gap" in gap_result.exclusion_counts

    missing_rows = [row(i, **({"cpu_usage_cores": None} if i == 3 else {})) for i in range(8)]
    missing_result = windows(missing_rows)
    assert not missing_result.eligible_windows
    assert "cpu_missing_or_invalid" in missing_result.exclusion_counts


def test_out_of_order_input_is_normalized_by_timestamp():
    result = windows([row(i) for i in reversed(range(8))])
    window = result.eligible_windows[0]
    assert window.source_indices == tuple(reversed(range(8)))
    assert window.source_timestamps_utc == tuple(row(i)["timestamp_utc"] for i in range(8))


@pytest.mark.parametrize("kwargs", [
    {"window_size": 1}, {"window_size": 2.5}, {"cadence_seconds": 0},
    {"cadence_seconds": float("nan")}, {"tolerance_seconds": -1},
    {"cadence_seconds": 1, "tolerance_seconds": 1},
])
def test_invalid_window_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        windows([row(0)], **kwargs)


def test_input_is_unchanged_results_are_deterministic_and_no_files_are_written():
    rows = [row(i) for i in range(8)]
    before = copy.deepcopy(rows)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        before_paths = tuple(directory.iterdir())
        first = windows(rows)
        second = windows(rows)
        assert tuple(directory.iterdir()) == before_paths
        assert first == second
    assert rows == before

