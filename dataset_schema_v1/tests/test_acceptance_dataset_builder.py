from __future__ import annotations

import json

import pytest

from dataset_schema_v1 import acceptance_dataset_builder as integration
from dataset_schema_v1.dataset_builder import DatasetBuildResult
from test_acceptance_records import make_record


RUN_ID = "synthetic-stable-001"
OUTPUT = "validated/synthetic-stable-001"


def _fixture_identity(monkeypatch):
    import test_acceptance_records as record_fixtures

    monkeypatch.setattr(record_fixtures, "RUN_ID", RUN_ID)
    monkeypatch.setattr(record_fixtures, "PATTERN", "stable")
    monkeypatch.setattr(record_fixtures, "OUTPUT", OUTPUT)


def _inputs(root, *, second_run=False):
    output = root / OUTPUT
    sources = {RUN_ID: output}
    metadata = {RUN_ID: {"pattern": "stable", "designated_output_dir": OUTPUT}}
    if second_run:
        other = "synthetic-stable-unauthorized"
        sources[other] = root / "validated" / other
        metadata[other] = {"pattern": "stable", "designated_output_dir": f"validated/{other}"}
    return sources, metadata


def _write_record(root, record):
    store = root / "dataset_schema_v1" / "acceptance_records"
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{record['acceptance_record_id']}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def test_only_authorized_source_dirs_reach_builder_and_parameters_are_forwarded(tmp_path, monkeypatch):
    _fixture_identity(monkeypatch)
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    _write_record(tmp_path, record)
    sources, metadata = _inputs(tmp_path, second_run=True)
    observed = {}

    def builder(source_dirs, accepted, eligibility, expected, **kwargs):
        observed.update(source_dirs=dict(source_dirs), accepted=accepted, kwargs=kwargs)
        return DatasetBuildResult(())

    monkeypatch.setattr(integration, "build_in_memory", builder)
    result = integration.build_from_acceptance_store(
        sources, {}, metadata, project_root=tmp_path,
        purpose="cpu_memory_classification", window_size=4,
        cadence_seconds=7.5, tolerance_seconds=0.02,
    )

    assert set(observed["source_dirs"]) == {RUN_ID}
    assert observed["accepted"] == {RUN_ID}
    assert observed["kwargs"] == {
        "project_root": tmp_path.resolve(), "window_size": 4,
        "cadence_seconds": 7.5, "tolerance_seconds": 0.02,
    }
    assert [run.status for run in result.runs] == ["blocked", "blocked"]
    unauthorized = next(run for run in result.runs if run.run_id != RUN_ID)
    assert unauthorized.input_observation_count == 0


@pytest.mark.parametrize("purpose", ["", "training", None, []])
def test_invalid_purpose_fails_closed_without_calling_builder(tmp_path, monkeypatch, purpose):
    sources, metadata = _inputs(tmp_path)
    monkeypatch.setattr(integration, "build_in_memory", lambda *args, **kwargs: pytest.fail("builder called"))
    result = integration.build_from_acceptance_store(sources, {}, metadata, project_root=tmp_path,
                                                     purpose=purpose)
    assert result.runs[0].status == "blocked"
    assert "purpose_invalid" in result.runs[0].reasons[0]


@pytest.mark.parametrize("root", ["missing-project-root", None])
def test_missing_or_invalid_project_root_fails_closed(tmp_path, monkeypatch, root):
    sources, metadata = _inputs(tmp_path)
    monkeypatch.setattr(integration, "build_in_memory", lambda *args, **kwargs: pytest.fail("builder called"))
    result = integration.build_from_acceptance_store(sources, {}, metadata,
                                                     project_root=root, purpose="cpu_memory_classification")
    assert result.runs[0].status == "blocked"
    assert "project_root_invalid" in result.runs[0].reasons[0]


def test_missing_approval_blocks_and_never_passes_source_to_builder(tmp_path, monkeypatch):
    _fixture_identity(monkeypatch)
    record, _ = make_record(tmp_path, status="eligible_for_review")
    _write_record(tmp_path, record)
    sources, metadata = _inputs(tmp_path)
    calls = []
    monkeypatch.setattr(integration, "build_in_memory", lambda source_dirs, *a, **k: (calls.append(dict(source_dirs)) or DatasetBuildResult(())))

    result = integration.build_from_acceptance_store(sources, {}, metadata, project_root=tmp_path,
                                                     purpose="cpu_memory_classification")
    assert calls == [{}]
    assert result.runs[0].status == "blocked"
    assert result.runs[0].input_observation_count == 0


@pytest.mark.parametrize("failure", ["wrong_output", "hash_mismatch", "conflicting_history", "superseded"])
def test_invalid_or_noncurrent_record_never_reaches_builder(tmp_path, monkeypatch, failure):
    _fixture_identity(monkeypatch)
    record, _ = make_record(tmp_path, status="accepted_for_cpu_memory_classification")
    _write_record(tmp_path, record)
    sources, metadata = _inputs(tmp_path)
    if failure == "wrong_output":
        metadata[RUN_ID]["designated_output_dir"] = "validated/elsewhere"
    elif failure == "hash_mismatch":
        (tmp_path / OUTPUT / "observations.csv").write_text("changed", encoding="utf-8")
    elif failure == "conflicting_history":
        second, _ = make_record(tmp_path, status="eligible_for_review", record_id="record-b")
        _write_record(tmp_path, second)
    elif failure == "superseded":
        second, _ = make_record(tmp_path, status="eligible_for_review", record_id="record-b")
        second["supersedes_record_id"] = record["acceptance_record_id"]
        _write_record(tmp_path, second)

    calls = []
    monkeypatch.setattr(integration, "build_in_memory", lambda source_dirs, *a, **k: (calls.append(dict(source_dirs)) or DatasetBuildResult(())))
    result = integration.build_from_acceptance_store(sources, {}, metadata, project_root=tmp_path,
                                                     purpose="cpu_memory_classification")
    # A malformed/conflicting complete store blocks before the builder; other
    # invalid records may reach it only with an empty source mapping.
    assert calls in ([], [{}])
    assert result.runs[0].status == "blocked"
    assert result.runs[0].input_observation_count == 0
