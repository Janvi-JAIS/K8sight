#!/usr/bin/env python3
"""Read-only evidence audit for designated K8sight validated runs.

This reports evidence quality and recorded training metadata. It does not
accept/reject runs for training and never writes project files.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dataset_schema_v1.validate_run import COLS, load_run


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    pattern: str
    output_dirs: tuple[str, ...]
    metadata_path: str


RUN_SPECS = (
    RunSpec("stable-low-r1-seed42-20261013-day3-01", "stable", ("dataset_schema_v1/validated_runs_day3/stable-low-r1-seed42-20261013-day3-01-outcome-fix",), "run_records/stable-low-r1-seed42-20261013-day3-01/experiment_metadata.json"),
    RunSpec("stable-low-r2-seed123-20261013-day3-02", "stable", ("dataset_schema_v1/validated_runs_day3/stable-low-r2-seed123-20261013-day3-02-context-enriched",), "run_records/stable-low-r2-seed123-20261013-day3-02/experiment_metadata.json"),
    RunSpec("stable-low-r3-seed20261018-day5-01-retry01", "stable", ("dataset_schema_v1/validated_runs_day5/stable-low-r3-seed20261018-day5-01-retry01-schema-validated-01",), "run_records/stable-low-r3-seed20261018-day5-01-retry01/experiment_metadata.json"),
    RunSpec("stable-low-r4-seed20261019-day5-02", "stable", ("dataset_schema_v1/validated_runs_day5/stable-low-r4-seed20261019-day5-02-schema-validated-01",), "run_records/stable-low-r4-seed20261019-day5-02/experiment_metadata.json"),
    RunSpec("increasing-low-r1-seed2026-20261014-day4-01", "increasing", ("dataset_schema_v1/validated_runs_day4/increasing-low-r1-seed2026-20261014-day4-01",), "run_records/increasing-low-r1-seed2026-20261014-day4-01/experiment_metadata.json"),
    RunSpec("increasing-low-r2-seed20261020-20261018-day5-03", "increasing", ("dataset_schema_v1/validated_runs_day5/increasing-low-r2-seed20261020-20261018-day5-03-schema-validated-01",), "run_records/increasing-low-r2-seed20261020-20261018-day5-03/experiment_metadata.json"),
    RunSpec("increasing-low-r3-seed20261021-20261018-day5-04", "increasing", ("dataset_schema_v1/validated_runs_day5/increasing-low-r3-seed20261021-20261018-day5-04-schema-validated-01",), "run_records/increasing-low-r3-seed20261021-20261018-day5-04/experiment_metadata.json"),
    RunSpec("increasing-low-r4-seed20261022-20261018-day5-05", "increasing", ("dataset_schema_v1/validated_runs_day5/increasing-low-r4-seed20261022-20261018-day5-05-schema-validated-01",), "run_records/increasing-low-r4-seed20261022-20261018-day5-05/experiment_metadata.json"),
    RunSpec("periodic-low-r1-seed20261014-day4-01", "periodic", ("dataset_schema_v1/validated_runs_day4/periodic-low-r1-seed20261014-day4-01-context-validated-01",), "run_records/periodic-low-r1-seed20261014-day4-01/experiment_metadata.json"),
    RunSpec("periodic-low-r2-seed20261015-20261018-day5-06", "periodic", ("dataset_schema_v1/validated_runs_day5/periodic-low-r2-seed20261015-20261018-day5-06-schema-validated-01",), "run_records/periodic-low-r2-seed20261015-20261018-day5-06/experiment_metadata.json"),
    RunSpec("periodic-low-r3-seed20261016-20261018-day5-07", "periodic", ("dataset_schema_v1/validated_runs_day5/periodic-low-r3-seed20261016-20261018-day5-07-schema-validated-01",), "run_records/periodic-low-r3-seed20261016-20261018-day5-07/experiment_metadata.json"),
    RunSpec("periodic-low-r4-seed20261017-20261018-day5-08", "periodic", ("dataset_schema_v1/validated_runs_day5/periodic-low-r4-seed20261017-20261018-day5-08-schema-validated-01",), "run_records/periodic-low-r4-seed20261017-20261018-day5-08/experiment_metadata.json"),
    RunSpec("spike-low-r1-seed20261014-day4-01", "spike", ("dataset_schema_v1/validated_runs_day4/spike-low-r1-seed20261014-day4-01-context-validated-01",), "run_records/spike-low-r1-seed20261014-day4-01/experiment_metadata.json"),
    RunSpec("spike-low-r2-seed20261015-20261018-day5-09", "spike", ("dataset_schema_v1/validated_runs_day5/spike-low-r2-seed20261015-20261018-day5-09-schema-validated-01",), "run_records/spike-low-r2-seed20261015-20261018-day5-09/experiment_metadata.json"),
    RunSpec("spike-low-r3-seed20261016-20261018-day5-10-retry01", "spike", ("dataset_schema_v1/validated_runs_day5/spike-low-r3-seed20261016-20261018-day5-10-retry01-schema-validated-01",), "run_records/spike-low-r3-seed20261016-20261018-day5-10-retry01/experiment_metadata.json"),
    RunSpec("spike-low-r4-seed20261017-20261018-day5-11", "spike", ("dataset_schema_v1/validated_runs_day5/spike-low-r4-seed20261017-20261018-day5-11-schema-validated-01",), "run_records/spike-low-r4-seed20261017-20261018-day5-11/experiment_metadata.json"),
    RunSpec("random_spike-low-r1-seed20261014-day4-01", "random_spike", ("dataset_schema_v1/validated_runs_day4/random_spike-low-r1-seed20261014-day4-01-context-validated-01",), "run_records/random_spike-low-r1-seed20261014-day4-01/experiment_metadata.json"),
    RunSpec("random_spike-low-r2-seed20261015-20261018-day5-12", "random_spike", ("dataset_schema_v1/validated_runs_day5/random_spike-low-r2-seed20261015-20261018-day5-12-schema-validated-01",), "run_records/random_spike-low-r2-seed20261015-20261018-day5-12/experiment_metadata.json"),
    RunSpec("random_spike-low-r3-seed20261016-20261018-day5-13", "random_spike", ("dataset_schema_v1/validated_runs_day5/random_spike-low-r3-seed20261016-20261018-day5-13-schema-validated-01",), "run_records/random_spike-low-r3-seed20261016-20261018-day5-13/experiment_metadata.json"),
    RunSpec("random_spike-low-r4-seed20261017-20261018-day5-14", "random_spike", ("dataset_schema_v1/validated_runs_day5/random_spike-low-r4-seed20261017-20261018-day5-14-schema-validated-01",), "run_records/random_spike-low-r4-seed20261017-20261018-day5-14/experiment_metadata.json"),
)

MEASUREMENT_COLUMNS = (
    "cpu_usage_cores", "memory_working_set_bytes",
    "cfs_throttled_periods_total", "cfs_periods_total",
    "cfs_throttled_seconds_total", "throttled_period_ratio_1m",
)
KNOWN_CAVEATS = {}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _parse_json_cell(value: str, expected_type, field: str, row_number: int):
    try:
        parsed = json.loads(value or ("[]" if expected_type is list else "{}"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"row {row_number}: malformed {field}: {exc}") from exc
    if not isinstance(parsed, expected_type):
        raise ValueError(f"row {row_number}: {field} must be {expected_type.__name__}")
    return parsed


_TERMINAL_AGGREGATE = re.compile(r"^\s*Aggregated\s+(\d+)\s+(\d+)\(")
_SHAPE_LOG_TIMESTAMP = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)\].*Shape test (starting|stopping)"
)


def _parse_nonnegative_integer(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} is not an integer") from None
    if parsed < 0:
        raise ValueError(f"{field} is negative")
    return parsed


def _read_terminal_summary(path: Path) -> tuple[dict, dict]:
    """Read the first Locust aggregate table and its percentile table."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {"status": "unavailable", "path": str(path), "reason": "file_missing"}, {
            "status": "unavailable", "path": str(path), "reason": "file_missing"
        }
    except (OSError, UnicodeError) as exc:
        return {"status": "malformed", "path": str(path), "reason": f"file_unreadable:{type(exc).__name__}"}, {
            "status": "malformed", "path": str(path), "reason": f"file_unreadable:{type(exc).__name__}"
        }

    count_row = next((match for line in lines if (match := _TERMINAL_AGGREGATE.match(line))), None)
    if count_row is None:
        count_result = {"status": "malformed", "path": str(path), "reason": "aggregate_request_summary_missing_or_malformed"}
    else:
        count_result = {
            "status": "observed",
            "path": str(path),
            "request_count": int(count_row.group(1)),
            "failure_count": int(count_row.group(2)),
        }

    percentile_result = {"status": "unavailable", "path": str(path), "reason": "aggregate_percentile_table_missing"}
    for index, line in enumerate(lines):
        columns = re.findall(r"\d+(?:\.\d+)?%", line)
        if not columns or "# reqs" not in line:
            continue
        for candidate in lines[index + 1:]:
            parts = candidate.split()
            if not parts or parts[0] != "Aggregated":
                continue
            values = parts[1:1 + len(columns)]
            if len(values) != len(columns):
                percentile_result = {"status": "malformed", "path": str(path), "reason": "aggregate_percentile_row_incomplete"}
            else:
                try:
                    percentile_result = {
                        "status": "observed",
                        "path": str(path),
                        "values": {column: float(value) for column, value in zip(columns, values)},
                    }
                except ValueError:
                    percentile_result = {"status": "malformed", "path": str(path), "reason": "aggregate_percentile_value_invalid"}
            break
        if percentile_result["status"] != "unavailable":
            break
    return count_result, percentile_result


def _read_csv_aggregate(path: Path, *, request_field: str, failure_field: str,
                        percentile_fields: bool = False) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            rows = list(reader)
    except FileNotFoundError:
        return {"status": "unavailable", "path": str(path), "reason": "file_missing"}
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"status": "malformed", "path": str(path), "reason": f"file_unreadable_or_invalid:{type(exc).__name__}"}

    required = {"Name", request_field, failure_field}
    if not required.issubset(fields):
        return {"status": "malformed", "path": str(path), "reason": "required_columns_missing"}
    aggregates = [row for row in rows if row.get("Name") == "Aggregated"]
    if not aggregates:
        return {"status": "malformed", "path": str(path), "reason": "aggregate_row_missing"}
    if percentile_fields and len(aggregates) != 1:
        return {"status": "malformed", "path": str(path), "reason": "aggregate_row_ambiguous"}
    row = aggregates[-1]
    try:
        result = {
            "status": "observed",
            "path": str(path),
            "request_count": _parse_nonnegative_integer(row.get(request_field, ""), request_field),
            "failure_count": _parse_nonnegative_integer(row.get(failure_field, ""), failure_field),
        }
        if percentile_fields:
            percentile_names = [field for field in fields if re.fullmatch(r"\d+(?:\.\d+)?%", field)]
            result["percentiles"] = {
                name: float(row[name]) for name in percentile_names if row.get(name, "") != ""
            }
            if len(result["percentiles"]) != len(percentile_names):
                return {"status": "malformed", "path": str(path), "reason": "percentile_value_missing"}
    except (TypeError, ValueError) as exc:
        return {"status": "malformed", "path": str(path), "reason": str(exc)}
    return result


def _request_and_percentile_evidence(run_dir: Path) -> dict:
    terminal, terminal_percentiles = _read_terminal_summary(run_dir / "locust_stderr.log")
    stats = _read_csv_aggregate(
        run_dir / "locust_stats.csv", request_field="Request Count",
        failure_field="Failure Count", percentile_fields=True,
    )
    history = _read_csv_aggregate(
        run_dir / "locust_stats_history.csv", request_field="Total Request Count",
        failure_field="Total Failure Count",
    )
    components = (terminal, stats, history)
    if any(item["status"] == "malformed" for item in components):
        overall = "malformed"
    elif any(item["status"] != "observed" for item in components):
        overall = "unavailable"
    else:
        counts = {
            (item["request_count"], item["failure_count"])
            for item in components
        }
        overall = "matched" if len(counts) == 1 else "unresolved_mismatch"

    if terminal_percentiles["status"] == "observed" and stats.get("status") == "observed":
        diffs = {
            name: {"terminal": value, "csv": stats["percentiles"][name]}
            for name, value in terminal_percentiles["values"].items()
            if name in stats["percentiles"] and value != stats["percentiles"][name]
        }
        percentile_status = "differences" if diffs else "matched"
    elif terminal_percentiles["status"] == "malformed" or stats.get("status") == "malformed":
        diffs, percentile_status = {}, "malformed"
    else:
        diffs, percentile_status = {}, "unavailable"
    return {
        "status": overall,
        "terminal": terminal,
        "csv": stats,
        "final_history": history,
        "percentile_comparison": {
            "status": percentile_status,
            "terminal": terminal_percentiles,
            "differences": diffs,
        },
    }


def _duration_evidence(metadata: dict, run_dir: Path) -> dict:
    configured = metadata.get("duration_seconds")
    boundaries = metadata.get("boundaries")
    launcher = {"status": "unavailable", "reason": "run_boundaries_missing"}
    if isinstance(boundaries, dict):
        try:
            start = datetime.fromisoformat(boundaries["run_start_utc"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(boundaries["workload_end_utc"].replace("Z", "+00:00"))
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError("timezone_missing")
            launcher = {
                "status": "observed",
                "start_utc": boundaries["run_start_utc"],
                "end_utc": boundaries["workload_end_utc"],
                "duration_seconds": (end - start).total_seconds(),
                "measurement_boundary": "run_start immediately before Locust subprocess invocation through workload_end after subprocess return",
            }
        except (KeyError, AttributeError, TypeError, ValueError):
            launcher = {"status": "malformed", "reason": "run_start_or_workload_end_timestamp_invalid"}

    shape_path = run_dir / "locust.log"
    shape = {"status": "unavailable", "path": str(shape_path), "reason": "file_missing"}
    try:
        lines = shape_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError) as exc:
        shape = {"status": "malformed", "path": str(shape_path), "reason": f"file_unreadable:{type(exc).__name__}"}
    else:
        events = []
        malformed_event = False
        for line in lines:
            match = _SHAPE_LOG_TIMESTAMP.match(line)
            if not match:
                if "Shape test starting" in line or "Shape test stopping" in line:
                    malformed_event = True
                continue
            try:
                timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            except ValueError:
                malformed_event = True
                continue
            events.append((timestamp, match.group(2)))
        start_event = next((event for event in events if event[1] == "starting"), None)
        stop_event = next((event for event in events if start_event and event[1] == "stopping" and event[0] >= start_event[0]), None)
        if malformed_event:
            shape = {"status": "malformed", "path": str(shape_path), "reason": "shape_event_timestamp_invalid_or_incomplete"}
        elif start_event and stop_event:
            shape = {
                "status": "observed",
                "path": str(shape_path),
                "start_log_timestamp": start_event[0].isoformat(timespec="milliseconds"),
                "stop_log_timestamp": stop_event[0].isoformat(timespec="milliseconds"),
                "duration_seconds": (stop_event[0] - start_event[0]).total_seconds(),
                "measurement_boundary": "Locust Shape test starting log event through Shape test stopping log event",
            }
        else:
            shape = {"status": "unavailable", "path": str(shape_path), "reason": "shape_start_or_stop_event_missing"}
    return {
        "configured_duration_seconds": configured if isinstance(configured, (int, float)) and not isinstance(configured, bool) else None,
        "launcher_subprocess_interval": launcher,
        "locust_shape_interval": shape,
        "interpretation": "These intervals have different start/stop events and are reported separately; launcher duration is not treated as exact shape duration.",
    }


def _outcome_evidence(manifest: dict, metadata: dict, run_dir: Path) -> dict:
    context_path = run_dir / "run_context.json"
    if not context_path.exists():
        context = {"status": "absent", "path": str(context_path)}
    elif not context_path.is_file():
        context = {"status": "malformed", "path": str(context_path), "reason": "not_a_regular_file"}
    else:
        try:
            document = json.loads(context_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                context = {"status": "malformed", "path": str(context_path), "reason": "document_not_object"}
            else:
                context = {"status": "present", "path": str(context_path)}
        except FileNotFoundError:
            context = {"status": "absent", "path": str(context_path)}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            context = {"status": "malformed", "path": str(context_path), "reason": f"file_unreadable_or_invalid:{type(exc).__name__}"}
    manifest_outcome = manifest.get("run_outcome")
    metadata_status = metadata.get("status")
    discrepancy = manifest_outcome == "unknown" and metadata_status == "completed"
    return {
        "manifest_run_outcome": manifest_outcome,
        "experiment_metadata_status": metadata_status,
        "locust_exit_code": metadata.get("locust_exit_code"),
        "collector_exit_code": metadata.get("collector_exit_code"),
        "run_context_file": context,
        "unknown_manifest_vs_completed_metadata": discrepancy,
        "interpretation": "Recorded values and run_context.json file state only; no missing manifest context or outcome is inferred.",
    }


def _blocked(spec: RunSpec, reason: str) -> dict:
    return {
        "run_id": spec.run_id,
        "expected_pattern": spec.pattern,
        "status": "blocked",
        "schema_validity": "not_checked",
        "training_eligibility": {"observed": None, "status": "unavailable"},
        "reasons": [reason],
        "acceptance_decision": "not_made",
    }


def audit_run(spec: RunSpec, project_root: Path) -> dict:
    """Audit one expected run. Input files are opened read-only."""
    root = Path(project_root).resolve()
    if len(spec.output_dirs) != 1:
        return _blocked(spec, f"Expected exactly one designated output; found {len(spec.output_dirs)}.")

    output = root / spec.output_dirs[0]
    metadata_path = root / spec.metadata_path
    run_dir = metadata_path.parent
    manifest_path = output / "manifest.json"
    observations_path = output / "observations.csv"
    for path in (manifest_path, observations_path, metadata_path):
        if not path.is_file():
            return _blocked(spec, f"Required file is missing: {path.relative_to(root)}")

    try:
        manifest = _read_json(manifest_path)
        metadata = _read_json(metadata_path)
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            raise ValueError("manifest and experiment metadata must be JSON objects")
        with observations_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLS:
                raise ValueError("observations.csv header does not match the v1 schema")
            rows = list(reader)
        if manifest.get("run_id") != spec.run_id:
            raise ValueError(f"Manifest run_id mismatch: expected {spec.run_id!r}, found {manifest.get('run_id')!r}")
        if manifest.get("pattern") != spec.pattern:
            raise ValueError(f"Manifest pattern mismatch: expected {spec.pattern!r}, found {manifest.get('pattern')!r}")
        if metadata.get("run_id") != spec.run_id:
            raise ValueError(f"Experiment metadata run_id mismatch: expected {spec.run_id!r}, found {metadata.get('run_id')!r}")
        if metadata.get("pattern") != spec.pattern:
            raise ValueError(f"Experiment metadata pattern mismatch: expected {spec.pattern!r}, found {metadata.get('pattern')!r}")
        dq = manifest.get("data_quality")
        if not isinstance(dq, dict):
            raise ValueError("Manifest data_quality must be an object")
        for field in ("expected_samples", "observed_samples"):
            recorded = dq.get(field)
            if isinstance(recorded, bool) or not isinstance(recorded, int):
                raise ValueError(f"Manifest {field} must be an integer")
            if recorded != len(rows):
                raise ValueError(f"Manifest {field}={recorded!r} disagrees with observations row count {len(rows)}")
        request_evidence = _request_and_percentile_evidence(run_dir)
        duration_evidence = _duration_evidence(metadata, run_dir)
        outcome_evidence = _outcome_evidence(manifest, metadata, run_dir)
        for index, row in enumerate(rows, start=2):
            if row.get("run_id") != spec.run_id or row.get("pattern") != spec.pattern:
                raise ValueError(f"row {index}: run_id or pattern disagrees with expected run")
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError, TypeError) as exc:
        return _blocked(spec, str(exc))

    validation_errors = load_run(output)
    schema_validity = "pass" if not validation_errors else "fail"
    if validation_errors:
        return {
            **_blocked(spec, "Official schema validation failed."),
            "schema_validity": schema_validity,
            "validation_errors": validation_errors,
        }

    phase_counts = collections.Counter(row["sample_phase"] for row in rows)
    missing_counts = collections.Counter()
    flag_counts = collections.Counter()
    missing_reason_counts = collections.Counter()
    try:
        for index, row in enumerate(rows, start=2):
            reasons = _parse_json_cell(row.get("missing_reasons", ""), dict, "missing_reasons", index)
            flags = _parse_json_cell(row.get("quality_flags", ""), list, "quality_flags", index)
            for column in MEASUREMENT_COLUMNS:
                if row.get(column, "") == "":
                    missing_counts[column] += 1
                    if column in reasons:
                        missing_reason_counts[f"{column}: {reasons[column]}"] += 1
            flag_counts.update(str(flag) for flag in flags)
    except ValueError as exc:
        return _blocked(spec, str(exc))

    dq = manifest.get("data_quality", {})
    source_integrity = dq.get("source_integrity")
    training_eligible = metadata.get("training_eligible")
    training_status = "observed" if isinstance(training_eligible, bool) else "unavailable"
    reasons = []
    if source_integrity != "verified":
        reasons.append(f"Source integrity is {source_integrity!r}, not verified.")
    if dq.get("gaps") or dq.get("duplicate_timestamps") or dq.get("counter_resets") or dq.get("container_changes"):
        reasons.append("Manifest reports gaps, duplicates, counter resets, or container changes.")
    feature_missing = {
        field: missing_counts[field]
        for field in ("cpu_usage_cores", "memory_working_set_bytes")
        if missing_counts[field]
    }
    if feature_missing:
        reasons.append("CPU or memory feature measurements are missing; inspect field counts and reasons.")
    elif missing_counts:
        reasons.append("Missingness is limited to counter/derived fields; CPU and memory feature measurements are present.")
    raw_counter_missing = {
        field: missing_counts[field]
        for field in ("cfs_throttled_periods_total", "cfs_periods_total", "cfs_throttled_seconds_total")
        if missing_counts[field]
    }
    derived_ratio_missing = missing_counts["throttled_period_ratio_1m"]
    if flag_counts:
        reasons.append("Observations contain quality flags; review their meaning before use.")
    throttling = manifest.get("throttling", {})
    if throttling.get("status") == "unknown":
        reasons.append("Throttling finding is unknown in the manifest.")

    configured_duration = duration_evidence["configured_duration_seconds"]
    launcher_interval = duration_evidence["launcher_subprocess_interval"]
    shape_interval = duration_evidence["locust_shape_interval"]
    if launcher_interval["status"] != "observed":
        reasons.append("Launcher run_start/workload_end duration evidence is " + launcher_interval["status"] + ".")
    elif configured_duration is None:
        reasons.append("Configured workload duration is unavailable or malformed.")
    elif launcher_interval["duration_seconds"] != configured_duration:
        shape_duration = (
            f" Locust shape log interval is {shape_interval['duration_seconds']:g}s."
            if shape_interval["status"] == "observed" else
            f" Locust shape duration evidence is {shape_interval['status']}."
        )
        reasons.append(
            f"Launcher subprocess interval is {launcher_interval['duration_seconds']:g}s versus "
            f"configured shape duration {configured_duration:g}s; these use different boundaries."
            + shape_duration
        )
    if shape_interval["status"] != "observed":
        reasons.append("Locust shape start/stop duration evidence is " + shape_interval["status"] + ".")

    if request_evidence["status"] == "unresolved_mismatch":
        reasons.append("Terminal, locust_stats.csv, and final history request/failure counts disagree; unresolved.")
    elif request_evidence["status"] in {"unavailable", "malformed"}:
        reasons.append("Locust count reconciliation evidence is " + request_evidence["status"] + "; unresolved.")
    percentile_comparison = request_evidence["percentile_comparison"]
    if percentile_comparison["status"] == "differences":
        rendered = ", ".join(
            f"{name} terminal {values['terminal']:g} vs CSV {values['csv']:g}"
            for name, values in sorted(percentile_comparison["differences"].items())
        )
        reasons.append("Terminal and CSV percentile values differ; unresolved: " + rendered + ".")
    elif percentile_comparison["status"] in {"unavailable", "malformed"}:
        reasons.append("Locust percentile comparison evidence is " + percentile_comparison["status"] + "; unresolved.")

    if outcome_evidence["unknown_manifest_vs_completed_metadata"]:
        reasons.append("Manifest run_outcome is unknown while experiment metadata status is completed; review recorded evidence.")
    reasons.extend(KNOWN_CAVEATS.get(spec.run_id, ()))
    if training_status != "observed":
        reasons.append("training_eligible is not recorded as a boolean in experiment metadata.")

    status = "review_required" if reasons else "pass"
    return {
        "run_id": spec.run_id,
        "expected_pattern": spec.pattern,
        "output_dir": str(output),
        "status": status,
        "schema_validity": schema_validity,
        "training_eligibility": {"observed": training_eligible, "status": training_status},
        "outcome": manifest.get("run_outcome"),
        "counts": {
            "observations_rows": len(rows),
            "manifest_expected_samples": dq.get("expected_samples"),
            "manifest_observed_samples": dq.get("observed_samples"),
            "phase_counts": dict(sorted(phase_counts.items())),
            "missing_feature_measurements": feature_missing,
            "raw_cfs_counter_missing": raw_counter_missing,
            "derived_throttling_ratio_missing": derived_ratio_missing,
            "missing_measurements": dict(sorted(missing_counts.items())),
            "missing_reasons": dict(sorted(missing_reason_counts.items())),
            "quality_flags": dict(sorted(flag_counts.items())),
        },
        "source_integrity": source_integrity,
        "throttling_status": throttling.get("status"),
        "locust_count_and_percentile_evidence": request_evidence,
        "duration_evidence": duration_evidence,
        "outcome_evidence": outcome_evidence,
        "reasons": reasons,
        "acceptance_decision": "not_made",
    }


def audit_runs(specifications: Iterable[RunSpec] = RUN_SPECS, project_root: Path | None = None) -> list[dict]:
    """Audit configured runs and block duplicate or ambiguous mappings."""
    specs = tuple(specifications)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    id_counts = collections.Counter(spec.run_id for spec in specs)
    output_owners: dict[str, list[str]] = collections.defaultdict(list)
    for spec in specs:
        for output in spec.output_dirs:
            output_owners[str(Path(output))].append(spec.run_id)
    results = []
    for spec in specs:
        if id_counts[spec.run_id] != 1:
            results.append(_blocked(spec, "Run ID appears in more than one audit mapping."))
        elif any(len(owners) != 1 for owners in (output_owners[str(Path(p))] for p in spec.output_dirs)):
            results.append(_blocked(spec, "A designated output directory is mapped to multiple run IDs."))
        else:
            results.append(audit_run(spec, root))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    results = audit_runs(project_root=args.project_root)
    print(json.dumps({"audit_type": "evidence_only_not_training_acceptance", "runs": results}, indent=2, sort_keys=True))
    return 1 if any(item["status"] == "blocked" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
