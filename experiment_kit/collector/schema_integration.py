"""
Offline integration between archived K8sight collector output and dataset_schema_v1.

This module must not contact Prometheus, launch workloads, or overwrite runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_collection_summary(summary_path: Path) -> dict:
    """Load an existing archived collection summary without modifying it."""
    summary_path = summary_path.expanduser().resolve(strict=True)
    if not summary_path.is_file():
        raise ValueError(f"Not a summary file: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    if not isinstance(summary, dict):
        raise ValueError("Collection summary must be a JSON object")
    if not isinstance(summary.get("metrics"), list):
        raise ValueError("Collection summary is missing its metrics list")
    if not isinstance(summary.get("target"), dict):
        raise ValueError("Collection summary is missing target identity")
    if not isinstance(summary.get("query_range"), dict):
        raise ValueError("Collection summary is missing query range")

    return summary



METRIC_COLUMNS = {
    "container_cpu_usage_seconds_total": "cpu_usage_cores",
    "container_memory_working_set_bytes": "memory_working_set_bytes",
    "container_cpu_cfs_throttled_periods_total": "cfs_throttled_periods_total",
    "container_cpu_cfs_periods_total": "cfs_periods_total",
    "container_cpu_cfs_throttled_seconds_total": "cfs_throttled_seconds_total",
}



def expected_timestamp_grid(query_range: dict) -> list[float]:
    """Build expected Prometheus evaluation timestamps from archived query metadata."""
    from datetime import datetime, timezone

    if not isinstance(query_range, dict):
        raise ValueError("query_range must be an object")

    def parse_utc(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a UTC timestamp")
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} is not a valid timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} must include a timezone")
        return parsed.astimezone(timezone.utc).timestamp()

    start = parse_utc(query_range.get("start_utc"), "start_utc")
    end = parse_utc(query_range.get("end_utc"), "end_utc")
    step = query_range.get("step_seconds")

    if end < start:
        raise ValueError("query end must not precede query start")
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise ValueError("step_seconds must be a positive number")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step_seconds must be a positive finite number")

    count = int(math.floor((end - start) / step + 1e-9)) + 1
    if count > 1_000_000:
        raise ValueError("Expected timestamp grid is too large")

    return [round(start + i * step, 3) for i in range(count)]


def calculate_sample_gaps(aligned: list[dict], expected_timestamps: list[float]) -> list[dict]:
    """Describe expected timestamps where one or more metrics are missing."""
    gaps = []
    by_timestamp = {
        round(float(row["timestamp_epoch"]), 3): row
        for row in aligned
    }

    for timestamp in expected_timestamps:
        row = by_timestamp.get(round(float(timestamp), 3))
        missing = [
            column
            for column in METRIC_COLUMNS.values()
            if row is None or row.get(column) is None
        ]
        if missing:
            gaps.append({
                "timestamp_utc": datetime.fromtimestamp(
                    timestamp, timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "missing_metrics": missing,
            })

    return gaps

def align_metric_samples(
    parsed_samples: dict,
    expected_timestamps=None,
) -> list[dict]:
    """Align metrics on exact timestamps, optionally preserving expected gaps."""
    import math

    if not isinstance(parsed_samples, dict):
        raise ValueError("parsed_samples must be a mapping")

    timestamps = set()
    for metric_name, samples in parsed_samples.items():
        if metric_name not in METRIC_COLUMNS:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if not isinstance(samples, dict):
            raise ValueError(f"{metric_name}: samples must be a mapping")

        for timestamp, value in samples.items():
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(f"{metric_name}: timestamp must be numeric")
            if not math.isfinite(timestamp):
                raise ValueError(f"{metric_name}: timestamp must be finite")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{metric_name}: sample value must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{metric_name}: sample value must be finite and nonnegative")
            if (
                metric_name == "container_memory_working_set_bytes"
                and not float(value).is_integer()
            ):
                raise ValueError("memory working-set samples must be whole bytes")
            timestamps.add(float(timestamp))

    if expected_timestamps is not None:
        expected = set()
        for timestamp in expected_timestamps:
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
            ):
                raise ValueError("Expected timestamps must be finite numbers")
            expected.add(round(float(timestamp), 3))

        actual = {round(timestamp, 3) for timestamp in timestamps}
        unexpected = actual - expected
        if unexpected:
            raise ValueError(
                "Metric timestamps fall outside the expected timestamp grid"
            )
        timestamps = expected

    rows = []
    for timestamp in sorted(timestamps):
        row = {"timestamp_epoch": timestamp}
        for metric_name, column in METRIC_COLUMNS.items():
            value = parsed_samples.get(metric_name, {}).get(timestamp)
            if value is None:
                # Numeric epoch keys such as 1000 and 1000.0 compare equally in Python.
                row[column] = None
            elif column == "memory_working_set_bytes":
                row[column] = int(value)
            else:
                row[column] = float(value)
        rows.append(row)

    return rows



OBSERVATION_COLUMNS = [
    "timestamp_utc", "run_id", "pattern", "namespace", "pod", "pod_uid",
    "container", "container_id", "sample_phase", "cpu_usage_cores",
    "cpu_rate_lookback_seconds", "cpu_rate_window_status",
    "memory_working_set_bytes", "cfs_throttled_periods_total",
    "cfs_periods_total", "cfs_throttled_seconds_total",
    "throttled_period_ratio_1m", "quality_flags", "missing_reasons",
]


def aligned_to_observation_rows(
    aligned: list[dict],
    *,
    run_id: str,
    pattern: str,
    target: dict,
    lookback_seconds: int,
) -> list[dict]:
    """Convert aligned samples into schema-shaped CSV rows without guessing boundaries."""
    from datetime import datetime, timezone

    allowed_patterns = {"stable", "increasing", "periodic", "spike", "random_spike"}
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if pattern not in allowed_patterns:
        raise ValueError("unsupported workload pattern")
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    if isinstance(lookback_seconds, bool) or not isinstance(lookback_seconds, int) or lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be a positive integer")

    rows = []
    for item in aligned:
        epoch = item.get("timestamp_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
            raise ValueError("timestamp_epoch must be numeric")
        if not __import__("math").isfinite(epoch):
            raise ValueError("timestamp_epoch must be finite")

        timestamp = datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        missing = {}
        row = {
            "timestamp_utc": timestamp,
            "run_id": run_id,
            "pattern": pattern,
            "namespace": target.get("namespace"),
            "pod": target.get("pod"),
            "pod_uid": target.get("pod_uid"),
            "container": target.get("container"),
            "container_id": target.get("container_id"),
            "sample_phase": "unknown",
            "cpu_usage_cores": item.get("cpu_usage_cores"),
            "cpu_rate_lookback_seconds": lookback_seconds,
            "cpu_rate_window_status": "unknown",
            "memory_working_set_bytes": item.get("memory_working_set_bytes"),
            "cfs_throttled_periods_total": item.get("cfs_throttled_periods_total"),
            "cfs_periods_total": item.get("cfs_periods_total"),
            "cfs_throttled_seconds_total": item.get("cfs_throttled_seconds_total"),
            "throttled_period_ratio_1m": None,
            "quality_flags": [],
            "missing_reasons": missing,
        }

        for column in (
            "cpu_usage_cores",
            "memory_working_set_bytes",
            "cfs_throttled_periods_total",
            "cfs_periods_total",
            "cfs_throttled_seconds_total",
            "throttled_period_ratio_1m",
        ):
            if row[column] is None:
                missing[column] = "metric sample unavailable at this timestamp"

        # The rate-window status is not_applicable when CPU itself is absent.
        if row["cpu_usage_cores"] is None:
            row["cpu_rate_window_status"] = "not_applicable"

        rows.append(row)

    return rows


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_archived_sources(summary_path: Path, metrics: list[dict], destination: Path) -> list[dict]:
    """Copy verified JSON evidence into a new run directory without altering sources."""
    source_root = summary_path.resolve(strict=True).parent
    copied_metrics = []
    raw_root = destination / "raw"
    raw_root.mkdir(exist_ok=True)

    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("Metric record must be an object")
        record = dict(metric)
        sources = record.get("raw_sources", [])
        if not isinstance(sources, list):
            raise ValueError("Metric raw_sources must be a list")
        copied_sources = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Raw source record must be an object")
            rel = Path(str(source.get("path", "")))
            if not rel.parts or rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe raw source path")
            src = (source_root / rel).resolve(strict=True)
            try:
                src.relative_to(source_root)
            except ValueError as exc:
                raise ValueError("Raw source escapes collection directory") from exc
            if not src.is_file() or source.get("media_type") != "application/json":
                raise ValueError("Raw source must be a JSON file")
            actual_hash = sha256_file(src)
            if actual_hash.lower() != str(source.get("sha256", "")).lower():
                raise ValueError(f"Raw source hash mismatch: {rel.as_posix()}")
            target = raw_root / src.name
            if target.exists():
                raise ValueError(f"Duplicate raw source filename: {target.name}")
            shutil.copyfile(src, target)
            copied_sources.append({
                "path": f"raw/{target.name}",
                "sha256": sha256_file(target),
                "media_type": "application/json",
            })
        record["raw_sources"] = copied_sources
        copied_metrics.append(record)
    return copied_metrics


def evidence(status: str, value=None, reason=None, source=None, observed_at_utc=None) -> dict:
    """Build an evidence field that conforms to manifest evidence rules."""
    if status not in {"observed", "unknown", "not_collected", "not_applicable"}:
        raise ValueError("Invalid evidence status")
    if status == "observed":
        if value is None:
            raise ValueError("Observed evidence requires a value")
        reason = None
    else:
        value = None
        if not reason:
            raise ValueError("Unavailable evidence requires a reason")
    return {
        "status": status,
        "value": value,
        "reason": reason,
        "source": source,
        "observed_at_utc": observed_at_utc,
    }


def build_manifest(summary: dict, metrics: list[dict], rows: list[dict], *, run_id: str, pattern: str, replicate: int) -> dict:
    """Build a conservative schema-v1 manifest from archived collector evidence."""
    allowed_patterns = {"stable", "increasing", "periodic", "spike", "random_spike"}
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if pattern not in allowed_patterns:
        raise ValueError("unsupported workload pattern")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 1:
        raise ValueError("replicate must be a positive integer")

    target = summary["target"]
    query_range = summary["query_range"]
    created = summary.get("created_at_utc")
    source = "collection_summary.json"
    unavailable = lambda reason: evidence("unknown", reason=reason, source=source, observed_at_utc=created)
    identity = {}
    for key in ("namespace", "pod", "container"):
        value = target.get(key)
        identity[key] = evidence("observed", value=value, source=source, observed_at_utc=created) if isinstance(value, str) and value else unavailable("Target label was not supplied")

    environment = {
        "kubernetes_context": unavailable("Kubernetes context was not recorded in the collection summary"),
        "namespace": identity["namespace"],
        "job": unavailable("Kubernetes job metadata was not collected"),
        "pod": identity["pod"],
        "pod_uid": unavailable("Pod UID was not collected"),
        "container": identity["container"],
        "container_id": unavailable("Container ID was not collected"),
        "image": unavailable("Container image was not collected"),
        "node": unavailable("Node identity was not collected"),
        "resources": {
            "cpu_request": unavailable("CPU request was not collected"),
            "memory_request": unavailable("Memory request was not collected"),
            "cpu_limit": unavailable("CPU limit was not collected"),
            "memory_limit": unavailable("Memory limit was not collected"),
        },
    }
    workload = {
        "generator": unavailable("Workload execution was not verified by the collector"),
        "generator_version": unavailable("Generator version was not recorded"),
        "generator_sha256": unavailable("Executed generator hash was not recorded"),
        "parameters": {},
        "random_seed": unavailable("Random seed was not recorded by the collector"),
    }
    boundaries = {
        "run_start_utc": None,
        "workload_end_utc": None,
        "cooldown_start_utc": None,
        "locust_shutdown_utc": None,
        "collection_start_utc": query_range.get("start_utc"),
        "collection_end_utc": query_range.get("end_utc"),
        "unknown_reasons": {
            "run_start_utc": "Workload start was not supplied",
            "workload_end_utc": "Workload end was not supplied",
            "cooldown_start_utc": "Cooldown start was not supplied",
            "locust_shutdown_utc": "Locust shutdown time was not supplied",
        },
    }
    units = {
        "cpu_usage_cores": "cores",
        "memory_working_set_bytes": "bytes",
        "cfs_throttled_periods_total": "periods",
        "cfs_periods_total": "periods",
        "cfs_throttled_seconds_total": "seconds",
        "throttled_period_ratio_1m": "ratio",
    }
    expected = expected_timestamp_grid(query_range)
    gaps = calculate_sample_gaps(
        [
            {
                "timestamp_epoch": datetime.fromisoformat(
                    row["timestamp_utc"].replace("Z", "+00:00")
                ).timestamp(),
                **{
                    column: row.get(column)
                    for column in METRIC_COLUMNS.values()
                },
            }
            for row in rows
        ],
        expected,
    )
    expected_count = len(expected)
    source_integrity = (
        "verified"
        if all(
            isinstance(metric, dict)
            and metric.get("availability") in {"available", "empty"}
            and isinstance(metric.get("raw_sources"), list)
            and bool(metric["raw_sources"])
            for metric in metrics
        )
        else "unknown"
    )
    step = query_range.get("step_seconds")
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "pattern": pattern,
        "replicate": evidence("observed", value=replicate, source="integration_cli"),
        "timezone": "UTC",
        "run_outcome": "unknown",
        "workload": workload,
        "environment": environment,
        "boundaries": boundaries,
        "observation_units": units,
        "scrape_interval": unavailable("Query evaluation step does not establish actual scrape cadence"),
        "metrics": metrics,
        "throttling": {
            "status": "unknown",
            "reason": "No throttling finding is inferred by offline integration",
            "coverage_start_utc": None,
            "coverage_end_utc": None,
            "covered_sample_count": None,
            "evidence_metrics": [],
        },
        "data_quality": {
            "expected_samples": expected_count,
            "observed_samples": len(rows),
            "gaps": gaps,
            "duplicate_timestamps": [],
            "counter_resets": [],
            "container_changes": [],
            "source_integrity": source_integrity,
        },
        "historical_audits": [],
    }


def apply_run_context(manifest: dict, rows: list[dict], context: dict, aligned: list[dict]) -> None:
    """Apply only explicitly recorded run-context evidence to an offline conversion."""
    if not isinstance(context, dict):
        raise ValueError("run context must be a JSON object")
    run_outcome = context.get("run_outcome")
    if (isinstance(run_outcome, dict)
            and run_outcome.get("status") == "observed"
            and isinstance(run_outcome.get("value"), str)
            and run_outcome["value"] in {"completed", "partial", "failed", "stopped", "unknown"}):
        manifest["run_outcome"] = run_outcome["value"]
    source = "run_context.json"
    for field in ("run_start_utc", "workload_end_utc", "cooldown_start_utc", "locust_shutdown_utc"):
        value = context.get("boundaries", {}).get(field)
        if value is not None:
            # Reuse UTC parsing and preserve only an explicitly supplied timestamp.
            expected_timestamp_grid({"start_utc": value, "end_utc": value, "step_seconds": 1})
            manifest["boundaries"][field] = value
            manifest["boundaries"]["unknown_reasons"].pop(field, None)
    workload = context.get("workload", {})
    for field in ("generator", "generator_version", "generator_sha256", "random_seed"):
        item = workload.get(field)
        if isinstance(item, dict) and item.get("status") == "observed" and item.get("value") is not None:
            manifest["workload"][field] = item
    if isinstance(workload.get("parameters"), dict):
        manifest["workload"]["parameters"] = workload["parameters"]
    environment = context.get("environment", {})
    for field in ("namespace", "job", "pod", "pod_uid", "container", "container_id", "image", "node"):
        item = environment.get(field)
        if isinstance(item, dict) and item.get("status") == "observed" and item.get("value") is not None:
            manifest["environment"][field] = item
    resources = environment.get("resources", {})
    for field in ("cpu_request", "memory_request", "cpu_limit", "memory_limit"):
        item = resources.get(field)
        if isinstance(item, dict) and item.get("status") == "observed" and item.get("value") is not None:
            manifest["environment"]["resources"][field] = item
    for row in rows:
        for field in ("namespace", "pod", "pod_uid", "container", "container_id"):
            item = manifest["environment"].get(field, {})
            if item.get("status") == "observed":
                row[field] = item["value"]
    identity_samples = context.get("identity_samples", [])
    if identity_samples:
        if not isinstance(identity_samples, list):
            raise ValueError("identity_samples must be a list")
        by_time = {sample.get("timestamp_utc"): sample for sample in identity_samples if isinstance(sample, dict)}
        for row in rows:
            sample = by_time.get(row["timestamp_utc"])
            if sample is not None:
                for field in ("namespace", "pod", "pod_uid", "container", "container_id"):
                    if field in sample and isinstance(sample[field], str):
                        row[field] = sample[field]
    history = context.get("historical_audits")
    if isinstance(history, list):
        manifest["historical_audits"] = history

    def epoch(field):
        value = manifest["boundaries"].get(field)
        if not value:
            return None
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    start, active_end, cool, shutdown = (epoch("run_start_utc"), epoch("workload_end_utc"), epoch("cooldown_start_utc"), epoch("locust_shutdown_utc"))
    collection_end = epoch("collection_end_utc")
    for row in rows:
        t = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).timestamp()

        active_ends = [
            value for value in (active_end, shutdown, cool)
            if value is not None
        ]
        effective_active_end = min(active_ends) if active_ends else None

        if start is not None and t < start:
            phase = "pre_start"
        elif start is not None and effective_active_end is not None and start <= t < effective_active_end:
            phase = "active"
        elif cool is not None and t >= cool and (collection_end is None or t <= collection_end) and (
            shutdown is None or cool >= shutdown or t < shutdown
        ):
            phase = "cooldown"
        elif shutdown is not None and t > shutdown and (cool is None or t < cool):
            phase = "post_shutdown"
        else:
            phase = "unknown"

        row["sample_phase"] = phase
        lookback = row["cpu_rate_lookback_seconds"]
        end_window = t
        begin_window = t - lookback
        crossed = [(b, state) for b, state in ((start, "crosses_pre_start"), (cool, "crosses_cooldown"), (shutdown, "crosses_shutdown")) if b is not None and begin_window < b <= end_window]
        row["cpu_rate_window_status"] = (
            min(crossed)[1]
            if crossed
            else (
                "fully_within_active"
                if start is not None
                and effective_active_end is not None
                and begin_window >= start
                and end_window <= effective_active_end
                else "unknown"
            )
        )

    reset_events = []
    changes = []
    counter_names = ("cfs_throttled_periods_total", "cfs_periods_total")
    ratio_name = "throttled_period_ratio_1m"
    for row in rows:
        row[ratio_name] = None
        row["missing_reasons"][ratio_name] = "complete aligned counters spanning exactly 60 seconds are unavailable"

    counter_steps = {
        metric.get("step_seconds")
        for metric in manifest.get("metrics", [])
        if isinstance(metric, dict)
        and metric.get("name") in {
            "container_cpu_cfs_throttled_periods_total",
            "container_cpu_cfs_periods_total",
        }
    }
    cadence = next(iter(counter_steps)) if len(counter_steps) == 1 else None
    if (isinstance(cadence, bool) or not isinstance(cadence, (int, float))
            or not math.isfinite(cadence) or cadence <= 0
            or abs(60 / cadence - round(60 / cadence)) > 1e-9):
        cadence = None

    # Record observed resets and only compute ratios from complete 60-second windows.
    for index in range(1, len(aligned)):
        prev, cur = aligned[index - 1], aligned[index]
        if any(prev.get(name) is not None and cur.get(name) is not None and cur[name] < prev[name] for name in counter_names):
            reset_events.append(rows[index]["timestamp_utc"])
            rows[index]["quality_flags"].append("counter_reset")
    parsed_times = []
    for row in rows:
        try:
            parsed_times.append(datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).timestamp())
        except (KeyError, TypeError, ValueError):
            parsed_times.append(None)
    for end_index, end_time in enumerate(parsed_times):
        if end_time is None or cadence is None:
            continue
        start_index = next((i for i in range(end_index - 1, -1, -1)
                            if parsed_times[i] is not None and abs((end_time - parsed_times[i]) - 60.0) < 0.001), None)
        if start_index is None:
            continue
        window = aligned[start_index:end_index + 1]
        window_times = parsed_times[start_index:end_index + 1]
        if (any(t is None for t in window_times)
                or len(window_times) != int(round(60 / cadence)) + 1
                or any(abs((b - a) - cadence) > 0.001 for a, b in zip(window_times, window_times[1:]))):
            continue
        if any(any(sample.get(name) is None for name in counter_names) for sample in window):
            continue

        # Never calculate a counter ratio across different container identities.
        identity_fields = ("pod", "pod_uid", "container", "container_id")
        identities = [
            tuple(rows[i].get(field) for field in identity_fields)
            for i in range(start_index, end_index + 1)
        ]
        if any(any(value is None for value in identity) for identity in identities):
            continue
        if any(identity != identities[0] for identity in identities[1:]):
            rows[end_index]["missing_reasons"][ratio_name] = (
                "container identity changed within the 60-second counter window"
            )
            continue

        if any(any(right[name] < left[name] for name in counter_names)
               for left, right in zip(window, window[1:])):
            continue
        numerator = window[-1][counter_names[0]] - window[0][counter_names[0]]
        denominator = window[-1][counter_names[1]] - window[0][counter_names[1]]
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            rows[end_index]["quality_flags"].append("throttling_counters_invalid_for_60s_ratio")
            continue
        rows[end_index][ratio_name] = numerator / denominator
        rows[end_index]["missing_reasons"].pop(ratio_name, None)
    identity_fields = ("pod", "pod_uid", "container", "container_id")
    for index in range(1, len(rows)):
        before = tuple(rows[index - 1].get(k) for k in identity_fields)
        after = tuple(rows[index].get(k) for k in identity_fields)
        if before != after:
            changes.append(rows[index]["timestamp_utc"])
    manifest["data_quality"]["counter_resets"] = reset_events
    manifest["data_quality"]["container_changes"] = changes
    if reset_events:
        manifest["throttling"].update(status="unknown", reason="Counter reset detected; throttling ratio unavailable")
    # Counter evidence is never interpreted as a clean pass when absent or incomplete.
    if not any(row["throttled_period_ratio_1m"] is not None for row in rows):
        manifest["throttling"].update(status="unknown", reason="Aligned throttling counters unavailable or incomplete")
    else:
        manifest["throttling"].update(status="unknown", reason="Throttling counters observed; finding requires complete validator evidence")


def write_run_files(destination: Path, manifest: dict, rows: list[dict]) -> None:
    """Write schema-shaped run files into an already-created new directory."""
    destination = destination.resolve(strict=True)
    if not destination.is_dir():
        raise ValueError("Destination must be a directory")
    for name in ("manifest.json", "observations.csv"):
        if (destination / name).exists():
            raise FileExistsError(f"Refusing to overwrite {name}")

    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    csv_path = destination / "observations.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OBSERVATION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if list(row) != OBSERVATION_COLUMNS:
                raise ValueError("Observation columns do not match schema order")
            serialized = dict(row)
            serialized["quality_flags"] = json.dumps(row["quality_flags"], separators=(",", ":"))
            serialized["missing_reasons"] = json.dumps(row["missing_reasons"], sort_keys=True, separators=(",", ":"))
            writer.writerow({key: "" if serialized.get(key) is None else serialized.get(key) for key in OBSERVATION_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--pattern",
        required=True,
        choices=("stable", "increasing", "periodic", "spike", "random_spike"),
    )
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--run-context", type=Path)
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "dataset_schema_v1" / "validate_run.py",
    )
    args = parser.parse_args()

    destination = args.output_dir.expanduser().resolve()
    try:
        summary_path = args.summary.expanduser().resolve(strict=True)
        summary = load_collection_summary(summary_path)
        validator_path = args.validator.expanduser().resolve(strict=True)

        if not validator_path.is_file():
            raise ValueError("Validator path is not a file")
        if not args.run_id.strip():
            raise ValueError("run_id must not be empty")
        if isinstance(args.replicate, bool) or args.replicate < 1:
            raise ValueError("replicate must be positive")

        source_root = summary_path.parent
        if destination == source_root or source_root in destination.parents:
            raise ValueError("Output directory must not be inside the source directory")
        if destination.exists():
            raise FileExistsError(
                f"Output directory already exists; refusing to overwrite: {destination}"
            )

        expected = expected_timestamp_grid(summary["query_range"])

        # Create a new, exclusive destination. If a later step fails, preserve
        # the partial output for inspection rather than silently deleting it.
        destination.mkdir(parents=True, exist_ok=False)
        metrics = copy_archived_sources(
            summary_path, summary["metrics"], destination
        )

        from experiment_kit.collector.live_collector import parse_single_series

        parsed_samples = {}
        for metric in metrics:
            name = metric.get("name")
            if name not in METRIC_COLUMNS:
                continue

            parsed_samples[name] = {}
            if metric.get("availability") != "available":
                continue

            sources = metric.get("raw_sources", [])
            if len(sources) != 1:
                metric["availability"] = "unknown"
                metric["error"] = "Expected exactly one archived response"
                continue

            raw_path = destination / sources[0]["path"]
            try:
                document = json.loads(raw_path.read_text(encoding="utf-8"))
                parsed_samples[name] = parse_single_series(document, name)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                metric["availability"] = "unknown"
                metric["error"] = f"Archived response could not be parsed: {exc}"
                parsed_samples[name] = {}

        aligned = align_metric_samples(
            parsed_samples, expected_timestamps=expected
        )

        cpu_metric = next(
            (
                metric for metric in metrics
                if metric.get("name") == "container_cpu_usage_seconds_total"
            ),
            {},
        )
        lookback = cpu_metric.get("lookback_seconds", 60)
        if (
            isinstance(lookback, bool)
            or not isinstance(lookback, (int, float))
            or not math.isfinite(lookback)
            or lookback <= 0
            or not float(lookback).is_integer()
        ):
            lookback = 60

        rows = aligned_to_observation_rows(
            aligned,
            run_id=args.run_id,
            pattern=args.pattern,
            target=summary["target"],
            lookback_seconds=int(lookback),
        )
        manifest = build_manifest(
            summary,
            metrics,
            rows,
            run_id=args.run_id,
            pattern=args.pattern,
            replicate=args.replicate,
        )
        if args.run_context:
            from experiment_kit.collector.run_context import load_run_context
            context = load_run_context(args.run_context)
            apply_run_context(manifest, rows, context, aligned)

        write_run_files(destination, manifest, rows)

        result = subprocess.run(
            [
                sys.executable,
                str(validator_path),
                "--run-dir",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            print(
                f"Validation failed; inspect preserved output: {destination}",
                file=sys.stderr,
            )
            return 1

        print(json.dumps({
            "status": "validated",
            "run_dir": str(destination),
            "expected_samples": manifest["data_quality"]["expected_samples"],
            "observed_samples": manifest["data_quality"]["observed_samples"],
            "gap_count": len(manifest["data_quality"]["gaps"]),
            "note": "Offline conversion only; not authorization to train models.",
        }, indent=2))
        return 0

    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Offline integration failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
