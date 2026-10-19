"""Offline-only experiment planning for K8sight.

This module plans runs without launching workloads, contacting Prometheus,
or modifying Kubernetes resources.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from experiment_kit.collector.config import CollectorConfig
from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.runner import validate_run_id


def build_run_plan(
    workload: WorkloadConfig,
    collector: CollectorConfig,
    *,
    planned_start_utc: datetime,
) -> dict[str, Any]:
    """Build a reproducible plan; this does not mean a run was executed."""
    validate_run_id(workload.run_id)
    workload.validate()

    if workload.run_id != collector.run_id:
        raise ValueError("workload and collector run IDs must match")
    if workload.pattern != collector.pattern:
        raise ValueError("workload and collector patterns must match")
    if workload.duration_seconds != collector.duration_seconds:
        raise ValueError("workload and collector durations must match")

    if planned_start_utc.tzinfo is None:
        raise ValueError("planned_start_utc must be timezone-aware")

    if collector.pre_start_seconds <= 0:
        raise ValueError("pre_start_seconds must be positive")
    if collector.cooldown_seconds <= 0:
        raise ValueError("cooldown_seconds must be positive")
    if collector.step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if collector.scrape_interval_seconds <= 0:
        raise ValueError("scrape_interval_seconds must be positive")
    if collector.cpu_lookback_seconds < 4 * collector.scrape_interval_seconds:
        raise ValueError("CPU lookback must be at least four scrape intervals")
    if collector.step_seconds < collector.scrape_interval_seconds:
        raise ValueError("query step must be at least the configured scrape interval")

    start = planned_start_utc.astimezone(timezone.utc)
    baseline_start = start
    run_start = start + timedelta(seconds=collector.pre_start_seconds)
    workload_end = run_start + timedelta(seconds=workload.duration_seconds)
    cooldown_end = workload_end + timedelta(seconds=collector.cooldown_seconds)
    collection_end = cooldown_end

    def stamp(value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "plan_version": "1.0.0",
        "status": "planned",
        "run_id": workload.run_id,
        "run_outcome": "not_started",
        "execution_enabled": False,
        "external_calls_enabled": False,
        "workload": workload.to_dict(),
        "collector": {
            "namespace": collector.namespace,
            "pod": collector.pod,
            "container": collector.container,
            "step_seconds": collector.step_seconds,
            "configured_scrape_interval_seconds": (
                collector.scrape_interval_seconds
            ),
            "cpu_lookback_seconds": collector.cpu_lookback_seconds,
            "pre_start_seconds": collector.pre_start_seconds,
            "cooldown_seconds": collector.cooldown_seconds,
        },
        "boundaries": {
            "collection_start_utc": stamp(baseline_start),
            "run_start_utc": stamp(run_start),
            "workload_end_utc": stamp(workload_end),
            "locust_shutdown_utc": None,
            "cooldown_start_utc": stamp(workload_end),
            "collection_end_utc": stamp(collection_end),
        },
        "schedule_hash": workload.compute_schedule_hash(),
        "notes": [
            "Plan only; no workload has been launched.",
            "Configured scrape interval is not live-verified.",
            "Planned boundaries are not observed lifecycle timestamps.",
            "Measurements have not been collected.",
        ],
    }


def write_run_plan(path: Path, plan: dict[str, Any]) -> Path:
    """Write a plan using exclusive creation; never replace existing evidence."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output
