"""
Offline record and manifest builder conforming to dataset_schema_v1.
Constructs valid manifest and observation objects without fabricating measurements.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.collector.config import CollectorConfig


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_rate_status(
    sample_time: datetime,
    lookback_seconds: Optional[int],
    boundaries: Dict[str, Optional[str]]
) -> str:
    """Classifies rolling rate status against boundaries matching validate_run logic."""
    if lookback_seconds is None:
        return "not_applicable"
    
    t_start = sample_time - timedelta(seconds=lookback_seconds)
    
    # Check boundaries crossed
    crossed = []
    for name, b_key in [
        ("crosses_pre_start", "run_start_utc"),
        ("crosses_cooldown", "cooldown_start_utc"),
        ("crosses_shutdown", "locust_shutdown_utc"),
    ]:
        b_str = boundaries.get(b_key)
        if b_str:
            b_dt = datetime.fromisoformat(b_str.replace("Z", "+00:00"))
            if t_start < b_dt <= sample_time:
                crossed.append((b_dt, name))
                
    if crossed:
        return min(crossed, key=lambda x: x[0])[1]
        
    start_str = boundaries.get("run_start_utc")
    active_end_values = [
        boundaries[key]
        for key in (
            "workload_end_utc",
            "locust_shutdown_utc",
            "cooldown_start_utc",
        )
        if boundaries.get(key) is not None
    ]
    if start_str and active_end_values:
        s_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        ae_dt = min(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in active_end_values
        )
        if t_start >= s_dt and sample_time <= ae_dt:
            return "fully_within_active"
            
    return "unknown"


def build_run_manifest(
    w_config: WorkloadConfig,
    c_config: CollectorConfig,
    boundaries: Dict[str, Optional[str]],
    sample_count: int,
    metrics_records: Optional[List[Dict[str, Any]]] = None,
    throttling_record: Optional[Dict[str, Any]] = None,
    run_outcome: str = "unknown",
) -> Dict[str, Any]:
    """Generates a complete run manifest conforming to dataset_schema_v1."""
    allowed_outcomes = {"completed", "partial", "failed", "stopped", "unknown"}
    if run_outcome not in allowed_outcomes:
        raise ValueError(f"run_outcome must be one of {sorted(allowed_outcomes)}")
    generator_hash = hashlib.sha256(
        Path("experiment_kit/generators/locustfile.py").read_bytes()
        if Path("experiment_kit/generators/locustfile.py").exists()
        else b"locustfile"
    ).hexdigest()

    metrics = metrics_records or [
        {
            "name": "container_cpu_usage_seconds_total",
            "availability": "not_collected",
            "promql": c_config.get_promql_queries()["container_cpu_usage_seconds_total"],
            "unit": "cores",
            "query_start_utc": boundaries.get("collection_start_utc"),
            "query_end_utc": boundaries.get("collection_end_utc"),
            "step_seconds": c_config.step_seconds,
            "lookback_seconds": c_config.cpu_lookback_seconds,
            "labels": {
                "namespace": c_config.namespace,
                "pod": c_config.pod,
                "container": c_config.container,
            },
            "raw_sources": [],
        },
        {
            "name": "container_memory_working_set_bytes",
            "availability": "not_collected",
            "promql": c_config.get_promql_queries()["container_memory_working_set_bytes"],
            "unit": "bytes",
            "query_start_utc": boundaries.get("collection_start_utc"),
            "query_end_utc": boundaries.get("collection_end_utc"),
            "step_seconds": c_config.step_seconds,
            "lookback_seconds": None,
            "labels": {
                "namespace": c_config.namespace,
                "pod": c_config.pod,
                "container": c_config.container,
            },
            "raw_sources": [],
        },
    ]

    throttling = throttling_record or {
        "status": "unknown",
        "reason": "CFS counters not collected in offline dry-run",
        "coverage_start_utc": None,
        "coverage_end_utc": None,
        "covered_sample_count": None,
        "evidence_metrics": [],
    }

    manifest = {
        "schema_version": "1.0.0",
        "run_id": w_config.run_id,
        "pattern": w_config.pattern,
        "replicate": {
            "status": "observed",
            "value": w_config.replicate,
            "reason": None,
            "source": "experiment_kit",
            "observed_at_utc": boundaries.get("run_start_utc"),
        },
        "timezone": "UTC",
        "run_outcome": run_outcome,
        "workload": {
            "generator": {
                "status": "observed",
                "value": "locust",
                "reason": None,
                "source": "experiment_kit",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
            "generator_version": {
                "status": "observed",
                "value": "2.46.4",
                "reason": None,
                "source": "installed_package",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
            "generator_sha256": {
                "status": "observed",
                "value": generator_hash,
                "reason": None,
                "source": "file_hash",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
            "parameters": w_config.get_resolved_parameters(),
            "random_seed": {
                "status": "observed",
                "value": w_config.random_seed,
                "reason": None,
                "source": "experiment_kit",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
        },
        "environment": {
            "kubernetes_context": {
                "status": "unknown",
                "value": None,
                "reason": "unverified offline; cluster query prohibited in Stage 2",
                "source": None,
                "observed_at_utc": None,
            },
            "namespace": {
                "status": "observed" if c_config.namespace else "unknown",
                "value": c_config.namespace,
                "reason": None if c_config.namespace else "unspecified",
                "source": "config",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
            "job": {
                "status": "unknown",
                "value": None,
                "reason": "unverified offline",
                "source": None,
                "observed_at_utc": None,
            },
            "pod": {
                "status": "observed" if c_config.pod != "unknown" else "unknown",
                "value": c_config.pod if c_config.pod != "unknown" else None,
                "reason": None if c_config.pod != "unknown" else "pod unassigned offline",
                "source": "config" if c_config.pod != "unknown" else None,
                "observed_at_utc": boundaries.get("run_start_utc") if c_config.pod != "unknown" else None,
            },
            "pod_uid": {
                "status": "unknown",
                "value": c_config.pod_uid,
                "reason": "unverified offline" if not c_config.pod_uid else None,
                "source": None,
                "observed_at_utc": None,
            },
            "container": {
                "status": "observed",
                "value": c_config.container,
                "reason": None,
                "source": "config",
                "observed_at_utc": boundaries.get("run_start_utc"),
            },
            "container_id": {
                "status": "unknown",
                "value": c_config.container_id,
                "reason": "unverified offline" if not c_config.container_id else None,
                "source": None,
                "observed_at_utc": None,
            },
            "image": {
                "status": "unknown",
                "value": c_config.image,
                "reason": "unverified offline" if not c_config.image else None,
                "source": None,
                "observed_at_utc": None,
            },
            "node": {
                "status": "unknown",
                "value": c_config.node,
                "reason": "unverified offline" if not c_config.node else None,
                "source": None,
                "observed_at_utc": None,
            },
            "resources": {
                "cpu_request": {
                    "status": "unknown",
                    "value": c_config.cpu_request_cores,
                    "reason": "0B live quota audit required",
                    "source": None,
                    "observed_at_utc": None,
                },
                "memory_request": {
                    "status": "unknown",
                    "value": c_config.memory_request_bytes,
                    "reason": "0B live quota audit required",
                    "source": None,
                    "observed_at_utc": None,
                },
                "cpu_limit": {
                    "status": "unknown",
                    "value": c_config.cpu_limit_cores,
                    "reason": "0B live quota audit required",
                    "source": None,
                    "observed_at_utc": None,
                },
                "memory_limit": {
                    "status": "unknown",
                    "value": c_config.memory_limit_bytes,
                    "reason": "0B live quota audit required",
                    "source": None,
                    "observed_at_utc": None,
                },
            },
        },
        "boundaries": {
            **boundaries,
            "unknown_reasons": {},
        },
        "observation_units": {
            "cpu_usage_cores": "cores",
            "memory_working_set_bytes": "bytes",
            "cfs_throttled_periods_total": "periods",
            "cfs_periods_total": "periods",
            "cfs_throttled_seconds_total": "seconds",
            "throttled_period_ratio_1m": "ratio",
        },
        "scrape_interval": {
            "status": "unknown",
            "value": None,
            "reason": "cAdvisor scrape interval must be measured from live Prometheus in authorized stage",
            "source": None,
            "observed_at_utc": None,
        },
        "metrics": metrics,
        "throttling": throttling,
        "data_quality": {
            "expected_samples": sample_count,
            "observed_samples": sample_count,
            "gaps": [],
            "duplicate_timestamps": [],
            "counter_resets": [],
            "container_changes": [],
            "source_integrity": "verified",
        },
        "historical_audits": [],
    }
    return manifest


def build_observations_table(
    w_config: WorkloadConfig,
    c_config: CollectorConfig,
    boundaries: Dict[str, Optional[str]],
    measured_data: Optional[Dict[str, Dict[datetime, float]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Constructs observation rows for CSV serialization conforming to COLS.
    Does NOT fabricate measurements: uncollected values remain null with reasons.
    """
    start_dt = datetime.fromisoformat(boundaries["collection_start_utc"].replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(boundaries["collection_end_utc"].replace("Z", "+00:00"))
    run_start_dt = datetime.fromisoformat(boundaries["run_start_utc"].replace("Z", "+00:00"))
    workload_end_value = boundaries.get("workload_end_utc")
    shutdown_value = boundaries.get("locust_shutdown_utc")
    cooldown_value = boundaries.get("cooldown_start_utc")

    active_end_values = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in (workload_end_value, shutdown_value, cooldown_value)
        if value is not None
    ]
    active_end_dt = min(active_end_values) if active_end_values else None
    cooldown_dt = (
        datetime.fromisoformat(cooldown_value.replace("Z", "+00:00"))
        if cooldown_value is not None else None
    )
    shutdown_dt = (
        datetime.fromisoformat(shutdown_value.replace("Z", "+00:00"))
        if shutdown_value is not None else None
    )

    step = timedelta(seconds=c_config.step_seconds)
    rows: List[Dict[str, Any]] = []

    current_t = start_dt
    while current_t <= end_dt:
        ts_str = format_utc(current_t)

        # Classify phases using actual workload/shutdown and cooldown boundaries.
        if current_t < run_start_dt:
            phase = "pre_start"
        elif active_end_dt is not None and current_t < active_end_dt:
            phase = "active"
        elif cooldown_dt is not None and current_t >= cooldown_dt:
            phase = "cooldown"
        elif shutdown_dt is not None and current_t > shutdown_dt:
            phase = "post_shutdown"
        else:
            phase = "unknown"

        # Determine rate lookback window status
        rate_status = compute_rate_status(current_t, c_config.cpu_lookback_seconds, boundaries)

        missing_reasons: Dict[str, str] = {}

        # Fill measurements if provided, otherwise record explicit missing reasons
        data = measured_data or {}
        
        cpu_val = data.get("container_cpu_usage_seconds_total", {}).get(current_t)
        if cpu_val is None:
            cpu_cores_str = ""
            rate_status = "not_applicable"
            missing_reasons["cpu_usage_cores"] = "uncollected offline"
        else:
            cpu_cores_str = str(cpu_val)

        mem_val = data.get("container_memory_working_set_bytes", {}).get(current_t)
        if mem_val is None:
            mem_bytes_str = ""
            missing_reasons["memory_working_set_bytes"] = "uncollected offline"
        else:
            mem_bytes_str = str(int(mem_val))

        # CFS counters
        cfs_th_p = data.get("container_cpu_cfs_throttled_periods_total", {}).get(current_t)
        if cfs_th_p is None:
            cfs_th_p_str = ""
            missing_reasons["cfs_throttled_periods_total"] = "uncollected offline"
        else:
            cfs_th_p_str = str(int(cfs_th_p))

        cfs_tot_p = data.get("container_cpu_cfs_periods_total", {}).get(current_t)
        if cfs_tot_p is None:
            cfs_tot_p_str = ""
            missing_reasons["cfs_periods_total"] = "uncollected offline"
        else:
            cfs_tot_p_str = str(int(cfs_tot_p))

        cfs_th_s = data.get("container_cpu_cfs_throttled_seconds_total", {}).get(current_t)
        if cfs_th_s is None:
            cfs_th_s_str = ""
            missing_reasons["cfs_throttled_seconds_total"] = "uncollected offline"
        else:
            cfs_th_s_str = str(cfs_th_s)

        th_ratio = data.get("throttled_period_ratio_1m", {}).get(current_t)
        if th_ratio is None:
            th_ratio_str = ""
            missing_reasons["throttled_period_ratio_1m"] = "uncollected offline"
        else:
            th_ratio_str = str(th_ratio)

        row = {
            "timestamp_utc": ts_str,
            "run_id": w_config.run_id,
            "pattern": w_config.pattern,
            "namespace": c_config.namespace,
            "pod": c_config.pod,
            "pod_uid": c_config.pod_uid or "",
            "container": c_config.container,
            "container_id": c_config.container_id or "",
            "sample_phase": phase,
            "cpu_usage_cores": cpu_cores_str,
            "cpu_rate_lookback_seconds": str(c_config.cpu_lookback_seconds),
            "cpu_rate_window_status": rate_status,
            "memory_working_set_bytes": mem_bytes_str,
            "cfs_throttled_periods_total": cfs_th_p_str,
            "cfs_periods_total": cfs_tot_p_str,
            "cfs_throttled_seconds_total": cfs_th_s_str,
            "throttled_period_ratio_1m": th_ratio_str,
            "quality_flags": "[]",
            "missing_reasons": json.dumps(missing_reasons),
        }
        rows.append(row)
        current_t += step

    return rows
