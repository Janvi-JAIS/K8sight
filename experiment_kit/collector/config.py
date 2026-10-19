"""
Configuration and query definitions for Prometheus and run-record collection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

DEFAULT_NAMESPACE = "default"
DEFAULT_CONTAINER = "hpa-example"
DEFAULT_STEP_SECONDS = 15
DEFAULT_SCRAPE_INTERVAL_SECONDS = 15
DEFAULT_CPU_LOOKBACK_SECONDS = 60
DEFAULT_PRE_START_SECONDS = 120  # 2-minute idle baseline
DEFAULT_COOLDOWN_SECONDS = 180   # 3-minute cooldown


@dataclass
class CollectorConfig:
    run_id: str
    pattern: str
    load_level: str = "high"
    replicate: int = 1
    random_seed: int = 42
    duration_seconds: int = 1200

    # Cluster identities (to be confirmed against live cluster in authorized stage)
    namespace: str = DEFAULT_NAMESPACE
    pod: str = "unknown"
    pod_uid: Optional[str] = None
    container: str = DEFAULT_CONTAINER
    container_id: Optional[str] = None
    node: Optional[str] = None
    image: Optional[str] = None

    # Timing and windows
    step_seconds: int = DEFAULT_STEP_SECONDS
    scrape_interval_seconds: int = DEFAULT_SCRAPE_INTERVAL_SECONDS
    cpu_lookback_seconds: int = DEFAULT_CPU_LOOKBACK_SECONDS
    pre_start_seconds: int = DEFAULT_PRE_START_SECONDS
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS

    # Resource requests/limits (unverified until 0B live cluster check)
    cpu_request_cores: Optional[float] = None
    cpu_limit_cores: Optional[float] = None
    memory_request_bytes: Optional[int] = None
    memory_limit_bytes: Optional[int] = None

    def get_promql_queries(self) -> Dict[str, str]:
        """Exact PromQL queries for the target container."""
        lbls = f'namespace="{self.namespace}",pod="{self.pod}",container="{self.container}"'
        lookback = f"{self.cpu_lookback_seconds}s"
        return {
            "container_cpu_usage_seconds_total": f"rate(container_cpu_usage_seconds_total{{{lbls}}}[{lookback}])",
            "container_memory_working_set_bytes": f"container_memory_working_set_bytes{{{lbls}}}",
            "container_cpu_cfs_throttled_periods_total": f"container_cpu_cfs_throttled_periods_total{{{lbls}}}",
            "container_cpu_cfs_periods_total": f"container_cpu_cfs_periods_total{{{lbls}}}",
            "container_cpu_cfs_throttled_seconds_total": f"container_cpu_cfs_throttled_seconds_total{{{lbls}}}",
        }
