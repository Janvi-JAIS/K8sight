"""
Configuration for parameterized Locust workload generators.
Supports explicit metadata, levels, seeds, and durations.
"""

from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional

PATTERNS = ("stable", "increasing", "periodic", "spike", "random_spike")
LOAD_LEVELS = ("low", "high")
DEFAULT_DURATION_SECONDS = 1200  # 20 minutes


@dataclass
class WorkloadConfig:
    run_id: str
    pattern: str
    load_level: str = "high"
    replicate: int = 1
    random_seed: int = 42
    duration_seconds: int = DEFAULT_DURATION_SECONDS
    target_host: str = "http://php-apache"
    dry_run: bool = False

    # Optional pattern-specific parameter overrides
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern '{self.pattern}' invalid; must be one of {PATTERNS}")
        if self.load_level not in LOAD_LEVELS and not self.custom_params:
            raise ValueError(f"load_level '{self.load_level}' invalid; must be one of {LOAD_LEVELS}")
        if self.replicate < 1:
            raise ValueError("replicate must be >= 1")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")

    def get_resolved_parameters(self) -> Dict[str, Any]:
        """Resolves default parameters based on pattern and load_level, applying any custom overrides."""
        self.validate()
        params: Dict[str, Any] = {
            "pattern": self.pattern,
            "load_level": self.load_level,
            "duration_seconds": self.duration_seconds,
            "replicate": self.replicate,
            "random_seed": self.random_seed,
        }

        lvl = self.load_level
        if self.pattern == "stable":
            params["users"] = 2 if lvl == "low" else 5
            params["spawn_rate"] = 5

        elif self.pattern == "increasing":
            params["start_users"] = 1
            params["end_users"] = 4 if lvl == "low" else 8
            params["spawn_rate"] = 5

        elif self.pattern == "periodic":
            params["min_users"] = 2 if lvl == "low" else 4
            params["max_users"] = 10 if lvl == "low" else 20
            params["period_seconds"] = 120
            params["spawn_rate"] = 10

        elif self.pattern == "spike":
            params["baseline_users"] = 2 if lvl == "low" else 4
            params["spike_users"] = 15 if lvl == "low" else 30
            params["spawn_rate"] = 20
            # Keep the spike inside the run; use up to 120 seconds.
            spike_duration = min(120, max(1, self.duration_seconds // 3))
            spike_start = min(
                int(self.duration_seconds * 0.33),
                self.duration_seconds - spike_duration,
            )
            spike_end = spike_start + spike_duration
            params["stages"] = [
                (0, spike_start, params["baseline_users"]),
                (spike_start, spike_end, params["spike_users"]),
                (spike_end, self.duration_seconds, params["baseline_users"]),
            ]

        elif self.pattern == "random_spike":
            params["baseline_users"] = 2 if lvl == "low" else 4
            params["spike_user_range"] = (8, 16) if lvl == "low" else (16, 32)
            params["check_interval"] = 15
            params["spike_chance"] = 0.20
            params["spawn_rate"] = 15

        # Apply any explicit user overrides
        params.update(self.custom_params)
        return params

    def compute_schedule_hash(self) -> str:
        """Deterministic SHA-256 hash of configuration parameters and seed."""
        params = self.get_resolved_parameters()
        encoded = json.dumps(params, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schedule_hash"] = self.compute_schedule_hash()
        d["resolved_parameters"] = self.get_resolved_parameters()
        return d

    @classmethod
    def from_env(cls) -> "WorkloadConfig":
        run_id = os.environ.get("K8SIGHT_RUN_ID", "local_dry_run")
        pattern = os.environ.get("K8SIGHT_PATTERN", "stable")
        load_level = os.environ.get("K8SIGHT_LOAD_LEVEL", "high")
        replicate = int(os.environ.get("K8SIGHT_REPLICATE", "1"))
        random_seed = int(os.environ.get("K8SIGHT_SEED", "42"))
        duration = int(os.environ.get("K8SIGHT_DURATION", str(DEFAULT_DURATION_SECONDS)))
        target_host = os.environ.get("K8SIGHT_TARGET_HOST", "http://php-apache")
        dry_run = os.environ.get("K8SIGHT_DRY_RUN", "0").lower() in ("1", "true", "yes")

        custom_str = os.environ.get("K8SIGHT_CUSTOM_PARAMS")
        custom_params = json.loads(custom_str) if custom_str else {}

        cfg = cls(
            run_id=run_id,
            pattern=pattern,
            load_level=load_level,
            replicate=replicate,
            random_seed=random_seed,
            duration_seconds=duration,
            target_host=target_host,
            dry_run=dry_run,
            custom_params=custom_params,
        )
        cfg.validate()
        return cfg
