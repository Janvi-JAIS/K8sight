"""
Parameterized LoadTestShape classes for the five K8sight workload patterns.
Provides fallback base class if Locust is not installed in local environment,
allowing offline testing and execution.
"""

from __future__ import annotations

import math
import random
import hashlib
from typing import Optional, Tuple, Type, Any, Dict

from experiment_kit.generators.config import WorkloadConfig

try:
    from locust import LoadTestShape as BaseLoadTestShape
except ImportError:
    class BaseLoadTestShape:  # type: ignore
        """Offline fallback stub for LoadTestShape when Locust is not installed."""
        def __init__(self) -> None:
            self._mock_run_time = 0.0

        def get_run_time(self) -> float:
            return self._mock_run_time

        def tick(self) -> Optional[Tuple[int, float]]:
            return None


class BaseConfiguredShape(BaseLoadTestShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__()
        self.config = config or WorkloadConfig.from_env()
        self.params = self.config.get_resolved_parameters()
        self.duration_seconds = self.config.duration_seconds
        self.spawn_rate = float(self.params.get("spawn_rate", 5.0))

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        """Calculates expected target user count at given run_time seconds. Used for dry-run and offline inspection."""
        raise NotImplementedError


class ParameterizedStableShape(BaseConfiguredShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__(config)
        self.users = int(self.params["users"])

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        if run_time > self.duration_seconds:
            return None
        return self.users

    def tick(self) -> Optional[Tuple[int, float]]:
        run_time = self.get_run_time()
        users = self.get_target_users_at_time(run_time)
        if users is None:
            return None
        return (users, self.spawn_rate)


class ParameterizedIncreasingShape(BaseConfiguredShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__(config)
        self.start_users = int(self.params["start_users"])
        self.end_users = int(self.params["end_users"])

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        if run_time > self.duration_seconds:
            return None
        progress = min(1.0, max(0.0, run_time / max(1.0, float(self.duration_seconds))))
        return int(self.start_users + progress * (self.end_users - self.start_users))

    def tick(self) -> Optional[Tuple[int, float]]:
        run_time = self.get_run_time()
        users = self.get_target_users_at_time(run_time)
        if users is None:
            return None
        return (users, self.spawn_rate)


class ParameterizedPeriodicShape(BaseConfiguredShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__(config)
        self.min_users = int(self.params["min_users"])
        self.max_users = int(self.params["max_users"])
        self.period_seconds = float(self.params["period_seconds"])

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        if run_time > self.duration_seconds:
            return None
        amplitude = (self.max_users - self.min_users) / 2.0
        midpoint = self.min_users + amplitude
        users = int(midpoint + amplitude * math.sin(2.0 * math.pi * run_time / self.period_seconds))
        return max(1, users)

    def tick(self) -> Optional[Tuple[int, float]]:
        run_time = self.get_run_time()
        users = self.get_target_users_at_time(run_time)
        if users is None:
            return None
        return (users, self.spawn_rate)


class ParameterizedSpikeShape(BaseConfiguredShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__(config)
        self.baseline_users = int(self.params["baseline_users"])
        self.spike_users = int(self.params["spike_users"])
        self.stages = self.params["stages"]

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        if run_time > self.duration_seconds:
            return None
        for start, end, users in self.stages:
            if start <= run_time < end:
                return int(users)
        return self.baseline_users

    def tick(self) -> Optional[Tuple[int, float]]:
        run_time = self.get_run_time()
        users = self.get_target_users_at_time(run_time)
        if users is None:
            return None
        return (users, self.spawn_rate)


class ParameterizedRandomSpikeShape(BaseConfiguredShape):
    def __init__(self, config: Optional[WorkloadConfig] = None) -> None:
        super().__init__(config)
        self.baseline_users = int(self.params["baseline_users"])
        self.spike_user_range = tuple(self.params["spike_user_range"])
        self.check_interval = float(self.params["check_interval"])
        self.spike_chance = float(self.params["spike_chance"])
        self.random_seed = int(self.params["random_seed"])

    def get_target_users_at_time(self, run_time: float) -> Optional[int]:
        if run_time > self.duration_seconds:
            return None
        interval_index = int(run_time // self.check_interval)
        # Hash seed and interval_index to decouple from pure index while remaining strictly reproducible
        h_str = f"seed={self.random_seed}:interval={interval_index}"
        derived_seed = int(hashlib.sha256(h_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(derived_seed)
        if rng.random() < self.spike_chance:
            return rng.randint(*self.spike_user_range)
        return self.baseline_users

    def tick(self) -> Optional[Tuple[int, float]]:
        run_time = self.get_run_time()
        users = self.get_target_users_at_time(run_time)
        if users is None:
            return None
        return (users, self.spawn_rate)


SHAPE_MAP: Dict[str, Type[BaseConfiguredShape]] = {
    "stable": ParameterizedStableShape,
    "increasing": ParameterizedIncreasingShape,
    "periodic": ParameterizedPeriodicShape,
    "spike": ParameterizedSpikeShape,
    "random_spike": ParameterizedRandomSpikeShape,
}


def get_shape_instance(config: Optional[WorkloadConfig] = None) -> BaseConfiguredShape:
    cfg = config or WorkloadConfig.from_env()
    shape_cls = SHAPE_MAP.get(cfg.pattern)
    if shape_cls is None:
        raise ValueError(f"No shape class registered for pattern '{cfg.pattern}'")
    return shape_cls(cfg)
