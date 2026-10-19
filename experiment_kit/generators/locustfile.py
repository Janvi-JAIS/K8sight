"""
Locust load-generation entrypoint for K8sight experiment kit.
Dynamically configures LoadTestShape and HttpUser from WorkloadConfig environment variables.

IMPORTANT ARCHITECTURAL NOTE ON CONCURRENCY & TRAFFIC PACING:
Locust's task pacing (such as `constant_throughput` or `constant_pacing`) does NOT guarantee
strictly open-loop traffic generation. In Locust, each virtual user is a coroutine executing
synchronous HTTP requests. If the target server's response time exceeds the pacing interval
(e.g. during pod CPU throttling, queuing, or latency spikes), the virtual user blocks until
the request completes before scheduling the next cycle. Consequently, achieved throughput
drifts downward under severe congestion.

Therefore:
1. Pacing is used as a rate-regulation target, NOT a proof of open-loop generation.
2. Achieved request rates must ALWAYS be measured from collected Locust statistics (RPS),
   never assumed to equal commanded rates.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.generators.shapes import get_shape_instance

try:
    from locust import HttpUser, task, between, constant_throughput, constant_pacing
    HAVE_LOCUST = True
except ImportError:
    HAVE_LOCUST = False
    # Offline stubs for environment where Locust is not installed
    class HttpUser:  # type: ignore
        abstract = True
        host = ""
        def __init__(self, *args, **kwargs): pass
    def task(f): return f
    def constant_pacing(p): return lambda self: p


config = WorkloadConfig.from_env()


class K8sightApacheUser(HttpUser):
    """
    Simulates user traffic against the microservice endpoint.
    If dry_run is True, no network requests are sent.
    """
    # Pacing: target 1 request per second per active virtual user.
    # Note: As documented above, actual rate will drop if latency > 1s.
    wait_time = constant_pacing(1.0) if HAVE_LOCUST else (lambda self: 1.0)

    @task
    def request_target(self) -> None:
        if config.dry_run:
            # Safe dry-run mode: no HTTP request dispatched
            return

        if hasattr(self, "client") and self.client:
            # Query the target service. Path defaults to "/"
            # The hpa-example image executes its CPU workload upon receiving any GET request.
            self.client.get("/")


# Instantiate the active shape class for Locust runtime
active_shape_instance = get_shape_instance(config)
ActiveWorkloadShape = active_shape_instance.__class__
