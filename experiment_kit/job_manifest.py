"""Pure Kubernetes Job and ConfigMap manifest construction; no cluster access."""
from __future__ import annotations

import hashlib
import json
import re


def build_job_manifests(*, run_id: str, image: str, namespace: str = "default",
                        config_files: dict[str, str], target_host: str,
                        pattern: str = "stable", seed: int = 42,
                        duration: int = 1200, load_level: str = "high",
                        replicate: int = 1, custom_params: dict | None = None,
                        active_deadline_seconds: int = 1800) -> list[dict]:
    """Return ConfigMap and Job dictionaries for review or serialization."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,50}", run_id.lower()):
        raise ValueError("run_id must be DNS-label compatible")
    if not image or not namespace or not config_files:
        raise ValueError("image, namespace and config_files are required")
    if active_deadline_seconds <= 0 or duration <= 0:
        raise ValueError("duration and active deadline must be positive")
    job_name = "k8sight-" + hashlib.sha256(run_id.encode()).hexdigest()[:12]
    config_name = job_name + "-files"
    items = [{"key": name, "path": name} for name in sorted(config_files)]
    env = [{"name": key, "value": str(value)} for key, value in {
        "K8SIGHT_RUN_ID": run_id, "K8SIGHT_PATTERN": pattern,
        "K8SIGHT_LOAD_LEVEL": load_level, "K8SIGHT_SEED": seed,
        "K8SIGHT_DURATION": duration, "K8SIGHT_TARGET_HOST": target_host,
        "K8SIGHT_REPLICATE": replicate,
        "K8SIGHT_CUSTOM_PARAMS": json.dumps(custom_params or {}, sort_keys=True),
    }.items()]
    config_map = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": config_name, "namespace": namespace},
                  "data": dict(config_files)}
    job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": job_name, "namespace": namespace},
           "spec": {"activeDeadlineSeconds": active_deadline_seconds, "ttlSecondsAfterFinished": 86400,
                    "template": {"spec": {"restartPolicy": "Never", "containers": [{
                        "name": "locust", "image": image,
                        "command": ["locust", "-f", "/opt/k8sight/generators/locustfile.py", "--headless",
                                    f"--run-time={duration}s", "--only-summary", "--csv=/logs/locust",
                                    "--csv-full-history", "--logfile=-"],
                        "env": env + [{"name": "PYTHONPATH", "value": "/opt/k8sight"}],
                        "volumeMounts": [{"name": "config", "mountPath": "/opt/k8sight", "readOnly": True}],
                    }], "volumes": [{"name": "config", "configMap": {"name": config_name, "items": items}}]}}}}
    return [config_map, job]
