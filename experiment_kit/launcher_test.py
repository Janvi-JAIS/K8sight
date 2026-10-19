"""Conservative launcher test for K8sight; not a training-data generator.

This script is intended to orchestrate a single stable Locust test,
record observed process boundaries, wait through cooldown, and archive
read-only Prometheus responses. It does not change Kubernetes resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
LOCUSTFILE = ROOT / "locustfile_stable.py"
DEFAULT_RUN_DIR = ROOT / "run_records" / "launcher-test-stable-20261013T134851Z"
PROMETHEUS_URL = "http://127.0.0.1:19091"
APP_URL = "http://127.0.0.1:18080/?value=10500"
NAMESPACE = "default"
CONTAINER = "hpa-example"
STEP_SECONDS = 15
SCRAPE_SECONDS = 15
LOOKBACK_SECONDS = 60
BASELINE_SECONDS = 120
WORKLOAD_SECONDS = 300
COOLDOWN_SECONDS = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def epoch_now() -> int:
    return int(time.time())


def run(command: list[str], *, timeout: int = 10) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def http_get(url: str, timeout: int = 5) -> tuple[int, str]:
    from urllib.request import Request
    request = Request(url, headers={"User-Agent": "k8sight-launcher-test/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def write_json_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def update_json_atomic(path: Path, data: dict) -> None:
    """Atomically replace a metadata file created by this run."""
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def utc_at_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def resource_cpu_millicores(value: str) -> float:
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000


def target_identity(pod_doc: dict) -> tuple[str | None, str | None]:
    items = pod_doc.get("items", [])
    if len(items) != 1:
        return None, None
    pod = items[0]
    statuses = pod.get("status", {}).get("containerStatuses", [])
    status = next((item for item in statuses if item.get("name") == CONTAINER), None)
    return pod.get("metadata", {}).get("uid"), status.get("containerID") if status else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser


def enforce_cpu_headroom(cpu_cores: float, cpu_limit: str) -> float:
    limit_millicores = resource_cpu_millicores(cpu_limit)
    allowed_cores = (limit_millicores - 300) / 1000
    if cpu_cores < 0 or cpu_cores > allowed_cores:
        raise RuntimeError(
            f"CPU safety gate failed: current CPU={cpu_cores * 1000:.1f}m; "
            f"need <= {allowed_cores * 1000:.1f}m to preserve >=300m "
            f"headroom against {cpu_limit}."
        )
    return limit_millicores


def persist_failure(metadata: dict, metadata_path: Path, failure_path: Path,
                    original_error: Exception) -> None:
    metadata["status"] = "failed_preflight_or_execution"
    metadata["failure"] = {"observed_at_utc": utc_now(), "error": str(original_error)}
    if metadata_path.is_file():
        try:
            update_json_atomic(metadata_path, metadata)
        except OSError as metadata_error:
            metadata["metadata_update_error"] = str(metadata_error)
            print(
                f"WARNING: could not update metadata after failure: {metadata_error}",
                file=sys.stderr,
            )
    if not failure_path.exists():
        try:
            write_json_new(failure_path, metadata)
        except OSError as failure_write_error:
            print(
                f"WARNING: could not preserve failure record: {failure_write_error}",
                file=sys.stderr,
            )


def main() -> int:
    args = build_parser().parse_args()

    run_dir = args.run_dir.resolve()
    try:
        run_dir.relative_to((ROOT / "run_records").resolve())
    except ValueError:
        print("ERROR: run directory must be inside run_records/", file=sys.stderr)
        return 2

    if not PYTHON.is_file() or not LOCUSTFILE.is_file():
        print("ERROR: expected project Python or stable Locust file is missing.", file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2
    if any(p.name not in {"run_id.txt", "run_dir.txt"} for p in run_dir.iterdir()):
        print("ERROR: run directory contains unexpected files; refusing to continue.", file=sys.stderr)
        return 2

    run_id = (run_dir / "run_id.txt").read_text(encoding="utf-8").strip()
    if not run_id.startswith("launcher-test-"):
        print("ERROR: run_id must start with launcher-test-.", file=sys.stderr)
        return 2

    artifacts = {
        "metadata": run_dir / "launcher_test_metadata.json",
        "locust_log": run_dir / "locust_console.log",
        "locust_csv_prefix": run_dir / "locust",
        "stdout": run_dir / "locust_stdout.log",
        "stderr": run_dir / "locust_stderr.log",
        "raw_dir": run_dir / "prometheus_raw",
        "context": run_dir / "run_context.json",
    }
    for key, path in artifacts.items():
        if key == "raw_dir":
            if path.exists():
                print(f"ERROR: output already exists: {path}", file=sys.stderr)
                return 2
        elif key == "locust_csv_prefix":
            if any(run_dir.glob("locust_*.csv")):
                print("ERROR: Locust CSV output already exists.", file=sys.stderr)
                return 2
        elif path.exists():
            print(f"ERROR: output already exists: {path}", file=sys.stderr)
            return 2

    metadata = {
        "run_id": run_id,
        "purpose": "launcher_test_only_not_training_data",
        "pattern": "stable",
        "status": "preflight",
        "created_at_utc": utc_now(),
        "run_directory": str(run_dir),
        "configuration": {
            "users": 1,
            "duration_seconds": WORKLOAD_SECONDS,
            "baseline_seconds": BASELINE_SECONDS,
            "cooldown_seconds": COOLDOWN_SECONDS,
            "prometheus_step_seconds": STEP_SECONDS,
            "cpu_lookback_seconds": LOOKBACK_SECONDS,
            "application_url": APP_URL,
            "prometheus_url": PROMETHEUS_URL,
        },
        "events": [],
        "training_excluded": True,
        "replicate": 0,
        "load_level": "launcher_test",
    }

    try:
        context = run(["kubectl", "config", "current-context"])
        if context != "docker-desktop":
            raise RuntimeError(f"Unexpected Kubernetes context: {context}")

        pod_json = run([
            "kubectl", "get", "pods", "-n", NAMESPACE,
            "-l", "app=php-apache", "-o", "json",
        ])
        pods = json.loads(pod_json).get("items", [])
        ready_pods = [
            pod for pod in pods
            if pod.get("status", {}).get("phase") == "Running"
            and any(
                c.get("name") == CONTAINER and c.get("ready")
                for c in pod.get("status", {}).get("containerStatuses", [])
            )
        ]
        if len(ready_pods) != 1:
            raise RuntimeError(f"Expected exactly one Ready target Pod; found {len(ready_pods)}")
        pod = ready_pods[0]
        pod_name = pod["metadata"]["name"]
        pod_uid = pod["metadata"]["uid"]
        container_status = next(
            c for c in pod["status"]["containerStatuses"] if c["name"] == CONTAINER
        )
        container_id = container_status["containerID"]
        restart_count = container_status["restartCount"]

        app_status, _ = http_get(APP_URL)
        if app_status != 200:
            raise RuntimeError(f"Application returned HTTP {app_status}")
        prom_status, _ = http_get(f"{PROMETHEUS_URL}/-/healthy")
        if prom_status != 200:
            raise RuntimeError(f"Prometheus health returned HTTP {prom_status}")

        # Read current CPU usage for the 300m headroom gate.
        selector = (
            f'container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",'
            f'pod="{pod_name}",container="{CONTAINER}"}}'
        )
        query = urlencode({"query": f"rate({selector}[1m])"})
        _, query_body = http_get(f"{PROMETHEUS_URL}/api/v1/query?{query}")
        query_result = json.loads(query_body)
        if query_result.get("status") != "success":
            raise RuntimeError("Prometheus CPU safety query failed")
        series = query_result.get("data", {}).get("result", [])
        if len(series) != 1 or len(series[0].get("value", [])) != 2:
            raise RuntimeError("CPU safety query did not return exactly one usable series")
        cpu_cores = float(series[0]["value"][1])
        spec_container = next(
            (item for item in pod.get("spec", {}).get("containers", [])
             if item.get("name") == CONTAINER), None
        )
        if spec_container is None:
            raise RuntimeError("Target container missing from Pod specification")
        resources = spec_container.get("resources", {})
        cpu_limit = resources.get("limits", {}).get("cpu")
        if cpu_limit is None:
            raise RuntimeError("Target Pod has no CPU limit; refusing the safety gate")
        # Preserve at least 300m headroom against the observed Pod CPU limit.
        cpu_limit_millicores = enforce_cpu_headroom(cpu_cores, cpu_limit)

        metadata["status"] = "preflight_passed"
        metadata["preflight"] = {
            "observed_at_utc": utc_now(),
            "kubernetes_context": context,
            "namespace": NAMESPACE,
            "pod": pod_name,
            "pod_uid": pod_uid,
            "container": CONTAINER,
            "container_id": container_id,
            "restart_count_before": restart_count,
            "application_http_status": app_status,
            "prometheus_healthy": True,
            "cpu_usage_cores": cpu_cores,
            "cpu_usage_millicores": round(cpu_cores * 1000, 3),
            "cpu_limit": cpu_limit,
            "cpu_limit_millicores": cpu_limit_millicores,
            "required_headroom_millicores": 300,
            "generator_sha256": hashlib.sha256(LOCUSTFILE.read_bytes()).hexdigest(),
        }
        metadata["events"].append({"event": "preflight_passed", "observed_at_utc": utc_now()})
        write_json_new(artifacts["metadata"], metadata)

        print(f"Preflight passed. Target Pod: {pod_name}")
        print(f"Current CPU: {cpu_cores * 1000:.1f}m; required headroom: >=300m")
        print(f"Waiting {BASELINE_SECONDS}s for pre-start baseline. No load is active.")
        metadata["events"].append({
            "event": "baseline_wait_started", "observed_at_utc": utc_now()
        })
        collection_start_epoch = epoch_now()
        metadata["collection_start_epoch"] = collection_start_epoch
        metadata["collection_start_utc"] = utc_at_epoch(collection_start_epoch)
        update_json_atomic(artifacts["metadata"], metadata)
        time.sleep(BASELINE_SECONDS)

        run_start_utc = utc_now()
        metadata["events"].append({
            "event": "locust_process_launch", "observed_at_utc": run_start_utc
        })
        metadata["status"] = "running"

        command = [
            str(PYTHON), "-m", "locust",
            "-f", str(LOCUSTFILE),
            "--headless",
            "--host", "http://127.0.0.1:18080",
            "--users", "1",
            "--spawn-rate", "1",
            "--run-time", f"{WORKLOAD_SECONDS}s",
            "--stop-timeout", "5",
            "--csv", str(artifacts["locust_csv_prefix"]),
            "--csv-full-history",
            "--logfile", str(artifacts["locust_log"]),
            "--only-summary",
        ]
        with artifacts["stdout"].open("x", encoding="utf-8") as stdout, \
             artifacts["stderr"].open("x", encoding="utf-8") as stderr:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            exit_code = proc.wait()

        process_exit_utc = utc_now()
        process_exit_epoch = epoch_now()
        metadata["events"].append({
            "event": "locust_process_exit",
            "observed_at_utc": process_exit_utc,
            "exit_code": exit_code,
        })
        metadata["status"] = "workload_process_exited"
        metadata["locust_exit_code"] = exit_code
        metadata["workload_end_boundary_note"] = (
            "Process-exit time is used conservatively as the workload-end boundary "
            "for this launcher test; do not use this run for training."
        )
        update_json_atomic(artifacts["metadata"], metadata)

        # The test is diagnostic only. Preserve a failed exit as failed evidence.
        if exit_code != 0:
            print(f"Locust exited with code {exit_code}; evidence will be preserved.")
        print(f"Locust process exited at {process_exit_utc}.")
        print(f"Waiting {COOLDOWN_SECONDS}s for cooldown before Prometheus collection.")
        time.sleep(COOLDOWN_SECONDS)

        identity_json = run([
            "kubectl", "get", "pods", "-n", NAMESPACE,
            "-l", "app=php-apache", "-o", "json",
        ])
        current_uid, current_container_id = target_identity(json.loads(identity_json))
        identity_matches = current_uid == pod_uid and current_container_id == container_id
        if not identity_matches:
            raise RuntimeError(
                "Target Pod identity changed during the run; refusing to accept metrics"
            )

        collection_end_epoch = epoch_now()
        collection_start_utc = utc_at_epoch(collection_start_epoch)
        collection_end_utc = utc_at_epoch(collection_end_epoch)
        collector_cmd = [
            str(PYTHON), "-m", "experiment_kit.collector.live_collector",
            "--prometheus-url", PROMETHEUS_URL,
            "--namespace", NAMESPACE,
            "--pod", pod_name,
            "--container", CONTAINER,
            "--start-epoch", str(collection_start_epoch),
            "--end-epoch", str(collection_end_epoch),
            "--step", str(STEP_SECONDS),
            "--scrape-interval", str(SCRAPE_SECONDS),
            "--lookback", str(LOOKBACK_SECONDS),
            "--output-dir", str(artifacts["raw_dir"]),
        ]
        collector_result = subprocess.run(
            collector_cmd, cwd=ROOT, text=True, capture_output=True,
            timeout=120, check=False,
        )
        (run_dir / "collector_stdout.log").write_text(
            collector_result.stdout, encoding="utf-8"
        )
        (run_dir / "collector_stderr.log").write_text(
            collector_result.stderr, encoding="utf-8"
        )

        collector_finished_utc = utc_now()
        context_doc = {
            "boundaries": {
                "run_start_utc": run_start_utc,
                "workload_end_utc": process_exit_utc,
                "locust_shutdown_utc": process_exit_utc,
                "cooldown_start_utc": process_exit_utc,
                "collection_start_utc": collection_start_utc,
                "collection_end_utc": collection_end_utc,
                "collector_finished_utc": collector_finished_utc,
            },
            "workload": {
                "generator": {
                    "status": "observed", "value": "Locust",
                    "reason": None, "source": "launcher process command",
                    "observed_at_utc": utc_now(),
                },
                "generator_version": {
                    "status": "observed", "value": run([str(PYTHON), "-m", "locust", "--version"]),
                    "reason": None, "source": "Locust version command",
                    "observed_at_utc": utc_now(),
                },
                "generator_sha256": {
                    "status": "observed",
                    "value": hashlib.sha256(LOCUSTFILE.read_bytes()).hexdigest(),
                    "reason": None, "source": "SHA-256 of locustfile_stable.py",
                    "observed_at_utc": utc_now(),
                },
                "random_seed": {
                    "status": "unknown", "value": None,
                    "reason": "Stable launcher test does not configure a random seed",
                    "source": "launcher configuration", "observed_at_utc": None,
                },
                "parameters": {
                    "pattern": "stable", "load_level": "launcher_test",
                    "replicate": 0, "duration_seconds": WORKLOAD_SECONDS,
                    "users": 1, "spawn_rate": 1,
                    "target_url": "http://127.0.0.1:18080",
                },
            },
            "environment": {
                "kubernetes_context": {"status": "observed", "value": context,
                    "reason": None, "source": "kubectl config current-context",
                    "observed_at_utc": utc_now()},
                "namespace": {"status": "observed", "value": NAMESPACE,
                    "reason": None, "source": "kubectl Pod inspection",
                    "observed_at_utc": utc_now()},
                "job": {"status": "not_applicable", "value": None,
                    "reason": "Locust runs locally, not as a Kubernetes Job",
                    "source": "launcher design", "observed_at_utc": utc_now()},
                "pod": {"status": "observed", "value": pod_name, "reason": None,
                    "source": "kubectl Pod inspection", "observed_at_utc": utc_now()},
                "pod_uid": {"status": "observed", "value": pod_uid, "reason": None,
                    "source": "kubectl Pod inspection", "observed_at_utc": utc_now()},
                "container": {"status": "observed", "value": CONTAINER, "reason": None,
                    "source": "kubectl Pod inspection", "observed_at_utc": utc_now()},
                "container_id": {"status": "observed", "value": container_id, "reason": None,
                    "source": "kubectl Pod inspection", "observed_at_utc": utc_now()},
                "image": {"status": "observed",
                    "value": pod["spec"]["containers"][0]["image"], "reason": None,
                    "source": "kubectl Pod specification", "observed_at_utc": utc_now()},
                "node": {"status": "observed", "value": pod["spec"].get("nodeName"), "reason": None,
                    "source": "kubectl Pod specification", "observed_at_utc": utc_now()},
                "resources": {
                    "cpu_request": {"status": "observed" if resources.get("requests", {}).get("cpu") is not None else "unknown", "value": resources.get("requests", {}).get("cpu"), "reason": None if resources.get("requests", {}).get("cpu") is not None else "not specified",
                        "source": "current Pod resource specification", "observed_at_utc": utc_now()},
                    "memory_request": {"status": "observed" if resources.get("requests", {}).get("memory") is not None else "unknown", "value": resources.get("requests", {}).get("memory"), "reason": None if resources.get("requests", {}).get("memory") is not None else "not specified",
                        "source": "current Pod resource specification", "observed_at_utc": utc_now()},
                    "cpu_limit": {"status": "observed", "value": cpu_limit, "reason": None,
                        "source": "current Pod resource specification", "observed_at_utc": utc_now()},
                    "memory_limit": {"status": "observed" if resources.get("limits", {}).get("memory") is not None else "unknown", "value": resources.get("limits", {}).get("memory"), "reason": None if resources.get("limits", {}).get("memory") is not None else "not specified",
                        "source": "current Pod resource specification", "observed_at_utc": utc_now()},
                },
            },
            "identity_samples": [],
            "historical_audits": [],
        }
        write_json_new(artifacts["context"], context_doc)

        metadata["status"] = (
            "completed" if exit_code == 0 and collector_result.returncode == 0
            else "failed_or_partial"
        )
        metadata["collection"] = {
            "exit_code": collector_result.returncode,
            "collection_start_epoch": collection_start_epoch,
            "collection_end_epoch": collection_end_epoch,
            "collection_start_utc": collection_start_utc,
            "collection_end_utc": collection_end_utc,
            "cooldown_seconds_requested": COOLDOWN_SECONDS,
            "finished_at_utc": collector_finished_utc,
            "target_identity_verified": identity_matches,
        }
        metadata["events"].append({
            "event": "collection_finished", "observed_at_utc": collector_finished_utc
        })
        update_json_atomic(artifacts["metadata"], metadata)

        print("\n=== Launcher test finished ===")
        print(f"Run directory: {run_dir}")
        print(f"Locust exit code: {exit_code}")
        print(f"Prometheus collector exit code: {collector_result.returncode}")
        print("This run is for launcher validation only and must not enter training.")
        print("\nCollector output:")
        print(collector_result.stdout.strip())
        if collector_result.stderr.strip():
            print("\nCollector errors/warnings:")
            print(collector_result.stderr.strip())
        return 0 if exit_code == 0 and collector_result.returncode == 0 else 1

    except (RuntimeError, OSError, ValueError, KeyError, TypeError, URLError,
            subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        persist_failure(
            metadata,
            artifacts["metadata"],
            run_dir / "launcher_test_failure.json",
            exc,
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        print("No schema integration or model training was started.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
