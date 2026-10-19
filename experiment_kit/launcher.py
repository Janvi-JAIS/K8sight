"""Run one parameterized K8sight experiment and archive its evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from experiment_kit.generators.config import LOAD_LEVELS, PATTERNS, WorkloadConfig


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "run_records"
PYTHON = ROOT / ".venv" / "bin" / "python"
GENERATOR = ROOT / "experiment_kit" / "generators" / "locustfile.py"
TARGET_URL = "http://127.0.0.1:18080"
PROMETHEUS_URL = "http://127.0.0.1:19091"
NAMESPACE = "default"
CONTAINER = "hpa-example"
BASELINE_SECONDS = 120
COOLDOWN_SECONDS = 180
STEP_SECONDS = 15
SCRAPE_SECONDS = 15
LOOKBACK_SECONDS = 60
CPU_OBSERVATION_MAX_AGE_SECONDS = 30
CPU_FRESHNESS_STEP_SECONDS = 5
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def epoch_now() -> int:
    return int(time.time())


def utc_at(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, document: dict) -> None:
    """Write a new JSON file exclusively, or atomically update its own file."""
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temp_path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value) or value.startswith("launcher-test"):
        raise ValueError("run_id must be a safe lowercase ID and cannot be a launcher-test ID")
    return value


def validate_config(args: argparse.Namespace) -> WorkloadConfig:
    validate_run_id(args.run_id)
    if args.target_url != TARGET_URL:
        raise ValueError(f"target URL must be the port-forward endpoint {TARGET_URL}")
    config = WorkloadConfig(
        run_id=args.run_id,
        pattern=args.pattern,
        load_level=args.load_level,
        replicate=args.replicate,
        random_seed=args.seed,
        duration_seconds=args.duration,
        target_host=TARGET_URL,
    )
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument("--load-level", choices=LOAD_LEVELS, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--target-url", default=TARGET_URL)
    return parser


def cpu_limit_millicores(value: str) -> float:
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1])
    if value.endswith("u"):
        return float(value[:-1]) / 1000
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000
    return float(value) * 1000


def enforce_cpu_headroom(cpu_cores: float, cpu_limit: str) -> float:
    try:
        limit_m = cpu_limit_millicores(cpu_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("CPU safety gate failed: observed CPU limit is invalid") from exc
    if (not math.isfinite(limit_m) or limit_m < 300 or not math.isfinite(cpu_cores)
            or cpu_cores < 0 or cpu_cores * 1000 > limit_m - 300):
        raise RuntimeError(
            f"CPU safety gate failed: observed {cpu_cores * 1000:.1f}m against "
            f"{cpu_limit}; at least 300m headroom is required"
        )
    return limit_m


def command_environment(config: WorkloadConfig, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.pop("K8SIGHT_CUSTOM_PARAMS", None)
    env.update({
        "K8SIGHT_RUN_ID": config.run_id,
        "K8SIGHT_PATTERN": config.pattern,
        "K8SIGHT_LOAD_LEVEL": config.load_level,
        "K8SIGHT_REPLICATE": str(config.replicate),
        "K8SIGHT_SEED": str(config.random_seed),
        "K8SIGHT_DURATION": str(config.duration_seconds),
        "K8SIGHT_TARGET_HOST": TARGET_URL,
        "K8SIGHT_DRY_RUN": "0",
    })
    return env


def locust_command(run_dir: Path, config: WorkloadConfig) -> list[str]:
    return [
        str(PYTHON), "-m", "locust", "-f", str(GENERATOR), "--headless",
        "--host", TARGET_URL, "--run-time", f"{config.duration_seconds}s",
        "--csv", str(run_dir / "locust"), "--csv-full-history",
        "--logfile", str(run_dir / "locust.log"), "--only-summary",
    ]


def run_command(command: list[str], timeout: int = 20) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def kubectl_target() -> dict:
    raw = run_command([
        "kubectl", "get", "pods", "-n", NAMESPACE, "-l", "app=php-apache", "-o", "json",
    ])
    pods = json.loads(raw).get("items", [])
    candidates = []
    for pod in pods:
        status = pod.get("status", {})
        target_status = next((item for item in status.get("containerStatuses", [])
                              if item.get("name") == CONTAINER and item.get("ready")), None)
        if status.get("phase") == "Running" and target_status:
            candidates.append((pod, target_status))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Ready target Pod; found {len(candidates)}")
    pod, status = candidates[0]
    spec = next((item for item in pod.get("spec", {}).get("containers", [])
                 if item.get("name") == CONTAINER), None)
    if spec is None:
        raise RuntimeError("Target container is absent from the Pod specification")
    return {
        "name": pod["metadata"]["name"],
        "uid": pod["metadata"]["uid"],
        "container_id": status["containerID"],
        "ready": status.get("ready") is True,
        "restart_count": status.get("restartCount"),
        "resources": spec.get("resources", {}),
        "image": spec.get("image"),
        "node": pod.get("spec", {}).get("nodeName"),
    }


def observe_cpu(pod_name: str, container_id: str, pod_uid: str) -> dict:
    requested_epoch = epoch_now()
    selector = (
        f'container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",'
        f'pod="{pod_name}",container="{CONTAINER}"}}'
    )
    query = urlencode({"query": f"rate({selector}[1m])", "time": requested_epoch})
    with urlopen(f"{PROMETHEUS_URL}/api/v1/query?{query}", timeout=5) as response:
        cpu_document = json.loads(response.read().decode("utf-8"))

    freshness_query = urlencode({
        "query": f"timestamp({selector})",
        "start": requested_epoch - CPU_OBSERVATION_MAX_AGE_SECONDS,
        "end": requested_epoch,
        "step": CPU_FRESHNESS_STEP_SECONDS,
    })
    with urlopen(
        f"{PROMETHEUS_URL}/api/v1/query_range?{freshness_query}", timeout=5
    ) as response:
        freshness_document = json.loads(response.read().decode("utf-8"))
    return parse_cpu_observation(
        cpu_document, freshness_document, pod_name, container_id, pod_uid,
        requested_epoch,
    )


def validate_cpu_series_identity(labels: dict, pod_name: str, container_id: str,
                                 pod_uid: str) -> None:
    if not pod_name or not container_id or not pod_uid:
        raise RuntimeError("CPU safety metric cannot be matched without Pod/container identity")
    if any(labels.get(key) != value for key, value in {
        "namespace": NAMESPACE, "pod": pod_name, "container": CONTAINER,
    }.items()):
        raise RuntimeError("CPU safety metric labels do not match the target container")
    runtime_id = container_id.rsplit("://", 1)[-1]
    cgroup_id = labels.get("id", "")
    container_matches = bool(runtime_id) and (
        labels.get("name") == runtime_id or runtime_id in cgroup_id
    )
    normalized_uid = pod_uid.replace("-", "").lower()
    normalized_cgroup = cgroup_id.replace("-", "").replace("_", "").lower()
    pod_uid_matches = not cgroup_id or normalized_uid in normalized_cgroup
    if not container_matches or not pod_uid_matches:
        raise RuntimeError("CPU safety metric does not identify the observed Pod/container")


def parse_cpu_observation(cpu_document: dict, freshness_document: dict,
                          pod_name: str, container_id: str, pod_uid: str,
                          requested_epoch: int) -> dict:
    cpu_results = cpu_document.get("data", {}).get("result", [])
    if cpu_document.get("status") != "success" or len(cpu_results) != 1:
        raise RuntimeError("CPU safety metric could not be verified uniquely")
    cpu_series = cpu_results[0]
    validate_cpu_series_identity(cpu_series.get("metric", {}), pod_name, container_id, pod_uid)
    cpu_value = cpu_series.get("value", [])
    if len(cpu_value) != 2:
        raise RuntimeError("CPU safety metric returned no usable observation")
    cpu_cores = float(cpu_value[1])
    if not math.isfinite(cpu_cores) or cpu_cores < 0:
        raise RuntimeError("CPU safety metric returned an invalid CPU value")

    data = freshness_document.get("data", {})
    fresh_results = data.get("result", [])
    if (freshness_document.get("status") != "success"
            or data.get("resultType") != "matrix" or len(fresh_results) != 1):
        raise RuntimeError("CPU source-sample timestamps could not be verified uniquely")
    fresh_series = fresh_results[0]
    fresh_labels = fresh_series.get("metric", {})
    validate_cpu_series_identity(fresh_labels, pod_name, container_id, pod_uid)
    if fresh_labels != cpu_series.get("metric", {}):
        raise RuntimeError("CPU and freshness queries returned different series identities")
    values = fresh_series.get("values", [])
    if not values:
        raise RuntimeError("CPU source-sample timestamp query returned no samples")
    source_epochs = []
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError("CPU source-sample timestamp query returned malformed data")
        evaluation_epoch = float(pair[0])
        source_epoch = float(pair[1])
        if (not math.isfinite(evaluation_epoch) or not math.isfinite(source_epoch)
                or evaluation_epoch > requested_epoch or source_epoch > evaluation_epoch):
            raise RuntimeError("CPU source-sample timestamp is invalid")
        source_epochs.append(source_epoch)
    newest_source_epoch = max(source_epochs)
    source_age = requested_epoch - newest_source_epoch
    if source_age < 0 or source_age > CPU_OBSERVATION_MAX_AGE_SECONDS:
        raise RuntimeError("CPU source samples are stale or have an invalid timestamp")
    return {
        "cpu_cores": cpu_cores,
        "source_sample_epoch": newest_source_epoch,
        "source_sample_utc": utc_at(int(newest_source_epoch)),
        "source_sample_age_seconds": source_age,
        "freshness_checked_at_epoch": requested_epoch,
        "freshness_checked_at_utc": utc_at(requested_epoch),
        "freshness_range_start_epoch": requested_epoch - CPU_OBSERVATION_MAX_AGE_SECONDS,
        "freshness_range_end_epoch": requested_epoch,
        "freshness_step_seconds": CPU_FRESHNESS_STEP_SECONDS,
        "series_labels": fresh_labels,
    }


def run_locust(command: list[str], env: dict[str, str], run_dir: Path,
               timeout: int) -> int:
    stdout_path = run_dir / "locust_stdout.log"
    stderr_path = run_dir / "locust_stderr.log"
    with stdout_path.open("x", encoding="utf-8") as stdout, \
         stderr_path.open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError(f"Locust exceeded timeout of {timeout}s")


def create_run_directory(path: Path) -> None:
    path.mkdir(parents=False, exist_ok=False)


def verify_same_target(expected: dict, actual: dict) -> None:
    if actual.get("ready") is not True:
        raise RuntimeError("Target container is no longer Ready")
    if (actual.get("uid"), actual.get("container_id")) != (
        expected.get("uid"), expected.get("container_id")
    ):
        raise RuntimeError("Target Pod UID or container ID changed")
    expected_limit = expected.get("resources", {}).get("limits", {}).get("cpu")
    actual_limit = actual.get("resources", {}).get("limits", {}).get("cpu")
    if not expected_limit or actual_limit != expected_limit:
        raise RuntimeError("Target Pod CPU limit changed or is missing")


def record_boundary(boundaries: dict, name: str, epoch: int) -> None:
    boundaries[f"{name}_epoch"] = epoch
    boundaries[f"{name}_utc"] = utc_at(epoch)


def run_preload_cpu_gate(expected_target: dict, metadata: dict) -> dict:
    """Recheck target identity and CPU headroom immediately before Locust."""
    started_epoch = epoch_now()
    gate = {
        "stage": "immediately_before_locust",
        "started_at_epoch": started_epoch,
        "started_at_utc": utc_at(started_epoch),
        "passed": False,
    }
    try:
        current = kubectl_target()
        verify_same_target(expected_target, current)
        observation = observe_cpu(
            current["name"], current["container_id"], current["uid"]
        )
        cpu = observation["cpu_cores"]
        limit = current.get("resources", {}).get("limits", {}).get("cpu")
        if limit is None:
            raise RuntimeError("Target Pod CPU limit is missing; refusing to start load")
        gate.update({
            "pod": current["name"],
            "pod_uid": current["uid"],
            "container_id": current["container_id"],
            "cpu_observed_cores": cpu,
            "cpu_limit": limit,
            "required_headroom_millicores": 300,
            "sample_observed_at_utc": observation["source_sample_utc"],
            "freshness_evidence": observation,
        })
        limit_m = cpu_limit_millicores(limit)
        headroom_m = limit_m - cpu * 1000
        if math.isfinite(limit_m) and math.isfinite(headroom_m):
            gate["cpu_limit_millicores"] = limit_m
            gate["cpu_headroom_millicores"] = headroom_m
        enforce_cpu_headroom(cpu, limit)
        gate["passed"] = True
        return gate
    except Exception as exc:
        gate["failure"] = str(exc)
        raise
    finally:
        gate["finished_at_utc"] = utc_at(epoch_now())
        metadata.setdefault("cpu_safety_checks", []).append(gate)


def persist_failure(metadata: dict, metadata_path: Path, original_error: Exception) -> None:
    metadata["status"] = "failed"
    metadata["failure"] = {"error": str(original_error), "observed_at_utc": utc_at(epoch_now())}
    try:
        atomic_json(metadata_path, metadata)
    except OSError as metadata_error:
        metadata["metadata_update_error"] = str(metadata_error)
        failure_path = metadata_path.with_name("failure.json")
        try:
            with failure_path.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except OSError as failure_error:
            print(f"Could not preserve failure record: {failure_error}", file=sys.stderr)
        print(f"Could not update run metadata: {metadata_error}", file=sys.stderr)


def run_experiment(config: WorkloadConfig, *, runs_dir: Path = RUNS_DIR,
                   baseline_seconds: int = BASELINE_SECONDS,
                   cooldown_seconds: int = COOLDOWN_SECONDS) -> int:
    run_dir = runs_dir / config.run_id
    create_run_directory(run_dir)
    metadata_path = run_dir / "experiment_metadata.json"
    metadata = {
        "run_id": config.run_id,
        "run_type": "experiment",
        "training_eligible": False,
        "training_acceptance_status": "pending_schema_validation_and_dataset_acceptance",
        "status": "preflight",
        "pattern": config.pattern,
        "load_level": config.load_level,
        "replicate": config.replicate,
        "seed": config.random_seed,
        "duration_seconds": config.duration_seconds,
        "target_url": TARGET_URL,
        "generator": str(GENERATOR.relative_to(ROOT)),
        "generator_sha256": None,
        "events": [],
    }
    atomic_json(metadata_path, metadata)
    try:
        if not PYTHON.is_file() or not GENERATOR.is_file():
            raise RuntimeError("Expected project Python or parameterized generator is missing")
        metadata["generator_sha256"] = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
        atomic_json(metadata_path, metadata)
        before = kubectl_target()
        observation = observe_cpu(before["name"], before["container_id"], before["uid"])
        cpu = observation["cpu_cores"]
        verify_same_target(before, kubectl_target())
        limits = before["resources"].get("limits", {})
        if limits.get("cpu") is None:
            raise RuntimeError("Target Pod CPU limit is missing; refusing to start load")
        limit_m = enforce_cpu_headroom(cpu, limits["cpu"])
        metadata["cpu_safety_checks"] = [{
            "stage": "preflight_before_baseline",
            "passed": True,
            "sample_observed_at_utc": observation["source_sample_utc"],
            "freshness_evidence": observation,
            "cpu_observed_cores": cpu,
            "cpu_limit": limits["cpu"],
            "cpu_limit_millicores": limit_m,
            "cpu_headroom_millicores": limit_m - cpu * 1000,
            "required_headroom_millicores": 300,
        }]
        metadata.update({
            "pod": before,
            "cpu_observation_cores": cpu,
            "cpu_limit_millicores": limit_m,
            "required_cpu_headroom_millicores": 300,
        })

        collection_start_epoch = epoch_now()
        metadata["boundaries"] = {}
        record_boundary(metadata["boundaries"], "baseline_start", collection_start_epoch)
        record_boundary(metadata["boundaries"], "collection_start", collection_start_epoch)
        metadata["events"].append({
            "event": "baseline_started", "observed_at_utc": utc_at(collection_start_epoch),
        })
        metadata["status"] = "baseline"
        atomic_json(metadata_path, metadata)
        time.sleep(baseline_seconds)

        baseline_end_epoch = epoch_now()
        record_boundary(metadata["boundaries"], "baseline_end", baseline_end_epoch)
        preload_gate = run_preload_cpu_gate(before, metadata)
        run_start_epoch = epoch_now()
        record_boundary(metadata["boundaries"], "run_start", run_start_epoch)
        metadata["preload_cpu_gate"] = preload_gate
        metadata["status"] = "running"
        metadata["events"].append({"event": "locust_started", "observed_at_utc": utc_at(run_start_epoch)})
        atomic_json(metadata_path, metadata)
        exit_code = run_locust(
            locust_command(run_dir, config), command_environment(config), run_dir,
            timeout=config.duration_seconds + 120,
        )
        workload_end_epoch = epoch_now()
        metadata["locust_exit_code"] = exit_code
        record_boundary(metadata["boundaries"], "workload_end", workload_end_epoch)
        record_boundary(metadata["boundaries"], "locust_shutdown", workload_end_epoch)
        record_boundary(metadata["boundaries"], "cooldown_start", workload_end_epoch)
        metadata["status"] = "cooldown"
        atomic_json(metadata_path, metadata)
        time.sleep(cooldown_seconds)
        cooldown_end_epoch = epoch_now()
        record_boundary(metadata["boundaries"], "cooldown_end", cooldown_end_epoch)
        atomic_json(metadata_path, metadata)

        after = kubectl_target()
        verify_same_target(before, after)

        collection_end_epoch = epoch_now()
        record_boundary(metadata["boundaries"], "collection_end", collection_end_epoch)
        atomic_json(metadata_path, metadata)
        collector_command = [
            str(PYTHON), "-m", "experiment_kit.collector.live_collector",
            "--prometheus-url", PROMETHEUS_URL, "--namespace", NAMESPACE,
            "--pod", before["name"], "--container", CONTAINER,
            "--start-epoch", str(collection_start_epoch),
            "--end-epoch", str(collection_end_epoch),
            "--step", str(STEP_SECONDS), "--scrape-interval", str(SCRAPE_SECONDS),
            "--lookback", str(LOOKBACK_SECONDS), "--output-dir", str(run_dir / "prometheus_raw"),
        ]
        result = subprocess.run(
            collector_command, cwd=ROOT, text=True, capture_output=True,
            timeout=180, check=False,
        )
        with (run_dir / "collector_stdout.log").open("x", encoding="utf-8") as stream:
            stream.write(result.stdout)
        with (run_dir / "collector_stderr.log").open("x", encoding="utf-8") as stream:
            stream.write(result.stderr)
        finished_epoch = epoch_now()
        record_boundary(metadata["boundaries"], "collector_finished", finished_epoch)
        metadata["collector_exit_code"] = result.returncode
        metadata["status"] = "completed" if exit_code == 0 and result.returncode == 0 else "partial"
        metadata["events"].append({"event": "collector_finished", "observed_at_utc": utc_at(finished_epoch)})
        atomic_json(metadata_path, metadata)
        return 0 if metadata["status"] == "completed" else 1
    except Exception as exc:
        persist_failure(metadata, metadata_path, exc)
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = validate_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return run_experiment(config)


if __name__ == "__main__":
    raise SystemExit(main())
