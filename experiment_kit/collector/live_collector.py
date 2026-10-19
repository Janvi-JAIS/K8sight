"""Read-only Prometheus range-query client for K8sight.

This module does not launch workloads or modify Kubernetes resources.
It archives raw HTTP responses and refuses to overwrite output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


METRICS = {
    "container_cpu_usage_seconds_total": {
        "unit": "cores",
        "query": lambda labels, lookback: (
            f"rate(container_cpu_usage_seconds_total{{{labels}}}[{lookback}s])"
        ),
        "lookback": True,
    },
    "container_memory_working_set_bytes": {
        "unit": "bytes",
        "query": lambda labels, lookback: (
            f"container_memory_working_set_bytes{{{labels}}}"
        ),
        "lookback": False,
    },
    "container_cpu_cfs_throttled_periods_total": {
        "unit": "periods",
        "query": lambda labels, lookback: (
            f"container_cpu_cfs_throttled_periods_total{{{labels}}}"
        ),
        "lookback": False,
    },
    "container_cpu_cfs_periods_total": {
        "unit": "periods",
        "query": lambda labels, lookback: (
            f"container_cpu_cfs_periods_total{{{labels}}}"
        ),
        "lookback": False,
    },
    "container_cpu_cfs_throttled_seconds_total": {
        "unit": "seconds",
        "query": lambda labels, lookback: (
            f"container_cpu_cfs_throttled_seconds_total{{{labels}}}"
        ),
        "lookback": False,
    },
}


def utc_string(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Prometheus URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Put credentials in a secure mechanism, not the URL")
    return value.rstrip("/")


def fetch_range(
    session: requests.Session,
    base_url: str,
    promql: str,
    start: float,
    end: float,
    step: int,
    timeout: float,
) -> dict:
    """Return the unmodified JSON response from Prometheus."""
    response = session.get(
        f"{base_url}/api/v1/query_range",
        params={
            "query": promql,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": step,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    document = response.json()
    if not isinstance(document, dict):
        raise ValueError("Prometheus returned a non-object JSON response")
    if document.get("status") != "success":
        raise ValueError(
            f"Prometheus query failed: {document.get('error', 'unknown error')}"
        )
    return document


def parse_single_series(document: dict, metric_name: str) -> dict[float, float]:
    """Parse one uniquely selected series; reject ambiguous selectors."""
    data = document.get("data")
    if not isinstance(data, dict) or data.get("resultType") != "matrix":
        raise ValueError(f"{metric_name}: expected a Prometheus matrix response")

    result = data.get("result")
    if not isinstance(result, list):
        raise ValueError(f"{metric_name}: missing matrix result list")
    if len(result) > 1:
        raise ValueError(
            f"{metric_name}: selector returned {len(result)} series; "
            "refusing to mix containers or pods"
        )
    if not result:
        return {}

    values = result[0].get("values")
    if not isinstance(values, list):
        raise ValueError(f"{metric_name}: missing sample values")

    samples = {}
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{metric_name}: malformed sample")
        timestamp, value = float(pair[0]), float(pair[1])
        if not math.isfinite(timestamp) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{metric_name}: invalid timestamp or metric value")
        if timestamp in samples:
            raise ValueError(f"{metric_name}: duplicate sample timestamp")
        samples[timestamp] = value
    return samples


def collect(
    *,
    prometheus_url: str,
    namespace: str,
    pod: str,
    container: str,
    start: float,
    end: float,
    step: int,
    lookback: int,
    output_dir: Path,
    scrape_interval: int | None = None,
    timeout: float = 10.0,
) -> dict:
    """Archive a bounded, read-only Prometheus collection in a new directory."""
    if not namespace or not pod or not container:
        raise ValueError("namespace, pod, and container must be explicit")
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} epoch must be a whole number of seconds")
    if start >= end:
        raise ValueError("start must be earlier than end")
    if step <= 0 or lookback <= 0 or timeout <= 0:
        raise ValueError("step, lookback, and timeout must be positive")
    if scrape_interval is not None:
        if scrape_interval <= 0 or step < scrape_interval:
            raise ValueError("step must be at least the scrape interval")
        if lookback < 4 * scrape_interval:
            raise ValueError("CPU lookback must be at least 4 times the scrape interval")

    base_url = validate_url(prometheus_url)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()

    labels = (
        f'namespace="{namespace}",pod="{pod}",container="{container}"'
    )
    session = requests.Session()
    session.headers["User-Agent"] = "K8sight-read-only-collector/1.0"
    records = []
    parsed_samples = {}

    try:
        for metric_name, spec in METRICS.items():
            promql = spec["query"](labels, lookback)
            record = {
                "name": metric_name,
                "promql": promql,
                "unit": spec["unit"],
                "labels": {
                    "namespace": namespace,
                    "pod": pod,
                    "container": container,
                },
                "query_start_utc": utc_string(start),
                "query_end_utc": utc_string(end),
                "step_seconds": step,
                "lookback_seconds": lookback if spec["lookback"] else None,
            }
            try:
                document = fetch_range(
                    session, base_url, promql, start, end, step, timeout
                )
                raw_path = raw_dir / f"{metric_name}.json"
                # Exclusive creation: never replace an archived response.
                with raw_path.open("x", encoding="utf-8") as stream:
                    json.dump(document, stream, sort_keys=True, indent=2)
                    stream.write("\n")

                record["availability"] = (
                    "available" if document["data"]["result"] else "empty"
                )
                record["raw_sources"] = [{
                    "path": f"raw/{raw_path.name}",
                    "sha256": sha256_file(raw_path),
                    "media_type": "application/json",
                }]
                parsed_samples[metric_name] = parse_single_series(
                    document, metric_name
                )
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                record["availability"] = "unknown"
                record["error"] = str(exc)
                # Preserve raw_sources if a response was archived before parsing failed.
                parsed_samples[metric_name] = {}
            records.append(record)
    finally:
        session.close()

    summary = {
        "collector": "k8sight-read-only-prometheus",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "prometheus_url": base_url,
        "target": {
            "namespace": namespace,
            "pod": pod,
            "container": container,
        },
        "query_range": {
            "start_utc": utc_string(start),
            "end_utc": utc_string(end),
            "step_seconds": step,
            "scrape_interval_seconds": scrape_interval,
        },
        "metrics": records,
        "sample_counts": {
            name: len(samples) for name, samples in parsed_samples.items()
        },
        "note": (
            "Raw query responses are archived. This is not yet a validated "
            "dataset-schema run manifest and must not be used for training."
        ),
    }
    summary_path = output_dir / "collection_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def required_metric_error(summary: dict) -> str | None:
    """Return an error when a required CPU or memory series was not collected."""
    available = {item.get("name") for item in summary.get("metrics", [])
                 if item.get("availability") == "available"}
    for name in ("container_cpu_usage_seconds_total", "container_memory_working_set_bytes"):
        if name not in available:
            return f"Required metric unavailable: {name}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus-url", default=os.environ.get(
        "K8SIGHT_PROMETHEUS_URL", "http://127.0.0.1:9090"
    ))
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--scrape-interval", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = collect(
            prometheus_url=args.prometheus_url,
            namespace=args.namespace,
            pod=args.pod,
            container=args.container,
            start=args.start_epoch,
            end=args.end_epoch,
            step=args.step,
            scrape_interval=args.scrape_interval,
            lookback=args.lookback,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
        unavailable = required_metric_error(summary)
        if unavailable:
            print(f"Collection failed: {unavailable}", file=sys.stderr)
            return 2
    except (OSError, ValueError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "metrics": [
            {
                "name": item["name"],
                "availability": item["availability"],
            }
            for item in summary["metrics"]
        ],
        "sample_counts": summary["sample_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
