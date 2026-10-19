#!/usr/bin/env python3
"""
K8sight - fix_random_spike_cpu_local.py

Rebuilds Random Spike CPU using the raw cumulative CPU counter already saved
locally in random_spike_cpu_raw.json. No kubectl, Prometheus port-forward,
or Locust rerun is required.

CPU calculation:
    cpu_cores(t) = (C(t) - C(t-60s)) / 60

The 15-second timestamp grid comes from random_spike_memory.json so the CPU
and memory timestamps stay aligned. Raw counter values are linearly
interpolated at the exact grid timestamps when needed.

Protected files (never written):
    stable_target.csv
    increasing_target.csv
    periodic_target.csv
    spike_target.csv
    random_spike_memory.json

Backups are made once for the current Random Spike CPU/CSV outputs before
replacement.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path.home() / "k8sight-locust"
START = "2026-09-27T20:35:59Z"
END = "2026-09-27T20:46:00Z"
STEP_SECONDS = 15
WINDOW_SECONDS = 60
TARGET_POD = "php-apache-5b5d776b6-llmk5"
NAMESPACE = "default"
CONTAINER = "hpa-example"

MEMORY_JSON = "random_spike_memory.json"
RAW_CPU_JSON = "random_spike_cpu_raw.json"
OUT_CPU_JSON = "random_spike_cpu.json"
OUT_DATASET = "random_spike_dataset.csv"
OUT_TARGET = "random_spike_target.csv"
OUT_COMBINED = "combined_all_patterns.csv"

PROTECTED = [
    "stable_target.csv",
    "increasing_target.csv",
    "periodic_target.csv",
    "spike_target.csv",
]
SCHEMA = [
    "timestamp", "pod", "cpu_cores", "cpu_millicores",
    "memory_bytes", "memory_mib", "pattern"
]


def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}")
    raise SystemExit(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def fmt_like_existing(t: float) -> str:
    s = str(pd.read_csv(PROTECTED[0], nrows=1)["timestamp"].iloc[0])
    try:
        x = float(s)
        if x.is_integer():
            return str(int(round(t)))
        return str(t)
    except ValueError:
        pass
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    if "T" in s:
        if s.endswith("Z"):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return dt.isoformat()
    if s.endswith("+00:00"):
        return dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_prom_matrix(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "success":
        fail(f"{path.name} does not contain a successful Prometheus response")
    return payload


def get_series(payload: dict, pod: str) -> list[tuple[float, float]]:
    matches = []
    for series in payload["data"]["result"]:
        if series.get("metric", {}).get("pod") == pod:
            matches.append(series)
    if len(matches) != 1:
        fail(f"Expected exactly one series for {pod}, found {len(matches)}")
    return sorted((float(t), float(v)) for t, v in matches[0]["values"])


def backup_once(name: str) -> None:
    p = Path(name)
    if not p.exists():
        return
    bak = Path(name + ".rate5m.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"    backup: {p.name} -> {bak.name}")
    else:
        print(f"    backup exists: {bak.name} (kept)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild-combined",
        action="store_true",
        help="rebuild combined_all_patterns.csv after correcting Random Spike",
    )
    args = ap.parse_args()

    print("\n=== 1/5 Checking local inputs ===")
    if not PROJECT_DIR.is_dir():
        fail(f"Project directory not found: {PROJECT_DIR}")
    # Work entirely inside the project directory.
    import os
    os.chdir(PROJECT_DIR)

    required = PROTECTED + [MEMORY_JSON, RAW_CPU_JSON]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        fail(f"Missing files: {missing}")

    before = {f: sha256(Path(f)) for f in PROTECTED + [MEMORY_JSON]}

    existing = {f: pd.read_csv(f) for f in PROTECTED}
    for f, df in existing.items():
        if list(df.columns) != SCHEMA:
            fail(f"{f} columns do not match expected schema")

    labels = sorted(set().union(*[set(df["pattern"].astype(str)) for df in existing.values()]))
    label = "random_spike" if any("_" in x for x in labels) else "random-spike"
    print(f"    project: {PROJECT_DIR}")
    print(f"    protected pattern labels: {labels}")
    print(f"    Random Spike label: {label}")

    memory_payload = load_prom_matrix(Path(MEMORY_JSON))
    mem_series = get_series(memory_payload, TARGET_POD)
    grid = np.array([t for t, _ in mem_series], dtype=float)
    mem_bytes = np.array([v for _, v in mem_series], dtype=float)

    t0, t1 = epoch(START), epoch(END)
    expected = np.arange(t0, t1 + 1e-9, STEP_SECONDS)
    if len(grid) != len(expected) or not np.allclose(grid, expected, atol=0.5):
        fail(
            f"Memory timestamps are not the expected {STEP_SECONDS}s grid "
            f"({len(grid)} vs {len(expected)})"
        )
    print(f"    memory grid: {len(grid)} points, {fmt_like_existing(grid[0])} -> {fmt_like_existing(grid[-1])}")

    print("\n=== 2/5 Reading raw CPU counter ===")
    raw_payload = load_prom_matrix(Path(RAW_CPU_JSON))
    samples = get_series(raw_payload, TARGET_POD)
    st = np.array([t for t, _ in samples], dtype=float)
    sv = np.array([v for _, v in samples], dtype=float)
    need_lo, need_hi = grid[0] - WINDOW_SECONDS, grid[-1]
    if st[0] > need_lo:
        fail(f"Raw counter starts too late: {st[0]} > required {need_lo}")
    if st[-1] < need_hi:
        fail(f"Raw counter ends too early: {st[-1]} < required {need_hi}")
    if np.any(np.diff(sv) < -1e-9):
        fail("Raw CPU counter decreases; possible counter reset/restart. Stop and inspect before using data.")
    gaps = np.diff(st)
    print(f"    raw samples: {len(st)}")
    print(f"    raw range: {datetime.fromtimestamp(st[0], timezone.utc)} -> {datetime.fromtimestamp(st[-1], timezone.utc)}")
    print(f"    raw scrape spacing: median {np.median(gaps):.0f}s, max {gaps.max():.0f}s")
    if gaps.max() > 90:
        fail(f"Raw counter contains a gap of {gaps.max():.0f}s; cannot safely compute a 60s difference.")
    if gaps.max() > 45:
        print(f"    WARNING: interpolating across a raw gap of {gaps.max():.0f}s")

    print("\n=== 3/5 Computing 1-minute CPU ===")
    c_hi = np.interp(grid, st, sv)
    c_lo = np.interp(grid - WINDOW_SECONDS, st, sv)
    cpu_cores = (c_hi - c_lo) / WINDOW_SECONDS
    if not np.isfinite(cpu_cores).all():
        fail("Computed CPU contains non-finite values")
    if np.any(cpu_cores < 0):
        fail("Computed CPU contains negative values")
    print(
        f"    CPU cores: min={cpu_cores.min():.6f}, "
        f"mean={cpu_cores.mean():.6f}, max={cpu_cores.max():.6f}"
    )

    print("\n=== 4/5 Building corrected Random Spike files ===")
    out = pd.DataFrame({
        "timestamp": [fmt_like_existing(t) for t in grid],
        "pod": TARGET_POD,
        "cpu_cores": cpu_cores,
        "cpu_millicores": cpu_cores * 1000.0,
        "memory_bytes": mem_bytes,
        "memory_mib": mem_bytes / (1024 * 1024),
        "pattern": label,
    })[SCHEMA]

    if len(out) != len(expected):
        fail(f"Output has {len(out)} rows; expected {len(expected)}")
    if not np.allclose(pd.to_datetime(out["timestamp"]).astype("int64") / 1e9, grid, atol=1):
        fail("Output timestamps do not match the memory grid")
    if np.any(np.diff(grid) != STEP_SECONDS):
        fail("Output timestamps are not exactly 15 seconds apart")
    if out.isna().any().any():
        fail("Output contains NaN values")
    if out.duplicated(["timestamp", "pod"]).any():
        fail("Output contains duplicate timestamp/pod rows")

    # Back up existing rate[5m]-based outputs before replacing them.
    for f in (OUT_CPU_JSON, OUT_DATASET, OUT_TARGET):
        backup_once(f)

    cpu_payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{
                "metric": {"pod": TARGET_POD, "namespace": NAMESPACE, "container": CONTAINER},
                "values": [
                    [int(t) if float(t).is_integer() else float(t), str(float(v))]
                    for t, v in zip(grid, cpu_cores)
                ],
            }],
        },
        "k8sight_derivation": {
            "method": "(counter(t) - counter(t-60s)) / 60, linear interpolation of raw samples",
            "source_metric": "container_cpu_usage_seconds_total",
            "raw_counter_file": RAW_CPU_JSON,
            "reason": "match the existing 1-minute CPU methodology used for the other workloads",
        },
    }
    Path(OUT_CPU_JSON).write_text(json.dumps(cpu_payload))
    out.to_csv(OUT_DATASET, index=False)
    out.to_csv(OUT_TARGET, index=False)
    print(f"    wrote {OUT_CPU_JSON}")
    print(f"    wrote {OUT_DATASET} ({len(out)} rows)")
    print(f"    wrote {OUT_TARGET} ({len(out)} rows)")

    if args.rebuild_combined:
        print("\n=== 5/5 Rebuilding combined_all_patterns.csv ===")
        combined = pd.concat(list(existing.values()) + [out], ignore_index=True)
        if combined.isna().any().any():
            fail("Combined dataset contains missing values")
        if combined.duplicated(["timestamp", "pod", "pattern"]).any():
            fail("Combined dataset contains duplicate rows")
        backup_once(OUT_COMBINED)
        combined.to_csv(OUT_COMBINED, index=False)
        print(f"    wrote {OUT_COMBINED} ({len(combined)} rows)")
    else:
        print("\n=== 5/5 Combined dataset not rebuilt yet ===")
        print("    Run with --rebuild-combined after verifying the corrected Random Spike file.")

    after = {f: sha256(Path(f)) for f in PROTECTED + [MEMORY_JSON]}
    if before != after:
        fail("A protected input file changed unexpectedly")

    print("\n=== Validation summary ===")
    print(f"    Random Spike rows                     : {len(out)} PASS")
    print(f"    15-second timestamp grid              : PASS")
    print(f"    no missing values                     : PASS")
    print(f"    no duplicate rows                     : PASS")
    print(f"    protected files unchanged             : PASS")
    print(f"    Random Spike CPU mean (mCPU)          : {cpu_cores.mean()*1000:.2f}")
    print(f"    Random Spike CPU max (mCPU)           : {cpu_cores.max()*1000:.2f}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
