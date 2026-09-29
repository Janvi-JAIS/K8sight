#!/usr/bin/env python3
"""
K8sight - finish_random_spike.py

One command to:
  1. verify the project + protected files
  2. auto port-forward Prometheus on a free local port
  3. pull Random Spike CPU + memory (query_range, 15s step)
  4. build random_spike_dataset.csv / random_spike_target.csv (same schema as the other patterns)
  5. validate everything
  6. build combined_all_patterns.csv (5 patterns)

The four existing *_target.csv files are only READ, never written.
A SHA-256 check before/after proves they were not modified.

Run from the project directory:   python finish_random_spike.py
"""

import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is missing. Run: pip install pandas   (inside your venv), then re-run.")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROJECT_DIR = Path.home() / "k8sight-locust"

START = "2026-09-27T20:35:59Z"
END = "2026-09-27T20:46:00Z"
STEP = "15s"

PROM_NAMESPACE = "default"
PROM_SERVICE = "svc/prometheus-kube-prometheus-prometheus"
PROM_REMOTE_PORT = 9090

CPU_QUERY = ('rate(container_cpu_usage_seconds_total{namespace="default",'
             'pod=~"php-apache.*",container="hpa-example"}[1m])')
MEM_QUERY = ('container_memory_working_set_bytes{namespace="default",'
             'pod=~"php-apache.*",container="hpa-example"}')

# php-apache pods only; excludes php-apache-2-* (the regex above matches both)
TARGET_POD_RE = re.compile(r"^php-apache-(?!2-)[a-z0-9]+-[a-z0-9]+$")

SCHEMA = ["timestamp", "pod", "cpu_cores", "cpu_millicores",
          "memory_bytes", "memory_mib", "pattern"]

PROTECTED = ["stable_target.csv", "increasing_target.csv",
             "periodic_target.csv", "spike_target.csv"]
REQUIRED_OTHER = ["combine_pattern.py", "locustfile_random_spike.py",
                  "locust-random-spike-job.yaml"]

OUT_CPU_JSON = "random_spike_cpu.json"
OUT_MEM_JSON = "random_spike_memory.json"
OUT_DATASET = "random_spike_dataset.csv"
OUT_TARGET = "random_spike_target.csv"
OUT_COMBINED = "combined_all_patterns.csv"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail(msg):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def step(msg):
    print(f"\n=== {msg}")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_port_forward(port):
    kubectl = shutil.which("kubectl")
    if not kubectl:
        fail("kubectl not found on PATH.")
    cmd = [kubectl, "port-forward", "-n", PROM_NAMESPACE, PROM_SERVICE,
           f"{port}:{PROM_REMOTE_PORT}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 40
    url = f"http://127.0.0.1:{port}/-/ready"
    while time.time() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            fail(f"port-forward exited early:\n{err}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(1)
    stop_port_forward(proc)
    fail("Prometheus did not become ready within 40s.")


def stop_port_forward(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def query_range(port, query):
    params = urllib.parse.urlencode({"query": query, "start": START, "end": END, "step": STEP})
    url = f"http://127.0.0.1:{port}/api/v1/query_range?{params}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def matrix_to_df(payload, name):
    if payload.get("status") != "success":
        fail(f"Prometheus status != success for {name}: {payload}")
    rows = []
    for series in payload["data"]["result"]:
        pod = series["metric"].get("pod", "unknown")
        for ts, val in series["values"]:
            rows.append({"ts": float(ts), "pod": pod, "value": float(val)})
    if not rows:
        fail(f"No data returned for {name}. Check the time window / port-forward.")
    return pd.DataFrame(rows)


def detect_ts_formatter(sample_csv):
    """Match the timestamp style already used in the existing pattern files."""
    s = str(pd.read_csv(sample_csv, nrows=1)["timestamp"].iloc[0])
    try:
        as_float = float(s)
        is_int = as_float.is_integer()
        print(f"    timestamp style detected: epoch ({'int' if is_int else 'float'})  e.g. {s}")
        return lambda t: int(round(t)) if is_int else t
    except ValueError:
        pass
    if "T" in s:
        z = s.endswith("Z")
        print(f"    timestamp style detected: ISO8601  e.g. {s}")

        def fmt(t):
            dt = datetime.fromtimestamp(t, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if z else dt.isoformat()
        return fmt
    plus = s.endswith("+00:00")
    print(f"    timestamp style detected: 'YYYY-MM-DD HH:MM:SS'  e.g. {s}")
    return lambda t: (datetime.fromtimestamp(t, tz=timezone.utc)
                      .strftime("%Y-%m-%d %H:%M:%S") + ("+00:00" if plus else ""))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    errors, warnings = [], []

    # 1. project + files ------------------------------------------------
    step("1/6 Checking project directory and required files")
    if not PROJECT_DIR.is_dir():
        fail(f"Project directory not found: {PROJECT_DIR}")
    import os
    os.chdir(PROJECT_DIR)
    missing = [f for f in PROTECTED + REQUIRED_OTHER if not Path(f).exists()]
    if missing:
        fail(f"Missing required files: {missing}")
    print(f"    OK - {PROJECT_DIR}")

    hashes_before = {f: sha256(f) for f in PROTECTED}

    existing = {f: pd.read_csv(f) for f in PROTECTED}
    for f, df in existing.items():
        if list(df.columns) != SCHEMA:
            fail(f"{f} columns {list(df.columns)} != expected schema {SCHEMA}")
    labels = sorted(set().union(*[set(df["pattern"].unique()) for df in existing.values()]))
    print(f"    existing pattern labels: {labels}")
    hyphen = any("-" in l for l in labels) and not any("_" in l for l in labels)
    label = "random-spike" if hyphen else "random_spike"
    print(f"    Random Spike label will be: '{label}'")
    fmt_ts = detect_ts_formatter("stable_target.csv")

    # 2-4. Prometheus ---------------------------------------------------
    step("2/6 Starting Prometheus port-forward (auto free port)")
    port = find_free_port()
    print(f"    using localhost:{port}")
    proc = start_port_forward(port)
    print("    Prometheus ready")

    try:
        step("3/6 Extracting CPU + memory from Prometheus")
        cpu_payload = query_range(port, CPU_QUERY)
        mem_payload = query_range(port, MEM_QUERY)
    finally:
        stop_port_forward(proc)
        print("    port-forward terminated")

    Path(OUT_CPU_JSON).write_text(json.dumps(cpu_payload))
    Path(OUT_MEM_JSON).write_text(json.dumps(mem_payload))
    print(f"    saved {OUT_CPU_JSON}, {OUT_MEM_JSON}")

    # 5. build dataset --------------------------------------------------
    step("4/6 Building and validating Random Spike dataset")
    cpu = matrix_to_df(cpu_payload, "CPU").rename(columns={"value": "cpu_cores"})
    mem = matrix_to_df(mem_payload, "memory").rename(columns={"value": "memory_bytes"})

    all_pods = sorted(set(cpu["pod"]) | set(mem["pod"]))
    keep = [p for p in all_pods if TARGET_POD_RE.match(p)]
    dropped = [p for p in all_pods if p not in keep]
    print(f"    pods returned: {all_pods}")
    if dropped:
        print(f"    excluded non-target pods: {dropped}")
    if not keep:
        fail("No php-apache target pod found in the results.")
    if len(keep) > 1:
        warnings.append(f"more than one target pod matched: {keep}")
    cpu = cpu[cpu["pod"].isin(keep)]
    mem = mem[mem["pod"].isin(keep)]

    # alignment
    cpu_keys = set(zip(cpu["ts"], cpu["pod"]))
    mem_keys = set(zip(mem["ts"], mem["pod"]))
    if cpu_keys != mem_keys:
        errors.append(f"CPU/memory timestamps do not align "
                      f"(cpu-only={len(cpu_keys - mem_keys)}, mem-only={len(mem_keys - cpu_keys)})")

    # duplicates in the raw series
    if cpu.duplicated(["ts", "pod"]).any() or mem.duplicated(["ts", "pod"]).any():
        errors.append("duplicate (timestamp, pod) rows in Prometheus output")

    df = cpu.merge(mem, on=["ts", "pod"], how="inner").sort_values(["pod", "ts"]).reset_index(drop=True)

    # window check
    t0 = datetime.fromisoformat(START.replace("Z", "+00:00")).timestamp()
    t1 = datetime.fromisoformat(END.replace("Z", "+00:00")).timestamp()
    if (df["ts"] < t0).any() or (df["ts"] > t1).any():
        errors.append("rows fall outside the requested start/end window")

    # missing values
    if df[["cpu_cores", "memory_bytes"]].isna().any().any():
        errors.append("missing (NaN) values in CPU or memory")

    expected = int((t1 - t0) // 15) + 1
    per_pod = df.groupby("pod").size().to_dict()
    for pod, n in per_pod.items():
        if abs(n - expected) > 2:
            warnings.append(f"{pod}: {n} rows, expected about {expected}")

    out = pd.DataFrame({
        "timestamp": df["ts"].map(fmt_ts),
        "pod": df["pod"],
        "cpu_cores": df["cpu_cores"],
        "cpu_millicores": df["cpu_cores"] * 1000.0,
        "memory_bytes": df["memory_bytes"],
        "memory_mib": df["memory_bytes"] / (1024 * 1024),
        "pattern": label,
    })[SCHEMA]

    if out.duplicated(["timestamp", "pod"]).any():
        errors.append("duplicate (timestamp, pod) rows in final dataset")
    if list(out.columns) != SCHEMA:
        errors.append("final columns do not match schema")

    if errors:
        print("\n[VALIDATION ERRORS]")
        for e in errors:
            print(f"  - {e}")
        fail("Random Spike validation failed. No CSVs were written.")

    out.to_csv(OUT_DATASET, index=False)
    out.to_csv(OUT_TARGET, index=False)
    print(f"    wrote {OUT_DATASET} and {OUT_TARGET}  ({len(out)} rows)")

    # 6. combined -------------------------------------------------------
    step("5/6 Building combined five-pattern dataset")
    parts = list(existing.values()) + [out]
    combined = pd.concat(parts, ignore_index=True)
    if combined.isna().any().any():
        fail("combined dataset contains missing values")
    if combined.duplicated(["timestamp", "pod", "pattern"]).any():
        fail("combined dataset contains duplicate rows")
    combined.to_csv(OUT_COMBINED, index=False)
    print(f"    wrote {OUT_COMBINED}")

    # protected-file integrity -----------------------------------------
    hashes_after = {f: sha256(f) for f in PROTECTED}
    unchanged = hashes_before == hashes_after
    if not unchanged:
        fail("A protected file changed. This should never happen - check your files.")

    # summary -----------------------------------------------------------
    step("6/6 Validation summary")
    print(f"    status=success (CPU + memory)        : PASS")
    print(f"    CPU/memory timestamps aligned        : PASS")
    print(f"    no duplicate rows                    : PASS")
    print(f"    no missing values                    : PASS")
    print(f"    target pod(s)                        : {keep}")
    print(f"    schema matches existing patterns     : PASS")
    print(f"    protected files unchanged (SHA-256)  : {'PASS' if unchanged else 'FAIL'}")
    print("\n    Row counts")
    for name, df_ in list(existing.items()) + [(OUT_TARGET, out)]:
        print(f"      {name:<24}{len(df_):>6}")
    print(f"      {OUT_COMBINED:<24}{len(combined):>6}")
    print("\n    Rows per pattern in combined dataset")
    for lbl, n in combined["pattern"].value_counts().sort_index().items():
        print(f"      {lbl:<24}{n:>6}")
    if warnings:
        print("\n    Warnings")
        for w in warnings:
            print(f"      - {w}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
