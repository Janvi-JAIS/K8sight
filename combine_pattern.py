import json
import csv
import argparse
from datetime import datetime, timezone


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    if data.get("status") != "success":
        raise RuntimeError(f"{filename} does not contain a successful Prometheus response")

    return data["data"]["result"]


def timestamp_to_string(ts):
    return datetime.fromtimestamp(
        float(ts), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--pattern", required=True)

    args = parser.parse_args()

    cpu_series = load_json(args.cpu)
    memory_series = load_json(args.memory)

    cpu_data = {}
    memory_data = {}

    # CPU
    for series in cpu_series:
        pod = series["metric"].get("pod", "unknown")

        for timestamp, value in series["values"]:
            cpu_data[(float(timestamp), pod)] = float(value)

    # Memory
    for series in memory_series:
        pod = series["metric"].get("pod", "unknown")

        for timestamp, value in series["values"]:
            memory_data[(float(timestamp), pod)] = float(value)

    rows = []

    for key in sorted(cpu_data):
        timestamp, pod = key

        if key not in memory_data:
            continue

        cpu_cores = cpu_data[key]
        memory_bytes = memory_data[key]

        rows.append([
            timestamp_to_string(timestamp),
            pod,
            cpu_cores,
            cpu_cores * 1000,
            memory_bytes,
            memory_bytes / (1024 * 1024),
            args.pattern
        ])

    raw_output = f"{args.pattern}_dataset.csv"

    with open(raw_output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "pod",
            "cpu_cores",
            "cpu_millicores",
            "memory_bytes",
            "memory_mib",
            "pattern"
        ])

        writer.writerows(rows)

    print(f"Created: {raw_output}")
    print(f"Rows: {len(rows)}")

    # Keep only the target workload pod
    target_pods = [
        pod for pod in sorted({row[1] for row in rows})
        if pod.startswith("php-apache-")
        and not pod.startswith("php-apache-2-")
    ]

    if len(target_pods) != 1:
        raise RuntimeError(
            f"Expected exactly one target php-apache pod, found: {target_pods}"
        )

    target_pod = target_pods[0]

    target_rows = [
        row for row in rows
        if row[1] == target_pod
    ]

    target_output = f"{args.pattern}_target.csv"

    with open(target_output, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "pod",
            "cpu_cores",
            "cpu_millicores",
            "memory_bytes",
            "memory_mib",
            "pattern"
        ])

        writer.writerows(target_rows)

    print(f"Created: {target_output}")
    print(f"Target pod: {target_pod}")
    print(f"Target rows: {len(target_rows)}")


if __name__ == "__main__":
    main()
