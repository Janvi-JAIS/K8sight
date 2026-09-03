import json
import csv
from datetime import datetime, timezone

CPU_FILE = "stable_cpu.json"
MEMORY_FILE = "stable_memory.json"
OUTPUT_FILE = "stable_dataset.csv"

STEP_SECONDS = 15


def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    if data.get("status") != "success":
        raise RuntimeError(f"{filename} contains an unsuccessful response")

    return data["data"]["result"]


def timestamp_to_string(timestamp):
    return datetime.fromtimestamp(
        float(timestamp), tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


cpu_series = load_json(CPU_FILE)
memory_series = load_json(MEMORY_FILE)

cpu_data = {}
memory_data = {}

for series in cpu_series:
    pod = series["metric"].get("pod", "unknown")

    for timestamp, value in series["values"]:
        timestamp_key = float(timestamp)

        cpu_data[(timestamp_key, pod)] = float(value)


for series in memory_series:
    pod = series["metric"].get("pod", "unknown")

    for timestamp, value in series["values"]:
        timestamp_key = float(timestamp)

        memory_data[(timestamp_key, pod)] = float(value)


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
        "stable",
    ])


with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "pod",
        "cpu_cores",
        "cpu_millicores",
        "memory_bytes",
        "memory_mib",
        "pattern",
    ])

    writer.writerows(rows)


print(f"Created {OUTPUT_FILE}")
print(f"Rows: {len(rows)}")
