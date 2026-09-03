import json
import csv
from datetime import datetime, timezone

INPUT_FILE = "stable_cpu.json"
OUTPUT_FILE = "stable.csv"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

if data["status"] != "success":
    raise RuntimeError(data)

rows = []

for series in data["data"]["result"]:
    pod = series["metric"].get("pod", "unknown")

    for timestamp, value in series["values"]:
        ts = datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        cpu_cores = float(value)
        cpu_millicores = cpu_cores * 1000

        rows.append([
            ts,
            pod,
            cpu_cores,
            cpu_millicores,
            "stable"
        ])

rows.sort(key=lambda row: (row[0], row[1]))

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "pod",
        "cpu_cores",
        "cpu_millicores",
        "pattern"
    ])
    writer.writerows(rows)

print(f"Created {OUTPUT_FILE}")
print(f"Rows: {len(rows)}")
