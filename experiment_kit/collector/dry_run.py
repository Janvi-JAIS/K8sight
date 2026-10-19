"""
Offline dry-run collector utility.
Generates and validates complete run records adhering to dataset_schema_v1
without contacting Kubernetes, Prometheus, or running workloads.
"""

from __future__ import annotations

import csv
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.generators.schedule import generate_schedule_summary
from experiment_kit.collector.config import CollectorConfig
from experiment_kit.collector.record_builder import (
    format_utc,
    build_run_manifest,
    build_observations_table,
)

# Reference dataset_schema_v1 validator
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT / "dataset_schema_v1") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_schema_v1"))
from validate_run import validate_run, COLS


def generate_dry_run_record(
    w_config: WorkloadConfig,
    c_config: CollectorConfig,
    output_dir: Path,
    base_time: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Constructs and writes a fully valid offline run record into output_dir.
    Runs schema validator to ensure 100% compliance.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    t0 = base_time or datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)

    # Calculate boundaries
    col_start = t0
    run_start = col_start + timedelta(seconds=c_config.pre_start_seconds)
    workload_end = run_start + timedelta(seconds=w_config.duration_seconds)
    cooldown_start = workload_end
    shutdown = cooldown_start + timedelta(seconds=c_config.cooldown_seconds)
    col_end = shutdown

    boundaries = {
        "run_start_utc": format_utc(run_start),
        "workload_end_utc": format_utc(workload_end),
        "cooldown_start_utc": format_utc(cooldown_start),
        "locust_shutdown_utc": format_utc(shutdown),
        "collection_start_utc": format_utc(col_start),
        "collection_end_utc": format_utc(col_end),
    }

    # Build observations table
    rows = build_observations_table(w_config, c_config, boundaries)

    # Build manifest
    manifest = build_run_manifest(
        w_config=w_config,
        c_config=c_config,
        boundaries=boundaries,
        sample_count=len(rows),
        run_outcome="completed",
    )

    # Write observations.csv
    csv_path = output_dir / "observations.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)

    # Write manifest.json
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2))

    # Write generator config & schedule summary
    sched_summary = generate_schedule_summary(w_config, step_seconds=c_config.step_seconds)
    with (output_dir / "schedule_summary.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(sched_summary, indent=2))

    # Run read-only validator
    validation_errors = validate_run(manifest, rows, output_dir)

    # Write README
    readme_text = f"""# K8sight Dry-Run Record: {w_config.run_id}

- Pattern: `{w_config.pattern}`
- Level: `{w_config.load_level}`
- Duration: `{w_config.duration_seconds}s`
- Pre-start baseline: `{c_config.pre_start_seconds}s`
- Cooldown: `{c_config.cooldown_seconds}s`
- Total samples: `{len(rows)}` (step `{c_config.step_seconds}s`)
- Schema Validation: `{'PASS' if not validation_errors else 'FAIL'}`
"""
    with (output_dir / "README.md").open("x", encoding="utf-8") as f:
        f.write(readme_text)

    return manifest, rows, validation_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate an offline dry-run record.")
    parser.add_argument("--pattern", default="stable", choices=["stable", "increasing", "periodic", "spike", "random_spike"])
    parser.add_argument("--level", default="high", choices=["low", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args()

    w_cfg = WorkloadConfig(
        run_id=f"dryrun_{args.pattern}_{args.level}_s{args.seed}",
        pattern=args.pattern,
        load_level=args.level,
        random_seed=args.seed,
        duration_seconds=args.duration,
        dry_run=True,
    )
    c_cfg = CollectorConfig(
        run_id=w_cfg.run_id,
        pattern=w_cfg.pattern,
        load_level=w_cfg.load_level,
        random_seed=w_cfg.random_seed,
        duration_seconds=w_cfg.duration_seconds,
        pod="target-pod",
    )

    manifest, rows, errors = generate_dry_run_record(w_cfg, c_cfg, args.output_dir)

    print(f"Generated dry-run record in: {args.output_dir}")
    print(f"Observations: {len(rows)} rows")
    if errors:
        print("VALIDATION ERRORS:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("Schema Validation: PASS (conforms to dataset_schema_v1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
