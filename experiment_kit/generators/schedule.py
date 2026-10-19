"""
Offline workload schedule calculation and dry-run utilities.
Computes and validates target concurrency time series without sending any HTTP traffic.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from typing import Dict, Any, List, Optional

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.generators.shapes import get_shape_instance


def calculate_schedule_timeline(config: WorkloadConfig, step_seconds: int = 15) -> List[Dict[str, Any]]:
    """Calculates the time-indexed schedule of target users for the configured duration."""
    shape = get_shape_instance(config)
    timeline: List[Dict[str, Any]] = []

    for t in range(0, config.duration_seconds + 1, step_seconds):
        users = shape.get_target_users_at_time(float(t))
        if users is not None:
            timeline.append({
                "time_offset_seconds": t,
                "target_users": users,
            })
    return timeline


def generate_schedule_summary(config: WorkloadConfig, step_seconds: int = 15) -> Dict[str, Any]:
    """Generates an offline dry-run summary of the planned workload."""
    timeline = calculate_schedule_timeline(config, step_seconds=step_seconds)
    user_counts = [row["target_users"] for row in timeline]

    min_u = min(user_counts) if user_counts else 0
    max_u = max(user_counts) if user_counts else 0
    mean_u = sum(user_counts) / len(user_counts) if user_counts else 0.0

    # Deterministic hash of the exact evaluated schedule series
    series_repr = ",".join(str(u) for u in user_counts)
    series_hash = hashlib.sha256(series_repr.encode("utf-8")).hexdigest()

    summary: Dict[str, Any] = {
        "run_id": config.run_id,
        "pattern": config.pattern,
        "load_level": config.load_level,
        "replicate": config.replicate,
        "random_seed": config.random_seed,
        "duration_seconds": config.duration_seconds,
        "step_seconds": step_seconds,
        "total_schedule_points": len(timeline),
        "min_target_users": min_u,
        "max_target_users": max_u,
        "mean_target_users": round(mean_u, 2),
        "evaluated_series_hash": series_hash,
        "config_schedule_hash": config.compute_schedule_hash(),
        "resolved_parameters": config.get_resolved_parameters(),
        "dry_run": config.dry_run,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate and inspect workload schedule offline.")
    parser.add_argument("--pattern", default="stable", choices=["stable", "increasing", "periodic", "spike", "random_spike"])
    parser.add_argument("--level", default="high", choices=["low", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")

    args = parser.parse_args()
    cfg = WorkloadConfig(
        run_id=f"dry_run_{args.pattern}_{args.level}_s{args.seed}",
        pattern=args.pattern,
        load_level=args.level,
        replicate=args.replicate,
        random_seed=args.seed,
        duration_seconds=args.duration,
        dry_run=True,
    )

    summary = generate_schedule_summary(cfg, step_seconds=args.step)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("=== Workload Schedule Summary (Dry-Run) ===")
        print(f"Pattern:          {summary['pattern']} ({summary['load_level']} level)")
        print(f"Duration:         {summary['duration_seconds']}s ({summary['total_schedule_points']} points at {summary['step_seconds']}s step)")
        print(f"Users (Min/Mean/Max): {summary['min_target_users']} / {summary['mean_target_users']} / {summary['max_target_users']}")
        print(f"Series Hash:      {summary['evaluated_series_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
