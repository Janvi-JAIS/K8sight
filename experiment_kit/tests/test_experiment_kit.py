import tempfile
import unittest
from pathlib import Path

from experiment_kit.generators.config import WorkloadConfig, PATTERNS
from experiment_kit.generators.schedule import calculate_schedule_timeline
from experiment_kit.collector.dry_run import generate_dry_run_record


class WorkloadConfigTests(unittest.TestCase):
    def test_all_patterns_produce_valid_schedules(self):
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                cfg = WorkloadConfig(
                    run_id=f"test_{pattern}",
                    pattern=pattern,
                    load_level="low",
                    duration_seconds=60,
                    dry_run=True,
                )
                timeline = calculate_schedule_timeline(cfg, step_seconds=5)

                self.assertTrue(timeline)
                self.assertEqual(timeline[0]["time_offset_seconds"], 0)
                self.assertLessEqual(
                    timeline[-1]["time_offset_seconds"], cfg.duration_seconds
                )
                self.assertTrue(
                    all(row["target_users"] >= 0 for row in timeline)
                )

    def test_short_spike_stays_within_duration(self):
        cfg = WorkloadConfig(
            run_id="test_short_spike",
            pattern="spike",
            load_level="low",
            duration_seconds=60,
            dry_run=True,
        )
        params = cfg.get_resolved_parameters()

        for start, end, _users in params["stages"]:
            self.assertLessEqual(start, end)
            self.assertLessEqual(end, cfg.duration_seconds)

        timeline = calculate_schedule_timeline(cfg, step_seconds=5)
        self.assertEqual(max(row["target_users"] for row in timeline), 15)

    def test_invalid_duration_is_rejected(self):
        cfg = WorkloadConfig(
            run_id="invalid_duration",
            pattern="stable",
            duration_seconds=0,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_schedule_hash_is_deterministic(self):
        kwargs = dict(
            run_id="same_run",
            pattern="random_spike",
            load_level="low",
            duration_seconds=60,
            random_seed=42,
        )
        first = WorkloadConfig(**kwargs).compute_schedule_hash()
        second = WorkloadConfig(**kwargs).compute_schedule_hash()
        self.assertEqual(first, second)


class DryRunRecordTests(unittest.TestCase):
    def test_all_patterns_generate_valid_dry_run_records(self):
        with tempfile.TemporaryDirectory(prefix="k8sight-test-") as temp:
            for pattern in PATTERNS:
                with self.subTest(pattern=pattern):
                    output_dir = Path(temp) / pattern
                    cfg = WorkloadConfig(
                        run_id=f"dry_{pattern}",
                        pattern=pattern,
                        load_level="low",
                        duration_seconds=60,
                        random_seed=42,
                        dry_run=True,
                    )
                    from experiment_kit.collector.config import CollectorConfig
                    collector_cfg = CollectorConfig(
                        run_id=cfg.run_id,
                        pattern=pattern,
                        pod="target-pod",
                    )

                    manifest, rows, errors = generate_dry_run_record(
                        cfg, collector_cfg, output_dir
                    )

                    self.assertEqual(errors, [], msg="; ".join(errors))
                    self.assertEqual(len(rows), 25)
                    self.assertTrue((output_dir / "observations.csv").is_file())
                    self.assertTrue((output_dir / "manifest.json").is_file())
                    self.assertEqual(manifest["run_id"], cfg.run_id)


if __name__ == "__main__":
    unittest.main()
