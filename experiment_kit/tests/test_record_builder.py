import unittest

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.collector.config import CollectorConfig
from experiment_kit.collector.record_builder import build_run_manifest


class RunOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.workload = WorkloadConfig(
            run_id="outcome_test",
            pattern="stable",
            load_level="low",
            duration_seconds=60,
            dry_run=True,
        )
        self.collector = CollectorConfig(
            run_id="outcome_test",
            pattern="stable",
            pod="target-pod",
        )
        self.boundaries = {
            "collection_start_utc": "2026-10-12T10:00:00Z",
            "run_start_utc": "2026-10-12T10:02:00Z",
            "workload_end_utc": "2026-10-12T10:03:00Z",
            "locust_shutdown_utc": "2026-10-12T10:03:00Z",
            "cooldown_start_utc": "2026-10-12T10:03:00Z",
            "collection_end_utc": "2026-10-12T10:06:00Z",
        }

    def build(self, **kwargs):
        return build_run_manifest(
            self.workload,
            self.collector,
            self.boundaries,
            sample_count=1,
            **kwargs,
        )

    def test_default_outcome_is_unknown(self):
        manifest = self.build()
        self.assertEqual(manifest["run_outcome"], "unknown")

    def test_all_supported_outcomes_are_accepted(self):
        for outcome in (
            "completed",
            "partial",
            "failed",
            "stopped",
            "unknown",
        ):
            with self.subTest(outcome=outcome):
                manifest = self.build(run_outcome=outcome)
                self.assertEqual(manifest["run_outcome"], outcome)

    def test_invalid_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(run_outcome="successful")


if __name__ == "__main__":
    unittest.main()
