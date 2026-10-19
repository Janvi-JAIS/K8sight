import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiment_kit.collector.config import CollectorConfig
from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.orchestrator import build_run_plan, write_run_plan


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.workload = WorkloadConfig(
            run_id="offline-test-stable-r1",
            pattern="stable",
            load_level="low",
            replicate=1,
            random_seed=42,
            duration_seconds=1200,
        )
        self.collector = CollectorConfig(
            run_id=self.workload.run_id,
            pattern=self.workload.pattern,
            load_level=self.workload.load_level,
            replicate=self.workload.replicate,
            random_seed=self.workload.random_seed,
            duration_seconds=self.workload.duration_seconds,
            pod="test-pod",
        )
        self.start = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)

    def make_plan(self):
        return build_run_plan(
            self.workload,
            self.collector,
            planned_start_utc=self.start,
        )

    def test_plan_is_marked_not_started(self):
        plan = self.make_plan()
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["run_outcome"], "not_started")
        self.assertFalse(plan["execution_enabled"])
        self.assertFalse(plan["external_calls_enabled"])
        self.assertIsNone(plan["boundaries"]["locust_shutdown_utc"])

    def test_plan_is_reproducible(self):
        self.assertEqual(self.make_plan(), self.make_plan())

    def test_planned_boundaries_are_ordered(self):
        boundaries = self.make_plan()["boundaries"]
        self.assertLess(
            boundaries["collection_start_utc"], boundaries["run_start_utc"]
        )
        self.assertLess(
            boundaries["run_start_utc"], boundaries["workload_end_utc"]
        )
        self.assertLess(
            boundaries["workload_end_utc"], boundaries["collection_end_utc"]
        )

    def test_naive_start_time_is_rejected(self):
        naive = datetime(2026, 10, 12, 12, 0)
        with self.assertRaises(ValueError):
            build_run_plan(self.workload, self.collector, planned_start_utc=naive)

    def test_mismatched_run_ids_are_rejected(self):
        self.collector.run_id = "different-run"
        with self.assertRaises(ValueError):
            self.make_plan()

    def test_invalid_cpu_lookback_is_rejected(self):
        self.collector.cpu_lookback_seconds = 30
        with self.assertRaises(ValueError):
            self.make_plan()

    def test_plan_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            plan = self.make_plan()
            write_run_plan(path, plan)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), plan
            )

            with self.assertRaises(FileExistsError):
                write_run_plan(path, {"status": "replacement"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), plan
            )


if __name__ == "__main__":
    unittest.main()
