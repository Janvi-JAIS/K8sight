"""Synthetic regression fixtures for approved offline Task A behavior."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiment_kit.collector.live_collector import collect, required_metric_error
from experiment_kit.collector.dry_run import generate_dry_run_record
from experiment_kit.collector.config import CollectorConfig
from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.collector.schema_integration import apply_run_context
from experiment_kit.collector.run_context import load_run_context
from experiment_kit.job_manifest import build_job_manifests


class TaskARegressionTests(unittest.TestCase):
    def test_run_context_loader_requires_json_object(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_context.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_run_context(path)

    def test_live_rejects_fractional_epoch_and_interval_constraints(self):
        common = dict(prometheus_url="http://127.0.0.1:9090", namespace="n", pod="p", container="c",
                      start=1, end=11, step=4, scrape_interval=2, lookback=8, output_dir=Path("unused"))
        with self.assertRaisesRegex(ValueError, "whole number"):
            collect(**{**common, "start": 1.5})
        with self.assertRaisesRegex(ValueError, "4 times"):
            collect(**{**common, "lookback": 7})
        with self.assertRaisesRegex(ValueError, "step must be at least"):
            collect(**{**common, "step": 1})

    def test_mocked_http_unavailable_cpu_or_memory_fails(self):
        def fake_get(url, params, timeout):
            name = params["query"].split("{", 1)[0]
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"status": "success", "data": {"resultType": "matrix", "result": [] if name in {
                "container_cpu_usage_seconds_total", "container_memory_working_set_bytes"} else [{"metric": {}, "values": [[1, "1"]]}]}}
            return response
        with tempfile.TemporaryDirectory() as temp, patch("experiment_kit.collector.live_collector.requests.Session") as factory:
            factory.return_value.get.side_effect = fake_get
            summary = collect(prometheus_url="http://127.0.0.1:9090", namespace="n", pod="p", container="c",
                              start=1, end=5, step=1, scrape_interval=1, lookback=4, output_dir=Path(temp) / "run")
            self.assertIn("Required metric unavailable", required_metric_error(summary))

    def test_dry_run_refuses_existing_directory_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            output.mkdir()
            sentinel = output / "preserve"
            sentinel.write_text("synthetic")
            wc = WorkloadConfig(run_id="synthetic", pattern="stable", dry_run=True)
            cc = CollectorConfig(run_id="synthetic", pattern="stable", pod="pod")
            with self.assertRaises(FileExistsError):
                generate_dry_run_record(wc, cc, output)
            self.assertEqual(sentinel.read_text(), "synthetic")

    def apply_counter_window(self, offsets, throttled, periods):
        manifest = {"boundaries": {"unknown_reasons": {"run_start_utc": "unknown"}},
                    "workload": {}, "environment": {"resources": {}}, "historical_audits": [],
                    "metrics": [{"name": "container_cpu_cfs_throttled_periods_total", "step_seconds": 15},
                                {"name": "container_cpu_cfs_periods_total", "step_seconds": 15}],
                    "data_quality": {"counter_resets": [], "container_changes": []},
                    "throttling": {"status": "unknown", "reason": "unknown"}}
        base = 1767225600
        rows = [{"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(base + offset)),
                 "pod": "php-apache", "pod_uid": "pod-uid-1",
                 "container": "hpa-example", "container_id": "container-old",
                 "quality_flags": [], "missing_reasons": {}, "cpu_rate_lookback_seconds": 4,
                 "throttled_period_ratio_1m": None, "sample_phase": "unknown"} for offset in offsets]
        aligned = [{"timestamp_epoch": base + offset, "cfs_throttled_periods_total": n, "cfs_periods_total": d}
                   for offset, n, d in zip(offsets, throttled, periods)]
        context = {"workload": {}, "environment": {}}
        apply_run_context(manifest, rows, context, aligned)
        self.assertEqual(manifest["throttling"]["status"], "unknown")
        return rows, manifest

    def test_ratio_requires_exact_60_second_interval(self):
        rows, manifest = self.apply_counter_window([0, 15, 30, 45, 60], [2, 3, 4, 5, 8], [10, 20, 30, 40, 50])
        self.assertEqual(rows[-1]["throttled_period_ratio_1m"], 6 / 40)
        self.assertNotIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])
        self.assertEqual(manifest["throttling"]["status"], "unknown")

    def test_container_identity_change_invalidates_throttling_window(self):
        rows, manifest = self.apply_counter_window(
            [0, 15, 30, 45, 60],
            [2, 3, 4, 5, 8],
            [10, 20, 30, 40, 50],
        )

        # Simulate a container replacement at the midpoint.
        for index, row in enumerate(rows):
            row.update({
                "pod": "php-apache",
                "pod_uid": "pod-uid-1",
                "container": "hpa-example",
                "container_id": "container-old" if index < 2 else "container-new",
            })

        # Re-run with the changed identities present in the rows.
        # The ratio must not span different container identities.
        from experiment_kit.collector.schema_integration import apply_run_context
        aligned = [
            {"timestamp_epoch": 1767225600 + offset,
             "cfs_throttled_periods_total": n,
             "cfs_periods_total": d}
            for offset, n, d in zip(
                [0, 15, 30, 45, 60],
                [2, 3, 4, 5, 8],
                [10, 20, 30, 40, 50],
            )
        ]
        manifest["data_quality"]["container_changes"] = []
        apply_run_context(
            manifest, rows,
            {"workload": {}, "environment": {}},
            aligned,
        )

        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn(
            "throttled_period_ratio_1m",
            rows[-1]["missing_reasons"],
        )
        self.assertTrue(manifest["data_quality"]["container_changes"])


    def test_missing_interior_timestamp_makes_ratio_unavailable(self):
        rows, _ = self.apply_counter_window([0, 15, 45, 60], [2, 3, 5, 8], [10, 20, 40, 50])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])

    def test_one_second_interval_is_unavailable(self):
        rows, _ = self.apply_counter_window([0, 1], [2, 3], [10, 20])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])

    def test_missing_counter_inside_interval_is_unavailable(self):
        rows, _ = self.apply_counter_window([0, 15, 30, 45, 60], [2, 3, None, 5, 8], [10, 20, 30, 40, 50])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])

    def test_counter_reset_inside_interval_is_unavailable(self):
        rows, manifest = self.apply_counter_window([0, 15, 30, 45, 60], [2, 4, 1, 5, 8], [10, 20, 5, 40, 50])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])
        self.assertEqual(len(manifest["data_quality"]["counter_resets"]), 1)

    def test_invalid_counter_deltas_are_unavailable(self):
        rows, _ = self.apply_counter_window([0, 15, 30, 45, 60], [2, 3, 4, 5, 50], [10, 20, 30, 40, 50])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])

    def test_missing_counter_endpoint_is_unavailable(self):
        rows, _ = self.apply_counter_window([0, 15, 30, 45, 60], [2, 3, 4, 5, 8], [10, 20, 30, 40, None])
        self.assertIsNone(rows[-1]["throttled_period_ratio_1m"])
        self.assertIn("throttled_period_ratio_1m", rows[-1]["missing_reasons"])

    def test_manifest_is_pure_and_has_required_job_settings(self):
        config, job = build_job_manifests(run_id="run-123", image="locust:test", config_files={"generators/locustfile.py": "pass"}, target_host="http://target")
        self.assertEqual(config["kind"], "ConfigMap")
        pod = job["spec"]["template"]["spec"]
        self.assertEqual(job["spec"]["ttlSecondsAfterFinished"], 86400)
        self.assertGreater(job["spec"]["activeDeadlineSeconds"], 0)
        self.assertEqual(pod["volumes"][0]["configMap"]["items"][0]["path"], "generators/locustfile.py")
        env = {x["name"]: x["value"] for x in pod["containers"][0]["env"]}
        self.assertEqual(env["PYTHONPATH"], "/opt/k8sight")
        self.assertIn("K8SIGHT_RUN_ID", env)
        self.assertIn("--csv-full-history", pod["containers"][0]["command"])


    def test_late_cooldown_phase_and_rate_status(self):
        from datetime import datetime, timezone

        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        def stamp(seconds):
            from datetime import timedelta
            return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")

        manifest = {
            "boundaries": {
                "run_start_utc": stamp(0),
                "workload_end_utc": stamp(120),
                "locust_shutdown_utc": stamp(120),
                "cooldown_start_utc": stamp(240),
                "collection_end_utc": stamp(240),
                "unknown_reasons": {},
            },
            "workload": {},
            "environment": {"resources": {}},
            "historical_audits": [],
            "metrics": [],
            "data_quality": {"counter_resets": [], "container_changes": []},
            "throttling": {"status": "unknown", "reason": "synthetic test"},
        }
        rows = [
            {
                "timestamp_utc": stamp(195),
                "cpu_rate_lookback_seconds": 30,
                "sample_phase": "unknown",
                "cpu_rate_window_status": "unknown",
                "missing_reasons": {},
                "quality_flags": [],
                "throttled_period_ratio_1m": None,
                "pod": "test-pod",
                "pod_uid": "test-uid",
                "container": "test-container",
                "container_id": "test-container-id",
            },
            {
                "timestamp_utc": stamp(240),
                "cpu_rate_lookback_seconds": 30,
                "sample_phase": "unknown",
                "cpu_rate_window_status": "unknown",
                "missing_reasons": {},
                "quality_flags": [],
                "throttled_period_ratio_1m": None,
                "pod": "test-pod",
                "pod_uid": "test-uid",
                "container": "test-container",
                "container_id": "test-container-id",
            },
        ]
        apply_run_context(manifest, rows, {"workload": {}, "environment": {}}, [])

        self.assertEqual(rows[0]["sample_phase"], "post_shutdown")
        self.assertEqual(rows[0]["cpu_rate_window_status"], "unknown")
        self.assertEqual(rows[1]["sample_phase"], "cooldown")
        self.assertNotEqual(rows[1]["cpu_rate_window_status"], "fully_within_active")


if __name__ == "__main__":
    unittest.main()
