import json
import tempfile
import unittest
from pathlib import Path

from experiment_kit.collector.schema_integration import load_collection_summary


class SchemaIntegrationTests(unittest.TestCase):
    def test_loads_valid_summary(self):
        summary = {
            "metrics": [{"name": "sample_metric"}],
            "target": {
                "namespace": "default",
                "pod": "test-pod",
                "container": "app",
            },
            "query_range": {
                "start_utc": "2026-01-01T00:00:00Z",
                "end_utc": "2026-01-01T00:01:00Z",
                "step_seconds": 30,
            },
        }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "collection_summary.json"
            original = json.dumps(summary, indent=2)
            path.write_text(original, encoding="utf-8")

            loaded = load_collection_summary(path)

            self.assertEqual(loaded, summary)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_rejects_missing_metrics(self):
        summary = {
            "target": {"namespace": "default"},
            "query_range": {},
        }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "collection_summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metrics list"):
                load_collection_summary(path)

    def test_rejects_non_object_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "collection_summary.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_collection_summary(path)

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing.json"

            with self.assertRaises(FileNotFoundError):
                load_collection_summary(path)


    def test_aligns_by_exact_timestamp_and_preserves_missing_values(self):
        from experiment_kit.collector.schema_integration import align_metric_samples

        aligned = align_metric_samples({
            "container_cpu_usage_seconds_total": {1000: 0.25, 1030: 0.5},
            "container_memory_working_set_bytes": {1000: 1048576},
        })

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0]["timestamp_epoch"], 1000.0)
        self.assertEqual(aligned[0]["cpu_usage_cores"], 0.25)
        self.assertEqual(aligned[0]["memory_working_set_bytes"], 1048576)
        self.assertIsNone(aligned[0]["cfs_periods_total"])
        self.assertIsNone(aligned[1]["memory_working_set_bytes"])
        self.assertEqual(aligned[1]["cpu_usage_cores"], 0.5)

    def test_alignment_rejects_negative_values(self):
        from experiment_kit.collector.schema_integration import align_metric_samples

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            align_metric_samples({
                "container_cpu_usage_seconds_total": {1000: -0.1},
            })

    def test_alignment_rejects_fractional_memory_bytes(self):
        from experiment_kit.collector.schema_integration import align_metric_samples

        with self.assertRaisesRegex(ValueError, "whole bytes"):
            align_metric_samples({
                "container_memory_working_set_bytes": {1000: 1.5},
            })


    def test_observation_conversion_has_schema_columns_and_missing_reasons(self):
        from experiment_kit.collector.schema_integration import (
            OBSERVATION_COLUMNS,
            aligned_to_observation_rows,
        )

        rows = aligned_to_observation_rows(
            [{
                "timestamp_epoch": 1000.0,
                "cpu_usage_cores": 0.25,
                "memory_working_set_bytes": None,
                "cfs_throttled_periods_total": None,
                "cfs_periods_total": None,
                "cfs_throttled_seconds_total": None,
            }],
            run_id="offline-test",
            pattern="stable",
            target={"namespace": "default", "pod": "test-pod", "container": "app"},
            lookback_seconds=60,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0]), OBSERVATION_COLUMNS)
        self.assertEqual(rows[0]["timestamp_utc"], "1970-01-01T00:16:40Z")
        self.assertEqual(rows[0]["sample_phase"], "unknown")
        self.assertEqual(rows[0]["cpu_rate_window_status"], "unknown")
        self.assertIn("memory_working_set_bytes", rows[0]["missing_reasons"])
        self.assertIsNone(rows[0]["memory_working_set_bytes"])

    def test_missing_cpu_is_not_applicable_for_rate_window(self):
        from experiment_kit.collector.schema_integration import aligned_to_observation_rows

        row = aligned_to_observation_rows(
            [{
                "timestamp_epoch": 1000.0,
                "cpu_usage_cores": None,
                "memory_working_set_bytes": None,
                "cfs_throttled_periods_total": None,
                "cfs_periods_total": None,
                "cfs_throttled_seconds_total": None,
            }],
            run_id="offline-test",
            pattern="stable",
            target={},
            lookback_seconds=60,
        )[0]

        self.assertEqual(row["cpu_rate_window_status"], "not_applicable")
        self.assertIn("cpu_usage_cores", row["missing_reasons"])


    def test_expected_grid_preserves_timestamp_missing_from_all_metrics(self):
        from experiment_kit.collector.schema_integration import align_metric_samples

        aligned = align_metric_samples(
            {
                "container_cpu_usage_seconds_total": {
                    1000.0: 0.25,
                    1030.0: 0.5,
                },
                "container_memory_working_set_bytes": {
                    1000.0: 1048576,
                    1030.0: 2097152,
                },
            },
            expected_timestamps=[1000.0, 1015.0, 1030.0],
        )

        self.assertEqual(
            [row["timestamp_epoch"] for row in aligned],
            [1000.0, 1015.0, 1030.0],
        )
        self.assertIsNone(aligned[1]["cpu_usage_cores"])
        self.assertIsNone(aligned[1]["memory_working_set_bytes"])
        self.assertIsNone(aligned[1]["cfs_periods_total"])


    def test_gap_report_lists_missing_metrics_at_expected_timestamps(self):
        from experiment_kit.collector.schema_integration import (
            align_metric_samples,
            calculate_sample_gaps,
            expected_timestamp_grid,
        )

        grid = expected_timestamp_grid({
            "start_utc": "2026-01-01T00:00:00Z",
            "end_utc": "2026-01-01T00:01:00Z",
            "step_seconds": 15,
        })
        names = [
            "container_cpu_usage_seconds_total",
            "container_memory_working_set_bytes",
            "container_cpu_cfs_throttled_periods_total",
            "container_cpu_cfs_periods_total",
            "container_cpu_cfs_throttled_seconds_total",
        ]
        samples = {name: {} for name in names}

        for i in (0, 2, 4):
            samples[names[0]][grid[i]] = 0.25
            samples[names[1]][grid[i]] = (i + 1) * 1024
            samples[names[2]][grid[i]] = i + 10
            samples[names[3]][grid[i]] = i + 100
            samples[names[4]][grid[i]] = float(i)

        aligned = align_metric_samples(samples, expected_timestamps=grid)
        gaps = calculate_sample_gaps(aligned, grid)

        self.assertEqual(len(aligned), 5)
        self.assertEqual(len(gaps), 2)
        self.assertEqual(
            [gap["timestamp_utc"] for gap in gaps],
            ["2026-01-01T00:00:15Z", "2026-01-01T00:00:45Z"],
        )
        self.assertTrue(all(len(gap["missing_metrics"]) == 5 for gap in gaps))

    def test_run_context_uses_earliest_active_end_boundary(self):
        from experiment_kit.collector.schema_integration import apply_run_context

        manifest = {
            "boundaries": {
                "unknown_reasons": {},
                "collection_end_utc": "2026-01-01T00:05:00Z",
            },
            "workload": {},
            "environment": {"resources": {}},
            "metrics": [],
            "historical_audits": [],
            "data_quality": {
                "counter_resets": [],
                "container_changes": [],
            },
            "throttling": {"status": "unknown", "reason": "test fixture"},
        }
        context = {
            "boundaries": {
                "run_start_utc": "2026-01-01T00:00:00Z",
                "workload_end_utc": "2026-01-01T00:02:00Z",
                "locust_shutdown_utc": "2026-01-01T00:02:00Z",
                "cooldown_start_utc": "2026-01-01T00:04:00Z",
            },
            "workload": {},
            "environment": {},
        }
        row = {
            "timestamp_utc": "2026-01-01T00:03:00Z",
            "cpu_rate_lookback_seconds": 30,
            "cpu_rate_window_status": "unknown",
            "sample_phase": "unknown",
            "missing_reasons": {},
        }

        apply_run_context(manifest, [row], context, [])

        self.assertEqual(row["sample_phase"], "post_shutdown")
        self.assertNotEqual(
            row["cpu_rate_window_status"], "fully_within_active"
        )
        self.assertEqual(row["cpu_rate_window_status"], "unknown")

    def apply_outcome_context(self, context):
        from experiment_kit.collector.schema_integration import apply_run_context

        manifest = {
            "run_outcome": "unknown",
            "boundaries": {"unknown_reasons": {}, "collection_end_utc": None},
            "workload": {},
            "environment": {"resources": {}},
            "metrics": [],
            "historical_audits": [],
            "data_quality": {"counter_resets": [], "container_changes": []},
            "throttling": {"status": "unknown", "reason": "No finding inferred"},
        }
        apply_run_context(manifest, [], context, [])
        self.assertEqual(manifest["throttling"]["status"], "unknown")
        return manifest["run_outcome"]

    def test_run_context_applies_explicit_observed_completed_outcome(self):
        self.assertEqual(
            self.apply_outcome_context({
                "run_outcome": {
                    "status": "observed", "value": "completed",
                    "reason": None, "source": "experiment_metadata.json",
                    "observed_at_utc": "2026-10-13T17:04:42Z",
                },
            }),
            "completed",
        )

    def test_run_context_missing_outcome_remains_unknown(self):
        self.assertEqual(self.apply_outcome_context({}), "unknown")

    def test_run_context_invalid_or_unobserved_outcome_remains_unknown(self):
        for evidence in (
            {"status": "observed", "value": "successful"},
            {"status": "unknown", "value": "completed"},
            {"status": "observed", "value": ""},
        ):
            with self.subTest(evidence=evidence):
                self.assertEqual(
                    self.apply_outcome_context({"run_outcome": evidence}),
                    "unknown",
                )

    def test_end_to_end_offline_conversion_validates_run(self):
        import hashlib
        import subprocess
        import sys
        from datetime import datetime, timezone, timedelta

        from experiment_kit.collector.schema_integration import (
            METRIC_COLUMNS,
            OBSERVATION_COLUMNS,
        )

        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            output = root / "converted"

            start = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
            end = start + timedelta(seconds=60)
            start_epoch = start.timestamp()
            end_epoch = end.timestamp()
            names = list(METRIC_COLUMNS)
            units = {
                "container_cpu_usage_seconds_total": "cores",
                "container_memory_working_set_bytes": "bytes",
                "container_cpu_cfs_throttled_periods_total": "periods",
                "container_cpu_cfs_periods_total": "periods",
                "container_cpu_cfs_throttled_seconds_total": "seconds",
            }
            metrics = []

            for name in names:
                value = {
                    "container_cpu_usage_seconds_total": 0.02,
                    "container_memory_working_set_bytes": 1048576,
                    "container_cpu_cfs_throttled_periods_total": 0,
                    "container_cpu_cfs_periods_total": 100,
                    "container_cpu_cfs_throttled_seconds_total": 0,
                }[name]
                raw = {
                    "status": "success",
                    "data": {
                        "resultType": "matrix",
                        "result": [{
                            "metric": {
                                "namespace": "default",
                                "pod": "test-pod",
                                "container": "test-container",
                            },
                            "values": [
                                [start_epoch + i * 15, str(value)]
                                for i in range(5)
                            ],
                        }],
                    },
                }
                filename = f"{name}.json"
                raw_path = source / filename
                raw_path.write_text(json.dumps(raw), encoding="utf-8")
                metrics.append({
                    "name": name,
                    "availability": "available",
                    "promql": name,
                    "unit": units[name],
                    "query_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "query_end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "step_seconds": 15,
                    "lookback_seconds": 60,
                    "labels": {
                        "namespace": "default",
                        "pod": "test-pod",
                        "container": "test-container",
                    },
                    "raw_sources": [{
                        "path": filename,
                        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        "media_type": "application/json",
                    }],
                })

            summary = {
                "created_at_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "target": {
                    "namespace": "default",
                    "pod": "test-pod",
                    "container": "test-container",
                },
                "query_range": {
                    "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "step_seconds": 15,
                },
                "metrics": metrics,
            }
            summary_path = source / "collection_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            command = [
                sys.executable, "-B", "-m",
                "experiment_kit.collector.schema_integration",
                "--summary", str(summary_path),
                "--output-dir", str(output),
                "--run-id", "temporary-integration-test",
                "--pattern", "stable",
            ]
            result = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("Valid run:", result.stdout)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "observations.csv").is_file())

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["data_quality"]["expected_samples"], 5)
            self.assertEqual(manifest["data_quality"]["observed_samples"], 5)
            self.assertEqual(manifest["data_quality"]["gaps"], [])

            with (output / "observations.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                import csv
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, OBSERVATION_COLUMNS)
                self.assertEqual(len(list(reader)), 5)

            # A second conversion must refuse to overwrite this run.
            second = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            self.assertEqual(second.returncode, 2)
            self.assertIn("refusing to overwrite", second.stderr.lower())


if __name__ == "__main__":
    unittest.main()
