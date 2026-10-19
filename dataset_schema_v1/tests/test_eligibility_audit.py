"""Offline tests for the read-only dataset evidence eligibility audit."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dataset_schema_v1.eligibility_audit import RUN_SPECS, RunSpec, audit_run, audit_runs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "dataset_schema_v1/tests/fixtures/valid_run"


class EligibilityAuditTests(unittest.TestCase):
    def workspace(self, temp: str, run_id: str = "synthetic-increasing-001", pattern: str = "increasing"):
        root = Path(temp)
        output = root / "validated" / "run"
        output.parent.mkdir()
        shutil.copytree(FIXTURE, output)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_id"] = run_id
        manifest["pattern"] = pattern
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        observations_path = output / "observations.csv"
        with observations_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
            fields = rows[0].keys()
        for row in rows:
            row["run_id"] = run_id
            row["pattern"] = pattern
        with observations_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "run_id": run_id,
            "pattern": pattern,
            "training_eligible": False,
            "duration_seconds": 180,
            "status": "completed",
            "locust_exit_code": 0,
            "collector_exit_code": 0,
            "boundaries": {
                "run_start_utc": "2026-01-01T00:00:00Z",
                "workload_end_utc": "2026-01-01T00:03:00Z",
            },
        }
        metadata_path = root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.write_locust_artifacts(root)
        spec = RunSpec(run_id, pattern, ("validated/run",), "metadata.json")
        return root, output, spec, manifest_path, observations_path, metadata_path

    @staticmethod
    def write_locust_artifacts(root: Path, *, terminal_count: int = 20,
                               csv_count: int = 20, history_count: int = 20,
                               terminal_80: int = 200, csv_80: int = 200):
        (root / "locust_stderr.log").write_text(
            "Type     Name     # reqs      # fails\n"
            f"         Aggregated {terminal_count}     0(0.00%) |\n"
            "Type     Name     50%    80% # reqs\n"
            f"         Aggregated 100 {terminal_80} {terminal_count}\n",
            encoding="utf-8",
        )
        with (root / "locust_stats.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("Name", "Request Count", "Failure Count", "50%", "80%"))
            writer.writeheader()
            writer.writerow({"Name": "Aggregated", "Request Count": csv_count, "Failure Count": 0, "50%": 100, "80%": csv_80})
        with (root / "locust_stats_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("Name", "Total Request Count", "Total Failure Count"))
            writer.writeheader()
            writer.writerow({"Name": "Aggregated", "Total Request Count": history_count, "Total Failure Count": 0})
        (root / "locust.log").write_text(
            "[2026-01-01 12:00:00,000] INFO locust.runners: Shape test starting.\n"
            "[2026-01-01 12:03:00,250] INFO locust.runners: Shape test stopping\n",
            encoding="utf-8",
        )

    def test_manifest_id_and_pattern_match_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            result = audit_run(spec, root)
            self.assertEqual(result["schema_validity"], "pass")
            self.assertEqual(result["status"], "review_required")
            self.assertEqual(result["run_id"], spec.run_id)
            self.assertEqual(result["expected_pattern"], spec.pattern)
            self.assertEqual(result["acceptance_decision"], "not_made")

    def test_missing_manifest_or_observations_is_blocked(self):
        for missing in ("manifest.json", "observations.csv"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temp:
                root, output, spec, *_ = self.workspace(temp)
                (output / missing).unlink()
                result = audit_run(spec, root)
                self.assertEqual(result["status"], "blocked")
                self.assertIn("Required file is missing", result["reasons"][0])

    def test_manifest_id_or_pattern_mismatch_is_blocked(self):
        for field, wrong in (("run_id", "different-run"), ("pattern", "spike")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root, output, spec, *_ = self.workspace(temp)
                path = output / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest[field] = wrong
                path.write_text(json.dumps(manifest), encoding="utf-8")
                result = audit_run(spec, root)
                self.assertEqual(result["status"], "blocked")
                self.assertIn("Manifest " + field + " mismatch", result["reasons"][0])

    def test_manifest_csv_row_count_disagreement_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, spec, *_ = self.workspace(temp)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["data_quality"]["observed_samples"] = 99
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = audit_run(spec, root)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("disagrees with observations row count", result["reasons"][0])

    def test_missing_values_and_quality_flags_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root, output, spec, *_ = self.workspace(temp)
            path = output / "observations.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
                fields = rows[0].keys()
            rows[0]["quality_flags"] = '["review_me"]'
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result = audit_run(spec, root)
            self.assertEqual(result["status"], "review_required")
            self.assertEqual(result["counts"]["missing_measurements"]["cfs_periods_total"], 2)
            self.assertEqual(result["counts"]["raw_cfs_counter_missing"], {
                "cfs_periods_total": 2,
                "cfs_throttled_periods_total": 2,
                "cfs_throttled_seconds_total": 2,
            })
            self.assertEqual(result["counts"]["derived_throttling_ratio_missing"], 2)
            self.assertEqual(result["counts"]["quality_flags"], {"review_me": 1})
            self.assertEqual(result["counts"]["phase_counts"], {"active": 2})

    def test_training_eligibility_is_separate_from_schema_validity(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            result = audit_run(spec, root)
            self.assertEqual(result["schema_validity"], "pass")
            self.assertEqual(result["training_eligibility"], {"observed": False, "status": "observed"})
            self.assertEqual(result["acceptance_decision"], "not_made")

    def test_duplicate_or_ambiguous_mappings_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            duplicate_id = audit_runs((spec, spec), root)
            self.assertTrue(all(item["status"] == "blocked" for item in duplicate_id))
            ambiguous = RunSpec(spec.run_id, spec.pattern, ("validated/run", "validated/other"), spec.metadata_path)
            self.assertEqual(audit_run(ambiguous, root)["status"], "blocked")

    def test_audit_does_not_modify_inputs_or_training_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, manifest_path, observations_path, metadata_path = self.workspace(temp)
            paths = (manifest_path, observations_path, metadata_path)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            before_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result = audit_run(spec, root)
            after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            after_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(before, after)
            self.assertEqual(before_metadata, after_metadata)
            self.assertEqual(result["training_eligibility"]["observed"], False)
            self.assertEqual(result["acceptance_decision"], "not_made")

    def test_selected_mapping_covers_all_twenty_runs_and_retry01_outputs(self):
        self.assertEqual(len(RUN_SPECS), 20)
        self.assertEqual(len({spec.run_id for spec in RUN_SPECS}), 20)
        by_id = {spec.run_id: spec for spec in RUN_SPECS}
        self.assertEqual(
            by_id["stable-low-r3-seed20261018-day5-01-retry01"].output_dirs,
            ("dataset_schema_v1/validated_runs_day5/stable-low-r3-seed20261018-day5-01-retry01-schema-validated-01",),
        )
        self.assertEqual(
            by_id["spike-low-r3-seed20261016-20261018-day5-10-retry01"].output_dirs,
            ("dataset_schema_v1/validated_runs_day5/spike-low-r3-seed20261016-20261018-day5-10-retry01-schema-validated-01",),
        )

    def test_all_four_known_request_count_mismatches_are_unresolved(self):
        cases = (
            ("stable-low-r1-seed42-20261013-day3-01", "stable", 458, 457, 457),
            ("increasing-low-r4-seed20261022-20261018-day5-05", "increasing", 423, 422, 422),
            ("periodic-low-r3-seed20261016-20261018-day5-07", "periodic", 586, 585, 585),
            ("random_spike-low-r3-seed20261016-20261018-day5-13", "random_spike", 449, 448, 448),
        )
        for run_id, pattern, terminal, csv_count, history_count in cases:
            with self.subTest(run_id=run_id), tempfile.TemporaryDirectory() as temp:
                root, _, spec, *_ = self.workspace(temp, run_id, pattern)
                self.write_locust_artifacts(
                    root, terminal_count=terminal, csv_count=csv_count,
                    history_count=history_count,
                )
                result = audit_run(spec, root)
                evidence = result["locust_count_and_percentile_evidence"]
                self.assertEqual(evidence["status"], "unresolved_mismatch")
                self.assertEqual(evidence["terminal"]["request_count"], terminal)
                self.assertEqual(evidence["csv"]["request_count"], csv_count)
                self.assertEqual(evidence["final_history"]["request_count"], history_count)
                self.assertEqual(evidence["terminal"]["failure_count"], 0)
                self.assertEqual(evidence["csv"]["failure_count"], 0)
                self.assertEqual(evidence["final_history"]["failure_count"], 0)
                self.assertTrue(any("disagree" in reason for reason in result["reasons"]))
                self.assertEqual(result["acceptance_decision"], "not_made")

    def test_periodic_r3_terminal_80th_percentile_difference_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(
                temp, "periodic-low-r3-seed20261016-20261018-day5-07", "periodic"
            )
            self.write_locust_artifacts(root, terminal_80=4800, csv_80=4900)
            result = audit_run(spec, root)
            comparison = result["locust_count_and_percentile_evidence"]["percentile_comparison"]
            self.assertEqual(comparison["status"], "differences")
            self.assertEqual(comparison["differences"]["80%"], {"terminal": 4800.0, "csv": 4900.0})
            self.assertTrue(any("80% terminal 4800 vs CSV 4900" in reason for reason in result["reasons"]))

    def test_launcher_and_shape_durations_report_distinct_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ , metadata_path = self.workspace(temp)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["boundaries"]["workload_end_utc"] = "2026-01-01T00:03:03Z"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = audit_run(spec, root)
            duration = result["duration_evidence"]
            self.assertEqual(duration["launcher_subprocess_interval"]["duration_seconds"], 183)
            self.assertEqual(duration["locust_shape_interval"]["duration_seconds"], 180.25)
            self.assertIn("different start/stop events", duration["interpretation"])
            self.assertTrue(any("different boundaries" in reason for reason in result["reasons"]))

    def test_day5_unknown_manifest_outcome_is_compared_to_metadata_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, manifest_path, _, metadata_path = self.workspace(
                temp, "increasing-low-r2-seed20261020-20261018-day5-03", "increasing"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_outcome"] = "unknown"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = audit_run(spec, root)
            outcome = result["outcome_evidence"]
            self.assertEqual(outcome["manifest_run_outcome"], "unknown")
            self.assertEqual(outcome["experiment_metadata_status"], "completed")
            self.assertEqual(outcome["locust_exit_code"], 0)
            self.assertEqual(outcome["collector_exit_code"], 0)
            self.assertEqual(outcome["run_context_file"]["status"], "absent")
            self.assertTrue(outcome["unknown_manifest_vs_completed_metadata"])
            self.assertEqual(result["acceptance_decision"], "not_made")

    def test_missing_or_malformed_locust_evidence_is_explicit_and_not_a_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            (root / "locust_stats.csv").unlink()
            result = audit_run(spec, root)
            evidence = result["locust_count_and_percentile_evidence"]
            self.assertEqual(evidence["status"], "unavailable")
            self.assertEqual(evidence["csv"]["status"], "unavailable")
            self.assertTrue(any("unavailable" in reason for reason in result["reasons"]))

        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            (root / "locust_stderr.log").write_bytes(b"\xff\xfe")
            result = audit_run(spec, root)
            evidence = result["locust_count_and_percentile_evidence"]
            self.assertEqual(evidence["status"], "malformed")
            self.assertEqual(evidence["terminal"]["status"], "malformed")
            self.assertTrue(any("malformed" in reason for reason in result["reasons"]))

    def test_missing_shape_log_is_reported_as_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            (root / "locust.log").unlink()
            result = audit_run(spec, root)
            self.assertEqual(result["duration_evidence"]["locust_shape_interval"]["status"], "unavailable")
            self.assertTrue(any("shape start/stop duration evidence is unavailable" in reason for reason in result["reasons"]))

    def test_malformed_shape_timestamp_is_reported_as_malformed(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, spec, *_ = self.workspace(temp)
            (root / "locust.log").write_text(
                "[bad-time] INFO locust.runners: Shape test starting.\n",
                encoding="utf-8",
            )
            result = audit_run(spec, root)
            self.assertEqual(result["duration_evidence"]["locust_shape_interval"]["status"], "malformed")
            self.assertTrue(any("shape start/stop duration evidence is malformed" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
