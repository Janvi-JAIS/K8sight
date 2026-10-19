"""Offline safety tests for the real experiment wrapper."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiment_kit.generators.config import WorkloadConfig
from experiment_kit.launcher import (
    GENERATOR,
    TARGET_URL,
    command_environment,
    create_run_directory,
    enforce_cpu_headroom,
    locust_command,
    parse_cpu_observation,
    persist_failure,
    record_boundary,
    run_experiment,
    utc_at,
    verify_same_target,
)


class LauncherOfflineTests(unittest.TestCase):
    @staticmethod
    def target(**changes):
        target = {
            "name": "pod-a",
            "uid": "pod-uid-a",
            "container_id": "containerd://container-a",
            "ready": True,
            "resources": {"limits": {"cpu": "500m"}},
        }
        target.update(changes)
        return target

    def execute_mocked_run(self, root, targets, cpu_results, trace):
        runs_dir = root / "run_records"
        runs_dir.mkdir()
        target_iter = iter(targets)
        cpu_iter = iter(cpu_results)

        def fake_target():
            trace.append("target")
            value = next(target_iter)
            if isinstance(value, Exception):
                raise value
            return value

        def fake_cpu(*args):
            trace.append("cpu")
            value = next(cpu_iter)
            if isinstance(value, Exception):
                raise value
            return {
                "cpu_cores": value,
                "source_sample_epoch": 1_791_904_800,
                "source_sample_utc": "2026-10-13T15:20:00Z",
                "source_sample_age_seconds": 5,
                "freshness_checked_at_epoch": 1_791_904_805,
                "freshness_checked_at_utc": "2026-10-13T15:20:05Z",
                "freshness_range_start_epoch": 1_791_904_775,
                "freshness_range_end_epoch": 1_791_904_805,
                "freshness_step_seconds": 5,
                "series_labels": {},
            }

        def fake_locust(*args, **kwargs):
            trace.append("locust")
            return 0

        collector_result = Mock(returncode=0, stdout="collector ok", stderr="")
        config = WorkloadConfig(
            run_id="stable-low-offline-001", pattern="stable", load_level="low",
            replicate=1, random_seed=42, duration_seconds=300, target_host=TARGET_URL,
        )
        with (
            patch("experiment_kit.launcher.PYTHON", Path(sys.executable)),
            patch("experiment_kit.launcher.kubectl_target", side_effect=fake_target),
            patch("experiment_kit.launcher.observe_cpu", side_effect=fake_cpu),
            patch("experiment_kit.launcher.time.sleep", side_effect=lambda seconds: trace.append(("sleep", seconds))),
            patch("experiment_kit.launcher.run_locust", side_effect=fake_locust),
            patch("experiment_kit.launcher.subprocess.run", return_value=collector_result),
        ):
            result = run_experiment(
                config, runs_dir=runs_dir, baseline_seconds=120, cooldown_seconds=180
            )
        run_dir = runs_dir / config.run_id
        metadata = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
        return result, metadata, trace

    def test_existing_run_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "existing-run"
            run_dir.mkdir()
            artifact = run_dir / "keep.txt"
            artifact.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_run_directory(run_dir)
            self.assertEqual(artifact.read_text(encoding="utf-8"), "preserve")

    def test_cpu_gate_requires_300m_headroom_and_finite_observation(self):
        self.assertEqual(enforce_cpu_headroom(0.2, "500m"), 500)
        for observed in (0.201, float("nan")):
            with self.assertRaisesRegex(RuntimeError, "CPU safety gate failed"):
                enforce_cpu_headroom(observed, "500m")
        for limit in ("250m", "nan"):
            with self.assertRaisesRegex(RuntimeError, "CPU safety gate failed"):
                enforce_cpu_headroom(0.0, limit)

    def test_cpu_metric_must_be_fresh_unique_and_labeled_for_target(self):
        labels = {
            "namespace": "default", "pod": "pod-a", "container": "hpa-example",
            "name": "cid-123",
        }
        cpu_document = {
            "status": "success",
            "data": {"result": [{"metric": labels, "value": [1010, "0.1"]}]},
        }
        fresh_document = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [{
                "metric": labels, "values": [[1000, "1000"], [1010, "1005"]],
            }]},
        }
        self.assertEqual(
            parse_cpu_observation(
                cpu_document, fresh_document, "pod-a", "containerd://cid-123",
                "pod-uid", 1010,
            )["cpu_cores"], 0.1,
        )
        # The instant result timestamp is recent (1010), but the actual source
        # timestamp returned as the timestamp() range-query value is stale (950).
        stale_source = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [{
                "metric": labels, "values": [[980, "950"], [1010, "950"]],
            }]},
        }
        with self.assertRaisesRegex(RuntimeError, "source samples are stale"):
            parse_cpu_observation(
                cpu_document, stale_source, "pod-a", "containerd://cid-123",
                "pod-uid", 1010,
            )

        empty_source = {"status": "success", "data": {"resultType": "matrix", "result": []}}
        ambiguous_source = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [
                fresh_document["data"]["result"][0], fresh_document["data"]["result"][0],
            ]},
        }
        mismatch_source = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [{
                "metric": {**labels, "name": "other-container"},
                "values": [[1000, "1000"], [1010, "1005"]],
            }]},
        }
        for broken in (
            empty_source, ambiguous_source, mismatch_source,
            {"status": "success", "data": {"resultType": "matrix", "result": [{
                "metric": labels, "values": [],
            }]}},
        ):
            with self.assertRaises(RuntimeError):
                parse_cpu_observation(
                    cpu_document, broken, "pod-a", "containerd://cid-123",
                    "pod-uid", 1010,
                )

    def test_cpu_metric_matches_live_cgroup_container_and_pod_uid_labels(self):
        runtime_id = "cb643dc8cd90083ed93af50b3919eb3f4b35903961f9ab89434828f2e32530a9"
        pod_uid = "b0a1e8c1-37ea-4995-881b-c8eb50b5de1d"
        labels = {
            "namespace": "default",
            "pod": "php-apache-85668bb794-z45cl",
            "container": "hpa-example",
            "name": runtime_id,
            "id": (
                "/kubelet.slice/kubelet-kubepods.slice/"
                "/kubelet-kubepods-burstable-podb0a1e8c1_37ea_4995_881b_"
                "c8eb50b5de1d.slice/cri-containerd-"
                f"{runtime_id}.scope"
            ),
        }
        cpu_document = {
            "status": "success",
            "data": {"result": [{"metric": labels, "value": [1_791_904_805, "0.000102"]}]},
        }
        freshness_document = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [{
                "metric": labels,
                "values": [[1_791_904_800, "1791904800"], [1_791_904_805, "1791904803"]],
            }]},
        }
        observation = parse_cpu_observation(
            cpu_document, freshness_document,
            "php-apache-85668bb794-z45cl",
            f"containerd://{runtime_id}", pod_uid, 1_791_904_805,
        )
        self.assertAlmostEqual(observation["cpu_cores"], 0.000102)
        self.assertEqual(observation["source_sample_epoch"], 1_791_904_803)
        self.assertEqual(observation["source_sample_age_seconds"], 2)
        with self.assertRaisesRegex(RuntimeError, "Pod/container"):
            parse_cpu_observation(
                cpu_document,
                freshness_document,
                "php-apache-85668bb794-z45cl",
                f"containerd://{runtime_id}",
                "different-pod-uid",
                1_791_904_805,
            )

    def test_identity_must_match_both_pod_uid_and_container_id(self):
        target = self.target()
        verify_same_target(target, dict(target))
        with self.assertRaisesRegex(RuntimeError, "UID or container ID"):
            verify_same_target(target, self.target(uid="new-uid"))
        with self.assertRaisesRegex(RuntimeError, "UID or container ID"):
            verify_same_target(target, self.target(container_id="new-container"))

    def test_second_cpu_gate_runs_after_baseline_before_locust_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = []
            target = self.target()
            result, metadata, trace = self.execute_mocked_run(
                Path(temp), [target, target, target, target], [0.1, 0.15], trace
            )
            baseline_index = trace.index(("sleep", 120))
            second_cpu_index = [i for i, event in enumerate(trace) if event == "cpu"][1]
            locust_index = trace.index("locust")
            self.assertEqual(result, 0)
            self.assertLess(baseline_index, second_cpu_index)
            self.assertLess(second_cpu_index, locust_index)
            self.assertFalse(metadata["training_eligible"])
            gate = metadata["cpu_safety_checks"][1]
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["cpu_observed_cores"], 0.15)
            self.assertEqual(gate["cpu_limit"], "500m")
            self.assertEqual(gate["cpu_headroom_millicores"], 350)
            self.assertTrue(gate["sample_observed_at_utc"].endswith("Z"))
            self.assertTrue(gate["finished_at_utc"].endswith("Z"))

    def test_insufficient_second_gate_prevents_locust_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = []
            target = self.target()
            result, metadata, trace = self.execute_mocked_run(
                Path(temp), [target, target, target], [0.1, 0.25], trace
            )
            self.assertEqual(result, 1)
            self.assertNotIn("locust", trace)
            self.assertEqual(metadata["status"], "failed")
            self.assertFalse(metadata["cpu_safety_checks"][1]["passed"])
            self.assertIn("300m", metadata["cpu_safety_checks"][1]["failure"])

    def test_stale_or_mismatched_second_sample_prevents_locust(self):
        for message in ("stale CPU sample", "mismatched CPU identity"):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                trace = []
                target = self.target()
                result, metadata, trace = self.execute_mocked_run(
                    Path(temp), [target, target, target], [0.1, RuntimeError(message)], trace
                )
                self.assertEqual(result, 1)
                self.assertNotIn("locust", trace)
                self.assertEqual(metadata["status"], "failed")
                self.assertIn(message, metadata["cpu_safety_checks"][1]["failure"])

    def test_changed_identity_or_cpu_limit_prevents_second_gate_and_locust(self):
        for key, changed_value in (
            ("uid", "pod-uid-new"),
            ("container_id", "containerd://container-new"),
            ("resources", {"limits": {"cpu": "600m"}}),
            ("ready", False),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                trace = []
                initial = self.target()
                changed = self.target(**{key: changed_value})
                result, metadata, trace = self.execute_mocked_run(
                    Path(temp), [initial, initial, changed], [0.1, 0.1], trace
                )
                self.assertEqual(result, 1)
                self.assertNotIn("locust", trace)
                self.assertEqual(metadata["status"], "failed")
                self.assertFalse(metadata["cpu_safety_checks"][1]["passed"])

    def test_collection_boundaries_have_consistent_epoch_and_utc_values(self):
        boundaries = {}
        expected = {
            "baseline_start": 1_791_904_800,
            "collection_start": 1_791_904_800,
            "baseline_end": 1_791_904_920,
            "run_start": 1_791_904_921,
            "workload_end": 1_791_905_221,
            "locust_shutdown": 1_791_905_221,
            "cooldown_start": 1_791_905_221,
            "cooldown_end": 1_791_905_401,
            "collection_end": 1_791_905_402,
            "collector_finished": 1_791_905_410,
        }
        for name, epoch in expected.items():
            record_boundary(boundaries, name, epoch)
            self.assertEqual(boundaries[f"{name}_utc"], utc_at(epoch))
            self.assertEqual(boundaries[f"{name}_epoch"], epoch)
        self.assertEqual(boundaries["collection_start_epoch"], expected["baseline_start"])
        self.assertLess(boundaries["baseline_end_epoch"], boundaries["run_start_epoch"])
        self.assertEqual(boundaries["workload_end_epoch"], boundaries["cooldown_start_epoch"])
        self.assertLess(boundaries["cooldown_end_epoch"], boundaries["collection_end_epoch"])
        self.assertLess(boundaries["collection_end_epoch"], boundaries["collector_finished_epoch"])

    def test_failure_is_recorded_and_never_marked_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata_path = Path(temp) / "experiment_metadata.json"
            metadata_path.write_text('{"status":"running"}\n', encoding="utf-8")
            metadata = {"status": "running"}
            persist_failure(metadata, metadata_path, RuntimeError("identity changed"))
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["error"], "identity changed")

    def test_failed_metadata_update_preserves_original_error_in_failure_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata_path = Path(temp) / "experiment_metadata.json"
            metadata_path.write_text('{"status":"running"}\n', encoding="utf-8")
            metadata = {"status": "running"}
            with patch(
                "experiment_kit.launcher.atomic_json",
                side_effect=OSError("metadata disk error"),
            ), contextlib.redirect_stderr(io.StringIO()):
                persist_failure(metadata, metadata_path, RuntimeError("CPU gate failed"))
            failure = json.loads((Path(temp) / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["failure"]["error"], "CPU gate failed")
            self.assertEqual(failure["metadata_update_error"], "metadata disk error")
            self.assertEqual(failure["status"], "failed")

    def test_generator_parameters_and_target_are_passed_explicitly(self):
        config = WorkloadConfig(
            run_id="experiment-001", pattern="spike", load_level="low",
            replicate=2, random_seed=17, duration_seconds=240,
            target_host=TARGET_URL,
        )
        env = command_environment(config, base={"K8SIGHT_CUSTOM_PARAMS": "{}"})
        command = locust_command(Path("/tmp/fresh-run"), config)
        self.assertEqual(env["K8SIGHT_PATTERN"], "spike")
        self.assertEqual(env["K8SIGHT_LOAD_LEVEL"], "low")
        self.assertEqual(env["K8SIGHT_REPLICATE"], "2")
        self.assertEqual(env["K8SIGHT_SEED"], "17")
        self.assertEqual(env["K8SIGHT_DURATION"], "240")
        self.assertEqual(env["K8SIGHT_TARGET_HOST"], TARGET_URL)
        self.assertNotIn("K8SIGHT_CUSTOM_PARAMS", env)
        self.assertEqual(command[command.index("-f") + 1], str(GENERATOR))
        self.assertEqual(command[command.index("--host") + 1], TARGET_URL)
        self.assertIn("240s", command)


if __name__ == "__main__":
    unittest.main()
