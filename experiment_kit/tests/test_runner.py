"""Contract tests for the K8sight run orchestrator."""
import tempfile
import unittest
from pathlib import Path

from experiment_kit.runner import (
    create_run_directory,
    outcome_from_exit_code,
    validate_run_id,
    validate_timeline,
    write_run_metadata,
)


class RunnerContractTests(unittest.TestCase):
    def test_valid_run_ids(self):
        for run_id in ("stable-low-r1-20261012", "spike-r2-test_01"):
            with self.subTest(run_id=run_id):
                self.assertIsNone(validate_run_id(run_id))

    def test_invalid_run_ids_are_rejected(self):
        for run_id in ("", ".", "..", "../other", "nested/run", "/tmp/run", "a b"):
            with self.subTest(run_id=run_id):
                with self.assertRaises((ValueError, TypeError)):
                    validate_run_id(run_id)

    def test_existing_run_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing-run"
            output.mkdir()
            sentinel = output / "preserve.txt"
            sentinel.write_text("original evidence", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_run_directory(output)

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "original evidence",
            )
            self.assertEqual(
                sorted(item.name for item in output.iterdir()),
                ["preserve.txt"],
            )

    def test_new_run_directory_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-run"
            result = create_run_directory(output)
            self.assertTrue(output.is_dir())
            self.assertEqual(Path(result), output)

    def test_valid_lifecycle_timeline(self):
        boundaries = {
            "run_start_utc": 100,
            "workload_end_utc": 220,
            "locust_shutdown_utc": 221,
            "cooldown_start_utc": 240,
            "collection_end_utc": 420,
        }
        self.assertIsNone(validate_timeline(boundaries))

    def test_invalid_lifecycle_order_is_rejected(self):
        boundaries = {
            "run_start_utc": 100,
            "workload_end_utc": 220,
            "locust_shutdown_utc": 210,
            "cooldown_start_utc": 240,
            "collection_end_utc": 420,
        }
        with self.assertRaises(ValueError):
            validate_timeline(boundaries)

    def test_failed_exit_is_not_completed(self):
        self.assertNotEqual(outcome_from_exit_code(1), "completed")
        self.assertNotEqual(outcome_from_exit_code(-1), "completed")

    def test_successful_exit_can_be_completed(self):
        self.assertEqual(outcome_from_exit_code(0), "completed")




    def test_write_run_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "metadata.json"
            metadata = {"run_id": "stable-low-r1-test", "pattern": "stable"}

            write_run_metadata(output, metadata)

            import json
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), metadata)

    def test_write_run_metadata_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "metadata.json"
            output.write_text('{"original": true}', encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_run_metadata(output, {"run_id": "replacement"})

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '{"original": true}',
            )
if __name__ == "__main__":
	unittest.main()
