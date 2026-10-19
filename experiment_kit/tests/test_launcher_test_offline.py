"""Offline unit tests for launcher metadata and collection boundaries."""
import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_kit.launcher_test import (
    build_parser,
    enforce_cpu_headroom,
    persist_failure,
    update_json_atomic,
    utc_at_epoch,
)


class LauncherOfflineTests(unittest.TestCase):
    def test_metadata_update_replaces_only_existing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = Path(temp) / "launcher_test_metadata.json"
            metadata.write_text('{"status":"preflight"}\n', encoding="utf-8")
            update_json_atomic(metadata, {"status": "running", "events": [1]})

            self.assertEqual(
                json.loads(metadata.read_text(encoding="utf-8")),
                {"status": "running", "events": [1]},
            )
            self.assertEqual([path.name for path in Path(temp).iterdir()], [metadata.name])

    def test_utc_boundaries_are_derived_from_collector_epochs(self):
        collection_start_epoch = 1_791_904_800
        collection_end_epoch = 1_791_905_100
        collection_start_utc = utc_at_epoch(collection_start_epoch)
        collection_end_utc = utc_at_epoch(collection_end_epoch)

        self.assertEqual(collection_start_utc, "2026-10-13T15:20:00Z")
        self.assertEqual(collection_end_utc, "2026-10-13T15:25:00Z")
        self.assertEqual(
            collection_end_epoch - collection_start_epoch,
            300,
        )

    def test_cli_has_no_cpu_gate_bypass_option(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                build_parser().parse_args(["--skip-cpu-gate"])
        self.assertEqual(raised.exception.code, 2)

    def test_cpu_gate_enforces_300m_headroom_against_observed_limit(self):
        self.assertEqual(enforce_cpu_headroom(0.2, "500m"), 500)
        with self.assertRaisesRegex(RuntimeError, "300m headroom"):
            enforce_cpu_headroom(0.201, "500m")

    def test_metadata_write_failure_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata_path = root / "launcher_test_metadata.json"
            failure_path = root / "launcher_test_failure.json"
            metadata_path.write_text('{"status":"running"}\n', encoding="utf-8")
            metadata = {"status": "running"}

            with patch(
                "experiment_kit.launcher_test.update_json_atomic",
                side_effect=OSError("atomic replacement failed"),
            ), contextlib.redirect_stderr(io.StringIO()):
                persist_failure(
                    metadata, metadata_path, failure_path,
                    RuntimeError("Pod identity changed"),
                )

            record = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(record["failure"]["error"], "Pod identity changed")
            self.assertEqual(record["metadata_update_error"], "atomic replacement failed")


if __name__ == "__main__":
    unittest.main()
