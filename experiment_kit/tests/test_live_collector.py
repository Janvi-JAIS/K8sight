import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from experiment_kit.collector.live_collector import (
    collect,
    parse_single_series,
)


def response_for(metric_name, start, end, step):
    values = []
    t = start
    while t <= end:
        if metric_name == "container_memory_working_set_bytes":
            value = "1048576"
        elif metric_name == "container_cpu_usage_seconds_total":
            value = "0.05"
        else:
            value = "3"
        values.append([t, value])
        t += step

    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{
                "metric": {
                    "__name__": metric_name,
                    "namespace": "default",
                    "pod": "test-pod",
                    "container": "app",
                },
                "values": values,
            }],
        },
    }


class LiveCollectorTests(unittest.TestCase):
    def test_parse_single_series(self):
        doc = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {},
                    "values": [[1000, "0.25"], [1030, "0.5"]],
                }],
            },
        }
        self.assertEqual(
            parse_single_series(doc, "sample_metric"),
            {1000.0: 0.25, 1030.0: 0.5},
        )

    def test_ambiguous_series_rejected(self):
        doc = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {}, "values": [[1000, "1"]]},
                    {"metric": {}, "values": [[1000, "2"]]},
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "returned 2 series"):
            parse_single_series(doc, "sample_metric")

    def test_duplicate_timestamp_rejected(self):
        doc = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {},
                    "values": [[1000, "1"], [1000, "2"]],
                }],
            },
        }
        with self.assertRaisesRegex(ValueError, "duplicate sample timestamp"):
            parse_single_series(doc, "sample_metric")

    def test_collect_archives_responses_and_hashes(self):
        start, end, step = 1000, 1060, 30

        def fake_get(url, params, timeout):
            metric_name = params["query"].split("{", 1)[0]
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = response_for(
                metric_name, start, end, step
            )
            return response

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-run"
            with patch("experiment_kit.collector.live_collector.requests.Session") as session_factory:
                session = session_factory.return_value
                session.get.side_effect = fake_get

                summary = collect(
                    prometheus_url="http://127.0.0.1:9090",
                    namespace="default",
                    pod="test-pod",
                    container="app",
                    start=start,
                    end=end,
                    step=step,
                    lookback=60,
                    output_dir=output,
                )

            self.assertEqual(len(summary["metrics"]), 5)
            self.assertTrue((output / "collection_summary.json").is_file())

            for metric in summary["metrics"]:
                self.assertEqual(metric["availability"], "available")
                source = metric["raw_sources"][0]
                raw_file = output / source["path"]
                self.assertTrue(raw_file.is_file())
                actual_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
                self.assertEqual(source["sha256"], actual_hash)
                json.loads(raw_file.read_text(encoding="utf-8"))

    def test_parse_failure_preserves_archived_raw_evidence(self):
        start, end, step = 1000, 1060, 30

        def fake_get(url, params, timeout):
            metric_name = params["query"].split("{", 1)[0]
            response = Mock()
            response.raise_for_status.return_value = None
            document = response_for(metric_name, start, end, step)

            # Return two series for one metric to force parsing to fail
            # after the raw response has already been archived.
            if metric_name == "container_memory_working_set_bytes":
                document["data"]["result"].append(
                    document["data"]["result"][0].copy()
                )

            response.json.return_value = document
            return response

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new-run"

            with patch(
                "experiment_kit.collector.live_collector.requests.Session"
            ) as session_factory:
                session = session_factory.return_value
                session.get.side_effect = fake_get

                summary = collect(
                    prometheus_url="http://127.0.0.1:9090",
                    namespace="default",
                    pod="test-pod",
                    container="app",
                    start=start,
                    end=end,
                    step=step,
                    lookback=60,
                    output_dir=output,
                )

            metric = next(
                item for item in summary["metrics"]
                if item["name"] == "container_memory_working_set_bytes"
            )
            self.assertEqual(metric["availability"], "unknown")
            self.assertTrue(metric.get("error"))

            # The evidence reference and hash must survive the parse failure.
            self.assertEqual(len(metric["raw_sources"]), 1)
            source_info = metric["raw_sources"][0]
            raw_file = output / source_info["path"]
            self.assertTrue(raw_file.is_file())
            self.assertEqual(
                source_info["sha256"],
                hashlib.sha256(raw_file.read_bytes()).hexdigest(),
            )

    def test_existing_output_directory_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                collect(
                    prometheus_url="http://127.0.0.1:9090",
                    namespace="default",
                    pod="test-pod",
                    container="app",
                    start=1000,
                    end=1060,
                    step=30,
                    lookback=60,
                    output_dir=output,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
