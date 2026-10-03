import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_dpa import load_records, percentile, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/performance/results/data"
RAW_V1 = DATA / "2026-09-14-rtx5070ti-dpa.raw.jsonl"
RAW_V2 = DATA / "2026-09-15-rtx5070ti-splitkv.raw.jsonl"


class DpaSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = list(load_records(RAW_V1))
        cls.v2 = list(load_records(RAW_V2))

    def test_percentile_matches_cpp_linear_interpolation(self):
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.1), 1.3)
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.9), 3.7)

    def test_v1_archive_is_complete(self):
        summary = summarize(self.v1)
        self.assertEqual(sum(summary["record_counts"].values()), 5025)
        self.assertEqual(summary["record_counts"]["sample"], 4800)
        self.assertEqual(len(summary["shapes"]), 32)
        self.assertEqual(summary["equivalence"]["failures"], 0)

    def test_v2_archive_recomputes_large_window_and_convergence(self):
        summary = summarize(self.v2)
        self.assertEqual(sum(summary["record_counts"].values()), 20385)
        self.assertEqual(summary["record_counts"]["sample"], 19200)
        self.assertEqual(summary["equivalence"]["splitkv_records"], 480)
        self.assertEqual(summary["convergence"]["converged_shapes"], 4)
        large = next(
            shape
            for shape in summary["shapes"]
            if shape["shape"]["visible_tokens"] == 2048
            and shape["shape"]["block_size"] == 16
        )
        self.assertAlmostEqual(
            large["speedup"]["splitkv_vs_direct"]["16"], 6.02807, places=5
        )
        self.assertAlmostEqual(
            large["speedup"]["direct_splitkv_vs_legacy_splitkv"]["16"],
            1.26884,
            places=5,
        )
        short = summary["shapes"][0]
        self.assertFalse(short["converged"])
        self.assertIsNone(short["speedup"]["splitkv_vs_direct"]["16"])

    def test_missing_sample_is_rejected(self):
        records = list(self.v1)
        records.pop(
            next(i for i, record in enumerate(records) if record["type"] == "sample")
        )
        with self.assertRaisesRegex(ValueError, "incomplete sample count"):
            summarize(records)

    def test_incorrect_output_is_rejected(self):
        records = copy.deepcopy(self.v2)
        record = next(
            record for record in records if record["type"] == "equivalence_splitkv"
        )
        record["bitwise_equal"] = False
        with self.assertRaisesRegex(ValueError, "correctness failed"):
            summarize(records)

    def test_corrupted_aggregate_is_rejected(self):
        records = copy.deepcopy(self.v1)
        record = next(record for record in records if record["type"] == "path_stats")
        record["median_ms"] *= 2
        with self.assertRaisesRegex(ValueError, "logged median_ms disagrees"):
            summarize(records)

    def test_duplicate_correctness_record_is_rejected(self):
        records = list(self.v1)
        records.append(
            next(record for record in records if record["type"] == "equivalence")
        )
        with self.assertRaisesRegex(ValueError, "duplicate equivalence"):
            summarize(records)

    def test_unsupported_schema_is_rejected(self):
        records = copy.deepcopy(self.v1)
        records[0]["schema"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported raw schema"):
            summarize(records)

    def test_conflicting_completed_correctness_is_rejected(self):
        records = copy.deepcopy(self.v2)
        next(record for record in records if record["type"] == "shape_summary")[
            "equiv_ok"
        ] = False
        with self.assertRaisesRegex(
            ValueError, "completed shape reports correctness failure"
        ):
            summarize(records)

    def test_corrupted_repeat_median_is_rejected(self):
        records = copy.deepcopy(self.v1)
        next(record for record in records if record["type"] == "path_stats")[
            "repeat_medians_ms"
        ][0] *= 2
        with self.assertRaisesRegex(ValueError, "logged repeat_medians_ms disagrees"):
            summarize(records)

    def test_nonfinite_or_negative_latency_is_rejected(self):
        for latency in (math.nan, math.inf, -1.0, 0.0):
            with self.subTest(latency=latency):
                records = copy.deepcopy(self.v1)
                next(record for record in records if record["type"] == "sample")[
                    "ms"
                ] = latency
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    summarize(records)

    def test_cli_is_deterministic_and_refuses_to_overwrite(self):
        command = [sys.executable, str(ROOT / "scripts/summarize_dpa.py"), str(RAW_V1)]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(len(json.loads(first.stdout)["source_sha256"]), 64)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            subprocess.run([*command, "--output", str(output)], check=True)
            original = output.read_bytes()
            rejected = subprocess.run(
                [*command, "--output", str(output)], capture_output=True, check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
