"""检查统计定义与输入边界；测试夹具不是项目的机器人性能结果。"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lidar_evaluation.metrics import (
    CSV_COLUMNS,
    TrialRecord,
    read_trials_csv,
    summarize_trials,
    trajectory_metrics,
)
from lidar_evaluation.report import create_report


class EvaluationTests(unittest.TestCase):
    def write_csv(self, directory, rows, columns=CSV_COLUMNS):
        path = Path(directory) / "input.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, **updates):
        value = dict.fromkeys(CSV_COLUMNS, "")
        value.update(
            trial_id="one",
            condition="baseline",
            seed="1",
            success="false",
            collision="0",
            duration_s="5",
        )
        value.update(updates)
        return value

    def test_failure_stays_in_denominator_and_missing_error_is_not_zero(self):
        records = [
            TrialRecord("1", "baseline", 1, True, False, 2, final_position_error_m=0.2),
            TrialRecord("2", "baseline", 2, False, True, 8),
        ]
        result = summarize_trials(records)[0]
        self.assertEqual(result["trial_count"], 2)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["collision_rate"], 0.5)
        self.assertEqual(result["duration_s_count"], 2)
        self.assertEqual(result["duration_s_mean"], 5)
        self.assertEqual(result["duration_s_std"], 3)
        self.assertEqual(result["final_position_error_m_count"], 1)
        self.assertEqual(result["final_position_error_m_mean"], 0.2)
        self.assertIsNone(result["estimation_rmse_m_mean"])

    def test_false_string_and_blank_metric_read_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = read_trials_csv(self.write_csv(directory, [self.row()]))
        self.assertFalse(rows[0].success)
        self.assertFalse(rows[0].collision)
        self.assertIsNone(rows[0].tracking_rmse_m)

    def test_duplicate_id_within_condition_rejected_but_paired_conditions_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, [self.row(), self.row()])
            with self.assertRaisesRegex(ValueError, "duplicate trial_id"):
                read_trials_csv(path)
            path = self.write_csv(directory, [self.row(), self.row(condition="noise")])
            self.assertEqual(len(read_trials_csv(path)), 2)

    def test_invalid_values_are_not_silently_accepted(self):
        for field, value in (
            ("success", "maybe"),
            ("collision", ""),
            ("duration_s", "nan"),
            ("duration_s", "-1"),
            ("estimation_rmse_m", "inf"),
            ("seed", "1.5"),
            ("condition", " "),
            ("trial_id", ""),
        ):
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.write_csv(directory, [self.row(**{field: value})])
                    with self.assertRaisesRegex(ValueError, "CSV line 2"):
                        read_trials_csv(path)

    def test_empty_input_and_header_only_are_not_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no header"):
                read_trials_csv(path)
            with self.assertRaisesRegex(ValueError, "no trial records"):
                read_trials_csv(self.write_csv(directory, []))
        with self.assertRaisesRegex(ValueError, "zero trials"):
            summarize_trials([])

    def test_known_trajectory_errors_and_heading_wrap(self):
        truth = np.array([[0, 0, math.radians(179)], [3, 4, math.radians(-179)]])
        estimates = np.array([[0, 0, math.radians(-179)], [0, 0, math.radians(179)]])
        reference = np.array([[0, 0, 0], [3, 5, math.pi]])
        result = trajectory_metrics(truth, estimates, reference)
        self.assertAlmostEqual(result["estimation_rmse_m"], math.sqrt(12.5))
        self.assertAlmostEqual(result["estimation_heading_rmse_deg"], 2)
        self.assertAlmostEqual(result["tracking_rmse_m"], math.sqrt(0.5))
        self.assertAlmostEqual(result["final_position_error_m"], 1)
        self.assertAlmostEqual(result["final_heading_error_deg"], 1)

    def test_trajectory_shape_alignment_and_missing_data(self):
        with self.assertRaises(ValueError):
            trajectory_metrics(np.empty((0, 3)))
        with self.assertRaises(ValueError):
            trajectory_metrics([[0, 0, 0]], estimates=[[0, 0, 0], [1, 0, 0]])
        with self.assertRaises(ValueError):
            trajectory_metrics([[float("nan"), 0, 0]])
        self.assertNotIn("final_position_error_m", trajectory_metrics([[0, 0, 0]]))
        self.assertEqual(
            trajectory_metrics([[0, 0, 0]], goal=[3, 4, 0])["final_position_error_m"],
            5,
        )

    def test_report_has_traceable_source_and_blank_missing_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_csv(directory, [self.row()])
            result = create_report(source, Path(directory) / "report")
            with Path(result["summary_csv"]).open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["final_position_error_m_mean"], "")
            metadata = json.loads(
                Path(result["metadata_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["trial_count"], 1)
            self.assertEqual(len(metadata["input_sha256"]), 64)
            self.assertIn("including failures", metadata["rate_denominator"])

    def test_report_refuses_to_overwrite_its_input_log(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_csv(directory, [self.row()])
            target = Path(directory) / "summary.csv"
            source.rename(target)
            original = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                create_report(target, directory)
            self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
