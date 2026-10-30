"""日志记录器的软件测试；这些固定夹具不代表小车导航成绩。"""

import csv
import json
import math
import tempfile
import unittest

import numpy as np

from lidar_evaluation.logging import RAW_COLUMNS, TrialLogger
from lidar_evaluation.metrics import read_trials_csv


class TrialLoggerTests(unittest.TestCase):
    def make_logger(
        self, directory, trial_id="001", condition="software_fixture", **kwargs
    ):
        return TrialLogger(directory, trial_id, condition, 7, **kwargs)

    def sample(
        self, logger, time_s=0.0, truth=(0, 0, 0), estimate=None, covariance=None
    ):
        logger.record_sample(
            time_s,
            truth,
            truth if estimate is None else estimate,
            truth,
            np.eye(3) * 0.01 if covariance is None else covariance,
            control=(0.1, -0.2),
        )

    def test_known_metrics_raw_uncertainty_and_config_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"noise": {"std_m": 0.01}}
            logger = self.make_logger(directory, config=config)
            config["noise"]["std_m"] = 99
            self.sample(logger, 4, truth=(0, 0, 0), estimate=(0.1, 0, 0))
            self.sample(
                logger,
                6,
                truth=(1, 0, math.radians(179)),
                estimate=(1.1, 0, math.radians(-179)),
            )
            result = logger.finish([1, 0, math.radians(-179)], heading_tolerance_deg=3)
            self.assertTrue(result.success)
            self.assertEqual(result.duration_s, 2)
            self.assertAlmostEqual(result.estimation_rmse_m, 0.1)
            self.assertAlmostEqual(result.final_heading_error_deg, 2)
            with logger.raw_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), RAW_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["p00"]), 0.01)
            self.assertEqual(float(rows[0]["angular_command_rad_s"]), -0.2)
            manifest = json.loads(logger.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["noise"]["std_m"], 0.01)
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(len(manifest["raw_sha256"]), 64)
            self.assertEqual(read_trials_csv(logger.trials_path)[0], result)

    def test_collision_or_timeout_cannot_be_success_even_at_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            for identifier, kwargs in (
                ("collision", {"collision": True}),
                ("timeout", {"timed_out": True}),
            ):
                logger = self.make_logger(directory, trial_id=identifier)
                self.sample(logger)
                result = logger.finish([0, 0, 0], **kwargs)
                self.assertFalse(result.success)
                manifest = json.loads(logger.manifest_path.read_text(encoding="utf-8"))
                self.assertIn(identifier, manifest["computed_failure_reasons"])

    def test_goal_tolerances_are_checked_from_actual_final_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make_logger(directory)
            self.sample(logger, truth=(0.2, 0, math.radians(20)))
            result = logger.finish([0, 0, 0])
            self.assertFalse(result.success)
            self.assertAlmostEqual(result.final_position_error_m, 0.2)
            self.assertAlmostEqual(result.final_heading_error_deg, 20)

    def test_samples_are_copied_and_time_must_increase(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make_logger(directory)
            truth = np.zeros(3)
            covariance = np.eye(3)
            logger.record_sample(1, truth, truth, truth, covariance)
            truth[:] = 99
            covariance[:] = 99
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                self.sample(logger, time_s=1)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                self.sample(logger, time_s=0)
            result = logger.finish([0, 0, 0])
            self.assertTrue(result.success)
            with logger.raw_path.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["truth_x_m"]), 0)
            self.assertEqual(float(row["p00"]), 1)

    def test_covariance_and_shapes_rejected_before_sample_is_added(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make_logger(directory)
            for covariance in (
                np.eye(2),
                np.diag([1, 1, -0.1]),
                [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                np.full((3, 3), np.nan),
            ):
                with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                    self.sample(logger, covariance=covariance)
            with self.assertRaises(ValueError):
                self.sample(logger, truth=[0, 0])
            self.assertEqual(logger.sample_count, 0)

    def test_duplicate_finish_never_changes_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_logger(directory)
            self.sample(first)
            first.finish([0, 0, 0])
            paths = [first.raw_path, first.manifest_path, first.trials_path]
            snapshots = {path: path.read_bytes() for path in paths}
            second = self.make_logger(directory)
            self.sample(second, truth=[1, 0, 0])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                second.finish([0, 0, 0])
            for path in paths:
                self.assertEqual(path.read_bytes(), snapshots[path])
            with self.assertRaises(RuntimeError):
                first.finish([0, 0, 0])

    def test_append_preserves_prior_trials_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_logger(directory)
            self.sample(first)
            first.finish([0, 0, 0])
            with first.trials_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames + ["operator_note"]
                rows = list(reader)
            rows[0]["operator_note"] = "keep this original note"
            with first.trials_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            second = self.make_logger(directory, trial_id="002")
            self.sample(second)
            second.finish([0, 0, 0], timed_out=True)
            self.assertEqual(len(read_trials_csv(second.trials_path)), 2)
            with second.trials_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["operator_note"], "keep this original note")
            self.assertEqual(rows[1]["operator_note"], "")

    def test_unsafe_identifiers_and_invalid_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for identifier in ("../x", "a/b", r"a\b", "a__b", "", "x" * 65):
                with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                    self.make_logger(directory, trial_id=identifier)
            with self.assertRaises(ValueError):
                self.make_logger(directory, config={"noise": float("nan")})
            with self.assertRaises(ValueError):
                self.make_logger(directory, config={"array": np.zeros(2)})

    def test_empty_finish_invalid_flags_and_finished_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make_logger(directory)
            with self.assertRaisesRegex(ValueError, "without recorded"):
                logger.finish([0, 0, 0])
            self.sample(logger)
            with self.assertRaises(ValueError):
                logger.finish([0, 0, 0], collision="false")
            with self.assertRaises(ValueError):
                logger.finish([0, 0, 0], heading_tolerance_deg=181)
            logger.finish([0, 0, 0])
            with self.assertRaises(RuntimeError):
                self.sample(logger, 1)


if __name__ == "__main__":
    unittest.main()
