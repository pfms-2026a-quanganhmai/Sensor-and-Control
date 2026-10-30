"""图表集成的软件夹具测试；不产生可冒充导航实验的交付数据。"""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lidar_evaluation.logging import TrialLogger
from lidar_evaluation.plotting import plot_logged_trial
from lidar_evaluation.report import create_report


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib"), "optional matplotlib is not installed"
)
class PlottingTests(unittest.TestCase):
    def log_fixture(self, directory):
        logger = TrialLogger(directory, "fixture001", "software_test_only", 0)
        for stamp, x in ((0, 0), (1, 1)):
            logger.record_sample(
                stamp, [x, 0, 0], [x + 0.05, 0, 0], [x, 0, 0], np.eye(3) * 0.01
            )
        logger.finish([1, 0, 0])
        return logger

    def test_report_and_uncertainty_plots_from_logged_known_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.log_fixture(directory)
            output = Path(directory) / "plot_output"
            images = plot_logged_trial(logger.raw_path, output)
            create_report(logger.trials_path, output, plots=True)
            images += [output / "success_rate.png", output / "final_position_error.png"]
            for image in images:
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(image.stat().st_size, 1000)

    def test_invalid_covariance_in_external_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.log_fixture(directory)
            with logger.raw_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["p00"] = -1
            broken = Path(directory) / "broken.csv"
            with broken.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "positive semidefinite"):
                plot_logged_trial(broken, Path(directory) / "plots")


if __name__ == "__main__":
    unittest.main()
