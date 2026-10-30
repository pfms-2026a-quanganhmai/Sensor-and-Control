"""检查校正方向、验证集误差和无效输入；人工数据仅用于软件单元测试。"""

import unittest

import numpy as np

from lidar_evaluation.calibration import (
    RangeCalibration,
    evaluate_calibration,
    fit_range_calibration,
    range_error_metrics,
)


class CalibrationTests(unittest.TestCase):
    def test_fit_maps_measured_to_reference(self):
        reference = np.array([0.5, 1.0, 2.0, 3.0])
        measured = reference * 1.08 + 0.03
        calibration = fit_range_calibration(reference, measured)
        self.assertAlmostEqual(calibration.slope, 1 / 1.08)
        self.assertAlmostEqual(calibration.intercept, -0.03 / 1.08)
        np.testing.assert_allclose(calibration.apply(measured), reference, atol=1e-12)

    def test_two_distinct_points_are_sufficient_for_linear_fit(self):
        calibration = fit_range_calibration([1.0, 2.0], [1.1, 2.2])
        self.assertAlmostEqual(float(calibration.apply(1.65)), 1.5)

    def test_error_sign_and_population_standard_deviation(self):
        metrics = range_error_metrics([1.0, 1.0], [0.9, 1.3])
        self.assertEqual(metrics.count, 2)
        self.assertAlmostEqual(metrics.mae_m, 0.2)
        self.assertAlmostEqual(metrics.rmse_m, np.sqrt(0.05))
        self.assertAlmostEqual(metrics.bias_m, 0.1)
        self.assertAlmostEqual(metrics.std_m, 0.2)

    def test_independent_validation_set(self):
        calibration = fit_range_calibration([0.5, 1.0, 2.0], [0.6, 1.1, 2.1])
        # 这里的验证距离与上面的拟合距离不同，避免只检查拟合残差。
        report = evaluate_calibration([0.8, 1.6], [0.9, 1.7], calibration)
        self.assertAlmostEqual(report["before"].bias_m, 0.1)
        self.assertLess(report["after"].rmse_m, 1e-12)

    def test_apply_preserves_missing_measurement_markers(self):
        calibration = RangeCalibration(1.0, -0.1)
        result = calibration.apply([1.1, np.nan, np.inf, -np.inf])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isposinf(result[2]))
        self.assertTrue(np.isneginf(result[3]))

    def test_degenerate_and_reversed_fits_are_rejected(self):
        for reference, measured in (([1, 2], [1, 1]), ([1, 2], [2, 1])):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                fit_range_calibration(reference, measured)

    def test_bad_pairs_are_rejected_without_silent_filtering(self):
        pairs = [([], []), ([1, 2], [1]), ([[1]], [[1]]), ([1], [np.nan]), ([-1], [1])]
        for reference, measured in pairs:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                range_error_metrics(reference, measured)
        with self.assertRaises(ValueError):
            fit_range_calibration([1], [1])

    def test_invalid_calibration_parameters_are_rejected(self):
        for slope, intercept in ((0, 0), (-1, 0), (np.inf, 0), (1, np.nan)):
            with self.subTest(slope=slope), self.assertRaises(ValueError):
                RangeCalibration(slope, intercept)


if __name__ == "__main__":
    unittest.main()
