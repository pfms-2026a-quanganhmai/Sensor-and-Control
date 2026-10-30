"""激光处理边界测试；这些是软件测试，不是机器人任务成功率。"""

import unittest

import numpy as np

from lidar_evaluation.lidar import ScanData, evaluate_front


class FrontObstacleTests(unittest.TestCase):
    def make_scan(self, ranges, angles=None):
        if angles is None:
            angles = np.linspace(-0.4, 0.4, len(ranges))
        return ScanData(angles=angles, ranges=ranges, range_min=0.1, range_max=5.0)

    def test_clear_scan_reports_nearest_and_coverage(self):
        result = evaluate_front(self.make_scan([1.5, 1.0, 2.0]))
        self.assertEqual(result.status, "clear")
        self.assertTrue(result.safe_to_move)
        self.assertEqual(result.nearest_distance_m, 1.0)
        self.assertEqual((result.valid_count, result.total_count), (3, 3))
        self.assertEqual(result.valid_fraction, 1.0)

    def test_threshold_is_inclusive(self):
        result = evaluate_front(self.make_scan([0.45, 2.0]))
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.safe_to_move)

    def test_close_obstacle_wins_even_with_low_coverage(self):
        result = evaluate_front(self.make_scan([np.nan, np.inf, 0.2, np.nan]))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.valid_fraction, 0.25)

    def test_below_minimum_front_hit_cannot_be_hidden_by_good_coverage(self):
        for under_range in (0.0, 0.02):
            with self.subTest(under_range=under_range):
                result = evaluate_front(self.make_scan([2.0, under_range, 2.0]))
                self.assertGreater(result.valid_fraction, 0.5)
                self.assertEqual(result.status, "unknown")
                self.assertFalse(result.safe_to_move)

    def test_valid_close_hit_takes_precedence_over_blind_zone_reading(self):
        result = evaluate_front(self.make_scan([0.02, 0.2, 2.0]))
        self.assertEqual(result.status, "blocked")

    def test_under_range_behind_robot_does_not_change_front_result(self):
        result = evaluate_front(self.make_scan([2.0, 0.02], [0.0, np.pi]))
        self.assertEqual(result.status, "clear")

    def test_stop_threshold_may_equal_sensor_range_limits(self):
        scan = self.make_scan([1.0])
        self.assertEqual(evaluate_front(scan, stop_distance=0.1).status, "clear")
        self.assertEqual(evaluate_front(scan, stop_distance=5.0).status, "blocked")

    def test_missing_beams_are_kept_in_coverage_denominator(self):
        result = evaluate_front(self.make_scan([2.0, np.nan, np.inf, -1.0]))
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.total_count, 4)
        self.assertFalse(result.safe_to_move)

    def test_all_invalid_including_no_return_is_unknown(self):
        for ranges in ([np.inf, np.inf], [np.nan, -np.inf], [0.05, 6.0]):
            with self.subTest(ranges=ranges):
                result = evaluate_front(self.make_scan(ranges))
                self.assertEqual(result.status, "unknown")
                self.assertIsNone(result.nearest_distance_m)
                self.assertFalse(result.safe_to_move)

    def test_empty_and_no_forward_beams_are_unknown(self):
        for scan in (self.make_scan([]), self.make_scan([0.2], [np.pi])):
            result = evaluate_front(scan)
            self.assertEqual(result.status, "unknown")
            self.assertEqual(result.total_count, 0)

    def test_back_obstacle_does_not_block_front_sector(self):
        scan = self.make_scan([2.0, 0.2, 2.0], [0.0, np.pi, 2 * np.pi - 0.2])
        result = evaluate_front(scan)
        self.assertEqual(result.status, "clear")
        self.assertEqual(result.total_count, 2)

    def test_both_angle_conventions_and_sector_edges(self):
        angles = np.deg2rad([330, 0, 30, 180])
        result = evaluate_front(self.make_scan([1.0, 1.0, 1.0, 0.2], angles))
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.status, "clear")

    def test_input_is_copied_and_read_only(self):
        ranges = np.array([1.0])
        scan = self.make_scan(ranges, [0.0])
        ranges[0] = 0.1
        self.assertEqual(scan.ranges[0], 1.0)
        with self.assertRaises(ValueError):
            scan.ranges[0] = 0.1

    def test_bad_shapes_angles_and_range_limits_are_rejected(self):
        inputs = [
            dict(angles=[0], ranges=[1, 2]),
            dict(angles=[[0]], ranges=[[1]]),
            dict(angles=[np.nan], ranges=[1]),
            dict(angles=[0], ranges=[1], range_min=-1),
            dict(angles=[0], ranges=[1], range_min=1, range_max=1),
            dict(angles=[0], ranges=[1], range_max=np.inf),
        ]
        for values in inputs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ScanData(**values)

    def test_bad_configuration_is_rejected(self):
        scan = self.make_scan([1.0])
        inputs = [
            {"half_angle_deg": 0},
            {"half_angle_deg": 181},
            {"half_angle_deg": np.nan},
            {"stop_distance": -1},
            {"stop_distance": np.inf},
            {"stop_distance": 0.05},
            {"stop_distance": 5.01},
            {"min_valid_fraction": 0},
            {"min_valid_fraction": 1.1},
        ]
        for values in inputs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_front(scan, **values)


if __name__ == "__main__":
    unittest.main()
