"""规划、控制、估计的独立已知答案与回归检查。"""

import unittest

import numpy as np

from navigation.analysis import analyse
from navigation.control import WheelServo, visual_dock
from navigation.estimation import NavigationEKF
from navigation.planner import OccupancyPlanner


class NavigationTests(unittest.TestCase):
    def test_moving_model_controllable_observable_and_lqr_stable(self):
        result = analyse()
        self.assertEqual(result["controllability_rank"], 3)
        self.assertEqual(result["camera_observability_rank"], 3)
        self.assertEqual(result["stationary_linearisation_controllability_rank"], 2)
        self.assertLess(result["closed_loop_spectral_radius"], 1)

    def test_planner_routes_around_measured_occupancy(self):
        planner = OccupancyPlanner()
        for x in np.arange(0.4, 0.65, 0.05):
            for y in np.arange(-0.1, 0.2, 0.05):
                planner.logodds[planner.cell([x, y])] = 5
        route = planner.plan(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        self.assertIsNotNone(route)
        self.assertTrue(np.max(np.abs(route[:, 1])) > 0.25)
        blocked = planner.blocked()
        for a, b in zip(route[:-1], route[1:]):
            self.assertFalse(
                any(blocked[c] for c in planner.line(planner.cell(a), planner.cell(b)))
            )

    def test_occupied_destination_is_rejected(self):
        planner = OccupancyPlanner()
        planner.logodds[planner.cell([1, 0])] = 5
        self.assertIsNone(planner.plan(np.array([0.0, 0.0]), np.array([1.0, 0.0])))

    def test_ekf_pose_update_reduces_error_and_rejects_outlier(self):
        ekf = NavigationEKF([0, 0, 0])
        ekf.predict(0.05, 0.05, 0.02)
        self.assertTrue(ekf.update([0.04, 0, 0]))
        self.assertLess(abs(ekf.x[0] - 0.04), 0.01)
        snapshot = ekf.x.copy()
        self.assertFalse(ekf.update([20, 20, 2]))
        np.testing.assert_array_equal(snapshot, ekf.x)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(ekf.P)), 0)

    def test_wheel_servo_compensates_breakaway_without_changing_physics(self):
        servo = WheelServo()
        command = servo.command(0.02, 0, np.zeros(2))
        self.assertTrue(np.all(command > 1))
        np.testing.assert_array_equal(servo.command(0, 0, np.zeros(2)), [0, 0])

    def test_docking_alignment_at_goal(self):
        v, w = visual_dock(np.array([1.05, 0, 0.2]), np.array([1.05, 0, 0]))
        self.assertEqual(v, 0)
        self.assertLess(w, 0)

    def test_missing_lidar_always_stops_forward_and_reverse(self):
        from lidar_evaluation import ScanData
        from navigation.runner import safe_command

        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        scan = ScanData(angles, np.full(180, np.nan))
        for command in ((0.1, 0), (-0.04, 0), (0, 0.5)):
            self.assertEqual(safe_command(command, scan)[0], (0, 0))

    def test_full_camera_transform_handles_body_pitch(self):
        from pathlib import Path

        import mujoco

        from navigation.runner import scene
        from navigation.vision import CalibratedCamera

        root = Path(__file__).resolve().parents[1]
        model, _ = scene(root, "nominal", 0)
        data = mujoco.MjData(model)
        camera = CalibratedCamera(model)
        try:
            data.qpos[0] = 0.6
            data.qpos[3:7] = [np.cos(0.025), 0, np.sin(0.025), 0]
            mujoco.mj_forward(model, data)
            pose = camera.read(data)
            self.assertIsNotNone(pose)
            self.assertLess(np.linalg.norm(pose[:2] - [0.6, 0]), 0.025)
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
