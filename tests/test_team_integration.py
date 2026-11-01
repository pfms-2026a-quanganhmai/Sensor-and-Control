"""实际模型的联调回归测试；真值只在断言中作为验证依据。"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and (ROOT / "models").exists(),
    "integration requires the latest team model and mujoco",
)
class TeamIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import run_integrated

        cls.integration = run_integrated

    def make(self, obstacle=False):
        import mujoco

        model, robot_geoms = self.integration.build_model(obstacle)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data, robot_geoms

    def test_all_robot_geometries_excluded_and_walls_visible(self):
        module = self.integration
        model, data, robot_geoms = self.make()
        self.assertTrue(np.all(model.geom_group[robot_geoms] == 1))
        scan = module.MujocoPlanarLidar(
            model, "evaluation_lidar", excluded_geom_groups=(1,)
        ).read(data)
        result = module.evaluate_front(scan, half_angle_deg=35, stop_distance=0.35)
        self.assertEqual(result.status, "clear")
        self.assertEqual(result.valid_fraction, 1.0)

    def test_actual_obstacle_seen_before_contact(self):
        import mujoco

        module = self.integration
        model, data, _ = self.make(obstacle=True)
        data.qpos[0:2] = [0.3, 0.15]
        yaw = np.arctan2(0.5, 1.0)
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        scan = module.MujocoPlanarLidar(
            model, "evaluation_lidar", excluded_geom_groups=(1,)
        ).read(data)
        result = module.evaluate_front(scan, half_angle_deg=35, stop_distance=0.35)
        self.assertEqual(result.status, "blocked")
        self.assertGreater(result.nearest_distance_m, scan.range_min)

    def test_iterative_camera_pose_matches_independent_known_setup(self):
        import mujoco

        module = self.integration
        model, data, _ = self.make()
        camera = module.CameraPose(model)
        try:
            for x, y, yaw in ((0, 0, 0), (0.2, 0, 0), (0, 0.05, 0.05)):
                data.qpos[0:2] = [x, y]
                data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_forward(model, data)
                measured = camera.read(data)
                self.assertIsNotNone(measured)
                self.assertLess(np.linalg.norm(measured[:2] - [x, y]), 0.10)
                error = module.team.wrap_angle(measured[2] - yaw)
                self.assertLess(abs(error), np.deg2rad(8))
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
