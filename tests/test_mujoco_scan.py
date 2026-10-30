"""通过解析上可知的场景验证 MuJoCo 传感器适配器的坐标系和量程。"""

import importlib.util
import unittest

import numpy as np

from lidar_evaluation import evaluate_front


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"), "optional mujoco is not installed"
)
class MujocoLidarTests(unittest.TestCase):
    def make(self, body: str):
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            f"<mujoco><worldbody>{body}</worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def test_known_front_face_is_surface_distance_not_center_distance(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make("""<site name="laser"/>
            <geom type="box" pos="1.1 0 0" size="0.1 1 1"/>""")
        scan = MujocoPlanarLidar(model, "laser", beam_count=4).read(data)
        self.assertAlmostEqual(scan.ranges[2], 1.0)

    def test_rotated_sensor_uses_local_forward_axis(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make("""<site name="laser" euler="0 0 90"/>
            <geom type="box" pos="0 2.1 0" size="1 0.1 1"/>""")
        scan = MujocoPlanarLidar(model, "laser", beam_count=4).read(data)
        self.assertAlmostEqual(scan.ranges[2], 2.0)

    def test_no_echo_is_unknown_instead_of_infinite_clearance(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make('<site name="laser"/>')
        scan = MujocoPlanarLidar(model, "laser").read(data)
        self.assertTrue(np.all(np.isinf(scan.ranges)))
        self.assertEqual(evaluate_front(scan).status, "unknown")

    def test_mount_yaw_converts_sensor_angles_to_robot_angles(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make("""<site name="laser" euler="0 0 90"/>
            <geom type="box" pos="0 0.3 0" size="1 0.05 1"/>
            <geom type="box" pos="2.1 0 0" size="0.1 1 1"/>""")
        scan = MujocoPlanarLidar(
            model, "laser", beam_count=4, mount_yaw_rad=np.pi / 2
        ).read(data)
        # 传感器正前方的近障碍在车体左方；车体正前方仍是两米外的平面。
        self.assertAlmostEqual(scan.angles[2], np.pi / 2)
        self.assertEqual(evaluate_front(scan).status, "clear")
        self.assertAlmostEqual(evaluate_front(scan).nearest_distance_m, 2.0)

    def test_excluded_geometry_group_removes_self_hit(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make("""<site name="laser"/>
            <geom type="box" pos="0.2 0 0" size="0.05 0.1 0.1" group="1"/>
            <geom type="box" pos="1.1 0 0" size="0.1 1 1"/>""")
        scan = MujocoPlanarLidar(
            model, "laser", beam_count=4, excluded_geom_groups=(1,)
        ).read(data)
        self.assertAlmostEqual(scan.ranges[2], 1.0)

    def test_maximum_range_clips_far_return_to_unknown(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, data = self.make("""<site name="laser"/>
            <geom type="box" pos="3 0 0" size="0.1 1 1"/>""")
        scan = MujocoPlanarLidar(model, "laser", beam_count=4, range_max=2).read(data)
        self.assertTrue(np.isinf(scan.ranges[2]))

    def test_bad_site_and_scanner_configuration_fail_early(self):
        from lidar_evaluation.mujoco_scan import MujocoPlanarLidar

        model, _ = self.make('<site name="laser"/>')
        for kwargs in (
            {"site_name": "absent"},
            {"site_name": "laser", "beam_count": 3},
            {"site_name": "laser", "excluded_geom_groups": (6,)},
            {"site_name": "laser", "range_min": 2, "range_max": 1},
        ):
            with self.assertRaises(ValueError):
                MujocoPlanarLidar(model, **kwargs)


if __name__ == "__main__":
    unittest.main()
