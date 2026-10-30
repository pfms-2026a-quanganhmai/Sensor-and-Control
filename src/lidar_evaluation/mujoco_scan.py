"""MuJoCo 平面激光适配器：用射线生成测距，不把真实位姿传给控制器。"""

from __future__ import annotations

import numpy as np

from .lidar import ScanData


class MujocoPlanarLidar:
    """从一个 site 发出平面射线，局部 +X 为前方，+Y 为左方。

    site 的位置和旋转由仿真器用于生成传感器数据，属于传感器模型内部。
    对外只返回角度与距离。调用者必须先运行 mj_forward 或 mj_step。
    扫描平面必须与车体 XY 平面平行；mount_yaw_rad 是标定得到的安装偏航角，
    将传感器局部光束角度转换到车体角度，避免把传感器前方误当作车体前方。
    如果安装在小车上，须把全部车体 geom 放到单独的 group，并通过
    excluded_geom_groups 排除；默认不擅自假设团队模型的 geom 分组。
    """

    def __init__(
        self,
        model,
        site_name: str,
        beam_count: int = 360,
        range_min: float = 0.05,
        range_max: float = 8.0,
        excluded_geom_groups: tuple[int, ...] = (),
        mount_yaw_rad: float = 0.0,
    ) -> None:
        import mujoco

        if (
            isinstance(beam_count, bool)
            or not isinstance(beam_count, int)
            or beam_count < 4
        ):
            raise ValueError("beam_count must be an integer >= 4")
        # 复用统一扫描类型对量程的检查，避免两个接口有不同定义。
        ScanData(np.array([]), np.array([]), range_min, range_max)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"LiDAR site not found: {site_name}")
        self.model = model
        self.range_min = range_min
        self.range_max = range_max
        if not np.isfinite(mount_yaw_rad):
            raise ValueError("mount_yaw_rad must be finite")
        self.mount_yaw_rad = float(mount_yaw_rad)
        self.angles = np.linspace(-np.pi, np.pi, beam_count, endpoint=False)
        self.geom_groups = np.ones(6, dtype=np.uint8)
        for group in excluded_geom_groups:
            if (
                isinstance(group, bool)
                or not isinstance(group, int)
                or not 0 <= group < 6
            ):
                raise ValueError("excluded geometry groups must be integers in [0, 5]")
            self.geom_groups[group] = 0

    def read(self, data) -> ScanData:
        """生成一次扫描；无回波或超最大量程返回 Inf，由处理层保守解释。"""
        import mujoco

        origin = np.array(data.site_xpos[self.site_id], dtype=np.float64)
        rotation = data.site_xmat[self.site_id].reshape(3, 3)
        ranges = np.full(self.angles.shape, np.inf)
        geom_id = np.empty(1, dtype=np.int32)
        for index, angle in enumerate(self.angles):
            local_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            world_direction = np.ascontiguousarray(rotation @ local_direction)
            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                world_direction,
                self.geom_groups,
                True,
                -1,
                geom_id,
            )
            if 0 <= distance <= self.range_max:
                # 小于 range_min 的距离也保留，交给统一有效量程检查处理。
                ranges[index] = distance
        return ScanData(
            self.angles + self.mount_yaw_rad, ranges, self.range_min, self.range_max
        )
