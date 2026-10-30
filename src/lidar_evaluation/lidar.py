"""激光扫描的独立处理模块：不读取仿真真实位姿，也不直接发送运动指令。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ScanData:
    """一次完整扫描；角度单位 rad、距离单位 m，机器人前方定义为 0 rad。

    无回波、丢失数据仍需保留对应角度，并将距离标为 NaN 或正无穷。
    不要先删除这些光束，否则无法计算扫描区域的数据有效率。
    """

    angles: np.ndarray
    ranges: np.ndarray
    range_min: float = 0.05
    range_max: float = 10.0

    def __post_init__(self) -> None:
        angles = np.asarray(self.angles, dtype=float)
        ranges = np.asarray(self.ranges, dtype=float)
        if angles.ndim != 1 or ranges.ndim != 1 or angles.shape != ranges.shape:
            raise ValueError(
                "angles and ranges must be one-dimensional and equal length"
            )
        if not np.all(np.isfinite(angles)):
            raise ValueError("all scan angles must be finite")
        if not np.isfinite(self.range_min) or self.range_min < 0:
            raise ValueError("range_min must be finite and non-negative")
        if not np.isfinite(self.range_max) or self.range_max <= self.range_min:
            raise ValueError("range_max must be finite and greater than range_min")

        # 保存副本并禁止就地修改，避免上游重复使用数组时改变本次测量。
        angles = angles.copy()
        ranges = ranges.copy()
        angles.setflags(write=False)
        ranges.setflags(write=False)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "ranges", ranges)


@dataclass(frozen=True)
class FrontObstacleResult:
    """前方区域的观测结论。unknown 表示证据不足，不能解释为没有障碍。

    nearest_distance_m 仅统计有效量程内的读数，不能代表盲区内的距离。
    """

    status: Literal["clear", "blocked", "unknown"]
    nearest_distance_m: float | None
    valid_fraction: float
    valid_count: int
    total_count: int

    @property
    def safe_to_move(self) -> bool:
        """仅表示本次前方检查通过；不替代完整路径、车身尺寸和制动分析。"""
        return self.status == "clear"


def evaluate_front(
    scan: ScanData,
    half_angle_deg: float = 30.0,
    stop_distance: float = 0.45,
    min_valid_fraction: float = 0.5,
) -> FrontObstacleResult:
    """检查前方扇区，返回 clear / blocked / unknown。

    默认值只是演示参数。实际停车阈值需结合传感器安装位置、车身外形、
    行驶速度、处理延迟和制动距离验证，不能直接当作安全保证。
    正无穷在不同驱动中含义可能不同；此处保守地作为未知测量处理。
    """
    if not np.isfinite(half_angle_deg) or not 0 < half_angle_deg <= 180:
        raise ValueError("half_angle_deg must be in (0, 180]")
    if not np.isfinite(stop_distance) or stop_distance <= 0:
        raise ValueError("stop_distance must be finite and positive")
    if not scan.range_min <= stop_distance <= scan.range_max:
        raise ValueError("stop_distance must lie within the sensor's valid range")
    if not np.isfinite(min_valid_fraction) or not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")

    # 把 0..2π 和 -π..π 两种输入统一到 -π..π，避免前方扇区被截断。
    wrapped = (scan.angles + np.pi) % (2 * np.pi) - np.pi
    half_angle = np.deg2rad(half_angle_deg)
    front_mask = np.abs(wrapped) <= half_angle + 1e-12
    total_count = int(np.count_nonzero(front_mask))

    # 分母是前方所有预期光束；NaN、Inf 和超量程光束不会从分母中删除。
    valid_mask = front_mask & np.isfinite(scan.ranges)
    valid_mask &= scan.ranges >= scan.range_min
    valid_mask &= scan.ranges <= scan.range_max
    valid_count = int(np.count_nonzero(valid_mask))
    valid_fraction = valid_count / total_count if total_count else 0.0
    nearest = float(np.min(scan.ranges[valid_mask])) if valid_count else None
    # 有些驱动会用低于最小量程的非负值表示过近物体。即使其他光束正常，
    # 这类盲区读数也不能被删除后解释为前方畅通；负数仍按坏数据处理。
    has_under_range = bool(
        np.any(
            front_mask
            & np.isfinite(scan.ranges)
            & (scan.ranges >= 0)
            & (scan.ranges < scan.range_min)
        )
    )

    # 先处理已发现的近障碍。即使多数光束丢失，已有危险证据仍应阻止前进。
    if nearest is not None and nearest <= stop_distance:
        status = "blocked"
    elif (
        has_under_range
        or total_count == 0
        or valid_count == 0
        or valid_fraction < min_valid_fraction
    ):
        status = "unknown"
    else:
        status = "clear"

    return FrontObstacleResult(
        status=status,
        nearest_distance_m=nearest,
        valid_fraction=valid_fraction,
        valid_count=valid_count,
        total_count=total_count,
    )
