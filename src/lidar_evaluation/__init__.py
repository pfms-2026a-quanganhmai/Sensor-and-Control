"""个人贡献：LiDAR 处理、距离校准及实验评估；无需连接实物机器人。"""

from .calibration import (
    RangeCalibration,
    evaluate_calibration,
    fit_range_calibration,
    range_error_metrics,
)
from .lidar import FrontObstacleResult, ScanData, evaluate_front

__all__ = [
    "FrontObstacleResult",
    "RangeCalibration",
    "ScanData",
    "evaluate_calibration",
    "evaluate_front",
    "fit_range_calibration",
    "range_error_metrics",
]
