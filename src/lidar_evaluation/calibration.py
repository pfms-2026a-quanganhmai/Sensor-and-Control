"""量程校正与误差统计；输入应来自已知距离实验，不在这里生成项目结果。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RangeCalibration:
    """线性校正：校正距离 = slope × 原始距离 + intercept，距离单位 m。"""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("calibration slope must be finite and positive")
        if not np.isfinite(self.intercept):
            raise ValueError("calibration intercept must be finite")

    def apply(self, measured: np.ndarray | list[float] | float) -> np.ndarray:
        """校正有效数字，保留 NaN / Inf 的无效数据标记。"""
        values = np.asarray(measured, dtype=float)
        corrected = values.copy()
        finite = np.isfinite(values)
        corrected[finite] = self.slope * values[finite] + self.intercept
        return corrected


@dataclass(frozen=True)
class RangeErrorMetrics:
    """误差定义为 measured-reference；std_m 为总体标准差（ddof=0）。"""

    count: int
    mae_m: float
    rmse_m: float
    bias_m: float
    std_m: float


def _paired_distances(
    reference: np.ndarray | list[float],
    measured: np.ndarray | list[float],
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(measured, dtype=float)
    if ref.ndim != 1 or obs.ndim != 1 or ref.shape != obs.shape:
        raise ValueError("reference and measured must be equal-length 1D arrays")
    if ref.size < minimum_count:
        raise ValueError(f"at least {minimum_count} paired measurements are required")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(obs)):
        raise ValueError("calibration measurements must all be finite")
    if np.any(ref < 0) or np.any(obs < 0):
        raise ValueError("distances must be non-negative")
    return ref, obs


def fit_range_calibration(
    reference: np.ndarray | list[float], measured: np.ndarray | list[float]
) -> RangeCalibration:
    """用校准集拟合原始距离到已知距离的线性映射。

    至少需要两个不同距离。为了报告可信结果，应另留独立验证集，
    不能把拟合所用数据上的误差下降当作独立验证证据。
    """
    ref, obs = _paired_distances(reference, measured, minimum_count=2)
    # 对测量值中心化后求最小二乘，降低大偏移情况下的数值误差。
    centered = obs - np.mean(obs)
    scale = max(1.0, float(np.max(np.abs(obs))))
    if np.ptp(obs) <= np.finfo(float).eps * scale:
        raise ValueError("measured distances must contain at least two distinct values")
    slope = float(np.dot(centered, ref - np.mean(ref)) / np.dot(centered, centered))
    intercept = float(np.mean(ref) - slope * np.mean(obs))
    return RangeCalibration(slope=slope, intercept=intercept)


def range_error_metrics(
    reference: np.ndarray | list[float], measured: np.ndarray | list[float]
) -> RangeErrorMetrics:
    """计算测距误差；拒绝坏数据，避免静默删除失败样本后得到过好结果。"""
    ref, obs = _paired_distances(reference, measured, minimum_count=1)
    error = obs - ref
    return RangeErrorMetrics(
        count=int(error.size),
        mae_m=float(np.mean(np.abs(error))),
        rmse_m=float(np.sqrt(np.mean(error**2))),
        bias_m=float(np.mean(error)),
        std_m=float(np.std(error, ddof=0)),
    )


def evaluate_calibration(
    reference: np.ndarray | list[float],
    measured: np.ndarray | list[float],
    calibration: RangeCalibration,
) -> dict[str, RangeErrorMetrics]:
    """在调用者提供的独立验证集上，对比校正前和校正后的误差。"""
    return {
        "before": range_error_metrics(reference, measured),
        "after": range_error_metrics(reference, calibration.apply(measured)),
    }
