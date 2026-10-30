"""离线评估：读取真实试验日志，并计算可解释的误差与成功率。

重要：这里可以使用仿真真值，因为本模块只用于评估，不能成为控制器的输入。
缺失的指标保持为 None；绝不能把未测量的误差补成 0。
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OPTIONAL_METRICS = (
    "final_position_error_m",
    "final_heading_error_deg",
    "tracking_rmse_m",
    "estimation_rmse_m",
)
CSV_COLUMNS = (
    "trial_id",
    "condition",
    "seed",
    "success",
    "collision",
    "duration_s",
    *OPTIONAL_METRICS,
)


@dataclass(frozen=True)
class TrialRecord:
    """一次完整试验；success 和 collision 应由团队事先约定的标准判定。"""

    trial_id: str
    condition: str
    seed: int
    success: bool
    collision: bool
    duration_s: float
    final_position_error_m: float | None = None
    final_heading_error_deg: float | None = None
    tracking_rmse_m: float | None = None
    estimation_rmse_m: float | None = None

    def __post_init__(self) -> None:
        for field in ("trial_id", "condition"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.success, bool) or not isinstance(self.collision, bool):
            raise ValueError("success and collision must be booleans")
        for field in ("duration_s", *OPTIONAL_METRICS):
            value = getattr(self, field)
            if value is None and field in OPTIONAL_METRICS:
                continue
            if (
                isinstance(value, (bool, str))
                or not np.isscalar(value)
                or not np.isreal(value)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} must be finite and non-negative")


def _parse_bool(value: str | None, field: str) -> bool:
    # 明确接受四种写法；例如字符串 'false' 不能用 bool('false') 转换。
    normal = (value or "").strip().lower()
    if normal in ("true", "1"):
        return True
    if normal in ("false", "0"):
        return False
    raise ValueError(f"{field} must be true/false or 1/0")


def _parse_number(
    value: str | None, field: str, optional: bool = False
) -> float | None:
    normal = (value or "").strip()
    if optional and not normal:
        return None
    try:
        result = float(normal)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def read_trials_csv(path: str | Path) -> list[TrialRecord]:
    """读取 UTF-8 CSV，拒绝不完整行、重复编号、NaN 和无穷大。

    同一个 condition 内 trial_id 必须唯一，不同条件可以使用相同编号和种子，
    以便比较算法时复用同一组初始条件。可选误差列可以留空，但列名必须存在。
    """
    records: list[TrialRecord] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names")
        missing = set(CSV_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        seen: set[tuple[str, str]] = set()
        for line, row in enumerate(reader, start=2):
            try:
                if None in row or any(
                    row.get(column) is None for column in CSV_COLUMNS
                ):
                    raise ValueError(
                        "row has a different number of fields from its header"
                    )
                trial_id = row["trial_id"].strip()
                condition = row["condition"].strip()
                raw_seed = row["seed"].strip()
                if not raw_seed.isascii() or not raw_seed.isdecimal():
                    raise ValueError("seed must be a non-negative integer")
                record = TrialRecord(
                    trial_id=trial_id,
                    condition=condition,
                    seed=int(raw_seed),
                    success=_parse_bool(row["success"], "success"),
                    collision=_parse_bool(row["collision"], "collision"),
                    duration_s=_parse_number(row["duration_s"], "duration_s"),
                    **{
                        field: _parse_number(row[field], field, optional=True)
                        for field in OPTIONAL_METRICS
                    },
                )
                key = (record.condition, record.trial_id)
                if key in seen:
                    raise ValueError(
                        f"duplicate trial_id {trial_id!r} in {condition!r}"
                    )
                seen.add(key)
                records.append(record)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
    if not records:
        raise ValueError("CSV contains no trial records; a template is not a result")
    return records


def summarize_trials(records: Iterable[TrialRecord]) -> list[dict]:
    """按测试条件汇总；标准差使用总体标准差 ddof=0。

    成功率、碰撞率的分母包含所有提交的试验，失败试验不会被偷偷删除。
    耗时均值也包含所有试验（包括超时和失败）；误差只汇总实际提供的值。
    """
    groups: dict[str, list[TrialRecord]] = {}
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, TrialRecord):
            raise ValueError("records must contain TrialRecord objects")
        key = (record.condition, record.trial_id)
        if key in seen:
            raise ValueError(
                f"duplicate trial_id {record.trial_id!r} in {record.condition!r}"
            )
        seen.add(key)
        groups.setdefault(record.condition, []).append(record)
    if not groups:
        raise ValueError("Cannot summarize zero trials")

    summaries = []
    for condition in sorted(groups):
        trials = groups[condition]
        count = len(trials)
        success_count = sum(record.success for record in trials)
        collision_count = sum(record.collision for record in trials)
        result = {
            "condition": condition,
            "trial_count": count,
            "success_count": success_count,
            "success_rate": success_count / count,
            "collision_count": collision_count,
            "collision_rate": collision_count / count,
        }
        for field in ("duration_s", *OPTIONAL_METRICS):
            values = np.asarray(
                [
                    getattr(record, field)
                    for record in trials
                    if getattr(record, field) is not None
                ],
                dtype=float,
            )
            result[f"{field}_count"] = len(values)
            result[f"{field}_mean"] = float(np.mean(values)) if len(values) else None
            result[f"{field}_std"] = (
                float(np.std(values, ddof=0)) if len(values) else None
            )
        summaries.append(result)
    return summaries


def _poses(values, name: str) -> np.ndarray:
    poses = np.asarray(values, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError(
            f"{name} must have non-empty shape (N, 3): x_m, y_m, heading_rad"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} must contain finite values")
    return poses


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    # 跨越 -pi/+pi 时，两次朝向相差应接近 0，而不是接近 2*pi。
    return (angle + np.pi) % (2 * np.pi) - np.pi


def trajectory_metrics(truth, estimates=None, reference=None, goal=None) -> dict:
    """由同步采样的平面轨迹计算评估指标，输入均为 (N, 3)，角度单位为弧度。

    truth: 仿真真值，仅允许用于本离线评估。
    estimates: EKF 输出，与 truth 一一对应，不能靠截断长度实现时间对齐。
    reference: 同一时刻的参考轨迹；计算的 tracking RMSE 是时间对应误差。
    goal: 最终目标 [x_m, y_m, heading_rad]；省略时使用 reference 的最后一点。
    未提供所需数据时，不输出相应指标；调用者可在 CSV 中留空。
    """
    actual = _poses(truth, "truth")
    result: dict = {"sample_count": len(actual)}
    for name, values in (("estimates", estimates), ("reference", reference)):
        if values is None:
            continue
        other = _poses(values, name)
        if other.shape != actual.shape:
            raise ValueError(f"{name} must match truth shape and sample times")
        position_error = np.linalg.norm(actual[:, :2] - other[:, :2], axis=1)
        heading_error = _wrap_angle(actual[:, 2] - other[:, 2])
        prefix = "estimation" if name == "estimates" else "tracking"
        result[f"{prefix}_rmse_m"] = float(np.sqrt(np.mean(position_error**2)))
        result[f"{prefix}_heading_rmse_deg"] = float(
            np.rad2deg(np.sqrt(np.mean(heading_error**2)))
        )
        if name == "reference" and goal is None:
            goal = other[-1]
    if goal is not None:
        target = np.asarray(goal, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("goal must be a finite pose [x_m, y_m, heading_rad]")
        result["final_position_error_m"] = float(
            np.linalg.norm(actual[-1, :2] - target[:2])
        )
        result["final_heading_error_deg"] = float(
            np.rad2deg(abs(_wrap_angle(actual[-1, 2] - target[2])))
        )
    return result
