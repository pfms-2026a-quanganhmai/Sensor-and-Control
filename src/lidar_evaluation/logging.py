"""记录团队闭环试验，保存原始轨迹、协方差、参数和可复核的结果。

本模块仅用于评估，真值绝不能传回控制器。一次进程依次写入同一输出目录；
不支持多个进程同时追加 trials.csv。请在所有试验开始前固定成功标准。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .metrics import CSV_COLUMNS, TrialRecord, read_trials_csv, trajectory_metrics

RAW_COLUMNS = (
    "time_s",
    "truth_x_m",
    "truth_y_m",
    "truth_heading_rad",
    "estimate_x_m",
    "estimate_y_m",
    "estimate_heading_rad",
    "reference_x_m",
    "reference_y_m",
    "reference_heading_rad",
    "p00",
    "p01",
    "p02",
    "p10",
    "p11",
    "p12",
    "p20",
    "p21",
    "p22",
    "linear_command_m_s",
    "angular_command_rad_s",
)


def _identifier(value: str, name: str) -> str:
    # 双下划线专门分隔 condition 和 trial_id，避免两组编号生成同一个文件名。
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value)
        or "__" in value
    ):
        raise ValueError(
            f"{name} must be an ASCII slug of 1-64 letters/digits/_/-, "
            "start with a letter/digit, and not contain '__'"
        )
    return value


def _finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite array of shape {shape}") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    # 复制输入，防止调用者下一步原地修改 EKF 数组，污染已经记录的样本。
    return array.copy()


def _nonnegative_number(value, name: str) -> float:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isreal(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


class TrialLogger:
    """缓存一次试验，在 finish 时写出完整文件并追加一条汇总记录。

    每个 condition + trial_id 只能使用一次。记录器不会根据激光检测结果
    冒充导航成功；success 由实际最终误差、碰撞、超时共同计算。
    """

    def __init__(
        self,
        output_dir: str | Path,
        trial_id: str,
        condition: str,
        seed: int,
        config: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.trial_id = _identifier(trial_id, "trial_id")
        self.condition = _identifier(condition, "condition")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a JSON-compatible dictionary")
        try:
            # JSON 往返转换同时检查配置和复制嵌套字典，不允许 NaN/Infinity。
            self.config = json.loads(json.dumps(config or {}, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "config must be a finite JSON-compatible dictionary"
            ) from exc
        self.seed = seed
        name = f"{self.condition}__{self.trial_id}"
        self.raw_path = self.output_dir / "raw" / f"{name}.csv"
        self.manifest_path = self.output_dir / "manifests" / f"{name}.json"
        self.trials_path = self.output_dir / "trials.csv"
        self._samples: list[dict] = []
        self._finished = False

    @property
    def sample_count(self) -> int:
        """已接收的有效同步样本数。"""
        return len(self._samples)

    def record_sample(
        self,
        time_s: float,
        truth_pose,
        estimated_pose,
        reference_pose,
        covariance,
        control=(0.0, 0.0),
    ) -> None:
        """记录同一仿真时刻的状态，位姿为 [x_m, y_m, heading_rad]。

        covariance 的状态顺序也必须为 [x, y, heading]。它应对称且半正定；
        允许浮点计算引起的极小误差。控制输入单位为 m/s、rad/s。
        """
        if self._finished:
            raise RuntimeError("Cannot add samples after finish")
        stamp = _nonnegative_number(time_s, "time_s")
        if self._samples and stamp <= self._samples[-1]["time_s"]:
            raise ValueError(
                "time_s must increase strictly; align samples before logging"
            )
        truth = _finite_array(truth_pose, (3,), "truth_pose")
        estimate = _finite_array(estimated_pose, (3,), "estimated_pose")
        reference = _finite_array(reference_pose, (3,), "reference_pose")
        matrix = _finite_array(covariance, (3, 3), "covariance")
        command = _finite_array(control, (2,), "control")
        if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        symmetric = (matrix + matrix.T) / 2
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(symmetric, ord=2)))
        if float(np.min(np.linalg.eigvalsh(symmetric))) < -tolerance:
            raise ValueError("covariance must be positive semidefinite")

        values = [stamp, *truth, *estimate, *reference, *matrix.ravel(), *command]
        self._samples.append(dict(zip(RAW_COLUMNS, map(float, values))))

    def finish(
        self,
        goal,
        collision: bool = False,
        timed_out: bool = False,
        position_tolerance_m: float = 0.1,
        heading_tolerance_deg: float = 10.0,
        failure_reason: str = "",
    ) -> TrialRecord:
        """结束一次试验；写出成功/失败结果，但不会删除旧试验。

        最终误差使用最后一个实际记录的真值，朝向误差取绝对最短角差。
        单个样本可记录为 0 秒，但正常运行应在开始和结束时都采样。
        failure_reason 作为团队补充说明；判定规则仍固定为容差、碰撞、超时。
        """
        if self._finished:
            raise RuntimeError("This trial has already been finished")
        if not self._samples:
            raise ValueError("Cannot finish a trial without recorded samples")
        if not isinstance(collision, bool) or not isinstance(timed_out, bool):
            raise ValueError("collision and timed_out must be booleans")
        if not isinstance(failure_reason, str):
            raise ValueError("failure_reason must be a string")
        target = _finite_array(goal, (3,), "goal")
        position_tolerance = _nonnegative_number(
            position_tolerance_m, "position_tolerance_m"
        )
        heading_tolerance = _nonnegative_number(
            heading_tolerance_deg, "heading_tolerance_deg"
        )
        if heading_tolerance > 180:
            raise ValueError("heading_tolerance_deg must not exceed 180")

        def poses(prefix: str) -> np.ndarray:
            return np.asarray(
                [
                    [
                        row[f"{prefix}_x_m"],
                        row[f"{prefix}_y_m"],
                        row[f"{prefix}_heading_rad"],
                    ]
                    for row in self._samples
                ]
            )

        metrics = trajectory_metrics(
            poses("truth"), poses("estimate"), poses("reference"), target
        )
        reasons = []
        if collision:
            reasons.append("collision")
        if timed_out:
            reasons.append("timeout")
        if metrics["final_position_error_m"] > position_tolerance:
            reasons.append("position_tolerance_exceeded")
        if metrics["final_heading_error_deg"] > heading_tolerance:
            reasons.append("heading_tolerance_exceeded")
        success = not reasons
        record = TrialRecord(
            trial_id=self.trial_id,
            condition=self.condition,
            seed=self.seed,
            success=success,
            collision=collision,
            duration_s=self._samples[-1]["time_s"] - self._samples[0]["time_s"],
            **{
                key: metrics[key]
                for key in (
                    "final_position_error_m",
                    "final_heading_error_deg",
                    "tracking_rmse_m",
                    "estimation_rmse_m",
                )
            },
        )
        # 在创建任何新文件前检查重复；错误的重复调用不能覆盖先前的证据。
        existing_rows = []
        fields = list(CSV_COLUMNS)
        if self.trials_path.exists():
            existing = read_trials_csv(self.trials_path)
            if any(
                row.condition == self.condition and row.trial_id == self.trial_id
                for row in existing
            ):
                raise ValueError(
                    "Duplicate condition + trial_id; previous trial preserved"
                )
            with self.trials_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames)
                existing_rows = list(reader)
        if self.raw_path.exists() or self.manifest_path.exists():
            raise ValueError(
                "Raw log or manifest already exists; previous files preserved"
            )

        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        summary_temp = self.output_dir / f".trials-{uuid.uuid4().hex}.tmp"
        created: list[Path] = []
        committed = False
        try:
            with self.raw_path.open("x", encoding="utf-8-sig", newline="") as stream:
                created.append(self.raw_path)
                writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
                writer.writeheader()
                writer.writerows(self._samples)
            manifest = {
                "schema_version": 1,
                "purpose": "evaluation only; truth must never enter controller inputs",
                "trial_id": self.trial_id,
                "condition": self.condition,
                "seed": self.seed,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": self.config,
                "goal": target.tolist(),
                "position_tolerance_m": position_tolerance,
                "heading_tolerance_deg": heading_tolerance,
                "success_rule": "within both tolerances, no collision, no timeout",
                "collision": collision,
                "timed_out": timed_out,
                "computed_failure_reasons": reasons,
                "team_note": failure_reason.strip(),
                "failure_reason": "; ".join(
                    reasons
                    + (
                        [failure_reason.strip()]
                        if failure_reason.strip() and not success
                        else []
                    )
                ),
                "sample_count": self.sample_count,
                "start_time_s": self._samples[0]["time_s"],
                "end_time_s": self._samples[-1]["time_s"],
                "raw_csv": str(self.raw_path.resolve()),
                "raw_sha256": hashlib.sha256(self.raw_path.read_bytes()).hexdigest(),
                "trials_csv": str(self.trials_path.resolve()),
                "raw_columns": list(RAW_COLUMNS),
                "result": asdict(record),
                "all_trajectory_metrics": metrics,
            }
            with self.manifest_path.open("x", encoding="utf-8") as stream:
                created.append(self.manifest_path)
                json.dump(
                    manifest, stream, ensure_ascii=False, indent=2, allow_nan=False
                )
                stream.write("\n")
            # 先完整写入同目录临时文件，再原子替换；保留原有 CSV 的额外列和值。
            with summary_temp.open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(asdict(record))
            os.replace(summary_temp, self.trials_path)
            committed = True
        finally:
            if summary_temp.exists():
                summary_temp.unlink()
            if not committed:
                # 仅撤回本次刚创建的文件；绝不删除原有日志或整个结果目录。
                for path in reversed(created):
                    path.unlink(missing_ok=True)
        self._finished = True
        return record
