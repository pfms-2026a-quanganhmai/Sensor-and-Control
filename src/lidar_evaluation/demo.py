"""可复现的部件验证；所有合成数据及注入噪声都明确标记，非导航成绩。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .calibration import evaluate_calibration, fit_range_calibration
from .lidar import ScanData, evaluate_front


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _calibration_check(
    output: Path,
    references: np.ndarray,
    seed: int,
    source: str,
    sensor_readings: np.ndarray | None = None,
) -> dict:
    """故意注入已知偏差和噪声，验证校正算法；训练集与验证集相互独立。"""
    rng = np.random.default_rng(seed)
    train_ref = np.repeat(references, 12)
    valid_ref = np.repeat(references, 8)
    clean = references if sensor_readings is None else sensor_readings
    # 这是人为设置的传感器误差模型，不能写成实测出的真实雷达噪声。
    train_measured = (
        1.02 * np.repeat(clean, 12) + 0.04 + rng.normal(0, 0.01, train_ref.size)
    )
    valid_measured = (
        1.02 * np.repeat(clean, 8) + 0.04 + rng.normal(0, 0.01, valid_ref.size)
    )
    calibration = fit_range_calibration(train_ref, train_measured)
    errors = evaluate_calibration(valid_ref, valid_measured, calibration)
    rows = []
    for split, truth, measured in (
        ("fit", train_ref, train_measured),
        ("held_out_validation", valid_ref, valid_measured),
    ):
        for reference, raw in zip(truth, measured):
            corrected = float(calibration.apply(np.array([raw]))[0])
            rows.append(
                {
                    "split": split,
                    "reference_m": float(reference),
                    "measured_m": float(raw),
                    "corrected_m": corrected,
                    "source": source,
                }
            )
    _write_csv(output / "calibration_samples.csv", rows)
    result = {
        "source": source,
        "seed": seed,
        "injected_error_model": "measured=1.02*reference+0.04+N(0,0.01^2) metres",
        "fit_count": int(train_ref.size),
        "validation_count": int(valid_ref.size),
        "correction": asdict(calibration),
        "held_out_errors": {key: asdict(value) for key, value in errors.items()},
    }
    _write_json(output / "calibration_result.json", result)
    return result


def _plot_checks(output: Path, rows: list[dict], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({row["condition"] for row in rows})
    counts = [
        sum(row["check_passed"] for row in rows if row["condition"] == label)
        for label in labels
    ]
    totals = [sum(row["condition"] == label for row in rows) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        labels, [count / total for count, total in zip(counts, totals)], color="#287b8e"
    )
    for bar, count, total in zip(bars, counts, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{count}/{total}",
            ha="center",
        )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Component check pass fraction")
    ax.set_title(title + "\nSensor component only; not navigation success")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / "component_checks.png", dpi=160)
    plt.close(fig)

    with (output / "calibration_samples.csv").open(encoding="utf-8-sig") as stream:
        samples = [
            row
            for row in csv.DictReader(stream)
            if row["split"] == "held_out_validation"
        ]
    references = np.array([float(row["reference_m"]) for row in samples])
    raw_errors = np.array([float(row["measured_m"]) for row in samples]) - references
    corrected_errors = (
        np.array([float(row["corrected_m"]) for row in samples]) - references
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(references, raw_errors, label="Before correction", alpha=0.65, s=22)
    ax.scatter(references, corrected_errors, label="After correction", alpha=0.65, s=22)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Reference distance (m)")
    ax.set_ylabel("Measured minus reference (m)")
    ax.set_title("Held-out calibration check: deliberately injected error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_validation.png", dpi=160)
    plt.close(fig)


def run_self_check(output_dir: str | Path, plots: bool = True) -> dict:
    """不需要机器人模型，用明确标记的合成扫描检查处理逻辑。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    front = np.abs(angles) <= np.deg2rad(30) + 1e-12
    rows = []
    for condition, expected in (
        ("clear", "clear"),
        ("near_obstacle", "blocked"),
        ("partial_missing", "unknown"),
        ("all_missing", "unknown"),
    ):
        ranges = np.full(angles.shape, 2.0)
        if condition == "near_obstacle":
            ranges[np.argmin(np.abs(angles))] = 0.25
        elif condition == "partial_missing":
            ranges[front] = np.nan
            ranges[np.argmin(np.abs(angles))] = 2.0
        elif condition == "all_missing":
            ranges[:] = np.nan
        result = evaluate_front(ScanData(angles, ranges, 0.05, 8.0))
        rows.append(
            {
                "condition": condition,
                "source": "synthetic_logic_check",
                "expected_status": expected,
                **asdict(result),
                "check_passed": result.status == expected,
            }
        )
    _write_csv(output / "component_checks.csv", rows)
    calibration = _calibration_check(
        output,
        np.array([0.3, 0.6, 1.0, 1.5, 2.0]),
        20261030,
        "synthetic_reference_and_injected_error_not_real_sensor_data",
    )
    result = {
        "source": "synthetic_component_check_not_navigation_results",
        "checks_passed": sum(row["check_passed"] for row in rows),
        "check_count": len(rows),
        "calibration": calibration,
    }
    _write_json(output / "run_summary.json", result)
    if plots:
        _plot_checks(output, rows, "Synthetic logic checks")
    return result


def run_mujoco_check(output_dir: str | Path, plots: bool = True) -> dict:
    """使用真实 MuJoCo 射线测距验证传感器组件，不依赖缺失的 TB3 资源。"""
    import mujoco

    from .mujoco_scan import MujocoPlanarLidar

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    center_measurements = []
    # 每个条件重复 20 个种子；这些是扫描检测实验，不是小车导航实验。
    for condition, distance, expected in (
        ("clear", 1.0, "clear"),
        ("near_obstacle", 0.25, "blocked"),
        ("partial_dropout", 1.0, "clear"),
        ("all_missing", 1.0, "unknown"),
    ):
        model = _make_calibration_scene(distance)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        lidar = MujocoPlanarLidar(model, "lidar")
        clean = lidar.read(data)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            ranges = clean.ranges.copy()
            finite = np.isfinite(ranges)
            ranges[finite] += rng.normal(0.0, 0.003, np.count_nonzero(finite))
            if condition == "partial_dropout":
                # 精确丢弃约 20% 的光束，使条件定义可复现。
                missing = rng.choice(len(ranges), size=len(ranges) // 5, replace=False)
                ranges[missing] = np.nan
            elif condition == "all_missing":
                ranges[:] = np.nan
            result = evaluate_front(ScanData(clean.angles, ranges, 0.05, 8.0))
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "source": "mujoco_static_sensor_component_with_injected_noise",
                    "reference_front_face_m": distance,
                    "expected_status": expected,
                    **asdict(result),
                    "check_passed": result.status == expected,
                }
            )
    _write_csv(output / "component_checks.csv", rows)

    # 单独验证已知平面距离：几何参考值只用于评估，不传给任何控制器。
    references = np.array([0.3, 0.6, 1.0, 1.5, 2.0])
    for distance in references:
        model = _make_calibration_scene(float(distance))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scan = MujocoPlanarLidar(model, "lidar").read(data)
        measured = float(scan.ranges[np.argmin(np.abs(scan.angles))])
        center_measurements.append(
            {
                "reference_m": float(distance),
                "measured_m": measured,
                "error_m": measured - float(distance),
            }
        )
    _write_csv(output / "geometric_range_validation.csv", center_measurements)
    measured_references = np.array([row["measured_m"] for row in center_measurements])
    calibration = _calibration_check(
        output,
        references,
        20261031,
        "mujoco_geometry_reference_with_deliberately_injected_sensor_error",
        sensor_readings=measured_references,
    )
    result = {
        "source": "mujoco_static_sensor_component_not_tb3_navigation",
        "mujoco_version": mujoco.__version__,
        "seeds": list(range(20)),
        "checks_passed": sum(row["check_passed"] for row in rows),
        "check_count": len(rows),
        "injected_scan_noise_std_m": 0.003,
        "geometric_max_abs_error_m": max(
            abs(row["error_m"]) for row in center_measurements
        ),
        "calibration": calibration,
    }
    _write_json(output / "run_summary.json", result)
    if plots:
        _plot_checks(output, rows, "MuJoCo ray-cast checks")
    return result


def _make_calibration_scene(front_distance: float):
    """小型静态验证场景：障碍前表面到激光原点的距离已知。"""
    import mujoco

    if not np.isfinite(front_distance) or front_distance <= 0:
        raise ValueError("front_distance must be finite and positive")
    xml = f'''<mujoco model="lidar_component_calibration">
      <worldbody>
        <site name="lidar" pos="0 0 0.2" size="0.01"/>
        <geom name="target" type="box" pos="{front_distance + 0.05} 0 0.2"
              size="0.05 2 0.2"/>
        <geom name="back" type="box" pos="-3 0 0.2" size="0.05 3 0.2"/>
        <geom name="left" type="box" pos="0 3 0.2" size="3 0.05 0.2"/>
        <geom name="right" type="box" pos="0 -3 0.2" size="3 0.05 0.2"/>
      </worldbody>
    </mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)
