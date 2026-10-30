"""从团队记录的同步轨迹画估计误差和不确定性，不生成虚构的运行结果。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def plot_logged_trial(input_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """画真实/估计/参考轨迹，以及各分量误差和 EKF 的正负两倍标准差。"""
    with Path(input_csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trajectory log contains no samples")
    try:
        time = np.array([float(row["time_s"]) for row in rows])
        poses = {}
        for name in ("truth", "estimate", "reference"):
            poses[name] = np.array(
                [
                    [
                        float(row[f"{name}_{key}"])
                        for key in ("x_m", "y_m", "heading_rad")
                    ]
                    for row in rows
                ]
            )
        covariance = np.array(
            [
                [[float(row[f"p{i}{j}"]) for j in range(3)] for i in range(3)]
                for row in rows
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid trajectory columns or values: {exc}") from exc
    if not all(
        np.all(np.isfinite(array)) for array in (time, covariance, *poses.values())
    ):
        raise ValueError("trajectory log must contain finite values")
    if np.any(np.diff(time) <= 0):
        raise ValueError("trajectory times must increase strictly")
    if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) < -1e-10):
        raise ValueError("covariance must be positive semidefinite")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "trajectory.png"
    uncertainty_path = destination / "estimation_uncertainty.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, label, style in (
        ("truth", "Simulator truth (evaluation only)", "-"),
        ("estimate", "Estimated state", "--"),
        ("reference", "Reference", ":"),
    ):
        ax.plot(poses[name][:, 0], poses[name][:, 1], style, label=label)
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Recorded trial trajectory")
    ax.axis("equal")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=160)
    plt.close(fig)

    error = poses["estimate"] - poses["truth"]
    error[:, 2] = (error[:, 2] + np.pi) % (2 * np.pi) - np.pi
    # 对角线为各状态方差；只画边际 ±2σ，不把它宣称为整体 95% 置信区域。
    twice_sigma = 2 * np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    error[:, 2] = np.rad2deg(error[:, 2])
    twice_sigma[:, 2] = np.rad2deg(twice_sigma[:, 2])
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for index, label in enumerate(
        ("x error (m)", "y error (m)", "Heading error (deg)")
    ):
        ax = axes[index]
        ax.fill_between(
            time,
            -twice_sigma[:, index],
            twice_sigma[:, index],
            color="#90caf9",
            alpha=0.5,
            label="Marginal +/-2 sigma",
        )
        ax.plot(time, error[:, index], color="#c84931", label="Estimate minus truth")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Recorded estimation errors and EKF uncertainty")
    axes[-1].set_xlabel("Simulation time (s)")
    fig.tight_layout()
    fig.savefig(uncertainty_path, dpi=160)
    plt.close(fig)
    return [trajectory_path, uncertainty_path]
