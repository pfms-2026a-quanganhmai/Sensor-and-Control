# ruff: noqa: E402
"""个人部分与队友主程序的联调入口：实际 TB3、视觉 EKF、LiDAR 和真实试验日志。

保留 main.py 的队友实现，控制器调用队友现有的临时航点控制器。
这不是 LQR 或完整充电站对接实现；障碍试验验证停车，不冒充绕障完成。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# 导入路径只在此仓库入口中设置；被导入的队友 main.py 不会自动启动窗口。
import main as team
from lidar_evaluation import ScanData, evaluate_front
from lidar_evaluation.logging import TrialLogger
from lidar_evaluation.mujoco_scan import MujocoPlanarLidar
from lidar_evaluation.plotting import plot_logged_trial
from lidar_evaluation.report import create_report


def build_model(obstacle: bool = False):
    """复用队友模型并在内存中添加激光和可见障碍，不改动原 XML。"""
    spec = mujoco.MjSpec.from_file(str(team.MODEL_PATH))
    base = spec.body("base")
    if base is None:
        raise ValueError("team model must contain body 'base'")
    base.add_site(
        name="evaluation_lidar",
        pos=[-0.049, 0, 0.20],
        size=[0.008] * 3,
        rgba=[0.1, 0.8, 0.1, 1],
    )
    # 有界房间提供可解释的量程回波，不能把无回波 Inf 偷换成通路畅通。
    for name, pos, size in (
        ("east", [2.5, 0, 0.3], [0.05, 1.5, 0.3]),
        ("west", [-1.5, 0, 0.3], [0.05, 1.5, 0.3]),
        ("north", [0.5, 1.5, 0.3], [2, 0.05, 0.3]),
        ("south", [0.5, -1.5, 0.3], [2, 0.05, 0.3]),
    ):
        spec.worldbody.add_geom(
            name=f"evaluation_wall_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.65, 0.7, 0.75, 1],
        )
    if obstacle:
        spec.worldbody.add_geom(
            name="evaluation_obstacle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.58, 0.27, 0.2],
            size=[0.08, 0.18, 0.2],
            rgba=[0.9, 0.25, 0.15, 1],
        )
    model = spec.compile()
    # 模型里车体既有 group 2/3，也有默认 group 0 的几何体；按 body 子树统一排除自身。
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot_bodies = {root_id}
    for body_id in range(root_id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in robot_bodies:
            robot_bodies.add(body_id)
    robot_geoms = np.array([int(body) in robot_bodies for body in model.geom_bodyid])
    model.geom_group[robot_geoms] = 1
    return model, robot_geoms


class CameraPose:
    """使用实际渲染图像与队友的 ArUco 坐标变换，不用真值伪造视觉测量。"""

    def __init__(self, model):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
        if camera_id < 0:
            raise ValueError("front_camera missing: update models from main")
        self.renderer = mujoco.Renderer(
            model, height=team.CAMERA_HEIGHT, width=team.CAMERA_WIDTH
        )
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )
        f = team.CAMERA_HEIGHT / (
            2 * math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2)
        )
        self.intrinsics = np.array(
            [
                [f, 0, (team.CAMERA_WIDTH - 1) / 2],
                [0, f, (team.CAMERA_HEIGHT - 1) / 2],
                [0, 0, 1],
            ]
        )
        half = team.ARUCO_MARKER_SIZE / 2
        self.points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        self.last_image = None

    def read(self, data):
        self.renderer.update_scene(data, camera="front_camera")
        image = self.renderer.render().copy()
        self.last_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        )
        if ids is None:
            return None
        cv2.aruco.drawDetectedMarkers(self.last_image, corners, ids)
        for corner, marker in zip(corners, ids.flatten()):
            if marker != 0:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                self.points,
                corner.reshape(4, 2).astype(np.float32),
                self.intrinsics,
                np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok and float(tvec[2, 0]) > 0:
                pose = np.array(team.aruco_to_robot_pose(rvec, tvec))
                if np.all(np.isfinite(pose)):
                    return pose
        return None

    def close(self):
        self.renderer.close()


def run_trial(
    output: Path,
    condition: str,
    seed: int,
    duration: float,
    gui: bool = False,
    plots: bool = True,
    run_id: str = "run",
) -> dict:
    model, robot_geoms = build_model(condition == "obstacle")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ekf = team.EKF()
    lidar = MujocoPlanarLidar(
        model,
        "evaluation_lidar",
        beam_count=180,
        range_max=5.0,
        excluded_geom_groups=(1,),
    )
    camera = CameraPose(model)
    rng = np.random.default_rng(seed)
    # 轮距由实际模型几何取得，其他参数记录下来，方便团队核对。
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_left")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_right")
    addresses = model.jnt_qposadr[[left, right]]
    previous_wheels = data.qpos[addresses].copy()
    logger = TrialLogger(
        output,
        f"{run_id}-{seed:03d}",
        condition,
        seed,
        config={
            "source": "actual_tb3_mujoco_with_team_visual_ekf",
            "controller": "team temporary waypoint controller, not LQR/docking",
            "model": str(team.MODEL_PATH),
            "condition": condition,
            "max_duration_s": duration,
            "lidar_rate_hz": 20,
            "lidar_noise_std_m": 0.003,
            "encoder_increment_noise_std_rad": 0.0005,
            "stop_distance_m": 0.35,
            "front_half_angle_deg": 35,
            "camera_dropout_interval_s": [2, 5]
            if condition == "camera_dropout"
            else [],
            "lidar_dropout_interval_s": [2, 5] if condition == "lidar_dropout" else [],
            "initial_pose_prior": [0, 0, 0],
        },
    )
    evidence_dir = output / "sensor_logs"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    sensor_path = evidence_dir / f"{condition}__{run_id}-{seed:03d}.csv"
    if (
        sensor_path.exists()
        or logger.raw_path.exists()
        or logger.manifest_path.exists()
    ):
        camera.close()
        raise ValueError("run_id already exists; choose a new --run-id")
    viewer = None
    if gui:
        import mujoco.viewer as viewer_module

        viewer = viewer_module.launch_passive(model, data)
    last_camera = last_sensor = last_log = -1.0
    front = None
    collision = False
    visual_updates = 0
    stopped_samples = 0
    commands = (0.0, 0.0)
    target = np.array([team.TARGET_X, team.TARGET_Y, 0.0])
    reached = False
    aborted = False
    fields = [
        "time_s",
        "status",
        "nearest_distance_m",
        "valid_fraction",
        "linear_command_m_s",
        "angular_command_rad_s",
        "visual_update",
    ]
    try:
        with sensor_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while data.time < duration:
                start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    aborted = True
                    break
                # 编码器增量来自轮关节；只加可复现的测量噪声，不读车体真实位姿。
                current = data.qpos[addresses].copy()
                delta = current - previous_wheels
                previous_wheels = current
                delta += rng.normal(0, 0.0005, size=2)
                ds = team.WHEEL_RADIUS * float(np.sum(delta)) / 2
                dh = (
                    team.WHEEL_RADIUS
                    * float(delta[1] - delta[0])
                    / team.WHEEL_SEPARATION
                )
                ekf.predict(ds, dh)
                visual_update = False
                if data.time - last_camera >= team.CAMERA_UPDATE_INTERVAL - 1e-9:
                    pose = camera.read(data)
                    missing = condition == "camera_dropout" and 2 <= data.time <= 5
                    if pose is not None and not missing:
                        ekf.update_pose(*pose)
                        visual_updates += 1
                        visual_update = True
                    last_camera = data.time
                if data.time - last_sensor >= 0.05 - 1e-9:
                    scan = lidar.read(data)
                    ranges = scan.ranges.copy()
                    valid = np.isfinite(ranges)
                    ranges[valid] += rng.normal(0, 0.003, np.count_nonzero(valid))
                    if condition == "lidar_dropout" and 2 <= data.time <= 5:
                        ranges[:] = np.nan
                    front = evaluate_front(
                        ScanData(scan.angles, ranges, scan.range_min, scan.range_max),
                        half_angle_deg=35,
                        stop_distance=0.35,
                    )
                    last_sensor = data.time
                linear, angular, reached = team.waypoint_controller(*ekf.x)
                # 数据不足或有近障碍时保守停止；本模块没有验证原地转向的全车身安全性。
                stale = data.time - last_sensor > 0.15
                if front is None or stale or front.status != "clear":
                    linear, angular = 0.0, 0.0
                    stopped_samples += 1
                commands = (linear, angular)
                left_speed, right_speed = team.body_velocity_to_wheel_speed(*commands)
                data.ctrl[:] = [left_speed, right_speed]
                # 接触真值仅用于评估；地板支撑接触不是碰撞。
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    if robot_geoms[a] != robot_geoms[b]:
                        other = b if robot_geoms[a] else a
                        if (
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                            != "floor"
                        ):
                            collision = True
                if data.time - last_log >= 0.05 - 1e-9:
                    truth = team.get_robot_pose(data)  # 唯一用途：传给评估日志。
                    # 参考为固定航点；不是时间参数化轨迹，RMSE 应解释为目标距离 RMSE。
                    reference = target.copy()
                    reference[2] = math.atan2(
                        target[1] - ekf.x[1], target[0] - ekf.x[0]
                    )
                    logger.record_sample(
                        float(data.time), truth, ekf.x, reference, ekf.P, commands
                    )
                    writer.writerow(
                        {
                            "time_s": data.time,
                            "status": front.status,
                            "nearest_distance_m": front.nearest_distance_m,
                            "valid_fraction": front.valid_fraction,
                            "linear_command_m_s": linear,
                            "angular_command_rad_s": angular,
                            "visual_update": visual_update,
                        }
                    )
                    last_log = data.time
                if gui:
                    viewer.sync()
                    cv2.imshow("Integrated TB3 - front camera", camera.last_image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        aborted = True
                        break
                if reached or collision:
                    break
                mujoco.mj_step(model, data)
                if gui:
                    time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))
            if data.time > last_log + 1e-9:
                # 把最后一个物理步的编码器也预测到终止时刻，避免日志时刻错位。
                delta = data.qpos[addresses] - previous_wheels
                delta += rng.normal(0, 0.0005, size=2)
                ekf.predict(
                    team.WHEEL_RADIUS * float(np.sum(delta)) / 2,
                    team.WHEEL_RADIUS
                    * float(delta[1] - delta[0])
                    / team.WHEEL_SEPARATION,
                )
                logger.record_sample(
                    float(data.time),
                    team.get_robot_pose(data),
                    ekf.x,
                    target,
                    ekf.P,
                    commands,
                )
            timed_out = not reached or aborted
            # 队友航点控制器没有最终朝向控制；仅报告航点位置成功，绝不称为 docking。
            record = logger.finish(
                target,
                collision=collision,
                timed_out=timed_out,
                position_tolerance_m=team.POSITION_TOLERANCE,
                heading_tolerance_deg=180,
                failure_reason="waypoint-only; no final docking heading controller",
            )
    finally:
        data.ctrl[:] = 0
        camera.close()
        if viewer is not None:
            viewer.close()
        if gui:
            cv2.destroyAllWindows()
    if plots:
        plot_logged_trial(
            logger.raw_path, output / "plots" / f"{condition}__{run_id}-{seed:03d}"
        )
    result = {
        "condition": condition,
        "seed": seed,
        "waypoint_success": record.success,
        "collision": collision,
        "visual_updates": visual_updates,
        "stopped_physics_steps": stopped_samples,
        "duration_s": record.duration_s,
        "final_position_error_m": record.final_position_error_m,
        "estimation_rmse_m": record.estimation_rmse_m,
        "raw_csv": str(logger.raw_path),
        "sensor_csv": str(sensor_path),
        "scope": "waypoint+lidar gate; not full avoidance, LQR or docking",
    }
    (
        output / "manifests" / f"{condition}__{run_id}-{seed:03d}_integration.json"
    ).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=["nominal", "obstacle", "camera_dropout", "lidar_dropout"],
        default="nominal",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "team_integration"
    )
    parser.add_argument(
        "--gui", action="store_true", help="显示 MuJoCo 与相机窗口；默认无界面运行"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("trials and duration must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.trials):
        print(
            json.dumps(
                run_trial(
                    args.output_dir,
                    args.condition,
                    seed,
                    args.duration,
                    args.gui,
                    not args.no_plots,
                    args.run_id,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    create_report(
        args.output_dir / "trials.csv",
        args.output_dir / "report",
        plots=not args.no_plots,
    )
    summarize_sensor_responses(args.output_dir, plots=not args.no_plots)
    print(f"真实联调日志：{args.output_dir.resolve()}")


def summarize_sensor_responses(output: Path, plots: bool = True):
    """单独评价检测后的停车响应，不把停车正确称为导航成功。"""
    summaries = []
    for path in sorted((output / "sensor_logs").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        nonclear = [row for row in rows if row["status"] != "clear"]
        violations = sum(
            abs(float(row["linear_command_m_s"])) > 1e-12
            or abs(float(row["angular_command_rad_s"])) > 1e-12
            for row in nonclear
        )
        summaries.append(
            {
                "sensor_log": path.name,
                "sample_count": len(rows),
                "blocked_samples": sum(row["status"] == "blocked" for row in rows),
                "unknown_samples": sum(row["status"] == "unknown" for row in rows),
                "nonclear_command_violations": violations,
                "interpretation": "sensor response only; stopping does not prove task success",
            }
        )
        if plots and nonclear:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            stamps = [float(row["time_s"]) for row in rows]
            statuses = [
                {"unknown": 0, "blocked": 1, "clear": 2}[row["status"]] for row in rows
            ]
            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            axes[0].step(stamps, statuses, where="post")
            axes[0].set_yticks([0, 1, 2], labels=["unknown", "blocked", "clear"])
            axes[0].set_title("Actual TB3 LiDAR response (not navigation success)")
            axes[1].plot(stamps, [float(row["linear_command_m_s"]) for row in rows])
            axes[1].set(xlabel="Simulation time (s)", ylabel="Linear command (m/s)")
            fig.tight_layout()
            plot_dir = output / "sensor_plots"
            plot_dir.mkdir(exist_ok=True)
            fig.savefig(plot_dir / f"{path.stem}.png", dpi=160)
            plt.close(fig)
    if summaries:
        with (output / "sensor_response_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    main()
