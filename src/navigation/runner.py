"""真实 MuJoCo 完整任务：LiDAR 导航、视觉 EKF、LQR 跟踪与 PBVS 对接。"""

import csv
import json
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lidar_evaluation import ScanData, evaluate_front
from lidar_evaluation.logging import TrialLogger
from lidar_evaluation.mujoco_scan import MujocoPlanarLidar
from lidar_evaluation.plotting import plot_logged_trial

from .control import DT, RADIUS, PathTracker, WheelServo, visual_dock, wrap
from .estimation import NavigationEKF
from .planner import OccupancyPlanner
from .vision import CalibratedCamera

GOAL = np.array([1.05, 0.0, 0.0])
MARKER = np.array([1.5, 0.0])


def scene(root, condition, seed):
    spec = mujoco.MjSpec.from_file(
        str(root / "models/robotis_tb3/scene_turtlebot3_waffle_pi.xml")
    )
    spec.body("base").add_site(
        name="navigation_lidar",
        pos=[-0.049, 0, 0.2],
        size=[0.008] * 3,
        rgba=[0, 1, 0, 1],
    )
    for name, pos, size in (
        ("east", [2.5, 0, 0.3], [0.05, 1.5, 0.3]),
        ("west", [-1.5, 0, 0.3], [0.05, 1.5, 0.3]),
        ("north", [0.5, 1.5, 0.3], [2, 0.05, 0.3]),
        ("south", [0.5, -1.5, 0.3], [2, 0.05, 0.3]),
    ):
        spec.worldbody.add_geom(
            name="room_" + name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.6, 0.65, 0.7, 1],
        )
    if condition != "nominal":
        random = np.random.default_rng(seed + 400)
        # 这里只构建实验环境；障碍几何参数绝不传给地图、规划器或控制器。
        pos = [
            0.58 + random.uniform(-0.025, 0.025),
            0.27 + random.uniform(-0.02, 0.02),
            0.2,
        ]
        spec.worldbody.add_geom(
            name="task_obstacle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=[0.08, 0.18, 0.2],
            rgba=[0.9, 0.25, 0.12, 1],
        )
    # 对接区和充电站背板清晰可见。地面色块仅用于显示，不提供控制观测。
    spec.worldbody.add_geom(
        name="charging_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.0, 0, 0.001],
        size=[0.23, 0.23, 0.001],
        rgba=[0.1, 0.4, 0.95, 1],
        contype=0,
        conaffinity=0,
        group=4,
    )
    spec.worldbody.add_geom(
        name="station_backboard",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.55, 0, 0.21],
        size=[0.02, 0.19, 0.21],
        rgba=[0.22, 0.3, 0.4, 1],
    )
    model = spec.compile()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    bodies = {base}
    for i in range(base + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    robot = np.array([int(b) in bodies for b in model.geom_bodyid])
    model.geom_group[robot] = 1
    return model, robot


def safe_command(command, scan):
    """原始测距上的短期轨迹/车体检查，不用真实障碍坐标。"""
    v, w = command
    forward = evaluate_front(
        scan, half_angle_deg=40, stop_distance=0.27, min_valid_fraction=0.6
    )
    if forward.status == "unknown":
        return (0.0, 0.0), "unknown"
    finite = (
        np.isfinite(scan.ranges)
        & (scan.ranges >= scan.range_min)
        & (scan.ranges <= scan.range_max)
    )
    points = np.column_stack(
        [
            scan.ranges[finite] * np.cos(scan.angles[finite]) - 0.049,
            scan.ranges[finite] * np.sin(scan.angles[finite]),
        ]
    )
    # 将激光原点移回车体参考点：p_base = p_sensor + [-.049,0]。
    if len(points) == 0:
        return (0.0, 0.0), "unknown"
    if abs(v) < 1e-7:
        return (
            ((0.0, 0.0), "rotation_blocked")
            if np.min(np.linalg.norm(points, axis=1)) < 0.21
            else ((v, w), "clear")
        )
    for t in np.linspace(0, 0.6, 8):
        center = (
            np.array([v * t, 0])
            if abs(w) < 1e-7
            else np.array([v / w * np.sin(w * t), v / w * (1 - np.cos(w * t))])
        )
        if np.min(np.linalg.norm(points - center, axis=1)) < 0.225:
            return (0.0, 0.0), "blocked"
    return (v, w), "clear"


def run(
    root,
    output,
    condition="obstacle",
    seed=0,
    duration=90.0,
    gui=False,
    speed=1.0,
    plots=True,
    run_id="trial",
    record_video=False,
):
    # 复用队友相机图像检测及坐标变换，不使用模拟真值生成测量。
    import main as team

    output = Path(output)
    model, robot_geoms = scene(root, condition, seed)
    data = mujoco.MjData(model)
    initial = (
        np.array([-0.08, 0.02, 0.04]) if condition == "shifted_start" else np.zeros(3)
    )
    # 已知初始位姿先验属于实验初始化，不在运行中读取真值反馈。
    data.qpos[:2] = initial[:2]
    data.qpos[3:7] = [np.cos(initial[2] / 2), 0, 0, np.sin(initial[2] / 2)]
    mujoco.mj_forward(model, data)
    ekf = NavigationEKF(initial)
    planner = OccupancyPlanner()
    tracker = PathTracker()
    servo = WheelServo()
    lidar = MujocoPlanarLidar(
        model,
        "navigation_lidar",
        beam_count=180,
        range_max=5,
        excluded_geom_groups=(1,),
    )
    camera = CalibratedCamera(model)
    recorder = None
    last_video = -1.0
    record = None
    if record_video:
        from .video import Recorder

        video_dir = Path(output) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        recorder = Recorder(model, video_dir / f"{condition}__{run_id}-{seed:03d}.mp4")
    joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("wheel_left", "wheel_right")
    ]
    addresses = model.jnt_qposadr[joints]
    rng = np.random.default_rng(seed)
    encoder_std = 0.0003 if condition != "noisy_encoder" else 0.0015
    previous = data.qpos[addresses].copy() + rng.normal(0, encoder_std, 2)
    logger = TrialLogger(
        output,
        f"{run_id}-{seed:03d}",
        condition,
        seed,
        config={
            "task": "LiDAR navigation + ArUco PBVS docking",
            "initial_prior": initial.tolist(),
            "goal": GOAL.tolist(),
            "controller": "discrete LQR path tracker + PBVS + encoder PI",
            "lqr_Q": [6, 40, 10],
            "lqr_R": [4, 2],
            "control_dt_s": DT,
            "wheel_friction_feedforward_rad_s": 1.0,
            "encoder_angle_noise_std_rad": encoder_std,
            "camera_R": ekf.R.tolist(),
            "max_duration_s": duration,
            "map_resolution_m": planner.resolution,
            "inflation_m": planner.inflation,
            "camera_dropout_s": [4, 7] if condition == "camera_dropout" else [],
            "lidar_dropout_s": [3, 5] if condition == "lidar_dropout" else [],
            "success_rule": "estimated pose held with fresh camera, plus independent true-pose evaluation",
        },
    )
    sensors = output / "sensor_logs"
    sensors.mkdir(parents=True, exist_ok=True)
    sensor_path = sensors / f"{condition}__{run_id}-{seed:03d}.csv"
    if sensor_path.exists():
        camera.close()
        raise ValueError("run-id already exists")
    viewer = None
    if gui:
        import mujoco.viewer as viewer_module

        viewer = viewer_module.launch_passive(model, data)
        with viewer.lock():
            viewer.opt.geomgroup[4] = 1
        viewer.cam.lookat[:] = [0.7, 0, 0.1]
        viewer.cam.distance = 3.6
        viewer.cam.azimuth = 110
        viewer.cam.elevation = -55
    wall_start = time.monotonic()
    next_display = wall_start
    last_control = -DT
    last_camera = -1.0
    last_visual = -100.0
    last_plan = -100.0
    last_visual_pose = None
    path = None
    stage = "NAVIGATE"
    hold = 0.0
    complete = False
    collision = False
    aborted = False
    commands = (0.0, 0.0)
    reference = GOAL.copy()
    safety = "initial"
    replans = 0
    stages = [stage]
    last_logged = -1.0
    scan = None
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "charging_pad")
    fields = [
        "time_s",
        "stage",
        "lidar_status",
        "linear_m_s",
        "angular_rad_s",
        "visual_fresh",
        "accepted_visual_updates",
        "estimated_x",
        "estimated_y",
        "estimated_heading",
        "path_points",
    ]
    try:
        with sensor_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while data.time < duration:
                if viewer is not None and not viewer.is_running():
                    aborted = True
                    break
                if data.time - last_control >= DT - 1e-9:
                    dt = DT if last_control < 0 else float(data.time - last_control)
                    observed = data.qpos[addresses].copy() + rng.normal(
                        0, encoder_std, 2
                    )
                    delta = observed - previous
                    previous = observed
                    ekf.predict(*(RADIUS * delta), dt)
                    last_control = float(data.time)
                    if data.time - last_camera >= 0.10 - 1e-9:
                        measured = camera.read(data)
                        dropped = condition == "camera_dropout" and 4 <= data.time <= 7
                        if (
                            measured is not None
                            and not dropped
                            and ekf.update(measured)
                        ):
                            last_visual = float(data.time)
                            last_visual_pose = measured.copy()
                        last_camera = float(data.time)
                    raw = lidar.read(data)
                    ranges = raw.ranges.copy()
                    valid = np.isfinite(ranges)
                    ranges[valid] += rng.normal(0, 0.002, np.count_nonzero(valid))
                    if condition == "lidar_dropout" and 3 <= data.time <= 5:
                        ranges[:] = np.nan
                    scan = ScanData(raw.angles, ranges, raw.range_min, raw.range_max)
                    planner.update(scan, ekf.x)
                    fresh = data.time - last_visual < 0.5
                    rho = np.linalg.norm(ekf.x[:2] - GOAL[:2])
                    heading = abs(wrap(ekf.x[2] - GOAL[2]))
                    old_stage = stage
                    # 先到站点中心线再切换视觉伺服，避免贴近站点后为修正横向误差转出相机视野。
                    if (
                        stage == "NAVIGATE"
                        and rho < 0.13
                        and (
                            abs(ekf.x[1] - GOAL[1]) < 0.03
                            or (path is not None and len(path) < 2)
                        )
                    ):
                        stage = "ACQUIRE_MARKER"
                    if stage == "ACQUIRE_MARKER":
                        bearing = math.atan2(MARKER[1] - ekf.x[1], MARKER[0] - ekf.x[0])
                        commands = (
                            0.0,
                            float(np.clip(1.8 * wrap(bearing - ekf.x[2]), -0.5, 0.5)),
                        )
                        reference = np.r_[ekf.x[:2], bearing]
                        if fresh:
                            stage = "VISUAL_DOCK"
                    if stage == "VISUAL_DOCK":
                        if not fresh:
                            stage = "ACQUIRE_MARKER"
                            hold = 0.0
                            commands = (0.0, 0.0)
                        elif (
                            rho < 0.12
                            and abs(
                                wrap(
                                    math.atan2(GOAL[1] - ekf.x[1], GOAL[0] - ekf.x[0])
                                    - ekf.x[2]
                                )
                            )
                            > 0.7
                        ):
                            # 差速车不能横移。近处有横向误差时先沿可感知通道退开，再斜向接近。
                            # 这样避免对着标记原地转出视野并在两个阶段来回振荡。
                            stage = "ALIGN_APPROACH"
                            hold = 0.0
                            commands = (0.0, 0.0)
                        else:
                            commands = visual_dock(ekf.x, GOAL)
                            reference = GOAL.copy()
                            visual_close = (
                                last_visual_pose is not None
                                and np.linalg.norm(last_visual_pose[:2] - GOAL[:2])
                                < 0.025
                                and abs(wrap(last_visual_pose[2] - GOAL[2]))
                                < np.deg2rad(3)
                            )
                            if (
                                rho < 0.020
                                and heading < np.deg2rad(2.5)
                                and visual_close
                            ):
                                hold += dt
                            else:
                                hold = 0.0
                            if hold >= 0.6:
                                complete = True
                                stage = "DOCKED"
                                commands = (0.0, 0.0)
                    if stage == "ALIGN_APPROACH":
                        reference = GOAL.copy()
                        reference[0] -= 0.15
                        if not fresh:
                            stage = "ACQUIRE_MARKER"
                            commands = (0.0, 0.0)
                        elif ekf.x[0] < GOAL[0] - 0.15:
                            stage = "VISUAL_DOCK"
                            commands = (0.0, 0.0)
                        else:
                            commands = (
                                -0.045,
                                float(
                                    np.clip(1.5 * wrap(GOAL[2] - ekf.x[2]), -0.3, 0.3)
                                ),
                            )
                    if stage == "NAVIGATE":
                        if data.time - last_plan >= 0.5 or path is None:
                            candidate = planner.plan(ekf.x[:2], np.array([1.0, 0.0]))
                            if candidate is not None:
                                path = candidate
                            last_plan = float(data.time)
                            replans += 1
                        if path is None:
                            commands = (0.0, 0.0)
                            reference = ekf.x.copy()
                        else:
                            commands, reference = tracker.command(ekf.x, path)
                    commands, safety = safe_command(commands, scan)
                    if safety != "clear":
                        hold = 0
                        if complete:
                            complete = False
                            stage = "VISUAL_DOCK"
                    if stage != old_stage:
                        stages.append(stage)
                    # 只有轮关节编码器参与内环，绝不以车体真值计算控制输入。
                    model_commands = servo.command(*commands, delta / dt, dt)
                    data.ctrl[:] = model_commands
                    # 真值只进入独立日志和任务验收，不反馈给任何控制模块。
                    truth = team.get_robot_pose(data)
                    logger.record_sample(
                        float(data.time), truth, ekf.x, reference, ekf.P, commands
                    )
                    last_logged = float(data.time)
                    writer.writerow(
                        dict(
                            zip(
                                fields,
                                [
                                    data.time,
                                    stage,
                                    safety,
                                    *commands,
                                    fresh,
                                    ekf.accepted,
                                    *ekf.x,
                                    len(path) if path is not None else 0,
                                ],
                            )
                        )
                    )
                    if complete:
                        model.geom_rgba[pad] = [0.05, 0.95, 0.2, 1]
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    if robot_geoms[a] != robot_geoms[b]:
                        other = b if robot_geoms[a] else a
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
                        if name != "floor":
                            collision = True
                now = time.monotonic()
                if recorder is not None and (
                    data.time - last_video >= 0.10 - 1e-9 or complete
                ):
                    recorder.write(
                        data,
                        camera.last_image,
                        ekf.x,
                        ekf.P,
                        team.get_robot_pose(data),
                        stage,
                        safety,
                        commands,
                    )
                    last_video = float(data.time)
                if gui and now >= next_display:
                    viewer.sync()
                    if camera.last_image is not None:
                        image = camera.last_image.copy()
                        cv2.putText(
                            image,
                            stage,
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            image,
                            f"{data.time:.1f}s | LiDAR {safety} | v={commands[0]:.3f}",
                            (10, 53),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                        )
                        cv2.imshow("Visual docking - ArUco camera", image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        aborted = True
                        break
                    next_display = now + 1 / 30
                if complete or collision:
                    break
                mujoco.mj_step(model, data)
                if gui:
                    delay = wall_start + data.time / speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            # 非完成出口把最后物理步也更新到日志时间；保持估计/真值时刻一致。
            if data.time > last_logged + 1e-9:
                observed = data.qpos[addresses].copy() + rng.normal(0, encoder_std, 2)
                ekf.predict(
                    *(RADIUS * (observed - previous)), float(data.time - last_control)
                )
                logger.record_sample(
                    float(data.time),
                    team.get_robot_pose(data),
                    ekf.x,
                    reference,
                    ekf.P,
                    commands,
                )
            record = logger.finish(
                GOAL,
                collision=collision,
                timed_out=not complete or aborted,
                position_tolerance_m=0.05,
                heading_tolerance_deg=5,
                failure_reason=""
                if complete
                else f"last stage={stage}; lidar={safety}",
            )
            # 保留最后实际渲染画面，视频制作和人工审查可以直接核对。
            frames = output / "frames"
            frames.mkdir(exist_ok=True)
            if camera.last_image is not None:
                cv2.imwrite(
                    str(frames / f"{condition}__{run_id}-{seed:03d}.png"),
                    camera.last_image,
                )
            if gui and complete:
                # 对接后原地保持，给用户时间看到绿色情况和最终验收，而不是立即关窗。
                end = time.monotonic() + 2.5
                while viewer.is_running() and time.monotonic() < end:
                    viewer.sync()
                    image = camera.last_image.copy()
                    cv2.putText(
                        image,
                        f"DOCKED: {record.final_position_error_m:.3f}m / {record.final_heading_error_deg:.1f}deg",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Visual docking - ArUco camera", image)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                    time.sleep(0.03)
    finally:
        data.ctrl[:] = 0
        camera.close()
        if recorder is not None:
            recorder.close(record)
        if viewer is not None:
            viewer.close()
        if gui:
            cv2.destroyAllWindows()
    if plots:
        plot_logged_trial(
            logger.raw_path, output / "plots" / f"{condition}__{run_id}-{seed:03d}"
        )
    map_dir = output / "maps"
    map_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        map_dir / f"{condition}__{run_id}-{seed:03d}.npz",
        logodds=planner.logodds,
        origin=planner.origin,
        resolution=planner.resolution,
        path=path if path is not None else np.empty((0, 2)),
    )
    result = {
        "condition": condition,
        "seed": seed,
        "success": record.success,
        "complete_state": complete,
        "collision": collision,
        "duration_s": record.duration_s,
        "final_position_error_m": record.final_position_error_m,
        "final_heading_error_deg": record.final_heading_error_deg,
        "estimation_rmse_m": record.estimation_rmse_m,
        "visual_updates": ekf.accepted,
        "rejected_visual": ekf.rejected,
        "replans": replans,
        "stages": stages,
        "last_stage": stage,
        "last_safety": safety,
        "wall_s": time.monotonic() - wall_start,
    }
    (
        output / "manifests" / f"{condition}__{run_id}-{seed:03d}_navigation.json"
    ).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
