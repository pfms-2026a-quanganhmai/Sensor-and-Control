"""基于估计状态的控制器；任何接口都不接收 MuJoCo 车体真值。"""

import math

import numpy as np

DT = 0.02
RADIUS = 0.033
TRACK = 0.287


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def linear_model(v=0.12, dt=DT):
    """运动参考坐标系的误差模型：ex、ey、eθ；输入为δv、δω。"""
    a = np.eye(3)
    a[1, 2] = v * dt
    b = np.array([[dt, 0], [0, 0], [0, dt]])
    return a, b


def lqr_gain(v=0.12, dt=DT):
    """用离散 Riccati 迭代计算增益，避免额外依赖 scipy。"""
    a, b = linear_model(v, dt)
    q = np.diag([6.0, 40.0, 10.0])
    r = np.diag([4.0, 2.0])
    p = q.copy()
    for _ in range(10000):
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        new = q + a.T @ p @ a - a.T @ p @ b @ k
        if np.max(np.abs(new - p)) < 1e-9:
            break
        p = new
    return k


class PathTracker:
    def __init__(self):
        self.gain = lqr_gain()

    def command(self, pose, path):
        if len(path) < 2:
            return (0.0, 0.0), np.array([*path[-1], pose[2]])
        p = np.asarray(pose[:2])
        best = None
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            delta = b - a
            length = np.linalg.norm(delta)
            if length < 1e-8:
                continue
            direction = delta / length
            along = np.clip(np.dot(p - a, direction), 0, length)
            projection = a + along * direction
            score = np.linalg.norm(p - projection)
            if best is None or score < best[0]:
                best = (score, i, projection, direction, length - along)
        _, index, projection, direction, remaining = best
        if remaining < 0.13 and index < len(path) - 2:
            direction = path[index + 2] - path[index + 1]
            direction /= np.linalg.norm(direction)
        theta = math.atan2(direction[1], direction[0])
        reference = projection + 0.07 * direction
        rotation = np.array(
            [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
        )
        error = np.r_[rotation @ (p - reference), wrap(pose[2] - theta)]
        if abs(error[2]) > 0.55:
            command = (0.0, float(np.clip(-1.8 * error[2], -0.75, 0.75)))
        else:
            command = np.array([0.10, 0.0]) - self.gain @ error
            command = (
                float(np.clip(command[0], 0, 0.14)),
                float(np.clip(command[1], -0.75, 0.75)),
            )
        return command, np.r_[reference, theta]


def visual_dock(pose, goal):
    """PBVS：对标记坐标系中的目标位姿进行闭环调节，最终同时检查位置和朝向。"""
    dx, dy = goal[:2] - pose[:2]
    rho = math.hypot(dx, dy)
    heading = wrap(goal[2] - pose[2])
    if rho < 0.012:
        return 0.0, float(np.clip(1.8 * heading, -0.35, 0.35))
    alpha = wrap(math.atan2(dy, dx) - pose[2])
    # 大角度先旋转，避免倒车或绕小圆圈接近站点。
    if abs(alpha) > 0.7:
        return 0.0, float(np.clip(1.8 * alpha, -0.55, 0.55))
    beta = wrap(goal[2] - pose[2] - alpha)
    return min(0.075, 0.7 * rho) * max(0, math.cos(alpha)), float(
        np.clip(2.3 * alpha - 0.6 * beta, -0.55, 0.55)
    )


class WheelServo:
    def __init__(self):
        self.integral = np.zeros(2)

    def command(self, v, w, measured_speed, dt=DT):
        desired = np.array([(v - w * TRACK / 2) / RADIUS, (v + w * TRACK / 2) / RADIUS])
        if np.max(np.abs(desired)) < 1e-6:
            self.integral[:] = 0
            return np.zeros(2)
        error = desired - measured_speed
        self.integral = np.clip(self.integral + error * dt, -2, 2)
        # 原模型摩擦0.1Nm、速度伺服kv=0.1，等效速度偏置为1rad/s。
        # 不删除摩擦；用编码器反馈和静摩擦前馈补偿，控制量仍受原执行器限幅。
        feedforward = np.sign(desired) * 1.0
        feedforward[np.abs(desired) < 1e-6] = 0
        return np.clip(
            desired + feedforward + 0.45 * error + 0.8 * self.integral, -7.88, 7.88
        )
