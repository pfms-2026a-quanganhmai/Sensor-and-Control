"""轮编码器/视觉 EKF：过程噪声按采样间隔和实际轮增量传播。"""

import numpy as np

from .control import wrap


class NavigationEKF:
    def __init__(self, prior):
        self.x = np.array(prior, dtype=float)
        self.P = np.diag([0.015**2, 0.015**2, np.deg2rad(2) ** 2])
        self.R = np.diag([0.025**2, 0.025**2, np.deg2rad(3) ** 2])
        self.accepted = 0
        self.rejected = 0

    def predict(self, left_distance, right_distance, dt):
        ds = (left_distance + right_distance) / 2
        dh = (right_distance - left_distance) / 0.287
        theta = self.x[2] + dh / 2
        f = np.eye(3)
        f[0, 2] = -ds * np.sin(theta)
        f[1, 2] = ds * np.cos(theta)
        self.x += np.array([ds * np.cos(theta), ds * np.sin(theta), dh])
        self.x[2] = wrap(self.x[2])
        # 轮子噪声为仿真注入/模型失配假设，非实物标定结果。
        variance = (0.00008 + 0.012 * abs(ds)) ** 2
        g = np.array(
            [
                [np.cos(theta) / 2, np.cos(theta) / 2],
                [np.sin(theta) / 2, np.sin(theta) / 2],
                [-1 / 0.287, 1 / 0.287],
            ]
        )
        # 基线试验存在轮胎滑移，不能只按编码器电子噪声给出过于乐观的P。
        q = g @ np.diag([variance, variance]) @ g.T + dt * np.diag(
            [0.02**2, 0.02**2, np.deg2rad(3) ** 2]
        )
        self.P = f @ self.P @ f.T + q

    def update(self, measurement):
        innovation = np.asarray(measurement) - self.x
        innovation[2] = wrap(innovation[2])
        s = self.P + self.R
        # 卡方门控只使用估计协方差与视觉创新，不允许使用真值挑选测量。
        if float(innovation @ np.linalg.solve(s, innovation)) > 16.27:
            self.rejected += 1
            return False
        k = np.linalg.solve(s.T, self.P.T).T
        self.x += k @ innovation
        self.x[2] = wrap(self.x[2])
        residual = np.eye(3) - k
        self.P = residual @ self.P @ residual.T + k @ self.R @ k.T
        self.P = (self.P + self.P.T) / 2
        self.accepted += 1
        return True
