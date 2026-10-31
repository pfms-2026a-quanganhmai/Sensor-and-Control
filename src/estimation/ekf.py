import math

import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class EKF:
    def __init__(self):
        # State:
        # [x, y, theta]
        self.x = np.zeros(3)

        # Initial covariance
        self.P = np.diag([
            0.01,
            0.01,
            math.radians(2.0) ** 2,
        ])

        # Process noise
        self.Q = np.diag([
            0.002,
            0.002,
            math.radians(1.0) ** 2,
        ])

        # Pose measurement noise:
        # [x, y, theta]
        self.R_pose = np.diag([
            0.03 ** 2,
            0.03 ** 2,
            math.radians(3.0) ** 2,
        ])

    def predict(
        self,
        delta_distance,
        delta_heading,
    ):
        x, y, theta = self.x

        theta_mid = (
            theta + delta_heading / 2.0
        )

        x_new = (
            x
            + delta_distance
            * math.cos(theta_mid)
        )

        y_new = (
            y
            + delta_distance
            * math.sin(theta_mid)
        )

        theta_new = wrap_angle(
            theta + delta_heading
        )

        self.x = np.array([
            x_new,
            y_new,
            theta_new,
        ])

        F = np.array([
            [
                1.0,
                0.0,
                -delta_distance
                * math.sin(theta_mid),
            ],
            [
                0.0,
                1.0,
                delta_distance
                * math.cos(theta_mid),
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ])

        self.P = (
            F @ self.P @ F.T
            + self.Q
        )

    def update_pose(
        self,
        measured_x,
        measured_y,
        measured_heading,
    ):
        z = np.array([
            measured_x,
            measured_y,
            measured_heading,
        ])

        # Measurement model:
        # z = Hx
        H = np.eye(3)

        innovation = z - self.x

        innovation[2] = wrap_angle(
            innovation[2]
        )

        S = (
            H @ self.P @ H.T
            + self.R_pose
        )

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        self.x[2] = wrap_angle(
            self.x[2]
        )

        I = np.eye(3)

        self.P = (
            I - K @ H
        ) @ self.P