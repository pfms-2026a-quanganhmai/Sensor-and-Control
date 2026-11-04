"""按完整相机外参执行位姿转换，避免把底盘俯仰误差当成平面位置。"""

import math

import cv2
import mujoco
import numpy as np


def quat_matrix(quat):
    flat = np.zeros(9)
    mujoco.mju_quat2Mat(flat, np.asarray(quat))
    return flat.reshape(3, 3)


class CalibratedCamera:
    def __init__(self, model):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "aruco_marker_0")
        if cid < 0 or mid < 0:
            raise ValueError("camera or known ArUco landmark absent")
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        f = 480 / (2 * math.tan(math.radians(float(model.cam_fovy[cid])) / 2))
        self.K = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1.0]])
        self.points = np.array(
            [[-0.09, 0.09, 0], [0.09, 0.09, 0], [0.09, -0.09, 0], [-0.09, -0.09, 0]],
            np.float32,
        )
        # 全部来自已知安装标定及静态地标图；没有读取运行中车体的真实位姿。
        self.R_BC = quat_matrix(model.cam_quat[cid]) @ np.diag([1, -1, -1])
        self.t_BC = model.cam_pos[cid].copy()
        self.R_WM = quat_matrix(model.geom_quat[mid])
        self.t_WM = model.geom_pos[mid].copy()
        self.last_image = None
        self.last_error = None

    def read(self, data):
        self.renderer.update_scene(data, camera="front_camera")
        rgb = self.renderer.render().copy()
        self.last_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        )
        if ids is None:
            return None
        cv2.aruco.drawDetectedMarkers(self.last_image, corners, ids)
        candidates = []
        for corner, identity in zip(corners, ids.flatten()):
            if identity != 0:
                continue
            points = corner.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self.points, points, self.K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok or not 0.12 < float(tvec[2, 0]) < 5:
                continue
            projected, _ = cv2.projectPoints(
                self.points, rvec, tvec, self.K, np.zeros(5)
            )
            error = float(
                np.mean(np.linalg.norm(projected.reshape(4, 2) - points, axis=1))
            )
            if error > 2:
                continue
            R_CM, _ = cv2.Rodrigues(rvec)
            R_WC = self.R_WM @ R_CM.T
            R_WB = R_WC @ self.R_BC.T
            t_WC = self.t_WM - R_WC @ tvec.reshape(3)
            t_WB = t_WC - R_WB @ self.t_BC
            if R_WB[2, 2] < 0.7:
                continue
            pose = np.array([t_WB[0], t_WB[1], math.atan2(R_WB[1, 0], R_WB[0, 0])])
            if np.all(np.isfinite(pose)):
                candidates.append((error, pose))
        if not candidates:
            return None
        self.last_error, pose = min(candidates, key=lambda item: item[0])
        return pose

    def close(self):
        self.renderer.close()
