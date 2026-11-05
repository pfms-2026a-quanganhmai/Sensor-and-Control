"""直接记录真实仿真帧与评价数值；真值显示仅用于演示和验收。"""

import cv2
import mujoco
import numpy as np


class Recorder:
    def __init__(self, model, path):
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.7, 0, 0.05]
        self.camera.distance = 3.6
        self.camera.azimuth = 105
        self.camera.elevation = -65
        self.option = mujoco.MjvOption()
        # 停靠区是无碰撞装饰组4，默认显示选项会隐藏它；录像需明确显示。
        self.option.geomgroup[4] = 1
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 640)
        )
        if not self.writer.isOpened():
            raise RuntimeError("Video encoder unavailable")
        self.last = None

    def write(self, data, image, estimate, covariance, truth, stage, safety, command):
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)
        world = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        canvas = np.full((640, 960, 3), 24, np.uint8)
        canvas[:480, :640] = world
        if image is not None:
            canvas[:240, 640:] = cv2.resize(image, (320, 240))
        lines = [
            f"SIM TIME {data.time:.1f} s",
            stage,
            f"LiDAR: {safety}",
            f"v {command[0]:.3f} m/s",
            f"w {command[1]:.3f} rad/s",
            "Blue pad: target",
            "Green pad: pose held",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (650, 275 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (220, 240, 240),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "Estimated / true state and marginal +/-2 sigma (truth: evaluation only)",
            (15, 508),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (0, 230, 230),
            1,
            cv2.LINE_AA,
        )
        sigma = 2 * np.sqrt(np.maximum(np.diag(covariance), 0))
        for i, label in enumerate(["x (m)", "y (m)", "yaw (deg)"]):
            scale = 180 / np.pi if i == 2 else 1
            text = f"{label}: estimate {estimate[i] * scale:.3f} | true {truth[i] * scale:.3f} | +/- {sigma[i] * scale:.3f}"
            cv2.putText(
                canvas,
                text,
                (20, 540 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        self.writer.write(canvas)
        self.last = canvas

    def close(self, result=None):
        if self.last is not None and result is not None:
            frame = self.last.copy()
            message = f"{'PASS' if result.success else 'FAIL'}: position {result.final_position_error_m:.3f} m, yaw {result.final_heading_error_deg:.2f} deg"
            cv2.rectangle(frame, (0, 0), (960, 42), (20, 20, 20), -1)
            cv2.putText(
                frame,
                message,
                (15, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if result.success else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            for _ in range(20):
                self.writer.write(frame)
        self.writer.release()
        self.renderer.close()
