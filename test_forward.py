"""队友的固定轮速前进检查；修复模型路径，增加无界面验证选项。"""

import argparse
import time
from pathlib import Path

import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.duration <= 60:
        parser.error("duration must be in (0,60]")
    # 最新 main 已自带全部资源，不再依赖用户工作目录或旁边的额外模型文件夹。
    scene = (
        Path(__file__).resolve().parent
        / "models/robotis_tb3/scene_turtlebot3_waffle_pi.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    viewer = None
    if not args.headless:
        import mujoco.viewer as viewer_module

        viewer = viewer_module.launch_passive(model, data)
    try:
        while data.time < args.duration and (viewer is None or viewer.is_running()):
            start = time.monotonic()
            data.ctrl[:] = 4.0
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - start)))
        displacement = float(data.qpos[0])
        print(f"Forward displacement: {displacement:.4f} m after {data.time:.3f} s")
        if displacement <= 0:
            raise RuntimeError("Robot did not move in the positive X direction")
    finally:
        data.ctrl[:] = 0
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
