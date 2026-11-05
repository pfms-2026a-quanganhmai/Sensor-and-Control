"""用独立已知位姿验证模拟相机与量程标定；真值不进入在线控制。"""

import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    from lidar_evaluation.mujoco_scan import MujocoPlanarLidar
    from navigation.control import wrap
    from navigation.runner import scene
    from navigation.vision import CalibratedCamera

    output = ROOT / "docs/results/calibration"
    output.mkdir(parents=True, exist_ok=True)
    model, _ = scene(ROOT, "nominal", 0)
    data = mujoco.MjData(model)
    camera = CalibratedCamera(model)
    lidar = MujocoPlanarLidar(model, "navigation_lidar", excluded_geom_groups=(1,))
    rows = []
    range_rows = []
    try:
        for x in (0.0, 0.3, 0.6, 0.9):
            for y, heading in ((-0.025, -0.025), (0.0, 0.0), (0.025, 0.025)):
                data.qpos[:2] = [x, y]
                data.qpos[3:7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
                mujoco.mj_forward(model, data)
                measured = camera.read(data)
                if measured is None:
                    raise RuntimeError("Known-pose validation failed to detect marker")
                rows.append(
                    {
                        "x_reference_m": x,
                        "y_reference_m": y,
                        "heading_reference_rad": heading,
                        "x_measured_m": measured[0],
                        "y_measured_m": measured[1],
                        "heading_measured_rad": measured[2],
                        "position_error_m": float(
                            np.linalg.norm(measured[:2] - [x, y])
                        ),
                        "heading_error_deg": float(
                            np.rad2deg(abs(wrap(measured[2] - heading)))
                        ),
                        "reprojection_error_px": camera.last_error,
                    }
                )
            data.qpos[:2] = [x, 0]
            data.qpos[3:7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            scan = lidar.read(data)
            distance = float(scan.ranges[np.argmin(abs(scan.angles))])
            reference = 1.5 - x + 0.049
            range_rows.append(
                {
                    "reference_m": reference,
                    "measured_m": distance,
                    "error_m": distance - reference,
                }
            )
    finally:
        camera.close()
    for filename, values in (
        ("camera_known_pose.csv", rows),
        ("lidar_known_surface.csv", range_rows),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = {
        "source": "independent configured virtual sensor calibration validation, not real hardware",
        "camera_intrinsics": camera.K.tolist(),
        "distortion": "zero by virtual camera design",
        "camera_pose_count": len(rows),
        "position_rmse_m": float(
            np.sqrt(np.mean([r["position_error_m"] ** 2 for r in rows]))
        ),
        "heading_rmse_deg": float(
            np.sqrt(np.mean([r["heading_error_deg"] ** 2 for r in rows]))
        ),
        "lidar_max_abs_error_m": float(max(abs(r["error_m"]) for r in range_rows)),
    }
    (output / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()
