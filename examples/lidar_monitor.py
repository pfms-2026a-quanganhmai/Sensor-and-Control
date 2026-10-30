"""读取 LiDAR 并显示结果，不主动驱动机器人；断开时已有接口会发送停车命令。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from lidar_evaluation import ScanData, evaluate_front
from rosbridge_api import RosbridgeRobot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "host", help="已有 rosbridge 服务的主机地址；不是 MuJoCo 场景路径"
    )
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--stop-distance", type=float, default=0.45)
    parser.add_argument("--half-angle", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--csv", type=Path, default=Path("results/lidar_monitor.csv"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    # 用 x 模式避免重复运行时悄悄覆盖已有测量证据。
    with args.csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample",
                "received_monotonic_s",
                "status",
                "nearest_distance_m",
                "valid_fraction",
            ),
        )
        writer.writeheader()
        with RosbridgeRobot(args.host, port=args.port) as robot:
            for index in range(args.samples):
                raw = robot.query_laser_scan(timeout=1.0, wait_for_new=True)
                if raw is None:
                    row = {
                        "sample": index,
                        "received_monotonic_s": time.monotonic(),
                        "status": "unknown",
                        "nearest_distance_m": None,
                        "valid_fraction": 0,
                    }
                else:
                    scan = ScanData(
                        raw.angles, raw.ranges, raw.range_min, raw.range_max
                    )
                    result = evaluate_front(scan, args.half_angle, args.stop_distance)
                    row = {
                        "sample": index,
                        "received_monotonic_s": time.monotonic(),
                        "status": result.status,
                        "nearest_distance_m": result.nearest_distance_m,
                        "valid_fraction": result.valid_fraction,
                    }
                # 接收时刻不等于传感器采样时刻，不能用来证明与相机/编码器同步。
                writer.writerow(row)
                stream.flush()
                print(row)


if __name__ == "__main__":
    main()
