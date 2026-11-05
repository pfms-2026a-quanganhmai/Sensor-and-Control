"""完整小车导航与视觉对接入口；默认带障碍，代码包含中文注释。"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    from lidar_evaluation.report import create_report
    from navigation.runner import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=[
            "nominal",
            "obstacle",
            "camera_dropout",
            "lidar_dropout",
            "noisy_encoder",
            "shifted_start",
        ],
        default="obstacle",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, default=0, help="首个噪声/场景随机种子，便于复现实验"
    )
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--run-id", default=time.strftime("nav-%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/navigation")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="记录实际场景、相机和估计不确定性到MP4",
    )
    args = parser.parse_args()
    if (
        args.seed < 0
        or args.trials < 1
        or not math.isfinite(args.duration)
        or args.duration <= 0
        or not math.isfinite(args.speed)
        or args.speed <= 0
    ):
        parser.error("trials, duration and speed must be positive")
    for seed in range(args.seed, args.seed + args.trials):
        result = run(
            ROOT,
            args.output_dir,
            args.condition,
            seed,
            args.duration,
            args.gui,
            args.speed,
            not args.no_plots,
            args.run_id,
            args.record_video,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    create_report(
        args.output_dir / "trials.csv",
        args.output_dir / "report",
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()


#cd .\Sensor-and-Control
#python .\run_navigation.py --gui --speed 2