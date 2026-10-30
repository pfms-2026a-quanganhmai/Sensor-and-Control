"""个人部分的一键入口：中文注释；先做部件验证，再接团队实际实验日志。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接在仓库根目录运行本脚本，无须先修改系统 PYTHONPATH。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "self-check", "mujoco", "report", "plot-trial"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "my_part")
    parser.add_argument(
        "--trials-csv", type=Path, help="团队实际实验记录，仅 report 模式使用"
    )
    parser.add_argument(
        "--trajectory-csv", type=Path, help="含真值、估计和协方差的原始轨迹日志"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="只生成 CSV/JSON，不绘图"
    )
    args = parser.parse_args()
    try:
        if args.mode in ("all", "self-check"):
            from lidar_evaluation.demo import run_self_check

            result = run_self_check(args.output_dir / "self_check", not args.no_plots)
            print("[合成部件自检 / synthetic component check]")
            print(f"检查通过 {result['checks_passed']}/{result['check_count']}")
            if result["checks_passed"] != result["check_count"]:
                return 1
        if args.mode in ("all", "mujoco"):
            from lidar_evaluation.demo import run_mujoco_check

            result = run_mujoco_check(
                args.output_dir / "mujoco_check", not args.no_plots
            )
            print("[MuJoCo 静态传感器验证 / static sensor check]")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["checks_passed"] != result["check_count"]:
                return 1
        if args.mode == "report":
            if args.trials_csv is None:
                parser.error("report 模式需要 --trials-csv，不能用部件自检代替导航成绩")
            from lidar_evaluation.report import create_report

            print(
                create_report(
                    args.trials_csv, args.output_dir / "team_report", not args.no_plots
                )
            )
        if args.mode == "plot-trial":
            if args.trajectory_csv is None:
                parser.error("plot-trial 模式需要 --trajectory-csv")
            from lidar_evaluation.plotting import plot_logged_trial

            print(
                plot_logged_trial(args.trajectory_csv, args.output_dir / "trial_plots")
            )
    except (ImportError, ValueError, OSError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        print(
            '请先运行 python -m pip install -e ".[evaluation,simulation]"',
            file=sys.stderr,
        )
        return 2
    print(f"输出位置：{args.output_dir.resolve()}")
    print("注意：部件验证不等于 TurtleBot3 导航或对接成功率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
