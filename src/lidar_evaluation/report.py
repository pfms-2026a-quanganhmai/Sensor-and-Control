"""从团队试验 CSV 生成汇总文件；此命令不会替你生成机器人成功结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .metrics import CSV_COLUMNS, read_trials_csv, summarize_trials


def create_report(
    input_csv: str | Path, output_dir: str | Path, plots: bool = False
) -> dict:
    """读取日志、验证数据，再写出汇总和可追溯的元数据。"""
    source = Path(input_csv)
    records = read_trials_csv(source)
    summaries = summarize_trials(records)
    # 使用者必须明确选 --plots；仅统计时无须安装或导入 matplotlib。
    if plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ValueError("Install matplotlib to use --plots") from exc

    destination = Path(output_dir)
    # 即使使用者把输入文件命名为 summary.csv，也不能覆盖原始日志。
    generated_names = ["summary.csv", "metadata.json"]
    if plots:
        generated_names += ["success_rate.png", "final_position_error.png"]
    if source.resolve() in {(destination / name).resolve() for name in generated_names}:
        raise ValueError("Report output files must not overwrite the input CSV")
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "input_csv": str(source.resolve()),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": (
            "user-provided trial records; provenance must be documented by the team"
        ),
        "trial_count": len(records),
        "conditions": [entry["condition"] for entry in summaries],
        "required_columns": list(CSV_COLUMNS),
        "rate_denominator": "all records in the condition, including failures",
        "duration_denominator": (
            "all records in the condition, including failures and timeouts"
        ),
        "optional_metric_denominator": "non-empty values only; see each metric_count",
        "standard_deviation": "population standard deviation, ddof=0",
        "missing_value_policy": "blank/None; never imputed as zero",
        "plots": plots,
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if plots:
        labels = [item["condition"] for item in summaries]
        width = max(6, min(18, len(labels) * 1.5))
        fig, ax = plt.subplots(figsize=(width, 4.5))
        bars = ax.bar(
            labels, [item["success_rate"] for item in summaries], color="#247ba0"
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Success rate")
        ax.set_title("Task success (all submitted trials)")
        ax.tick_params(axis="x", rotation=20)
        for bar, item in zip(bars, summaries):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{item['success_count']}/{item['trial_count']}",
                ha="center",
            )
        fig.tight_layout()
        fig.savefig(destination / "success_rate.png", dpi=160)
        plt.close(fig)

        # 没有最终误差的组不会被画成 0 米。图上的 n 与成功率的分母可能不同。
        measured = [
            item for item in summaries if item["final_position_error_m_count"] > 0
        ]
        fig, ax = plt.subplots(figsize=(width, 4.5))
        if measured:
            means = [item["final_position_error_m_mean"] for item in measured]
            stds = [item["final_position_error_m_std"] for item in measured]
            ax.bar(
                [item["condition"] for item in measured],
                means,
                yerr=stds,
                capsize=5,
                color="#43aa8b",
            )
            for index, item in enumerate(measured):
                ax.annotate(
                    f"n={item['final_position_error_m_count']}",
                    (
                        index,
                        item["final_position_error_m_mean"]
                        + item["final_position_error_m_std"],
                    ),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                )
            ax.margins(y=0.25)
            ax.set_ylim(bottom=0)
        else:
            ax.text(
                0.5,
                0.5,
                "No final-position-error values supplied",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set_ylabel("Final position error (m)")
        ax.set_title(
            "Final position error (available values)\nMean and population standard deviation"
        )
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(destination / "final_position_error.png", dpi=160)
        plt.close(fig)
    return {
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
        "trial_count": len(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV containing actual trial records")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also write PNG plots (requires matplotlib)",
    )
    args = parser.parse_args(argv)
    try:
        result = create_report(args.csv, args.output_dir, plots=args.plots)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    print(f"Validated {result['trial_count']} trial records.")
    print(f"Summary: {result['summary_csv']}")
    print(f"Metadata: {result['metadata_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
