"""汇总五种输入的测试结果，生成对比表与对比图。"""

import csv
import json
from pathlib import Path

from . import visualize
from .main import DATA_TYPES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def collect_rows():
    rows = []
    for data_type in sorted(DATA_TYPES):
        type_dir = OUTPUT_ROOT / data_type
        if not type_dir.is_dir():
            continue
        metrics_paths = sorted(type_dir.glob("*/test_metrics.json"))
        if not metrics_paths:
            continue
        latest = max(metrics_paths, key=lambda path: path.parent.name)
        metrics = json.loads(latest.read_text(encoding="utf-8"))
        rows.append(
            {
                "data_type": data_type,
                "label": DATA_TYPES[data_type]["label"],
                "name": DATA_TYPES[data_type]["name"],
                "test_accuracy": metrics["test_accuracy"],
                "test_correct": metrics["test_correct"],
                "test_total": metrics["test_total"],
                "best_val_accuracy": metrics.get("best_val_accuracy"),
                "parameter_count": metrics.get("parameter_count"),
            }
        )
    return rows


def write_comparison(rows):
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "data_type", "label", "test_accuracy", "test_correct", "test_total",
        "best_val_accuracy", "parameter_count",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    visualize.plot_comparison(rows, output_dir / "comparison.png")

    lines = [
        "# 五种输入对比（T=200 正负分离数据）",
        "",
        "| 输入 | 参数量 | 最佳验证准确率 | 测试准确率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        parameter_count = row["parameter_count"] or 0
        best_val = row["best_val_accuracy"]
        best_val_text = f"{best_val:.2%}" if best_val is not None else "-"
        lines.append(
            f"| {row['label']} | {parameter_count:,} | {best_val_text} "
            f"| {row['test_accuracy']:.2%}（{row['test_correct']}/{row['test_total']}） |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"对比结果已写入 {output_dir / 'comparison.md'}")


def main():
    rows = collect_rows()
    if not rows:
        raise RuntimeError("未找到任何 test_metrics.json，请先运行各数据类型的训练。")
    write_comparison(rows)


if __name__ == "__main__":
    main()
