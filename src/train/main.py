"""CapgMyo DB-a PLIF-CSNN 训练入口。

对五种正负分离输入分别训练同一个 PLIF-CSNN，并输出历史曲线、测试指标、
混淆矩阵、逐类别/逐被试准确率与最佳模型。
"""

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from ..data.dataset import load_dataset, normalize_and_move
from ..model.csnn import build_model, count_parameters
from . import visualize
from .metrics import compute_test_metrics
from .trainer import evaluate, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 五种输入与对应数据目录、输入类型、是否归一化。
DATA_TYPES = {
    "raw_polarity": {
        "data_root": "CapgMyo_data/raw_polarity_binned/T_200",
        "input_kind": "raw",
        "normalize": False,
        "label": "raw_polarity（无归一化）",
        "name": "Raw polarity",
    },
    "raw_polarity_norm": {
        "data_root": "CapgMyo_data/raw_polarity_binned/T_200",
        "input_kind": "raw",
        "normalize": True,
        "label": "raw_polarity（z-score 归一化）",
        "name": "Raw polarity (z-score)",
    },
    "neurophic_system_spike": {
        "data_root": "CapgMyo_data/neurophic_system_encoding_spike/T_200",
        "input_kind": "spike",
        "normalize": False,
        "label": "神经形态系统脉冲编码",
        "name": "Neuromorphic spike",
    },
    "delta_spike": {
        "data_root": "CapgMyo_data/delta_encoding_spike/T_200",
        "input_kind": "spike",
        "normalize": False,
        "label": "Delta 编码脉冲",
        "name": "Delta spike",
    },
    "threshold_spike": {
        "data_root": "CapgMyo_data/threshold_encoding_spike/T_200",
        "input_kind": "spike",
        "normalize": False,
        "label": "Threshold 编码脉冲",
        "name": "Threshold spike",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="CapgMyo PLIF-CSNN 训练")
    parser.add_argument("--data-type", choices=sorted(DATA_TYPES), required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="覆盖默认数据目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="默认 outputs/<data-type>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--widths", default="24,48,96")
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--pool", default="4,8")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def parse_tuple(text, expected, name):
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != expected or any(value <= 0 for value in values):
        raise ValueError(f"--{name} 必须为 {expected} 个逗号分隔的正整数")
    return values


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def write_confusion_csv(confusion, path):
    rows = []
    for true_label, row in enumerate(confusion.tolist()):
        record = {"true_gesture": true_label + 1}
        for predicted_label, value in enumerate(row):
            record[f"pred_{predicted_label + 1}"] = value
        rows.append(record)
    fieldnames = ["true_gesture"] + [f"pred_{i}" for i in range(1, 9)]
    write_csv(path, rows, fieldnames)


def final_test(model, datasets, output_dir, config, best_epoch, best_val_accuracy):
    evaluation = evaluate(model, datasets["test"], config["eval_batch_size"], collect=True)
    confusion, per_subject, per_class = compute_test_metrics(evaluation)

    write_confusion_csv(confusion, output_dir / "confusion_matrix.csv")
    visualize.plot_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
    write_csv(output_dir / "per_subject.csv", per_subject, list(per_subject[0]))
    write_csv(output_dir / "per_class.csv", per_class, list(per_class[0]))
    visualize.plot_per_class_accuracy(per_class, output_dir / "per_class.png")

    metrics = {
        "test_loss": evaluation["loss"],
        "test_accuracy": evaluation["accuracy"],
        "test_correct": evaluation["correct"],
        "test_total": evaluation["total"],
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "parameter_count": config["parameter_count"],
        "confusion_matrix": confusion.tolist(),
        "per_subject": per_subject,
        "per_class": per_class,
    }
    write_json(output_dir / "test_metrics.json", metrics)
    return evaluation, confusion, per_subject, per_class


def write_summary(output_dir, config, best_epoch, best_val_accuracy, evaluation, per_subject):
    lines = [
        "# CapgMyo CSNN 训练结果",
        "",
        f"- 输入：{config['label']}",
        f"- 测试准确率：{evaluation['accuracy']:.2%}（{evaluation['correct']}/{evaluation['total']}）",
        f"- 最佳验证准确率：{best_val_accuracy:.2%}（epoch {best_epoch}）",
        f"- 卷积宽度：{config['widths']}",
        f"- hidden / pool：{config['hidden']} / {config['pool']}",
        f"- 参数量：{config['parameter_count']:,}",
        f"- PLIF tau：{config['tau']}，dropout：{config['dropout']}",
        f"- 学习率：{config['lr']}，weight decay：{config['weight_decay']}",
        f"- 标签平滑：{config['label_smoothing']}，batch size：{config['batch_size']}",
        f"- 随机种子：{config['seed']}",
        "",
        "## 各被试测试准确率",
        "",
        "| 被试 | 正确/总数 | 准确率 |",
        "| ---: | ---: | ---: |",
    ]
    for row in per_subject:
        lines.append(f"| {row['subject_id']:02d} | {row['correct']}/{row['total']} | {row['accuracy']:.2%} |")
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `history.csv`：训练/验证历史",
            "- `training_curves.png`：loss 与 accuracy 曲线",
            "- `test_metrics.json`：完整测试指标与混淆矩阵",
            "- `confusion_matrix.csv` / `confusion_matrix.png`：混淆矩阵",
            "- `per_class.csv` / `per_class.png`：逐类别准确率",
            "- `per_subject.csv`：逐被试准确率",
            "- `best.pt`：最佳模型",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    data_type,
    seed,
    output_dir,
    data_root=None,
    batch_size=256,
    eval_batch_size=256,
    max_epochs=200,
    widths=(24, 48, 96),
    hidden=160,
    pool=(4, 8),
    tau=2.0,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    dropout=0.0,
):
    """训练一个 data_type + seed，把全部结果写入 output_dir，返回测试指标。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝在 CPU 上启动正式训练")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    spec = DATA_TYPES[data_type]
    data_root = data_root or (PROJECT_ROOT / spec["data_root"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "data_type": data_type,
        "label": spec["label"],
        "name": spec["name"],
        "input_kind": spec["input_kind"],
        "normalize": spec["normalize"],
        "widths": list(widths),
        "hidden": hidden,
        "pool": list(pool),
        "tau": tau,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "dropout": dropout,
        "seed": seed,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_epochs": max_epochs,
    }
    write_json(output_dir / "config.json", config)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 加载数据: {data_root}", flush=True)
    datasets, validation = load_dataset(data_root, spec["input_kind"])
    write_json(output_dir / "data_validation.json", validation)
    normalization = validation["normalization"]
    normalize_and_move(datasets, torch.device("cuda"), spec["input_kind"], spec["normalize"], normalization)

    device = datasets["train"]["data"].device
    model = build_model(tuple(widths), tau, device, dropout, hidden, pool)
    config["parameter_count"] = count_parameters(model)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 开始训练 {data_type} seed={seed}，参数量 {config['parameter_count']:,}", flush=True)
    history, best_epoch, best_val_accuracy = train_model(
        model, datasets, config, output_dir, config, normalization
    )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 最终测试", flush=True)
    evaluation, confusion, per_subject, per_class = final_test(
        model, datasets, output_dir, config, best_epoch, best_val_accuracy
    )

    visualize.plot_training_curves(
        output_dir / "history.csv",
        output_dir / "training_curves.png",
        title=config["name"],
    )
    write_summary(output_dir, config, best_epoch, best_val_accuracy, evaluation, per_subject)

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 完成 {data_type} seed={seed}："
        f"测试准确率 {evaluation['accuracy']:.2%}，输出目录 {output_dir}",
        flush=True,
    )
    return {
        "data_type": data_type,
        "seed": seed,
        "test_accuracy": evaluation["accuracy"],
        "test_correct": evaluation["correct"],
        "test_total": evaluation["total"],
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "parameter_count": config["parameter_count"],
        "output_dir": str(output_dir),
    }


def main():
    args = parse_args()
    widths = parse_tuple(args.widths, 3, "widths")
    pool = parse_tuple(args.pool, 2, "pool")

    output_root = args.output_dir or (PROJECT_ROOT / "outputs" / args.data_type)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root.resolve() / run_id

    run_training(
        args.data_type,
        args.seed,
        output_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        widths=widths,
        hidden=args.hidden,
        pool=pool,
        tau=args.tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
