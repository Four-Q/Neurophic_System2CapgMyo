"""训练与评估结果可视化。

遵循科研绘图规范：全英文标签、低饱和度配色、克制的网格与坐标轴。
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# 低饱和度科研配色（蓝 / 赭，色弱友好，克制且可打印）。
COLOR_TRAIN = "#4c78a8"
COLOR_VAL = "#c9763c"
COLOR_SERIES = "#4c78a8"
INK_PRIMARY = "#2b2b2b"
INK_SECONDARY = "#5f5f5f"
GRID_COLOR = "#dcdcdc"
SPINE_COLOR = "#c4c4c4"

# 顺序色带：从近白到低饱和度蓝，用于混淆矩阵这类连续计数。
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "muted_blue",
    ["#f5f8fc", "#d6e2f0", "#a9c0dc", "#7a9cc0", "#4c78a8"],
)


def _style_axes(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)


def plot_training_curves(history_path, output_path, title=None):
    """从 history.csv 绘制 loss 与 accuracy 曲线（不含 learning_rate）。"""
    history_path = Path(history_path).resolve()
    if not history_path.is_file():
        raise FileNotFoundError(f"history.csv 不存在: {history_path}")

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    with history_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_accuracy"]) * 100)
            val_acc.append(float(row["val_accuracy"]) * 100)

    if not epochs:
        raise RuntimeError("history.csv 为空")

    best_index = max(range(len(val_acc)), key=lambda index: val_acc[index])
    best_epoch = epochs[best_index]

    figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    if title:
        figure.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.02)

    loss_axis.plot(epochs, train_loss, label="Train", color=COLOR_TRAIN, linewidth=1.4)
    loss_axis.plot(epochs, val_loss, label="Validation", color=COLOR_VAL, linewidth=1.4)
    loss_axis.set_xlabel("Epoch", color=INK_PRIMARY, fontsize=10)
    loss_axis.set_ylabel("Cross-entropy loss", color=INK_PRIMARY, fontsize=10)
    loss_axis.set_title("Loss", color=INK_PRIMARY, fontsize=11)
    loss_axis.legend(frameon=False, fontsize=9)
    _style_axes(loss_axis)

    acc_axis.plot(epochs, train_acc, label="Train", color=COLOR_TRAIN, linewidth=1.4)
    acc_axis.plot(epochs, val_acc, label="Validation", color=COLOR_VAL, linewidth=1.4)
    acc_axis.axvline(best_epoch, color="#9a9a9a", linestyle="--", linewidth=1.0)
    acc_axis.scatter(
        [best_epoch],
        [val_acc[best_index]],
        color=COLOR_VAL,
        s=26,
        zorder=5,
        label=f"Best val {val_acc[best_index]:.2f}% @ epoch {best_epoch}",
    )
    acc_axis.set_xlabel("Epoch", color=INK_PRIMARY, fontsize=10)
    acc_axis.set_ylabel("Accuracy (%)", color=INK_PRIMARY, fontsize=10)
    acc_axis.set_title("Accuracy", color=INK_PRIMARY, fontsize=11)
    acc_axis.legend(frameon=False, fontsize=9)
    _style_axes(acc_axis)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_confusion_matrix(confusion, output_path):
    confusion = np.asarray(confusion)
    figure, axis = plt.subplots(figsize=(6.6, 5.8))
    image = axis.imshow(confusion, interpolation="nearest", cmap=BLUE_CMAP)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    axis.set(
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=np.arange(1, 9),
        yticklabels=np.arange(1, 9),
    )
    axis.set_xlabel("Predicted gesture", color=INK_PRIMARY, fontsize=10)
    axis.set_ylabel("True gesture", color=INK_PRIMARY, fontsize=10)
    axis.set_title("CapgMyo DB-a Test Confusion Matrix", color=INK_PRIMARY, fontsize=12)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)

    threshold = confusion.max() / 2
    for row in range(8):
        for column in range(8):
            value = int(confusion[row, column])
            axis.text(
                column, row, value, ha="center", va="center", fontsize=9,
                color="white" if value > threshold else INK_PRIMARY,
            )
    axis.grid(False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_per_class_accuracy(per_class, output_path):
    gestures = [row["gesture_id"] for row in per_class]
    accuracies = [row["accuracy"] * 100 for row in per_class]

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    bars = axis.bar(gestures, accuracies, color=COLOR_SERIES, alpha=0.9, width=0.62)
    axis.set_xlabel("Gesture", color=INK_PRIMARY, fontsize=10)
    axis.set_ylabel("Accuracy (%)", color=INK_PRIMARY, fontsize=10)
    axis.set_title("Per-class Test Accuracy", color=INK_PRIMARY, fontsize=12)
    axis.set_xticks(gestures)
    axis.set_ylim(0, 105)
    axis.tick_params(colors=INK_SECONDARY, labelsize=9)
    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{accuracy:.1f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
        )
    _style_axes(axis)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_comparison(rows, output_path):
    """绘制多种输入的测试准确率对比柱状图。"""
    names = [row.get("name", row["data_type"]) for row in rows]
    accuracies = [row["test_accuracy"] * 100 for row in rows]

    figure, axis = plt.subplots(figsize=(8.8, 4.4))
    bars = axis.bar(names, accuracies, color=COLOR_SERIES, alpha=0.9, width=0.62)
    axis.set_xlabel("Input", color=INK_PRIMARY, fontsize=10)
    axis.set_ylabel("Test accuracy (%)", color=INK_PRIMARY, fontsize=10)
    axis.set_title("CapgMyo DB-a CSNN Comparison (T=200)", color=INK_PRIMARY, fontsize=12)
    axis.set_ylim(0, 105)
    axis.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY)
    axis.tick_params(axis="y", labelsize=9, colors=INK_SECONDARY)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
            f"{accuracy:.2f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
        )
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
