"""CapgMyo DB-a 数据加载与校验。

统一处理两类输入：原始极性幅度（float32，可选 z-score 归一化）与二值脉冲
（bool，直接转换为 float32）。所有划分只按 repetition 编号固定划分，训练统计
量仅来自训练集。
"""

import re
from collections import Counter
from pathlib import Path

import torch

from ..model.csnn import INPUT_CHANNELS, TIME_STEPS


EXPECTED_SPLITS = {
    "train": {"count": 1008, "repetitions": {1, 3, 4, 5, 6, 7, 8}},
    "val": {"count": 144, "repetitions": {10}},
    "test": {"count": 288, "repetitions": {2, 9}},
}
REQUIRED_KEYS = {"data", "label", "subject_id", "gesture_id", "repetition_id"}
FILENAME_PATTERN = re.compile(r"s(\d+)_g(\d+)_r(\d+)\.pt$")


def _scalar_int(value, name, path):
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{path}: {name} 必须是单元素 Tensor")
    return int(value.item())


def load_dataset(data_root, input_kind):
    """加载并校验数据集。

    Parameters
    ----------
    data_root : Path
        包含 train/val/test 子目录的数据根目录。
    input_kind : str
        ``"raw"`` 表示原始幅度 float32；``"spike"`` 表示二值脉冲 bool。

    Returns
    -------
    (datasets, validation)
    """
    if input_kind not in ("raw", "spike"):
        raise ValueError(f"input_kind 必须是 raw 或 spike，实际为 {input_kind!r}")

    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    datasets = {}
    validation = {"data_root": str(data_root), "splits": {}}
    train_sum = 0.0
    train_sum_sq = 0.0
    train_value_count = 0

    for split, expected in EXPECTED_SPLITS.items():
        paths = sorted((data_root / split).glob("subject_*/*.pt"))
        if len(paths) != expected["count"]:
            raise ValueError(f"{split} 应有 {expected['count']} 个文件，实际为 {len(paths)}")

        data_items = []
        labels = []
        subjects = []
        gestures = []
        repetitions = []

        for path in paths:
            sample = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(sample, dict) or not REQUIRED_KEYS.issubset(sample):
                raise ValueError(f"{path}: 样本字段不完整")

            data = sample["data"]
            expected_shape = (TIME_STEPS, INPUT_CHANNELS, 8, 16)
            if not isinstance(data, torch.Tensor) or tuple(data.shape) != expected_shape:
                raise ValueError(
                    f"{path}: data shape 应为 [{TIME_STEPS},2,8,16]，"
                    f"实际为 {getattr(data, 'shape', None)}"
                )
            if input_kind == "spike" and data.dtype != torch.bool:
                raise ValueError(f"{path}: 脉冲数据 dtype 应为 bool，实际为 {data.dtype}")
            if input_kind == "raw" and data.dtype != torch.float32:
                raise ValueError(f"{path}: 原始幅度数据 dtype 应为 float32，实际为 {data.dtype}")

            label = _scalar_int(sample["label"], "label", path)
            subject = _scalar_int(sample["subject_id"], "subject_id", path)
            gesture = _scalar_int(sample["gesture_id"], "gesture_id", path)
            repetition = _scalar_int(sample["repetition_id"], "repetition_id", path)

            match = FILENAME_PATTERN.match(path.name)
            if match is None:
                raise ValueError(f"{path}: 文件名不符合 sXX_gXX_rXX.pt")
            file_subject, file_gesture, file_repetition = map(int, match.groups())
            if (subject, gesture, repetition) != (file_subject, file_gesture, file_repetition):
                raise ValueError(f"{path}: 文件名与样本元数据不一致")
            if not 0 <= label < 8 or label != gesture - 1:
                raise ValueError(f"{path}: label/gesture_id 不合法")
            if not 1 <= subject <= 18:
                raise ValueError(f"{path}: subject_id 不合法")

            if split == "train":
                if input_kind == "spike":
                    train_sum += data.sum(dtype=torch.float64).item()
                else:
                    values = data.to(dtype=torch.float64)
                    train_sum += values.sum().item()
                    train_sum_sq += (values * values).sum().item()
                train_value_count += data.numel()

            data_items.append(data)
            labels.append(label)
            subjects.append(subject)
            gestures.append(gesture)
            repetitions.append(repetition)

        repetition_set = set(repetitions)
        if repetition_set != expected["repetitions"]:
            raise ValueError(f"{split}: repetition 集合异常: {sorted(repetition_set)}")

        label_counts = Counter(labels)
        subject_counts = Counter(subjects)
        expected_label_count = expected["count"] // 8
        expected_subject_count = expected["count"] // 18
        if set(label_counts) != set(range(8)) or set(label_counts.values()) != {expected_label_count}:
            raise ValueError(f"{split}: 类别分布异常: {dict(label_counts)}")
        if set(subject_counts) != set(range(1, 19)) or set(subject_counts.values()) != {expected_subject_count}:
            raise ValueError(f"{split}: 被试分布异常: {dict(subject_counts)}")

        datasets[split] = {
            "data": torch.stack(data_items).contiguous(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "subjects": torch.tensor(subjects, dtype=torch.long),
            "gestures": torch.tensor(gestures, dtype=torch.long),
            "repetitions": torch.tensor(repetitions, dtype=torch.long),
        }
        validation["splits"][split] = {
            "count": len(paths),
            "shape": list(datasets[split]["data"].shape),
            "label_counts": dict(sorted(label_counts.items())),
            "subject_counts": dict(sorted(subject_counts.items())),
            "repetitions": sorted(repetition_set),
        }

    repetition_sets = [EXPECTED_SPLITS[name]["repetitions"] for name in ("train", "val", "test")]
    if any(repetition_sets[i] & repetition_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/val/test 的 repetition 存在交集")

    if input_kind == "spike":
        normalization = {
            "kind": "none_binary_spikes",
            "mean": 0.0,
            "std": 1.0,
            "value_count": train_value_count,
            "train_spike_rate": train_sum / train_value_count,
        }
    else:
        mean = train_sum / train_value_count
        variance = train_sum_sq / train_value_count - mean * mean
        std = float(max(variance, 1e-12) ** 0.5)
        normalization = {
            "kind": "raw_amplitude",
            "mean": mean,
            "std": std,
            "value_count": train_value_count,
        }

    validation["normalization"] = normalization
    return datasets, validation


def normalize_and_move(datasets, device, input_kind, normalize, normalization):
    """把数据常驻到设备，并按需执行 z-score 归一化。"""
    for split, values in datasets.items():
        data = values["data"].to(dtype=torch.float32)
        if input_kind == "raw" and normalize:
            data = (data - normalization["mean"]) / normalization["std"]
        values["data"] = data.contiguous()
        for key in ("data", "labels", "subjects", "gestures", "repetitions"):
            values[key] = values[key].to(device, non_blocking=True)
