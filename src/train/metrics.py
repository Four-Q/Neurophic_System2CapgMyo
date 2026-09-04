"""测试集指标计算：混淆矩阵、逐被试准确率、逐类别指标。"""

import torch


def compute_test_metrics(evaluation):
    predictions = evaluation["predictions"].long()
    labels = evaluation["labels"].long()
    subjects = evaluation["subjects"].long()

    confusion = torch.zeros((8, 8), dtype=torch.long)
    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label, predicted_label] += 1
    if int(confusion.sum()) != 288:
        raise RuntimeError(f"测试混淆矩阵应包含 288 个样本，实际为 {int(confusion.sum())}")

    per_subject = []
    for subject in range(1, 19):
        mask = subjects == subject
        total = int(mask.sum())
        correct = int((predictions[mask] == labels[mask]).sum())
        if total != 16:
            raise RuntimeError(f"subject_{subject:02d} 测试样本应为 16，实际为 {total}")
        per_subject.append(
            {"subject_id": subject, "correct": correct, "total": total, "accuracy": correct / total}
        )

    per_class = []
    for label in range(8):
        true_positive = int(confusion[label, label])
        false_positive = int(confusion[:, label].sum()) - true_positive
        false_negative = int(confusion[label, :].sum()) - true_positive
        support = int(confusion[label, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "label": label,
                "gesture_id": label + 1,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return confusion, per_subject, per_class
