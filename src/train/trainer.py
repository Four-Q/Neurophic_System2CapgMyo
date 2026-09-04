"""PLIF-CSNN 训练循环与评估。"""

import csv
import os
import random
import time

import numpy as np
import torch
from torch import nn
from spikingjelly.activation_based import functional

from ..model.csnn import forward_logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_indices(size, batch_size, shuffle, device):
    if shuffle:
        indices = torch.randperm(size, device=device)
    else:
        indices = torch.arange(size, device=device)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def evaluate(model, split_data, batch_size, collect=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_subjects = []

    for indices in batch_indices(len(split_data["labels"]), batch_size, False, split_data["data"].device):
        try:
            logits = forward_logits(model, split_data["data"][indices])
            labels = split_data["labels"][indices]
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            predictions = logits.argmax(dim=1)
        finally:
            functional.reset_net(model)
        total_loss += loss.item()
        total_correct += (predictions == labels).sum().item()
        total += len(indices)
        if collect:
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_subjects.append(split_data["subjects"][indices].cpu())

    result = {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "correct": total_correct,
        "total": total,
    }
    if collect:
        result.update(
            predictions=torch.cat(all_predictions),
            labels=torch.cat(all_labels),
            subjects=torch.cat(all_subjects),
        )
    return result


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def _atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train_model(model, datasets, config, run_dir, run_config, normalization):
    """训练一个候选配置，返回 (history, best_epoch, best_val_accuracy)。"""
    set_seed(config["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["max_epochs"],
        eta_min=1e-5,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = -1.0
    best_epoch = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_started = time.monotonic()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for indices in batch_indices(
            len(datasets["train"]["labels"]),
            config["batch_size"],
            True,
            datasets["train"]["data"].device,
        ):
            optimizer.zero_grad(set_to_none=True)
            try:
                labels = datasets["train"]["labels"][indices]
                batch_data = datasets["train"]["data"][indices]
                logits = forward_logits(model, batch_data)
                loss = nn.functional.cross_entropy(
                    logits, labels, label_smoothing=config["label_smoothing"]
                )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("loss 非有限值")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not torch.isfinite(gradient_norm).item():
                    raise FloatingPointError("梯度非有限值")
                optimizer.step()
            finally:
                # 在反向传播后重置，避免跨 batch 泄漏膜电位。
                functional.reset_net(model)
            train_loss_sum += loss.item() * len(indices)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += len(indices)

        scheduler.step()
        validation = evaluate(model, datasets["val"], config["eval_batch_size"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        _write_csv(run_dir / "history.csv", history, list(row))

        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_epoch = epoch
            _atomic_torch_save(
                run_dir / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "run_config": run_config,
                    "epoch": epoch,
                    "best_val_accuracy": best_val,
                    "history": history,
                    "normalization": normalization,
                },
            )

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch={epoch:02d} "
            f"train={row['train_accuracy']:.2%} val={row['val_accuracy']:.2%} "
            f"best={best_val:.2%} time={row['epoch_seconds']:.1f}s",
            flush=True,
        )

    return history, best_epoch, best_val
