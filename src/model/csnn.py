"""PLIF-CSNN 模型定义。

对正负分离的 sEMG 输入（``[T, 2, 8, 16]``，T 个时间步、2 通道、
8×16 电极）做逐步卷积处理，最后对完整时间轴上的 logits 求均值用于分类。
"""

import numpy as np

# SpikingJelly 0.0.0.0.14 的 CuPy 内核仍引用 np.int；只在当前进程补兼容别名。
if not hasattr(np, "int"):
    np.int = int

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


INPUT_CHANNELS = 2
TIME_STEPS = 200
NUM_CLASSES = 8


def make_plif(tau):
    return neuron.ParametricLIFNode(
        init_tau=tau,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )


class FullSequencePLIFCSNN(nn.Module):
    def __init__(self, widths, tau, dropout, hidden=0, pool=(4, 8)):
        super().__init__()
        c1, c2, c3 = widths
        hidden = max(256, c3 * 2) if hidden <= 0 else hidden
        # 顺序主干避免完整序列下残差膜电位叠加造成脉冲活动饱和。
        self.features = nn.Sequential(
            layer.Conv2d(INPUT_CHANNELS, c1, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            make_plif(tau),
            layer.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c1),
            make_plif(tau),
            layer.MaxPool2d(kernel_size=2),
            layer.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            make_plif(tau),
            layer.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c2),
            make_plif(tau),
            layer.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(c3),
            make_plif(tau),
            layer.AdaptiveAvgPool2d(pool),
            layer.Flatten(),
        )
        self.classifier = nn.Sequential(
            layer.Linear(c3 * pool[0] * pool[1], hidden, bias=False),
            make_plif(tau),
            nn.Dropout(dropout),
            layer.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, x_seq):
        if x_seq.shape[0] != TIME_STEPS:
            raise ValueError(
                f"模型必须逐步处理 {TIME_STEPS} 个时间步，实际输入 {x_seq.shape[0]}"
            )
        logits = self.classifier(self.features(x_seq))
        # 每个真实时间步独立产生 logits，最后沿完整时间轴求均值。
        return logits.mean(dim=0)


def build_model(widths, tau, device, dropout=0.25, hidden=0, pool=(4, 8)):
    model = FullSequencePLIFCSNN(widths, tau, dropout, hidden, pool).to(device)
    functional.set_step_mode(model, "m")
    functional.set_backend(model, "cupy", instance=neuron.ParametricLIFNode)
    return model


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def forward_logits(model, batch_data):
    if batch_data.shape[1] != TIME_STEPS:
        raise ValueError(f"时间步必须为 {TIME_STEPS}，实际为 {batch_data.shape[1]}")
    x_seq = batch_data.transpose(0, 1).contiguous()
    return model(x_seq)
