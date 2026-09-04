"""System_with_TIA 的批量脉冲编码后端。

该模块只保留最终二值脉冲，避免为全量数据保存约 50,001 点的中间波形。
参考波形与单通道基准仍由 :mod:`system_with_tia` 提供。
"""

from concurrent.futures import ProcessPoolExecutor
from math import ceil, exp
import os

import numpy as np
import torch

from .system_with_tia import _PARAMETERS, simulate_system


NATIVE_SAMPLE_COUNT = 1000
DURATION_S = 1.0
INTERNAL_STEP_COUNT = int(ceil(DURATION_S / _PARAMETERS.maximum_step))
INTERNAL_STEP_S = DURATION_S / INTERNAL_STEP_COUNT
PWL_POINT_COUNT = NATIVE_SAMPLE_COUNT + 1

_CUDA_EXTENSION = None


def _validate_batch_inputs(vin_values, T):
    vin = np.asarray(vin_values, dtype=np.float64)
    if vin.ndim != 2 or vin.shape[1] != PWL_POINT_COUNT:
        raise ValueError(
            f"vin_values 必须是 [N, {PWL_POINT_COUNT}]，实际为 {vin.shape}。"
        )
    if vin.shape[0] == 0:
        raise ValueError("vin_values 至少需要包含一条输入流。")
    if not np.all(np.isfinite(vin)):
        raise ValueError("vin_values 包含 NaN 或无穷大。")
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)):
        raise TypeError("T 必须是正整数。")
    if T <= 0:
        raise ValueError("T 必须是正整数。")
    return np.ascontiguousarray(vin), int(T)


def _advance_nbox_vector(voltage, high_resistance, input_voltage):
    """按参考求解器的阈值交点规则批量推进一个 NbOx 振荡器。"""
    parameters = _PARAMETERS
    resistance = np.where(
        high_resistance,
        parameters.rin_nbox,
        parameters.rme_nbox,
    )
    target = input_voltage * resistance / (parameters.nbox_rload + resistance)
    tau = parameters.cparal / (
        1.0 / parameters.nbox_rload + 1.0 / resistance
    )
    next_voltage = target + (voltage - target) * np.exp(-INTERNAL_STEP_S / tau)

    crossed = np.where(
        high_resistance,
        next_voltage >= parameters.vh,
        next_voltage <= parameters.vl,
    )
    if np.any(crossed):
        indices = np.flatnonzero(crossed)
        old_high = high_resistance[indices]
        threshold = np.where(old_high, parameters.vh, parameters.vl)
        ratio = (threshold - target[indices]) / (
            voltage[indices] - target[indices]
        )
        ratio = np.clip(ratio, np.finfo(np.float64).tiny, 1.0)
        crossing_time = np.clip(
            -tau[indices] * np.log(ratio),
            0.0,
            INTERNAL_STEP_S,
        )
        new_high = ~old_high
        new_resistance = np.where(
            new_high,
            parameters.rin_nbox,
            parameters.rme_nbox,
        )
        new_target = (
            input_voltage[indices]
            * new_resistance
            / (parameters.nbox_rload + new_resistance)
        )
        new_tau = parameters.cparal / (
            1.0 / parameters.nbox_rload + 1.0 / new_resistance
        )
        next_voltage[indices] = new_target + (
            threshold - new_target
        ) * np.exp(-(INTERNAL_STEP_S - crossing_time) / new_tau)
        high_resistance[indices] = new_high

    return next_voltage


def simulate_system_batch_spikes_numpy(vin_values, T=1000):
    """使用 NumPy 向量化同一电路方程，供无 CUDA 环境与冒烟测试使用。"""
    vin, T = _validate_batch_inputs(vin_values, T)
    parameters = _PARAMETERS
    run_count = vin.shape[0]

    vout = np.zeros(run_count, dtype=np.float64)
    final_out = np.zeros(run_count, dtype=np.float64)
    vtia = np.zeros(run_count, dtype=np.float64)
    state_x1 = np.zeros(run_count, dtype=np.float64)
    state_x2 = np.zeros(run_count, dtype=np.float64)
    state_x3 = np.zeros(run_count, dtype=np.float64)
    u1_high = np.ones(run_count, dtype=np.bool_)
    u2_high = np.ones(run_count, dtype=np.bool_)
    armed = np.zeros(run_count, dtype=np.bool_)
    last_spike_step = np.full(run_count, -10**9, dtype=np.int64)
    spikes = np.zeros((run_count, T), dtype=np.bool_)

    state_decays = (
        (
            exp(-INTERNAL_STEP_S / (parameters.ron1 * parameters.state_capacitance)),
            exp(-INTERNAL_STEP_S / (parameters.roff1 * parameters.state_capacitance)),
        ),
        (
            exp(-INTERNAL_STEP_S / (parameters.ron2 * parameters.state_capacitance)),
            exp(-INTERNAL_STEP_S / (parameters.roff2 * parameters.state_capacitance)),
        ),
        (
            exp(-INTERNAL_STEP_S / (parameters.ron3 * parameters.state_capacitance)),
            exp(-INTERNAL_STEP_S / (parameters.roff3 * parameters.state_capacitance)),
        ),
    )
    tia_decay = exp(-INTERNAL_STEP_S / (parameters.rf1 * parameters.cf1))
    gain = parameters.rf2 / parameters.rin2
    previous_vin = vin[:, 0].copy()
    refractory_steps = int(ceil(parameters.spike_refractory / INTERNAL_STEP_S))

    for step in range(1, INTERNAL_STEP_COUNT + 1):
        position = step * NATIVE_SAMPLE_COUNT / INTERNAL_STEP_COUNT
        left = min(int(position), NATIVE_SAMPLE_COUNT)
        fraction = position - left
        if left == NATIVE_SAMPLE_COUNT:
            current_vin = vin[:, -1]
        else:
            current_vin = vin[:, left] + fraction * (
                vin[:, left + 1] - vin[:, left]
            )
        interval_vin = 0.5 * (previous_vin + current_vin)
        previous_vin = current_vin

        vout = _advance_nbox_vector(vout, u1_high, interval_vin)
        normalized_gate = np.maximum(
            (vout - parameters.synapse_vth)
            / (parameters.synapse_vnorm - parameters.synapse_vth),
            0.0,
        ) ** parameters.synapse_pdrive
        gate_is_on = vout > parameters.synapse_vth

        decay = np.where(gate_is_on, state_decays[0][0], state_decays[0][1])
        state_x1 = np.where(gate_is_on, normalized_gate, 0.0) + (
            state_x1 - np.where(gate_is_on, normalized_gate, 0.0)
        ) * decay
        decay = np.where(gate_is_on, state_decays[1][0], state_decays[1][1])
        state_x2 = np.where(gate_is_on, normalized_gate, 0.0) + (
            state_x2 - np.where(gate_is_on, normalized_gate, 0.0)
        ) * decay
        decay = np.where(gate_is_on, state_decays[2][0], state_decays[2][1])
        state_x3 = np.where(gate_is_on, normalized_gate, 0.0) + (
            state_x3 - np.where(gate_is_on, normalized_gate, 0.0)
        ) * decay

        conductance = np.maximum(
            parameters.g11 * state_x1 * state_x1
            + parameters.g12 * state_x1 * state_x2
            + parameters.g13 * state_x1 * state_x3
            + parameters.g23 * state_x2 * state_x3
            + parameters.gv * normalized_gate,
            0.0,
        )
        tia_target = -parameters.rf1 * parameters.synapse_vds * conductance
        vtia = tia_target + (vtia - tia_target) * tia_decay
        vtia = np.clip(vtia, parameters.negative_rail, parameters.positive_rail)
        vdrive = (1.0 + gain) * parameters.vref - gain * vtia
        vdrive = np.clip(
            vdrive,
            parameters.negative_rail,
            parameters.positive_rail,
        )

        previous_final = final_out
        final_out = _advance_nbox_vector(final_out, u2_high, vdrive)
        armed |= final_out >= parameters.spike_arm_voltage
        falling = (
            (previous_final > parameters.spike_fire_voltage)
            & (final_out <= parameters.spike_fire_voltage)
        )
        fired = armed & falling & (
            step - last_spike_step >= refractory_steps
        )
        if np.any(fired) and step < INTERNAL_STEP_COUNT:
            spike_bin = int(np.floor((step / INTERNAL_STEP_COUNT) * T))
            spikes[np.flatnonzero(fired), spike_bin] = True
        last_spike_step[fired] = step
        armed[fired] = False

    return spikes


def _reference_worker(arguments):
    vin, T = arguments
    time = np.arange(PWL_POINT_COUNT, dtype=np.float64) / NATIVE_SAMPLE_COUNT
    return simulate_system(time, vin, T=T)["spikes"].astype(np.bool_)


def simulate_system_batch_spikes_reference(vin_values, T=1000, workers=1):
    """逐通道调用基准求解器，主要用于小批量校验。"""
    vin, T = _validate_batch_inputs(vin_values, T)
    workers = max(int(workers), 1)
    if workers == 1:
        rows = [_reference_worker((row, T)) for row in vin]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(
                _reference_worker,
                ((row, T) for row in vin),
                chunksize=max(1, len(vin) // (workers * 4)),
            ))
    return np.stack(rows)


def _cuda_sources():
    parameters = _PARAMETERS
    tau_high = parameters.cparal / (
        1.0 / parameters.nbox_rload + 1.0 / parameters.rin_nbox
    )
    tau_low = parameters.cparal / (
        1.0 / parameters.nbox_rload + 1.0 / parameters.rme_nbox
    )
    decay_high = exp(-INTERNAL_STEP_S / tau_high)
    decay_low = exp(-INTERNAL_STEP_S / tau_low)
    state_decays = [
        (
            exp(-INTERNAL_STEP_S / (ron * parameters.state_capacitance)),
            exp(-INTERNAL_STEP_S / (roff * parameters.state_capacitance)),
        )
        for ron, roff in (
            (parameters.ron1, parameters.roff1),
            (parameters.ron2, parameters.roff2),
            (parameters.ron3, parameters.roff3),
        )
    ]
    tia_decay = exp(-INTERNAL_STEP_S / (parameters.rf1 * parameters.cf1))

    cpp_source = r"""
#include <torch/extension.h>

torch::Tensor system_with_tia_spikes_cuda(torch::Tensor vin, int64_t bins);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "simulate_spikes",
        &system_with_tia_spikes_cuda,
        "Batched System_with_TIA spike encoding (CUDA)"
    );
}
"""

    cuda_source = f"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {{

constexpr int kPwlPoints = {PWL_POINT_COUNT};
constexpr int kNativeSamples = {NATIVE_SAMPLE_COUNT};
constexpr int kInternalSteps = {INTERNAL_STEP_COUNT};
constexpr double kDt = {INTERNAL_STEP_S:.17g};
constexpr double kVh = {parameters.vh:.17g};
constexpr double kVl = {parameters.vl:.17g};
constexpr double kRin = {parameters.rin_nbox:.17g};
constexpr double kRme = {parameters.rme_nbox:.17g};
constexpr double kRload = {parameters.nbox_rload:.17g};
constexpr double kTauHigh = {tau_high:.17g};
constexpr double kTauLow = {tau_low:.17g};
constexpr double kDecayHigh = {decay_high:.17g};
constexpr double kDecayLow = {decay_low:.17g};
constexpr double kSynapseVth = {parameters.synapse_vth:.17g};
constexpr double kSynapseVnorm = {parameters.synapse_vnorm:.17g};
constexpr double kSynapsePdrive = {parameters.synapse_pdrive:.17g};
constexpr double kX1OnDecay = {state_decays[0][0]:.17g};
constexpr double kX1OffDecay = {state_decays[0][1]:.17g};
constexpr double kX2OnDecay = {state_decays[1][0]:.17g};
constexpr double kX2OffDecay = {state_decays[1][1]:.17g};
constexpr double kX3OnDecay = {state_decays[2][0]:.17g};
constexpr double kX3OffDecay = {state_decays[2][1]:.17g};
constexpr double kG11 = {parameters.g11:.17g};
constexpr double kG12 = {parameters.g12:.17g};
constexpr double kG13 = {parameters.g13:.17g};
constexpr double kG23 = {parameters.g23:.17g};
constexpr double kGv = {parameters.gv:.17g};
constexpr double kSynapseVds = {parameters.synapse_vds:.17g};
constexpr double kRf1 = {parameters.rf1:.17g};
constexpr double kTiaDecay = {tia_decay:.17g};
constexpr double kVref = {parameters.vref:.17g};
constexpr double kGain = {parameters.rf2 / parameters.rin2:.17g};
constexpr double kNegativeRail = {parameters.negative_rail:.17g};
constexpr double kPositiveRail = {parameters.positive_rail:.17g};
constexpr double kArmVoltage = {parameters.spike_arm_voltage:.17g};
constexpr double kFireVoltage = {parameters.spike_fire_voltage:.17g};
constexpr int kRefractorySteps = {int(ceil(parameters.spike_refractory / INTERNAL_STEP_S))};

__device__ __forceinline__ double clamp_value(double value, double low, double high) {{
    return fmin(fmax(value, low), high);
}}

__device__ __forceinline__ void advance_nbox(
    double& voltage,
    bool& high_resistance,
    double input_voltage
) {{
    const double resistance = high_resistance ? kRin : kRme;
    const double target = input_voltage * resistance / (kRload + resistance);
    const double tau = high_resistance ? kTauHigh : kTauLow;
    const double decay = high_resistance ? kDecayHigh : kDecayLow;
    double next_voltage = target + (voltage - target) * decay;
    const bool crossed = high_resistance
        ? next_voltage >= kVh
        : next_voltage <= kVl;

    if (crossed) {{
        const double threshold = high_resistance ? kVh : kVl;
        double ratio = (threshold - target) / (voltage - target);
        ratio = clamp_value(ratio, 2.2250738585072014e-308, 1.0);
        double crossing_time = clamp_value(-tau * log(ratio), 0.0, kDt);
        high_resistance = !high_resistance;
        const double new_resistance = high_resistance ? kRin : kRme;
        const double new_target = (
            input_voltage * new_resistance / (kRload + new_resistance)
        );
        const double new_tau = high_resistance ? kTauHigh : kTauLow;
        next_voltage = new_target + (threshold - new_target) * exp(
            -(kDt - crossing_time) / new_tau
        );
    }}
    voltage = next_voltage;
}}

__global__ void simulate_kernel(
    const double* __restrict__ vin,
    bool* __restrict__ spikes,
    int64_t run_count,
    int bins
) {{
    const int64_t run = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (run >= run_count) {{
        return;
    }}

    const double* run_vin = vin + run * kPwlPoints;
    bool* run_spikes = spikes + run * bins;
    double previous_vin = run_vin[0];
    double vout = 0.0;
    double final_out = 0.0;
    double vtia = 0.0;
    double state_x1 = 0.0;
    double state_x2 = 0.0;
    double state_x3 = 0.0;
    bool u1_high = true;
    bool u2_high = true;
    bool armed = false;
    int last_spike_step = -1000000000;

    for (int step = 1; step <= kInternalSteps; ++step) {{
        const double position = (
            static_cast<double>(step) * kNativeSamples / kInternalSteps
        );
        int left = static_cast<int>(position);
        if (left > kNativeSamples) {{
            left = kNativeSamples;
        }}
        const double fraction = position - left;
        const double current_vin = left == kNativeSamples
            ? run_vin[kNativeSamples]
            : run_vin[left] + fraction * (run_vin[left + 1] - run_vin[left]);
        const double interval_vin = 0.5 * (previous_vin + current_vin);
        previous_vin = current_vin;

        advance_nbox(vout, u1_high, interval_vin);
        const double normalized_base = fmax(
            (vout - kSynapseVth) / (kSynapseVnorm - kSynapseVth),
            0.0
        );
        const double normalized_gate = pow(normalized_base, kSynapsePdrive);
        const bool gate_is_on = vout > kSynapseVth;
        const double equilibrium = gate_is_on ? normalized_gate : 0.0;
        state_x1 = equilibrium + (state_x1 - equilibrium) * (
            gate_is_on ? kX1OnDecay : kX1OffDecay
        );
        state_x2 = equilibrium + (state_x2 - equilibrium) * (
            gate_is_on ? kX2OnDecay : kX2OffDecay
        );
        state_x3 = equilibrium + (state_x3 - equilibrium) * (
            gate_is_on ? kX3OnDecay : kX3OffDecay
        );

        const double conductance = fmax(
            kG11 * state_x1 * state_x1
            + kG12 * state_x1 * state_x2
            + kG13 * state_x1 * state_x3
            + kG23 * state_x2 * state_x3
            + kGv * normalized_gate,
            0.0
        );
        const double tia_target = -kRf1 * kSynapseVds * conductance;
        vtia = tia_target + (vtia - tia_target) * kTiaDecay;
        vtia = clamp_value(vtia, kNegativeRail, kPositiveRail);
        double vdrive = (1.0 + kGain) * kVref - kGain * vtia;
        vdrive = clamp_value(vdrive, kNegativeRail, kPositiveRail);

        const double previous_final = final_out;
        advance_nbox(final_out, u2_high, vdrive);
        if (final_out >= kArmVoltage) {{
            armed = true;
        }}
        const bool falling = previous_final > kFireVoltage && final_out <= kFireVoltage;
        const bool fired = (
            armed && falling && step - last_spike_step >= kRefractorySteps
        );
        if (fired) {{
            if (step < kInternalSteps) {{
                const int spike_bin = static_cast<int>(floor(
                    (static_cast<double>(step) / kInternalSteps) * bins
                ));
                if (spike_bin >= 0 && spike_bin < bins) {{
                    run_spikes[spike_bin] = true;
                }}
            }}
            last_spike_step = step;
            armed = false;
        }}
    }}
}}

}}  // namespace

torch::Tensor system_with_tia_spikes_cuda(torch::Tensor vin, int64_t bins) {{
    TORCH_CHECK(vin.is_cuda(), "vin 必须位于 CUDA 设备。 ");
    TORCH_CHECK(vin.scalar_type() == at::kDouble, "vin 必须为 float64。 ");
    TORCH_CHECK(vin.is_contiguous(), "vin 必须连续。 ");
    TORCH_CHECK(vin.dim() == 2 && vin.size(1) == kPwlPoints, "vin shape 必须为 [N, 1001]。 ");
    TORCH_CHECK(bins > 0, "T 必须为正整数。 ");

    at::cuda::CUDAGuard device_guard(vin.device());
    auto output = torch::zeros(
        {{vin.size(0), bins}},
        vin.options().dtype(torch::kBool)
    );
    constexpr int threads = 128;
    const int blocks = static_cast<int>((vin.size(0) + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    simulate_kernel<<<blocks, threads, 0, stream>>>(
        vin.data_ptr<double>(),
        output.data_ptr<bool>(),
        vin.size(0),
        static_cast<int>(bins)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}}
"""
    return cpp_source, cuda_source


def _configure_cuda_arch_list():
    """未显式配置时仅为当前 CUDA 设备设置编译架构。"""
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return os.environ["TORCH_CUDA_ARCH_LIST"]


def load_cuda_backend(verbose=False):
    """按需编译并缓存 CUDA 扩展；CPU 环境导入模块时不会触发编译。"""
    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 环境未发现可用 CUDA 设备。")

    from torch.utils.cpp_extension import CUDA_HOME, load_inline

    if CUDA_HOME is None:
        raise RuntimeError("未找到 CUDA Toolkit，无法编译批量 CUDA 后端。")
    # 仅编译当前可见设备，避免扩展为无关架构重复构建。
    _configure_cuda_arch_list()
    cpp_source, cuda_source = _cuda_sources()
    _CUDA_EXTENSION = load_inline(
        name="capgmyo_system_with_tia_cuda_v1",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _CUDA_EXTENSION


def simulate_system_batch_spikes_cuda(vin_values, T=1000, verbose=False):
    """在 CUDA 上批量求解 1 秒电路并返回 `[N, T]` 布尔脉冲。"""
    vin, T = _validate_batch_inputs(vin_values, T)
    extension = load_cuda_backend(verbose=verbose)
    device = torch.device("cuda", torch.cuda.current_device())
    vin_tensor = torch.as_tensor(vin, dtype=torch.float64, device=device).contiguous()
    spikes = extension.simulate_spikes(vin_tensor, T)
    torch.cuda.synchronize(device)
    return spikes.cpu().numpy()


def simulate_system_batch_spikes(vin_values, T=1000, backend="auto", verbose=False):
    """选择 CUDA 或 NumPy 后端执行批量脉冲编码。"""
    backend = str(backend).lower()
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "numpy"
    if backend == "cuda":
        return simulate_system_batch_spikes_cuda(
            vin_values,
            T=T,
            verbose=verbose,
        )
    if backend == "numpy":
        return simulate_system_batch_spikes_numpy(vin_values, T=T)
    if backend == "reference":
        return simulate_system_batch_spikes_reference(vin_values, T=T)
    raise ValueError("backend 必须是 auto、cuda、numpy 或 reference。")


__all__ = [
    "DURATION_S",
    "INTERNAL_STEP_COUNT",
    "INTERNAL_STEP_S",
    "NATIVE_SAMPLE_COUNT",
    "PWL_POINT_COUNT",
    "load_cuda_backend",
    "simulate_system_batch_spikes",
    "simulate_system_batch_spikes_cuda",
    "simulate_system_batch_spikes_numpy",
    "simulate_system_batch_spikes_reference",
]
