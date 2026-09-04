# Threshold 编码

## 目的

把正负分离的 sEMG 时序数据 `raw_polarity` 从 `[1000, 2, 8, 16]` 浮点幅度转换为 `[T, 2, 8, 16]` 二值脉冲（本项目 `T=200`）。与 Delta 编码不同，Threshold 编码直接对幅度本身阈值化，保留「幅值是否超过阈值」的信息。

## 编码方案

原生 1000 个时刻通过下式划分到 `T` 个无重叠区间：

$$
B_k=\left\{t\mid \left\lfloor\frac{tT}{1000}\right\rfloor=k\right\}
$$

`raw_polarity` 保持正、负两个非负通道并分别编码：

$$
S_k^{\pm}=\mathbf{1}\left(\max_{t\in B_k}X_t^{\pm}>\theta\right)
$$

固定阈值时，区间最大值等价于先在原生时间轴产生脉冲，再在每个区间执行逻辑 OR。`T=1000` 时不进行时间聚合。

## 阈值估计

- `raw_polarity` 使用一个全局标量阈值 $\theta$；
- 阈值仅由训练集估计，验证集和测试集不参与；
- 目标发射率按训练集全部 trial、时间、通道和电极位置总体计算；
- 使用流式 65536-bin 直方图估计分位数，使训练集总体发放率逼近 `TARGET_SPIKE_RATE = 0.10`。

## 参数

```python
T = 200
TARGET_SPIKE_RATE = 0.10
```

`T` 支持 2～1000 的整数，包括不能整除 1000 的取值。

## 输出

```text
CapgMyo_data/threshold_encoding_spike/T_200_target_rate0.1/raw_polarity/
├── train/{subject_*}/sXX_gXX_rXX.pt
├── val/{subject_*}/sXX_gXX_rXX.pt
├── test/{subject_*}/sXX_gXX_rXX.pt
└── manifest.json（位于 T_200_target_rate0.1 层）
```

输出 `data` 使用 `torch.bool`，shape 为 `[200, 2, 8, 16]`；文件名、被试目录与全部标签元数据保持不变。
