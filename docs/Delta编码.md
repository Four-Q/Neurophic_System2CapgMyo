# Delta 编码

## 目的

把正负分离的 sEMG 时序数据 `raw_polarity` 从 `[1000, 2, 8, 16]` 浮点幅度转换为 `[T, 2, 8, 16]` 二值脉冲（本项目 `T=200`），只保留「相邻时刻变化是否发生」的信息，不区分变化的上升与下降方向，因此不增加通道数。

## 编码方案

编码首先在原生时间轴计算绝对差分：

$$
D_0=0,\qquad D_t=|X_t-X_{t-1}|
$$

随后把 1000 个差分时刻无重叠地划分到 `T` 个区间，在每个区间内取最大值并阈值化：

$$
B_k=\left\{t\mid \left\lfloor\frac{tT}{1000}\right\rfloor=k\right\}
$$

$$
M_k=\max_{t\in B_k}D_t,\qquad S_k=\mathbf{1}(M_k>\theta)
$$

固定阈值时，该过程等价于在原生时间轴产生绝对值 Delta 事件，再在每个目标时间区间内执行逻辑 OR。

## 阈值估计

- `raw_polarity` 使用训练集估计一个全局标量阈值 $\theta$；
- 验证集和测试集不参与阈值估计，避免数据泄漏；
- 使用流式高分辨率直方图（65536 bin）估计目标分位数，使训练集总体发放率逼近 `TARGET_SPIKE_RATE = 0.10`；
- 阈值确定后，对 train/val/test 三个划分统一应用。

## 参数

```python
T = 200
TARGET_SPIKE_RATE = 0.10
```

`T` 支持 2～1000 的整数，包括不能整除 1000 的取值。

## 输出

```text
CapgMyo_data/delta_encoding_spike/T_200/
├── train/{subject_*}/sXX_gXX_rXX.pt
├── val/{subject_*}/sXX_gXX_rXX.pt
├── test/{subject_*}/sXX_gXX_rXX.pt
└── manifest.json（位于 T_200 层）
```

输出 `data` 使用 `torch.bool`，shape 为 `[200, 2, 8, 16]`；文件名、被试目录与全部标签元数据保持不变。
