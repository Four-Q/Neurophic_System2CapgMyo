# Vin 输入编码方式

## 1. 目的

本方法把 CapgMyo 连续 sEMG 信号映射为适合 `System_with_TIA` 单极性输入端的电压序列（即神经形态电路的输入电压 Vin），并由电路动力学产生二值脉冲，作为脉冲编码的统一输入方案。

主要目标：

- 映射参数只由训练集计算一次，验证集和测试集严格复用；
- 所有样本、受试者、时间点、空间电极和极性流共享同一组分位点；
- 保留不同样本和不同电极之间的绝对幅值关系与空间活动分布；
- 将训练集低幅值区域映射到 NbOx 启动点以下，降低弱信号和噪声引起的发放；
- 为训练集主体幅值区间分配最大的电压动态范围；
- 压缩高幅值长尾，并把极端幅值限制在 2.5 V；
- 保持映射连续、单调、确定且易于在软件和近传感硬件中实现。

在本项目中，只对正负分离数据 `raw_polarity` 进行脉冲编码；映射统计量仍来自 `raw/train` 的原始有符号信号绝对幅值。

## 2. 输入定义

单个 CapgMyo 样本的 shape 为：

| 数据源 | 输入 shape | 映射后的电压 shape |
| --- | --- | --- |
| `raw` | `[1000,1,8,16]` | `[1000,1,8,16]` |
| `raw_polarity` | `[1000,2,8,16]` | `[1000,2,8,16]` |

其中：

- 1000 个时间点对应 1000 Hz、1 s 的真实时间轴；
- `raw` 为有符号 sEMG 信号；
- `raw_polarity[:,0]` 为 positive 幅值；
- `raw_polarity[:,1]` 为 negative magnitude；
- 映射不执行空间求和、空间平均、逐样本归一化或逐电极归一化。

## 3. 训练集统计口径

### 3.1 统计数据来源

映射参数只能由 `CapgMyo_data/raw/train` 中的训练集原始信号计算。将训练集全部样本、全部时间点和全部 `8×16` 电极位置的绝对幅值汇总为：

$$
\mathcal{A}_{\mathrm{train}}
=
\left\{|x|\mid x\text{ 来自 raw/train}\right\}
$$

统计时直接对原始有符号信号取绝对值，不在正负分离后的两个通道上计算分位点，避免 `raw_polarity` 中大量结构性零值改变分位点含义。

### 3.2 可调分位参数

默认使用：

$$
p_{\mathrm{low}}=0.10
$$

$$
p_{\mathrm{mid}}=0.90
$$

$$
p_{\mathrm{high}}=0.995
$$

三个概率值应作为数据准备流程的显式配置参数，且必须满足：

$$
0<p_{\mathrm{low}}<p_{\mathrm{mid}}<p_{\mathrm{high}}<1
$$

对应训练集全局分位点为：

$$
\boxed{
q_{\mathrm{low}}
=Q_{p_{\mathrm{low}}}(\mathcal{A}_{\mathrm{train}})
}
$$

$$
\boxed{
q_{\mathrm{mid}}
=Q_{p_{\mathrm{mid}}}(\mathcal{A}_{\mathrm{train}})
}
$$

$$
\boxed{
q_{\mathrm{high}}
=Q_{p_{\mathrm{high}}}(\mathcal{A}_{\mathrm{train}})
}
$$

参数必须满足：

$$
0<q_{\mathrm{low}}<q_{\mathrm{mid}}<q_{\mathrm{high}}
$$

若不满足该条件，应停止数据转换并检查训练数据、分位参数或统计实现，不允许回退到逐样本映射参数。

## 4. 固定电压锚点

使用以下四个电压锚点：

| 幅值锚点 | 电压 | 含义 |
| --- | ---: | --- |
| $0$ | 1.0 V | 零输入基线 |
| $q_{\mathrm{low}}$ | 1.8639 V | 第一级 NbOx 振荡器启动点 |
| $q_{\mathrm{mid}}$ | 2.4 V | 训练集主体高端电压 |
| $q_{\mathrm{high}}$ | 2.5 V | 高幅值上限 |

因此：

$$
V(0)=1.0\ \mathrm{V}
$$

$$
V(q_{\mathrm{low}})=1.8639\ \mathrm{V}
$$

$$
V(q_{\mathrm{mid}})=2.4\ \mathrm{V}
$$

$$
V(q_{\mathrm{high}})=2.5\ \mathrm{V}
$$

大于等于 $q_{\mathrm{high}}$ 的幅值均限制为 2.5 V。

## 5. 三段线性映射公式

对任意非负活动量 $a$，定义：

$$
\boxed{
V(a)=
\begin{cases}
1.0+(1.8639-1.0)\dfrac{a}{q_{\mathrm{low}}},
&0\le a\le q_{\mathrm{low}}\\[12pt]
1.8639+(2.4-1.8639)
\dfrac{a-q_{\mathrm{low}}}
{q_{\mathrm{mid}}-q_{\mathrm{low}}},
&q_{\mathrm{low}}<a\le q_{\mathrm{mid}}\\[12pt]
2.4+(2.5-2.4)
\dfrac{a-q_{\mathrm{mid}}}
{q_{\mathrm{high}}-q_{\mathrm{mid}}},
&q_{\mathrm{mid}}<a<q_{\mathrm{high}}\\[12pt]
2.5,
&a\ge q_{\mathrm{high}}
\end{cases}
}
$$

该映射在 $q_{\mathrm{low}}$、$q_{\mathrm{mid}}$ 和 $q_{\mathrm{high}}$ 处连续，并在完整定义域内单调不减。

### 5.1 低幅值区间

训练集 $0$ 至 $p_{\mathrm{low}}$ 分位的幅值映射到 1.0～1.8639 V，设计意图是使训练集中前 10% 的低幅值数据处于不起振或临界起振区，降低弱活动和噪声引起的输出发放。

### 5.2 主体幅值区间

训练集 $p_{\mathrm{low}}$ 至 $p_{\mathrm{mid}}$ 分位的主体幅值映射到 1.8639～2.4 V，获得最大的电压动态范围。

### 5.3 高幅值区间

训练集 $p_{\mathrm{mid}}$ 至 $p_{\mathrm{high}}$ 分位的强活动映射到 2.4～2.5 V，对高幅值长尾进行压缩，同时保留其强度顺序。

### 5.4 极端幅值

超过训练集 $p_{\mathrm{high}}$ 分位的幅值统一映射为 $V(a)=2.5\ \mathrm{V}$，仅对训练集最高 0.5% 的幅值执行饱和限制。

## 6. `raw_polarity` 映射方案

对原始有符号信号执行正负分离：

$$
x^+(t)=\max(x(t),0)
$$

$$
x^-(t)=\max(-x(t),0)
$$

positive 和 negative 两路分别映射：

$$
V^+(t)=V\left(x^+(t)\right)
$$

$$
V^-(t)=V\left(x^-(t)\right)
$$

输出保持 `[1000,2,8,16]`，通道顺序固定为 `[positive, negative]`。两路必须共享从 `raw/train` 计算的同一组 $q_{\mathrm{low}}$、$q_{\mathrm{mid}}$ 和 $q_{\mathrm{high}}$，不得分别估计 positive 和 negative 的尺度。当某一极性在当前时刻为零时，该路电压固定为 1.0 V。

## 7. 电路脉冲产生

映射得到的 Vin 电压序列以 1000 个采样点 + 1 个 PWL 拐点（重复末值）表示，随后送入 `neurophic_system_model` 中的 `System_with_TIA` 电路求解器。求解器模拟两级 NbOx 振荡器、突触与跨阻放大（TIA）电路，最终以滞回武装、下降沿触发和不定期检测 `final_out` 脉冲，并把连续脉冲时刻编码为长度 `T` 的二值脉冲数组（本项目 `T=200`）。

电路求解提供 NumPy 向量化与 CUDA 批量两个后端，二者经预检（精确 F1 与 ±1 bin 容差 F1）保持一致后才进行全量转换。

## 8. 数据划分与参数复用约束

映射流程必须遵守以下顺序：

1. 读取显式配置的 $p_{\mathrm{low}}$、$p_{\mathrm{mid}}$ 和 $p_{\mathrm{high}}$；
2. 只扫描 `raw/train`，计算 $q_{\mathrm{low}}$、$q_{\mathrm{mid}}$ 和 $q_{\mathrm{high}}$；
3. 保存分位参数、实际分位点、统计数量及源 manifest 指纹；
4. 使用同一组参数转换 `raw_polarity` 的 train、val、test；
5. 验证集和测试集不得参与分位点计算或重新标定电压范围。

以下做法均不允许：

- 按单个 trial 计算分位点；
- 按受试者计算独立分位点；
- 按空间电极计算独立分位点；
- 为 positive 和 negative 分别计算分位点；
- 使用验证集或测试集重新计算幅值分位点；
- 为达到指定输出脉冲率而逐样本调整映射参数。

## 9. 参数配置建议

默认配置为：

```text
P_LOW = 0.10
P_MID = 0.90
P_HIGH = 0.995

V_ZERO = 1.0 V
V_ON = 1.8639 V
V_MID = 2.4 V
V_MAX = 2.5 V
```

输出电压范围固定，优先只搜索分位概率。Vin 映射不保证最终电路输出具有固定脉冲率；输出发放率由固定映射和电路动力学共同决定，并作为验证指标记录。

## 10. 参考实现

```python
def map_activity_vin(activity, q_low, q_mid, q_high):
    v_zero = 1.0
    v_on = 1.8639
    v_mid = 2.4
    v_max = 2.5

    low_voltage = v_zero + (v_on - v_zero) * (activity / q_low)
    middle_voltage = v_on + (v_mid - v_on) * (
        (activity - q_low) / (q_mid - q_low)
    )
    high_voltage = v_mid + (v_max - v_mid) * (
        (activity - q_mid) / (q_high - q_mid)
    )

    voltage = np.where(
        activity <= q_low,
        low_voltage,
        np.where(
            activity <= q_mid,
            middle_voltage,
            high_voltage,
        ),
    )
    return np.clip(voltage, v_zero, v_max)
```

正式实现使用高分辨率直方图估计分位点，避免一次性把训练集全部数值复制到内存。输出 manifest 记录分位概率、实际分位点、电压锚点、统计数量、源数据指纹和参数哈希。
