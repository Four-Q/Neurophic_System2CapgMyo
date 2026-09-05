# CapgMyo DB-a 神经形态脉冲编码 CSNN

本项目在 CapgMyo DB-a 高密度表面肌电（HD-sEMG）数据上，对五种正负分离输入分别训练同一个脉冲卷积神经网络（PLIF-CSNN）做 8 类手势识别，并完整输出训练历史、测试指标、混淆矩阵、逐类别/逐被试准确率与最佳模型。

五种输入均为正负通道分离数据：

| 输入 | 数据目录 | dtype | 目标测试准确率 |
| --- | --- | --- | ---: |
| 原始极性数据（无归一化） | `raw_polarity_binned/T_200` | float32 | 95.83% |
| 原始极性数据（z-score 归一化） | `raw_polarity_binned/T_200` | float32 | 95.14% |
| 神经形态系统脉冲编码（Vin 输入） | `neurophic_system_encoding_spike/T_200` | bool | 92.36% |
| Delta 编码脉冲 | `delta_encoding_spike/T_200` | bool | 85.07% |
| Threshold 编码脉冲 | `threshold_encoding_spike/T_200` | bool | 86.11% |

## 项目结构

```text
.
├── CapgMyo_data/                    # 初始为空，由数据准备 notebook 生成
├── neurophic_system_model/          # System_with_TIA 电路仿真后端
│   ├── system_with_tia.py
│   └── system_with_tia_batch.py
├── src/
│   ├── data_prepare/                # 6 个数据准备 notebook（顺序执行）
│   │   ├── 1.prepare_raw_capgmyo_dba.ipynb
│   │   ├── 2.prepare_polarity_capgmyo_dba.ipynb
│   │   ├── 3.prepare_raw_polarity_binned.ipynb
│   │   ├── 4.prepare_neurophic_system_encoding_spike.ipynb
│   │   ├── 5.prepare_delta_encoding_spike.ipynb
│   │   └── 6.prepare_threshold_encoding_spike.ipynb
│   ├── data/dataset.py              # 数据加载与校验
│   ├── model/csnn.py                # PLIF-CSNN 模型
│   └── train/
│       ├── main.py                  # 训练入口
│       ├── trainer.py               # 训练循环与评估
│       ├── metrics.py               # 混淆矩阵 / 逐类别 / 逐被试指标
│       ├── visualize.py             # 训练可视化
│       └── compare.py               # 五种输入对比
├── docs/                            # 编码方式与数据集说明
├── outputs/                         # 训练输出（每种输入一个子目录）
├── requirements.txt
└── README.md
```

## 环境安装

建议 Python 3.10+，并在独立虚拟环境中安装依赖。训练需要 NVIDIA GPU（CUDA 12.x）与 CUDA Toolkit（`nvcc`，用于神经形态脉冲编码的 CUDA 后端编译）。

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` 只列包名，不固定版本。请在安装后确认 `torch` 与本机 CUDA 版本匹配，并确认 `spikingjelly`、`cupy-cuda12x` 与 `torch` 版本兼容。

## 数据准备

`CapgMyo_data` 初始为空，所有数据由 `src/data_prepare` 中的 notebook 生成。在项目根目录启动 Jupyter：

```bash
jupyter lab
```

按顺序运行：

1. `1.prepare_raw_capgmyo_dba.ipynb` —— 下载并校验 Figshare v1 的 18 个压缩包，转换为 `.pt` 并划分 train/val/test（约 3 GB）。
2. `2.prepare_polarity_capgmyo_dba.ipynb` —— 正负通道分离（约 2 GB）。
3. `3.prepare_raw_polarity_binned.ipynb` —— 原始极性数据分箱（T=200）。
4. `4.prepare_neurophic_system_encoding_spike.ipynb` —— Vin 输入脉冲编码（需 CUDA）。
5. `5.prepare_delta_encoding_spike.ipynb` —— Delta 编码脉冲。
6. `6.prepare_threshold_encoding_spike.ipynb` —— Threshold 编码脉冲。

notebook 会自动向上查找项目根目录（需同时包含 `src/data_prepare` 与 `CapgMyo_data`），支持复用已有有效文件，通常可安全重跑。下载若受网络影响，可设置 `CAPGMYO_USE_ENV_PROXY=1` 走系统代理。

## 训练

在项目根目录执行（需已完成对应数据准备）：

```bash
python -m src.train.main --data-type raw_polarity
python -m src.train.main --data-type raw_polarity_norm
python -m src.train.main --data-type neurophic_system_spike
python -m src.train.main --data-type delta_spike
python -m src.train.main --data-type threshold_spike
```

`--data-type` 取值为上述五种。也可用 `--data-root` 覆盖数据目录、用 `--output-dir` 覆盖输出目录；训练超参数（卷积宽度 `--widths 24,48,96`、`--hidden 160`、`--pool 4,8`、`--tau 2.0`、`--lr 1e-3` 等）默认即复现配方，可按需覆盖。

五种训练完成后，生成跨五种输入的对比表与对比图：

```bash
python -m src.train.compare
```

## 输出

每种输入的训练结果写入 `outputs/<data-type>/<时间戳>/`：

| 文件 | 说明 |
| --- | --- |
| `history.csv` | 每轮 train/val loss、accuracy、learning_rate |
| `training_curves.png` | loss 与 accuracy 曲线 |
| `test_metrics.json` | 测试准确率、混淆矩阵、逐类别/逐被试指标 |
| `confusion_matrix.csv` / `.png` | 混淆矩阵 |
| `per_class.csv` / `.png` | 逐类别准确率（及 precision/recall/f1） |
| `per_subject.csv` | 逐被试准确率 |
| `best.pt` | 最佳模型 |
| `summary.md` | 结果汇总 |

跨五种对比产物写入 `outputs/` 顶层：`comparison.csv`、`comparison.png`、`comparison.md`。

## 模型与训练配方

五种输入共用同一个 PLIF-CSNN，固定配方为：卷积宽度 `24-48-96`、分类头隐藏宽度 `160`、分类前空间池化 `4×8`、PLIF `tau=2.0`、AdamW（`lr=1e-3`、`weight_decay=1e-4`）、标签平滑 `0.05`、无 Dropout/Mixup/输入丢弃、随机种子 `42`、batch size `256`、200 epoch、CosineAnnealingLR、梯度裁剪 5.0。

模型对完整 200 个时间步逐步卷积处理，对时间轴上的 logits 求均值后分类。脉冲输入（bool）直接转换为 float32 送入网络；原始幅度输入默认不做归一化，`raw_polarity_norm` 使用训练集统计量做 z-score 标准化。

## 复现说明

- 五种输入分别产生测试准确率约 95.83% / 95.14% / 92.36% / 85.07% / 86.11%。
- 训练开启 cuDNN benchmark 与 TF32，即使固定随机种子 42，重跑也不保证 checkpoint 逐位一致；准确率预期在 ±0.5% 内浮动。
- 数据划分固定为受试者内按重复编号划分：训练 `1,3,4,5,6,7,8`、验证 `10`、测试 `2,9`，对应 1008/144/288 个样本。

## 编码方式文档

- [`docs/Vin输入编码方式.md`](docs/Vin输入编码方式.md)
- [`docs/Delta编码.md`](docs/Delta编码.md)
- [`docs/Threshold编码.md`](docs/Threshold编码.md)
- [`docs/数据集说明.md`](docs/数据集说明.md)

## 参考资料

- 数据集：[CapgMyo DB-a（Figshare）](https://doi.org/10.6084/m9.figshare.7210397.v1)
- 论文：[Gesture Recognition by Instantaneous Surface EMG Images](https://doi.org/10.1038/srep36571)
