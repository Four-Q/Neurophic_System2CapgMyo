"""对 10 个确定性随机种子 × 5 种数据分别训练，并汇总结果。

运行：`python -m src.train.run_multi_seed`（需要 CUDA 与已准备的数据）。
每个 (seed, data_type) 的完整训练结果写入
`outputs/10_random_seed/<seed>/<data_type>/`，全部结束后生成
`outputs/10_random_seed/summary.md` 汇总统计。
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from .main import DATA_TYPES, run_training, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "10_random_seed"
BASE_SEED = 42
NUM_SEEDS = 10
DATA_ORDER = [
    "raw_polarity",
    "raw_polarity_norm",
    "neurophic_system_spike",
    "delta_spike",
    "threshold_spike",
]


def generate_seeds(base_seed, num_seeds):
    """从 base_seed 确定性生成 num_seeds 个互不相同的随机种子。"""
    rng = np.random.default_rng(base_seed)
    seeds = []
    seen = set()
    while len(seeds) < num_seeds:
        value = int(rng.integers(0, 2**31 - 1))
        if value not in seen:
            seen.add(value)
            seeds.append(value)
    return seeds


def _pct(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_summary_md(seeds, results):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    names = [DATA_TYPES[dt]["name"] for dt in DATA_ORDER]
    lines = [
        "# 10 随机种子训练结果汇总",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- 基准种子（用于生成随机种子）：{BASE_SEED}",
        f"- 随机种子（{NUM_SEEDS} 个）：{', '.join(map(str, seeds))}",
        "",
        "## 各种子测试准确率",
        "",
    ]
    header = "| seed | " + " | ".join(names) + " |"
    separator = "| ---: | " + " | ".join(["---:"] * len(names)) + " |"
    lines.append(header)
    lines.append(separator)
    for seed in seeds:
        cells = [_pct(results[dt].get(seed)) for dt in DATA_ORDER]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")

    lines.extend(["", "## 各数据统计", ""])
    lines.append("| 数据 | mean | std | min | max | 最佳种子 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for dt in DATA_ORDER:
        values = [results[dt][s] for s in seeds if results[dt].get(s) is not None]
        if not values:
            lines.append(f"| {DATA_TYPES[dt]['name']} | N/A | N/A | N/A | N/A | N/A |")
            continue
        accs = np.array(values, dtype=float)
        best_seed = max((s for s in seeds if results[dt].get(s) is not None), key=lambda s: results[dt][s])
        lines.append(
            f"| {DATA_TYPES[dt]['name']} | {accs.mean():.2%} | {accs.std():.2%} "
            f"| {accs.min():.2%} | {accs.max():.2%} | {best_seed} |"
        )

    lines.extend(["", "## neurophic_system_spike 优于 delta/threshold 的种子（按差值降序，前 5）", ""])
    qualifying = []
    for seed in seeds:
        neurophic = results["neurophic_system_spike"].get(seed)
        delta = results["delta_spike"].get(seed)
        threshold = results["threshold_spike"].get(seed)
        if neurophic is None or delta is None or threshold is None:
            continue
        if neurophic > delta and neurophic > threshold:
            margin = neurophic - max(delta, threshold)
            qualifying.append((seed, neurophic, delta, threshold, margin))
    qualifying.sort(key=lambda row: -row[4])
    top5 = qualifying[:5]
    if top5:
        lines.append("| 排名 | seed | neurophic | delta | threshold | 差值 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for rank, (seed, neurophic, delta, threshold, margin) in enumerate(top5, 1):
            lines.append(
                f"| {rank} | {seed} | {neurophic:.2%} | {delta:.2%} "
                f"| {threshold:.2%} | {margin:.2%} |"
            )
        lines.extend(["", f"（共 {len(qualifying)} 个种子满足条件）"])
    else:
        lines.append("无满足条件的种子。")
    lines.append("")

    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总已写入 {OUTPUT_ROOT / 'summary.md'}", flush=True)


def main():
    seeds = generate_seeds(BASE_SEED, NUM_SEEDS)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "seeds.json", {"base_seed": BASE_SEED, "seeds": seeds})
    print(f"随机种子：{seeds}", flush=True)

    results = {dt: {} for dt in DATA_ORDER}
    total = len(seeds) * len(DATA_ORDER)
    done = 0
    for seed in seeds:
        for data_type in DATA_ORDER:
            done += 1
            print(f"\n===== [{done}/{total}] seed={seed} {data_type} =====", flush=True)
            output_dir = OUTPUT_ROOT / str(seed) / data_type
            try:
                metrics = run_training(data_type, seed, output_dir)
                results[data_type][seed] = metrics["test_accuracy"]
                print(
                    f"[{done}/{total}] seed={seed} {data_type} -> "
                    f"test={metrics['test_accuracy']:.2%}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] seed={seed} {data_type}: {exc}", flush=True)
                results[data_type][seed] = None

    write_summary_md(seeds, results)


if __name__ == "__main__":
    main()
