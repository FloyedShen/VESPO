"""
诊断信息分析工具

分析 generate_data.py 生成的诊断信息文件
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import seaborn as sns


def load_diagnostics(filepath: str) -> List[Dict]:
    """加载诊断信息文件"""
    diagnostics = []
    with open(filepath, 'r') as f:
        for line in f:
            diagnostics.append(json.loads(line))
    return diagnostics


def compute_statistics(diagnostics: List[Dict]) -> Dict:
    """计算统计信息"""
    # 提取所有字段
    fields = list(diagnostics[0].keys())
    stats = {}

    for field in fields:
        if field == "sample_idx":
            continue

        values = [d[field] for d in diagnostics if field in d]
        if values:
            stats[field] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75)
            }

    return stats


def print_summary(stats: Dict, filepath: str):
    """打印统计摘要"""
    print("\n" + "="*70)
    print(f"Diagnostics Summary: {Path(filepath).name}")
    print("="*70)

    for field, field_stats in stats.items():
        print(f"\n{field}:")
        print(f"  Mean:   {field_stats['mean']:8.4f}  ±  {field_stats['std']:.4f}")
        print(f"  Median: {field_stats['median']:8.4f}")
        print(f"  Range:  [{field_stats['min']:.4f}, {field_stats['max']:.4f}]")
        print(f"  IQR:    [{field_stats['q25']:.4f}, {field_stats['q75']:.4f}]")


def check_theoretical_conditions(stats: Dict):
    """检查理论条件"""
    print("\n" + "="*70)
    print("Theoretical Conditions Check")
    print("="*70)

    # 检查1: ESS ratio 应该接近1
    if "ess_ratio_mean" in stats:
        ess_ratio = stats["ess_ratio_mean"]["mean"]
        status = "✓ GOOD" if 0.8 <= ess_ratio <= 1.2 else "⚠ WARNING"
        print(f"\n1. ESS Ratio ≈ 1.0")
        print(f"   Measured: {ess_ratio:.4f}")
        print(f"   Status: {status}")

    # 检查2: Alpha 应该在合理范围
    if "alpha_mean" in stats:
        alpha = stats["alpha_mean"]["mean"]
        alpha_std = stats["alpha_mean"]["std"]
        status = "✓ GOOD" if 0.2 <= alpha <= 0.8 else "⚠ WARNING"
        print(f"\n2. Alpha ∈ [0.2, 0.8]")
        print(f"   Measured: {alpha:.4f} ± {alpha_std:.4f}")
        print(f"   Status: {status}")

    # 检查3: KL对称 (如果有kl_theta和kl_t)
    if "kl_theta_mean" in stats and "kl_t_mean" in stats:
        kl_theta = stats["kl_theta_mean"]["mean"]
        kl_t = stats["kl_t_mean"]["mean"]
        kl_diff = abs(kl_theta - kl_t)
        status = "✓ GOOD" if kl_diff < 0.05 else "⚠ WARNING"
        print(f"\n3. KL Symmetry: D_KL(q*||π_θ) ≈ D_KL(q*||π_t)")
        print(f"   D_KL(q*||π_θ) = {kl_theta:.4f}")
        print(f"   D_KL(q*||π_t)  = {kl_t:.4f}")
        print(f"   |Difference|    = {kl_diff:.4f}")
        print(f"   Status: {status}")


def plot_distributions(diagnostics: List[Dict], output_dir: str):
    """绘制分布图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 提取数据
    fields_to_plot = ["alpha_mean", "ess_ratio_mean", "kl_theta_mean", "kl_t_mean"]
    data = {}

    for field in fields_to_plot:
        values = [d[field] for d in diagnostics if field in d]
        if values:
            data[field] = values

    if not data:
        print("No data to plot")
        return

    # 创建子图
    n_plots = len(data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (field, values) in enumerate(data.items()):
        ax = axes[idx]

        # 直方图
        ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(values):.3f}')

        ax.set_xlabel(field, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {field}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / "distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Distributions plot saved to: {output_path}")

    # 绘制散点图: alpha vs ESS ratio
    if "alpha_mean" in data and "ess_ratio_mean" in data:
        plt.figure(figsize=(10, 8))
        plt.scatter(data["alpha_mean"], data["ess_ratio_mean"],
                   alpha=0.5, s=20, color='steelblue')
        plt.xlabel('Alpha', fontsize=12)
        plt.ylabel('ESS Ratio', fontsize=12)
        plt.title('Alpha vs ESS Ratio', fontsize=14, fontweight='bold')
        plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='ESS Ratio = 1.0')
        plt.grid(True, alpha=0.3)
        plt.legend()

        output_path = output_dir / "alpha_vs_ess.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Alpha vs ESS plot saved to: {output_path}")

    # 绘制时间序列 (如果样本是按顺序的)
    if "sample_idx" in diagnostics[0]:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Alpha 演化
        if "alpha_mean" in data:
            sample_indices = [d["sample_idx"] for d in diagnostics]
            axes[0].plot(sample_indices, data["alpha_mean"], alpha=0.7, linewidth=1)
            axes[0].set_xlabel('Sample Index', fontsize=12)
            axes[0].set_ylabel('Alpha', fontsize=12)
            axes[0].set_title('Alpha Evolution', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

        # ESS ratio 演化
        if "ess_ratio_mean" in data:
            sample_indices = [d["sample_idx"] for d in diagnostics]
            axes[1].plot(sample_indices, data["ess_ratio_mean"], alpha=0.7, linewidth=1)
            axes[1].axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
            axes[1].set_xlabel('Sample Index', fontsize=12)
            axes[1].set_ylabel('ESS Ratio', fontsize=12)
            axes[1].set_title('ESS Ratio Evolution', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / "evolution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Evolution plot saved to: {output_path}")


def compare_methods(filepaths: List[str]):
    """对比不同方法的诊断信息"""
    print("\n" + "="*70)
    print("Method Comparison")
    print("="*70)

    all_stats = {}
    for filepath in filepaths:
        diagnostics = load_diagnostics(filepath)
        stats = compute_statistics(diagnostics)
        method_name = Path(filepath).stem.replace('.diagnostics', '')
        all_stats[method_name] = stats

    # 对比表格
    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        "Method", "Alpha", "ESS Ratio", "KL Diff"))
    print("-" * 60)

    for method, stats in all_stats.items():
        alpha = stats.get("alpha_mean", {}).get("mean", 0)
        ess = stats.get("ess_ratio_mean", {}).get("mean", 0)

        kl_theta = stats.get("kl_theta_mean", {}).get("mean", 0)
        kl_t = stats.get("kl_t_mean", {}).get("mean", 0)
        kl_diff = abs(kl_theta - kl_t)

        print(f"{method:<20} {alpha:>12.4f} {ess:>12.4f} {kl_diff:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze diagnostics from data generation")

    parser.add_argument("files", nargs="+", help="Diagnostics files (.diagnostics.jsonl)")
    parser.add_argument("--output_dir", type=str, default="diagnostics_analysis",
                       help="Output directory for plots")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip plotting")

    args = parser.parse_args()

    # 单文件分析
    if len(args.files) == 1:
        filepath = args.files[0]
        print(f"\nAnalyzing: {filepath}")

        diagnostics = load_diagnostics(filepath)
        stats = compute_statistics(diagnostics)

        print_summary(stats, filepath)
        check_theoretical_conditions(stats)

        if not args.no_plots:
            plot_distributions(diagnostics, args.output_dir)

    # 多文件对比
    else:
        compare_methods(args.files)

        # 分别分析每个文件
        for filepath in args.files:
            print(f"\n{'='*70}")
            print(f"Detailed analysis: {Path(filepath).name}")
            print('='*70)

            diagnostics = load_diagnostics(filepath)
            stats = compute_statistics(diagnostics)
            print_summary(stats, filepath)
            check_theoretical_conditions(stats)


if __name__ == "__main__":
    main()
