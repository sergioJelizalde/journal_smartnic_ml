#!/usr/bin/env python3
"""
plot_ieee_letter.py - IEEE Letters style benchmark comparison plots

Reads aggregated CSV results from benchmark runs (x86/bf3, multiple kernels)
and generates publication-ready plots comparing:
  - Latency by model size (hand-rolled vs XNNPACK)
  - Throughput by model size
  - Normalized speedup vs scalar baseline
  - Intra-device and inter-device comparisons

Usage:
  python plot_ieee_letter.py ../results/summary.csv --output ../results/plots/
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path


# IEEE Letters standard dimensions
FIGURE_WIDTH = 3.5  # inches, single column
FIGURE_HEIGHT = 2.5  # inches, compact for letters
DPI = 300
FONT_SIZE_MAIN = 9
FONT_SIZE_LABEL = 8
FONT_SIZE_LEGEND = 7

# Color palette: distinguishable, colorblind-safe
COLORS = {
    'scalar': '#1f77b4',      # blue
    'avx': '#ff7f0e',         # orange
    'neon': '#2ca02c',        # green
    'xnnpack': '#d62728',     # red
}

MARKERS = {
    'scalar': 'o',
    'avx': 's',
    'neon': '^',
    'xnnpack': 'D',
}

plt.rcParams.update({
    'font.size': FONT_SIZE_MAIN,
    'font.family': 'sans-serif',
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.labelsize': FONT_SIZE_LABEL,
    'axes.titlesize': FONT_SIZE_MAIN,
    'xtick.labelsize': FONT_SIZE_LABEL,
    'ytick.labelsize': FONT_SIZE_LABEL,
    'legend.fontsize': FONT_SIZE_LEGEND,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
})


def load_summary_csv(summary_path):
    """Load aggregated results CSV."""
    df = pd.read_csv(summary_path)
    
    # Ensure expected columns
    expected = ['platform', 'model_size', 'kernel_type', 'avg_latency_ns', 'throughput_kips']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    
    return df


def extract_model_order(df):
    """Extract model sizes in order."""
    models = df['model_size'].unique()
    order = ['16_8', '32_16', '64_32', '128_64_16', '256_128_32']
    return [m for m in order if m in models]


def plot_latency_by_model(df, platform, output_dir):
    """
    Plot 1: Latency (ns) vs Model Size
    
    Shows each kernel type as a line, separated by platform (if multiple).
    """
    df_platform = df[df['platform'] == platform].copy()
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    models = extract_model_order(df_platform)
    model_labels = [m.replace('_', '-') for m in models]
    
    for kernel in sorted(df_platform['kernel_type'].unique()):
        kernel_data = df_platform[df_platform['kernel_type'] == kernel]
        kernel_data = kernel_data.set_index('model_size').loc[models]
        
        ax.plot(
            range(len(models)),
            kernel_data['avg_latency_ns'],
            marker=MARKERS.get(kernel, 'o'),
            color=COLORS.get(kernel, '#000000'),
            label=kernel.upper(),
            linewidth=1.2,
            markersize=4,
        )
    
    ax.set_xlabel('Model Size (hidden layers)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Latency (ns)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{platform.upper()} — Inference Latency', fontsize=FONT_SIZE_MAIN, weight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'latency_{platform}.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_throughput_by_model(df, platform, output_dir):
    """
    Plot 2: Throughput (kips) vs Model Size
    
    Shows throughput (inferences/sec) for each kernel, log scale.
    """
    df_platform = df[df['platform'] == platform].copy()
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    models = extract_model_order(df_platform)
    model_labels = [m.replace('_', '-') for m in models]
    
    for kernel in sorted(df_platform['kernel_type'].unique()):
        kernel_data = df_platform[df_platform['kernel_type'] == kernel]
        kernel_data = kernel_data.set_index('model_size').loc[models]
        
        ax.plot(
            range(len(models)),
            kernel_data['throughput_kips'],
            marker=MARKERS.get(kernel, 'o'),
            color=COLORS.get(kernel, '#000000'),
            label=kernel.upper(),
            linewidth=1.2,
            markersize=4,
        )
    
    ax.set_xlabel('Model Size (hidden layers)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Throughput (kInferences/sec)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{platform.upper()} — Inference Throughput', fontsize=FONT_SIZE_MAIN, weight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'throughput_{platform}.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_normalized_speedup(df, platform, output_dir):
    """
    Plot 3: Normalized Speedup vs Scalar Baseline
    
    Each kernel's speedup is: scalar_latency / kernel_latency.
    Bar plot grouped by model size.
    """
    df_platform = df[df['platform'] == platform].copy()
    
    models = extract_model_order(df_platform)
    model_labels = [m.replace('_', '-') for m in models]
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Compute speedups relative to scalar
    speedups = {}
    for kernel in df_platform['kernel_type'].unique():
        kernel_data = df_platform[df_platform['kernel_type'] == kernel]
        kernel_data = kernel_data.set_index('model_size').loc[models]
        
        if 'speedup_vs_scalar' in kernel_data.columns:
            speedups[kernel] = kernel_data['speedup_vs_scalar'].values
        else:
            # Compute on the fly: scalar_latency / kernel_latency
            scalar_data = df_platform[df_platform['kernel_type'] == 'scalar']
            scalar_data = scalar_data.set_index('model_size').loc[models]
            speedups[kernel] = scalar_data['avg_latency_ns'].values / kernel_data['avg_latency_ns'].values
    
    # Bar plot
    x = np.arange(len(models))
    width = 0.22
    
    for i, kernel in enumerate(sorted(speedups.keys())):
        if kernel != 'scalar':  # Skip scalar baseline (1.0)
            offset = (i - 1) * width  # Center around x
            ax.bar(
                x + offset,
                speedups[kernel],
                width,
                label=kernel.upper(),
                color=COLORS.get(kernel, '#000000'),
                edgecolor='black',
                linewidth=0.5,
            )
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Scalar baseline')
    ax.set_xlabel('Model Size (hidden layers)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Speedup', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{platform.upper()} — Speedup vs Scalar', fontsize=FONT_SIZE_MAIN, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_ylim([0, max([max(v) for v in speedups.values()]) * 1.1])
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'speedup_{platform}.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_interdevice_comparison(df, output_dir):
    """
    Plot 4: Inter-device Comparison
    
    Compare hand-rolled kernels (AVX on x86 vs NEON on BF3) side-by-side.
    Shows that custom kernels are faster than XNNPACK on both platforms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT))
    
    platforms = ['x86', 'bf3']
    optimized_kernels = {'x86': 'avx', 'bf3': 'neon'}
    
    for idx, (ax, platform) in enumerate(zip(axes, platforms)):
        df_platform = df[df['platform'] == platform].copy()
        models = extract_model_order(df_platform)
        model_labels = [m.replace('_', '-') for m in models]
        
        # Compare optimized kernel (AVX/NEON) vs XNNPACK
        for kernel in [optimized_kernels[platform], 'xnnpack']:
            kernel_data = df_platform[df_platform['kernel_type'] == kernel]
            if not kernel_data.empty:
                kernel_data = kernel_data.set_index('model_size').loc[models]
                
                ax.plot(
                    range(len(models)),
                    kernel_data['avg_latency_ns'],
                    marker=MARKERS.get(kernel, 'o'),
                    color=COLORS.get(kernel, '#000000'),
                    label=kernel.upper(),
                    linewidth=1.2,
                    markersize=4,
                )
        
        ax.set_xlabel('Model Size', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('Latency (ns)', fontsize=FONT_SIZE_LABEL)
        ax.set_title(f'{platform.upper()}: Custom vs XNNPACK', fontsize=FONT_SIZE_MAIN, weight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'interdevice_comparison.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot IEEE Letters style benchmark comparisons'
    )
    parser.add_argument('summary_csv', help='Path to aggregated results CSV')
    parser.add_argument('--output', '-o', default='plots',
                        help='Output directory for PNG files')
    args = parser.parse_args()
    
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {summary_path}")
    df = load_summary_csv(summary_path)
    print(f"Rows: {len(df)}, Columns: {df.columns.tolist()}\n")
    
    # Generate plots for each platform
    for platform in df['platform'].unique():
        print(f"\nGenerating plots for {platform.upper()}:")
        plot_latency_by_model(df, platform, output_dir)
        plot_throughput_by_model(df, platform, output_dir)
        plot_normalized_speedup(df, platform, output_dir)
    
    # Inter-device comparison
    if len(df['platform'].unique()) > 1:
        print(f"\nGenerating inter-device comparison:")
        plot_interdevice_comparison(df, output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
