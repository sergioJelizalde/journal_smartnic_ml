#!/usr/bin/env python3
"""
aggregate_results.py - Merge x86 and bf3 benchmark CSVs into one summary

Reads individual CSV files from results/x86/ and results/bf3/,
combines them with platform/device labels, and writes summary.csv
for downstream plotting.

Usage:
  python aggregate_results.py \
    ../results/x86/baseline_scalar.csv \
    ../results/x86/optimized_avx.csv \
    ../results/x86/xnnpack.csv \
    ../results/bf3/baseline_scalar.csv \
    ../results/bf3/optimized_neon.csv \
    ../results/bf3/xnnpack.csv \
    --output ../results/summary.csv
"""

import sys
import argparse
import pandas as pd
from pathlib import Path


def infer_platform_and_kernel(filepath):
    """
    Infer platform (x86 or bf3) and kernel type from file path.
    
    Examples:
      results/x86/optimized_avx.csv -> ('x86', 'avx')
      results/bf3/baseline_scalar.csv -> ('bf3', 'scalar')
    """
    path = Path(filepath)
    filename = path.stem  # e.g., "optimized_avx"
    parent = path.parent.name  # e.g., "x86" or "bf3"
    
    # Parse kernel type from filename
    if 'scalar' in filename:
        kernel = 'scalar'
    elif 'avx' in filename or 'optimized_avx' in filename:
        kernel = 'avx'
    elif 'neon' in filename or 'optimized_neon' in filename:
        kernel = 'neon'
    elif 'xnnpack' in filename:
        kernel = 'xnnpack'
    else:
        kernel = 'unknown'
    
    # Validate platform
    if parent not in ['x86', 'bf3']:
        parent = 'unknown'
    
    return parent, kernel


def compute_speedup_vs_scalar(group_df):
    """
    For a given platform, compute speedup vs scalar baseline.
    
    speedup = scalar_latency / kernel_latency
    """
    scalar_row = group_df[group_df['kernel_type'] == 'scalar']
    if scalar_row.empty:
        return None
    
    scalar_latency = scalar_row['avg_latency_ns'].values[0]
    group_df = group_df.copy()
    group_df['speedup_vs_scalar'] = scalar_latency / group_df['avg_latency_ns']
    return group_df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate x86 and bf3 benchmark CSVs'
    )
    parser.add_argument('csv_files', nargs='+', help='Input CSV files')
    parser.add_argument('--output', '-o', default='summary.csv',
                        help='Output summary CSV')
    args = parser.parse_args()
    
    all_dfs = []
    
    # Load each CSV and infer platform/kernel
    for csv_file in args.csv_files:
        path = Path(csv_file)
        if not path.exists():
            print(f"WARNING: {csv_file} not found, skipping", file=sys.stderr)
            continue
        
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        
        platform, kernel = infer_platform_and_kernel(csv_file)
        print(f"  → platform={platform}, kernel={kernel}")
        
        # Add platform and kernel columns (may override if already present)
        df['platform'] = platform
        df['kernel_type'] = kernel
        
        all_dfs.append(df)
    
    if not all_dfs:
        print("ERROR: No CSV files loaded", file=sys.stderr)
        sys.exit(1)
    
    # Concatenate all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")
    
    # Compute speedup_vs_scalar for each platform
    speedup_dfs = []
    for platform in combined['platform'].unique():
        platform_data = combined[combined['platform'] == platform]
        
        speedup_data = compute_speedup_vs_scalar(platform_data)
        if speedup_data is not None:
            speedup_dfs.append(speedup_data)
    
    if speedup_dfs:
        combined = pd.concat(speedup_dfs, ignore_index=True)
    else:
        combined['speedup_vs_scalar'] = 1.0  # Fallback
    
    # Sort by platform, model_size, kernel for readability
    combined['model_order'] = combined['model_size'].map({
        '16_8': 0,
        '32_16': 1,
        '64_32': 2,
        '128_64_16': 3,
        '256_128_32': 4,
    })
    combined = combined.sort_values(by=['platform', 'model_order', 'kernel_type'])
    combined = combined.drop(columns=['model_order'])
    
    # Write summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Wrote summary to: {output_path}")
    print(f"  Columns: {combined.columns.tolist()}")
    print(f"\nPreview:")
    print(combined.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
