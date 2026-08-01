#!/usr/bin/env python3
"""
Analyse Order Permutation Test Results
========================================

Loads all test1_order_*.csv files from the results/ subdirectory,
computes per-model variance statistics, and saves aggregate tables.

Usage:
    cd "results/perturbation/permutation (non-sequential)"
    python analyse_permutation.py

Outputs:
    figures/permutation_aggregate.csv
    figures/permutation_per_participant.csv
    figures/permutation_stats.json
"""

import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def get_model_family(name):
    n = name.lower()
    if 'llama' in n or 'centaur' in n:
        return 'llama'
    return 'qwen'


def is_finetuned(name):
    n = name.lower()
    return 'centaur' in n or 'qwentaur' in n or 'lora' in n


def model_size(name):
    m = re.search(r'(\d+\.?\d*)[Bb]', name)
    return float(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results():
    """Load all test1_order_*.csv files from results/ subdirectories."""
    model_dirs = sorted(glob(os.path.join(RESULTS_DIR, '*')))
    all_dfs = []
    for d in model_dirs:
        if not os.path.isdir(d):
            continue
        for csv_path in sorted(glob(os.path.join(d, 'test1_order_*.csv'))):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
            print(f"  Loaded {os.path.relpath(csv_path, SCRIPT_DIR)}: "
                  f"{len(df):,} rows, {df['participant_id'].nunique()} participants")
            del df
    if not all_dfs:
        raise FileNotFoundError(f"No test1_order_*.csv files found in {RESULTS_DIR}")
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_per_participant_variance(df):
    """Compute per-(model, experiment, participant) mean variance across permutations.

    For each (participant, target_idx): variance of each p_i across permutations,
    averaged across response tokens. Then averaged across targets per participant.
    """
    p_cols = [c for c in df.columns if c.startswith('p_')]
    rows = []

    for (model, experiment), grp in df.groupby(['model', 'experiment']):
        for pid, df_p in grp.groupby('participant_id'):
            target_variances = []
            for tidx, df_t in df_p.groupby('target_idx'):
                v = df_t[p_cols].var(ddof=1).dropna().values
                if len(v) > 0:
                    target_variances.append(np.nanmean(v))
            if target_variances:
                rows.append({
                    'model': model,
                    'experiment': experiment,
                    'participant_id': pid,
                    'mean_variance': np.mean(target_variances),
                })

    return pd.DataFrame(rows)


def compute_aggregate_stats(ppv):
    """Aggregate per-participant variances into per-(model, experiment) statistics."""
    results = []
    for (model, experiment), grp in ppv.groupby(['model', 'experiment']):
        v = grp['mean_variance'].values
        n = len(v)
        mean = np.mean(v)
        median = np.median(v)
        std = np.std(v, ddof=1)
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
        t_stat, p_val = stats.ttest_1samp(v, 0)

        results.append({
            'model': model,
            'experiment': experiment,
            'type': 'Finetuned' if is_finetuned(model) else 'Base',
            'family': get_model_family(model),
            'mean_variance': mean,
            'median_variance': median,
            'std': std,
            'se': se,
            'ci_low': mean - t_crit * se,
            'ci_high': mean + t_crit * se,
            'p_value': p_val,
            'n': n,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(agg):
    sep = '=' * 115
    dash = '-' * 115

    for experiment in sorted(agg['experiment'].unique()):
        df = agg[agg['experiment'] == experiment].copy()
        df = df.sort_values(
            by='model',
            key=lambda s: s.map(lambda m: (
                get_model_family(m),
                0 if is_finetuned(m) else 1,
                model_size(m),
            )))

        print(f"\n{sep}")
        print(f"  {experiment}")
        print(sep)
        print(f"{'Model':<28} {'Type':<10} {'Mean Var':>12} {'Median':>12} "
              f"{'SE':>10} {'95% CI':>28} {'n':>5}")
        print(dash)

        for _, row in df.iterrows():
            ci = f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
            print(f"{row['model']:<28} {row['type']:<10} "
                  f"{row['mean_variance']:>12.6f} {row['median_variance']:>12.6f} "
                  f"{row['se']:>10.6f} {ci:>28} {row['n']:>5d}")

        ft = df[df['type'] == 'Finetuned']
        base = df[df['type'] == 'Base']
        if len(ft) > 0 and len(base) > 0:
            f_avg = ft['mean_variance'].mean()
            b_avg = base['mean_variance'].mean()
            ratio = b_avg / f_avg if f_avg > 0 else float('inf')
            print(f"\n  Finetuned avg: {f_avg:.6f}")
            print(f"  Base avg:      {b_avg:.6f}")
            print(f"  Ratio (base / finetuned): {ratio:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    df_all = load_all_results()
    print(f"\nTotal: {len(df_all):,} rows, {df_all['model'].nunique()} models, "
          f"{df_all['experiment'].nunique()} experiments\n")

    print("Computing per-participant variances...")
    ppv = compute_per_participant_variance(df_all)
    del df_all

    print("Computing aggregate statistics...")
    agg = compute_aggregate_stats(ppv)

    print_summary(agg)

    # Save per-participant variances
    ppv_path = os.path.join(OUTPUT_DIR, 'permutation_per_participant.csv')
    ppv.to_csv(ppv_path, index=False)
    print(f"\n  Per-participant saved to {ppv_path}")

    # Save aggregate table
    csv_path = os.path.join(OUTPUT_DIR, 'permutation_aggregate.csv')
    agg.to_csv(csv_path, index=False)
    print(f"  Aggregate saved to {csv_path}")

    # Save stats to JSON
    stats_out = {'aggregate': agg.to_dict(orient='records')}
    json_path = os.path.join(OUTPUT_DIR, 'permutation_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Stats saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
