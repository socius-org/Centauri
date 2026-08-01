#!/usr/bin/env python3
"""
Psych-201 Out-of-Distribution Evaluation: Statistical Analysis
===============================================================

Computes statistics comparing finetuned vs base models on the Psych-201
dataset, which was NOT used during supervised fine-tuning.

Usage:
    cd "results/Psych201-RT (NLL)"
    python analyse_psych201.py

Input:
    results/*.csv   (per-model task-level NLL)

Output:
    figures/psych201_stats.json
    figures/psych201_model_summary.csv
    figures/psych201_task_summary.csv
"""

import json
import math
import os
import re
import glob

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..")
FIGURES_DIR = SCRIPT_DIR

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


def display_name(name):
    return (name
            .replace('socius-', '')
            .replace('unsloth-', '')
            .replace('-LoRA', '')
            .replace('-Base', '')
            .replace('Meta-', ''))


# ---------------------------------------------------------------------------
# Chance baseline: ln(k) per task
# ---------------------------------------------------------------------------

_ln = math.log
LN_K = {
    "anllo2024weird":                  _ln(2),
    "bavard2021range":                 _ln(2),
    "busch2024navon":                  _ln(2),
    "busch2024stroop":                 _ln(3),
    "castrorodrigues2022twostep":      _ln(4),
    "fan2022trait":                    _ln(2),
    "franke2024bayesian":              _ln(4),
    "frankedegen2016reasoning":        _ln(4),
    "guenther2020ts":                  _ln(2),
    "guenther2023grammaticality":      _ln(2),
    "palminteri2017confirmation":      _ln(2),
    "rutledge2023happiness":           None,      # mixed: binary lottery + free rating
    "shahar2019twosteptask":           _ln(2),
    "spektor2024lossaversion":         _ln(2),
    "tsvilodub2023xorsome":            None,      # continuous slider 0-100
    "vandendriessche2022depression":   _ln(2),
    "xu2023augmenting":                _ln(2),
    "zika2023traitanxiety":            None,      # continuous probability
}

# Tasks with a well-defined discrete response space
DISCRETE_TASKS = sorted(t for t, v in LN_K.items() if v is not None and v > 0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results():
    """Load all per-model CSV files into a single DataFrame."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        model = os.path.basename(f).replace('.csv', '')
        for _, r in df.iterrows():
            rows.append({
                'model': model,
                'task': r['task'],
                'loss': r['loss'],
            })
    df = pd.DataFrame(rows)
    df['type'] = df['model'].apply(lambda m: 'Finetuned' if is_finetuned(m) else 'Base')
    df['family'] = df['model'].apply(get_model_family)
    df['size'] = df['model'].apply(model_size)
    df['display_name'] = df['model'].apply(display_name)
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def overall_comparison(df):
    """Aggregate FT vs Base comparison across all tasks."""
    ft_means = df[df['type'] == 'Finetuned'].groupby('task')['loss'].mean()
    base_means = df[df['type'] == 'Base'].groupby('task')['loss'].mean()

    common = sorted(set(ft_means.index) & set(base_means.index))
    ft_vals = ft_means.loc[common].values
    base_vals = base_means.loc[common].values
    diffs = base_vals - ft_vals  # positive = FT better

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(base_vals, ft_vals)

    # Wilcoxon signed-rank test
    w_stat, w_pval = stats.wilcoxon(diffs)

    # Win count + binomial test
    n_ft_wins = int(np.sum(diffs > 0))
    n_tasks = len(common)
    binom_pval = float(stats.binomtest(n_ft_wins, n_tasks, 0.5).pvalue)

    # Effect size: Cohen's d (paired)
    d = float(np.mean(diffs) / np.std(diffs, ddof=1))

    return {
        'n_tasks': n_tasks,
        'ft_mean_nll': float(np.mean(ft_vals)),
        'base_mean_nll': float(np.mean(base_vals)),
        'mean_improvement': float(np.mean(diffs)),
        'median_improvement': float(np.median(diffs)),
        'paired_t_stat': float(t_stat),
        'paired_t_pval': float(t_pval),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pval': float(w_pval),
        'cohens_d': d,
        'ft_wins': n_ft_wins,
        'base_wins': n_tasks - n_ft_wins,
        'binomial_pval': binom_pval,
    }


def per_task_effects(df):
    """Per-task effect sizes and tests."""
    tasks = sorted(df['task'].unique())
    results = []
    for task in tasks:
        td = df[df['task'] == task]
        ft_vals = td[td['type'] == 'Finetuned']['loss'].values
        base_vals = td[td['type'] == 'Base']['loss'].values

        ft_mean = float(np.mean(ft_vals))
        base_mean = float(np.mean(base_vals))
        diff = base_mean - ft_mean

        # Effect size ratio
        ratio = ft_mean / base_mean if base_mean > 0 else np.nan

        # Mann-Whitney U (unpaired, since different models)
        if len(ft_vals) > 1 and len(base_vals) > 1:
            u_stat, u_pval = stats.mannwhitneyu(ft_vals, base_vals, alternative='less')
        else:
            u_stat, u_pval = np.nan, np.nan

        results.append({
            'task': task,
            'ft_mean': ft_mean,
            'base_mean': base_mean,
            'diff': diff,
            'ratio': ratio,
            'mannwhitney_u': float(u_stat) if not np.isnan(u_stat) else None,
            'mannwhitney_pval': float(u_pval) if not np.isnan(u_pval) else None,
        })

    return pd.DataFrame(results).sort_values('diff', ascending=False)


def matched_pair_comparison(df):
    """Compare matched base-FT pairs at each model size."""
    pairs = {
        'llama': {1: ('Llama-Centaur-1B', 'Llama-3.2-1B'),
                  3: ('Llama-Centaur-3B', 'Llama-3.2-3B'),
                  8: ('Llama-Centaur-8B', 'Llama-3.1-8B')},
        'qwen': {0.6: ('Qwentaur-0.6B', 'Qwen3-0.6B'),
                 1.7: ('Qwentaur-1.7B', 'Qwen3-1.7B'),
                 4: ('Qwentaur-4B', 'Qwen3-4B'),
                 8: ('Qwentaur-8B', 'Qwen3-8B'),
                 14: ('Qwentaur-14B', 'Qwen3-14B')},
    }

    results = []
    for family, size_pairs in pairs.items():
        for size, (ft_name, base_name) in size_pairs.items():
            ft_data = df[df['display_name'] == ft_name]
            base_data = df[df['display_name'] == base_name]

            if ft_data.empty or base_data.empty:
                continue

            ft_by_task = ft_data.set_index('task')['loss']
            base_by_task = base_data.set_index('task')['loss']
            common = sorted(set(ft_by_task.index) & set(base_by_task.index))

            ft_vals = ft_by_task.loc[common].values
            base_vals = base_by_task.loc[common].values
            diffs = base_vals - ft_vals

            t_stat, t_pval = stats.ttest_rel(base_vals, ft_vals)

            results.append({
                'family': family,
                'size': size,
                'ft_model': ft_name,
                'base_model': base_name,
                'ft_mean': float(np.mean(ft_vals)),
                'base_mean': float(np.mean(base_vals)),
                'mean_improvement': float(np.mean(diffs)),
                'pct_improvement': float(np.mean(diffs) / np.mean(base_vals) * 100),
                'ft_wins': int(np.sum(diffs > 0)),
                'n_tasks': len(common),
                'paired_t_pval': float(t_pval),
            })

    return pd.DataFrame(results)


def scaling_analysis(df):
    """Analyse how FT advantage changes with model size."""
    results = []
    for family in ['llama', 'qwen']:
        for model_type in ['Finetuned', 'Base']:
            sub = df[(df['family'] == family) & (df['type'] == model_type)]
            for size in sorted(sub['size'].unique()):
                mean_nll = sub[sub['size'] == size]['loss'].mean()
                results.append({
                    'family': family,
                    'type': model_type,
                    'size': size,
                    'mean_nll': float(mean_nll),
                })

    scaling_df = pd.DataFrame(results)

    # Compute FT advantage at each size point
    advantage = []
    for family in ['llama', 'qwen']:
        ft = scaling_df[(scaling_df['family'] == family) & (scaling_df['type'] == 'Finetuned')]
        base = scaling_df[(scaling_df['family'] == family) & (scaling_df['type'] == 'Base')]
        for _, ft_row in ft.iterrows():
            base_match = base[base['size'] == ft_row['size']]
            if not base_match.empty:
                advantage.append({
                    'family': family,
                    'size': ft_row['size'],
                    'ft_nll': ft_row['mean_nll'],
                    'base_nll': float(base_match['mean_nll'].iloc[0]),
                    'advantage': float(base_match['mean_nll'].iloc[0] - ft_row['mean_nll']),
                })

    return scaling_df, pd.DataFrame(advantage)


def normalised_comparison(df):
    """Compare FT vs Base using normalised metric (ln(k) - NLL) / ln(k).

    Restricted to the 15 discrete tasks with well-defined ln(k).
    """
    df_disc = df[df['task'].isin(DISCRETE_TASKS)].copy()
    df_disc['lnk'] = df_disc['task'].map(LN_K)
    df_disc['norm'] = (df_disc['lnk'] - df_disc['loss']) / df_disc['lnk']

    ft_means = df_disc[df_disc['type'] == 'Finetuned'].groupby('task')['norm'].mean()
    base_means = df_disc[df_disc['type'] == 'Base'].groupby('task')['norm'].mean()

    common = sorted(set(ft_means.index) & set(base_means.index))
    ft_vals = ft_means.loc[common].values
    base_vals = base_means.loc[common].values
    diffs = ft_vals - base_vals  # positive = FT better (higher norm)

    t_stat, t_pval = stats.ttest_rel(ft_vals, base_vals)
    w_stat, w_pval = stats.wilcoxon(diffs)
    n_ft_wins = int(np.sum(diffs > 0))
    n_tasks = len(common)
    binom_pval = float(stats.binomtest(n_ft_wins, n_tasks, 0.5).pvalue)
    d = float(np.mean(diffs) / np.std(diffs, ddof=1))

    return {
        'n_tasks': n_tasks,
        'ft_mean_norm': float(np.mean(ft_vals)),
        'base_mean_norm': float(np.mean(base_vals)),
        'mean_improvement': float(np.mean(diffs)),
        'paired_t_stat': float(t_stat),
        'paired_t_pval': float(t_pval),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_pval': float(w_pval),
        'cohens_d': d,
        'ft_wins': n_ft_wins,
        'base_wins': n_tasks - n_ft_wins,
        'binomial_pval': binom_pval,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    df = load_all_results()
    tasks = sorted(df['task'].unique())
    models = sorted(df['model'].unique())
    print(f"  {len(models)} models × {len(tasks)} tasks")

    # 1. Overall comparison
    print("\n=== Overall FT vs Base ===")
    overall = overall_comparison(df)
    print(f"  FT mean NLL:   {overall['ft_mean_nll']:.4f}")
    print(f"  Base mean NLL: {overall['base_mean_nll']:.4f}")
    print(f"  Improvement:   {overall['mean_improvement']:.4f} "
          f"({overall['mean_improvement']/overall['base_mean_nll']*100:.1f}%)")
    print(f"  Paired t-test: t={overall['paired_t_stat']:.3f}, "
          f"p={overall['paired_t_pval']:.2e}")
    print(f"  Wilcoxon:      W={overall['wilcoxon_stat']:.1f}, "
          f"p={overall['wilcoxon_pval']:.2e}")
    print(f"  Cohen's d:     {overall['cohens_d']:.3f}")
    print(f"  Win count:     {overall['ft_wins']}/{overall['n_tasks']} "
          f"(binomial p={overall['binomial_pval']:.4f})")

    # 2. Per-task effects
    print("\n=== Per-Task Effects ===")
    task_effects = per_task_effects(df)
    print(task_effects.to_string(index=False, float_format='{:.4f}'.format))

    # 3. Matched pairs
    print("\n=== Matched-Pair Comparison ===")
    matched = matched_pair_comparison(df)
    if not matched.empty:
        print(matched.to_string(index=False, float_format='{:.4f}'.format))

    # 4. Scaling
    print("\n=== Scaling Analysis ===")
    scaling_df, advantage_df = scaling_analysis(df)
    if not advantage_df.empty:
        print(advantage_df.to_string(index=False, float_format='{:.4f}'.format))

    # 5. Normalised comparison (discrete tasks only)
    print("\n=== Normalised (ln(k) - NLL) / ln(k) — 15 discrete tasks ===")
    norm = normalised_comparison(df)
    print(f"  FT mean norm:   {norm['ft_mean_norm']:.4f}")
    print(f"  Base mean norm: {norm['base_mean_norm']:.4f}")
    print(f"  Improvement:    {norm['mean_improvement']:.4f}")
    print(f"  Paired t-test:  t={norm['paired_t_stat']:.3f}, "
          f"p={norm['paired_t_pval']:.2e}")
    print(f"  Wilcoxon:       W={norm['wilcoxon_stat']:.1f}, "
          f"p={norm['wilcoxon_pval']:.2e}")
    print(f"  Cohen's d:      {norm['cohens_d']:.3f}")
    print(f"  Win count:      {norm['ft_wins']}/{norm['n_tasks']} "
          f"(binomial p={norm['binomial_pval']:.4f})")

    # 6. Model summary
    model_summary = (df.groupby(['model', 'type', 'family', 'size', 'display_name'])
                     ['loss'].agg(['mean', 'median', 'std'])
                     .reset_index()
                     .sort_values('mean'))

    # Save outputs
    model_summary.to_csv(os.path.join(FIGURES_DIR, 'psych201_model_summary.csv'),
                         index=False)
    task_effects.to_csv(os.path.join(FIGURES_DIR, 'psych201_task_summary.csv'),
                        index=False)

    all_stats = {
        'overall': overall,
        'normalised': norm,
        'matched_pairs': matched.to_dict('records') if not matched.empty else [],
        'scaling_advantage': advantage_df.to_dict('records') if not advantage_df.empty else [],
    }
    with open(os.path.join(FIGURES_DIR, 'psych201_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved to {FIGURES_DIR}/")
    print("  psych201_stats.json")
    print("  psych201_model_summary.csv")
    print("  psych201_task_summary.csv")


if __name__ == "__main__":
    main()
