#!/usr/bin/env python3
"""
Metabench Performance Visualisation — Nature Journal Style

Generates figures comparing base vs cognitively fine-tuned models on metabench.

Figures generated:
1. metabench_finetuned_only - Performance of fine-tuned models
2. metabench_delta - Performance change (Δ = finetuned − base)
3. metabench_by_family - Two-panel figure by model family
4. metabench_heatmap - Z-score heatmap with significance markers

Usage:
    python generate_metabench_plots.py

Input: JSON files in metabench/ directory
Output: PNG and PDF files in figures/metabench/
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import scienceplots  # noqa: F401
import warnings
warnings.filterwarnings('ignore')

plt.style.use(['nature'])

# =============================================================================
# Configuration
# =============================================================================

# Colour scheme
LLAMA = '#0082fb'
LLAMA_DARK = '#005bb5'
QWEN = '#7F6DEF'

# Directory paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / 'metabench'
OUTPUT_DIR = SCRIPT_DIR / 'figures' / 'metabench'


def apply_style(fs=8, fl=5.5):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': fs,
        'axes.labelsize': fs,
        'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1,
        'legend.fontsize': fl,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
    })


apply_style()


# File mappings: model_name -> filename (without path)
# Base models use unsloth- prefix, fine-tuned use socius- prefix
FILE_MAPPINGS = {
    # Qwen family - base models
    'Qwen3-0.6B': 'unsloth-Qwen3-0.6B-base_metabench.json',
    'Qwen3-1.7B': 'unsloth-Qwen3-1.7B-base_metabench.json',
    'Qwen3-4B': 'unsloth-Qwen3-4B-base_metabench.json',
    'Qwen3-8B': 'unsloth-Qwen3-8B-base_metabench.json',
    'Qwen3-14B': 'unsloth-Qwen3-14B-base_metabench.json',
    # Qwen family - fine-tuned models
    'Qwentaur-0.6B': 'socius-Qwentaur-0.6B_metabench.json',
    'Qwentaur-1.7B': 'socius-Qwentaur-1.7B_metabench.json',
    'Qwentaur-4B': 'socius-Qwentaur-4B_metabench.json',
    'Qwentaur-8B': 'socius-Qwentaur-8B_metabench.json',
    'Qwentaur-14B': 'socius-Qwentaur-14B_metabench.json',
    # Llama family - base models
    'Llama-3.2-1B': 'unsloth-Llama-3.2-1B-base_metabench.json',
    'Llama-3.2-3B': 'unsloth-Llama-3.2-3B-base_metabench.json',
    'Llama-3.1-8B': 'unsloth-Llama-3.1-8B-base_metabench.json',
    # Llama family - fine-tuned models
    'Llama-Centaur-1B': 'socius-Llama-Centaur-1B_metabench.json',
    'Llama-Centaur-3B': 'socius-Llama-Centaur-3B_metabench.json',
    'Llama-Centaur-8B': 'socius-Llama-Centaur-8B_metabench.json',
}

# Alternative: auto-discover files if FILE_MAPPINGS don't match
def discover_files(input_dir: Path) -> dict:
    """Auto-discover JSON files and map to model names."""
    mappings = {}

    for filepath in input_dir.glob('*.json'):
        filename = filepath.name
        stem = filepath.stem  # filename without extension

        # Remove _metabench suffix if present
        name_part = stem.replace('_metabench', '')

        # Handle unsloth- prefix (base models)
        if name_part.startswith('unsloth-'):
            model_name = name_part.replace('unsloth-', '').replace('-base', '')
            mappings[model_name] = filename

        # Handle socius- prefix (fine-tuned models)
        elif name_part.startswith('socius-'):
            model_name = name_part.replace('socius-', '')
            mappings[model_name] = filename

        else:
            # Unknown prefix, use as-is
            mappings[name_part] = filename

    return mappings

# Metrics extraction configuration
METRICS_MAPPING = {
    'ARC': ('metabench_arc', 'acc_norm,none', 'acc_norm_stderr,none'),
    'GSM8K': ('metabench_gsm8k', 'exact_match,flexible-extract', 'exact_match_stderr,flexible-extract'),
    'HellaSwag': ('metabench_hellaswag', 'acc_norm,none', 'acc_norm_stderr,none'),
    'MMLU': ('metabench_mmlu', 'acc,none', 'acc_stderr,none'),
    'TruthfulQA': ('metabench_truthfulqa', 'acc,none', 'acc_stderr,none'),
    'Winogrande': ('metabench_winogrande', 'acc,none', 'acc_stderr,none'),
}

BENCHMARKS = ['ARC', 'GSM8K', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande']

# =============================================================================
# Utility Functions
# =============================================================================

def tint(hex_color: str, amount: float = 0.4) -> str:
    """Lighten a colour by blending with white."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])


def shade(hex_color: str, amount: float = 0.3) -> str:
    """Darken a colour."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c * (1 - amount) for c in rgb])


def extract_metrics_with_stderr(filepath: Path) -> dict:
    """Extract benchmark metrics and standard errors from lm-eval JSON output."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = {}
    for metric_name, (task, metric_key, stderr_key) in METRICS_MAPPING.items():
        if task in data.get('results', {}):
            value = data['results'][task].get(metric_key)
            stderr = data['results'][task].get(stderr_key)
            if value is not None:
                results[metric_name] = {
                    'value': value,
                    'stderr': stderr if stderr else 0
                }
    return results


def load_all_results(input_dir: Path, file_mappings: dict = None) -> dict:
    """Load results from all JSON files."""
    if file_mappings is None:
        file_mappings = FILE_MAPPINGS

    all_results = {}
    missing_count = 0

    for model_name, filename in file_mappings.items():
        filepath = input_dir / filename
        try:
            all_results[model_name] = extract_metrics_with_stderr(filepath)
        except FileNotFoundError:
            print(f"Warning: File not found for {model_name}: {filepath}")
            missing_count += 1
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # If too many files missing, try auto-discovery
    if missing_count > len(file_mappings) // 2:
        print("\nToo many files missing. Attempting auto-discovery...")
        discovered = discover_files(input_dir)
        if discovered:
            print(f"Discovered {len(discovered)} files:")
            for name, fname in sorted(discovered.items()):
                print(f"  {name}: {fname}")
            print()
            # Reload with discovered mappings
            all_results = {}
            for model_name, filename in discovered.items():
                filepath = input_dir / filename
                try:
                    all_results[model_name] = extract_metrics_with_stderr(filepath)
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")

    return all_results


def compute_mean_with_se(values: list, errors: list) -> tuple:
    """Compute mean and propagated standard error."""
    mean_val = np.mean(values)
    mean_se = np.sqrt(np.sum(np.array(errors)**2)) / len(errors)
    return mean_val, mean_se


def compute_ztest(base_val: float, base_se: float,
                  ft_val: float, ft_se: float) -> tuple:
    """Compute z-test statistic and p-value for difference."""
    if base_se is None or ft_se is None or base_se == 0 or ft_se == 0:
        return None, None
    pooled_se = np.sqrt(base_se**2 + ft_se**2)
    z = (ft_val - base_val) / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def save_figure(fig: plt.Figure, output_dir: Path, basename: str):
    """Save figure as both PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'{basename}.{ext}',
                    dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")
    plt.close(fig)


# =============================================================================
# Figure 1: Fine-tuned Models Only
# =============================================================================

def create_finetuned_only_figure(all_results: dict, output_dir: Path):
    """
    Create figure showing only fine-tuned model performance.
    Cleaner visualisation for comparing across scales.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    benchmarks_with_mean = BENCHMARKS + ['Mean']
    n_benchmarks = len(benchmarks_with_mean)

    # Fine-tuned models ordered by family then size
    models_config = [
        ('Qwentaur-0.6B', 'Qwentaur-0.6B', tint(QWEN, 0.55)),
        ('Qwentaur-1.7B', 'Qwentaur-1.7B', tint(QWEN, 0.40)),
        ('Qwentaur-4B', 'Qwentaur-4B', tint(QWEN, 0.25)),
        ('Qwentaur-8B', 'Qwentaur-8B', tint(QWEN, 0.12)),
        ('Qwentaur-14B', 'Qwentaur-14B', QWEN),
        ('Llama-Centaur-1B', 'Llama-Centaur-1B', tint(LLAMA, 0.45)),
        ('Llama-Centaur-3B', 'Llama-Centaur-3B', tint(LLAMA, 0.25)),
        ('Llama-Centaur-8B', 'Llama-Centaur-8B', LLAMA),
    ]

    n_models = len(models_config)
    bar_width = 0.75 / n_models
    x = np.arange(n_benchmarks)

    for i, (model_key, label, color) in enumerate(models_config):
        values = []
        errors = []

        for bench in BENCHMARKS:
            data = all_results.get(model_key, {}).get(bench, {})
            values.append(data.get('value', 0))
            errors.append(data.get('stderr', 0))

        mean_val, mean_se = compute_mean_with_se(values, errors)
        values.append(mean_val)
        errors.append(mean_se)

        offset = (i - (n_models - 1) / 2) * bar_width

        ax.bar(x + offset, values, bar_width, yerr=errors,
               capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
               color=color, edgecolor='white', linewidth=0.5, label=label)

    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_with_mean)
    ax.set_ylim(0, 1.05)

    ax.axvline(x=len(BENCHMARKS) - 0.5, color='gray', linestyle='--',
               linewidth=0.6, alpha=0.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=5.5,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, 'metabench_finetuned_only')


# =============================================================================
# Figure 2: Delta Performance (Fine-tuned − Base)
# =============================================================================

def create_delta_figure(all_results: dict, output_dir: Path):
    """
    Create figure showing Δ Performance (fine-tuned − base) for all model pairs.
    Positive = improvement, Negative = degradation.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))

    benchmarks_with_mean = BENCHMARKS + ['Mean']
    n_benchmarks = len(benchmarks_with_mean)

    # Model pairs: (base_key, ft_key, label, colour)
    model_pairs = [
        ('Qwen3-0.6B', 'Qwentaur-0.6B', 'Qwentaur-0.6B', tint(QWEN, 0.55)),
        ('Qwen3-1.7B', 'Qwentaur-1.7B', 'Qwentaur-1.7B', tint(QWEN, 0.40)),
        ('Qwen3-4B', 'Qwentaur-4B', 'Qwentaur-4B', tint(QWEN, 0.25)),
        ('Qwen3-8B', 'Qwentaur-8B', 'Qwentaur-8B', tint(QWEN, 0.12)),
        ('Qwen3-14B', 'Qwentaur-14B', 'Qwentaur-14B', QWEN),
        ('Llama-3.2-1B', 'Llama-Centaur-1B', 'Llama-Centaur-1B', tint(LLAMA, 0.45)),
        ('Llama-3.2-3B', 'Llama-Centaur-3B', 'Llama-Centaur-3B', tint(LLAMA, 0.25)),
        ('Llama-3.1-8B', 'Llama-Centaur-8B', 'Llama-Centaur-8B', LLAMA),
    ]

    n_models = len(model_pairs)
    bar_width = 0.75 / n_models
    x = np.arange(n_benchmarks)

    for i, (base_key, ft_key, label, color) in enumerate(model_pairs):
        deltas = []
        errors = []

        for bench in BENCHMARKS:
            base_data = all_results.get(base_key, {}).get(bench, {})
            ft_data = all_results.get(ft_key, {}).get(bench, {})

            base_val = base_data.get('value', 0)
            ft_val = ft_data.get('value', 0)
            base_se = base_data.get('stderr', 0)
            ft_se = ft_data.get('stderr', 0)

            delta = ft_val - base_val
            pooled_se = np.sqrt(base_se**2 + ft_se**2) if (base_se and ft_se) else 0

            deltas.append(delta)
            errors.append(pooled_se)

        mean_delta, mean_se = compute_mean_with_se(deltas, errors)
        deltas.append(mean_delta)
        errors.append(mean_se)

        offset = (i - (n_models - 1) / 2) * bar_width

        ax.bar(x + offset, deltas, bar_width, yerr=errors,
               capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
               color=color, edgecolor='white', linewidth=0.5, label=label)

    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Vertical line before Mean
    ax.axvline(x=len(BENCHMARKS) - 0.5, color='gray', linestyle='--',
               linewidth=0.6, alpha=0.5)

    # Shading for degradation/improvement regions
    ylim = ax.get_ylim()
    ax.axhspan(ylim[0], 0, alpha=0.03, color='red')
    ax.axhspan(0, ylim[1], alpha=0.03, color='green')

    ax.set_ylabel('Delta Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_with_mean)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=5.5,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, 'metabench_delta')


# =============================================================================
# Figure 3: Grouped by Model Family (Two Panels)
# =============================================================================

def create_family_panels_figure(all_results: dict, output_dir: Path):
    """
    Two-panel figure: Qwen family (left) and Llama family (right).
    Hollow bars = base, filled bars = fine-tuned.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    benchmarks_with_mean = BENCHMARKS + ['Mean']
    n_benchmarks = len(benchmarks_with_mean)
    x = np.arange(n_benchmarks)

    # Qwen family pairs
    qwen_pairs = [
        ('Qwen3-0.6B', 'Qwentaur-0.6B', '0.6B'),
        ('Qwen3-1.7B', 'Qwentaur-1.7B', '1.7B'),
        ('Qwen3-4B', 'Qwentaur-4B', '4B'),
        ('Qwen3-8B', 'Qwentaur-8B', '8B'),
        ('Qwen3-14B', 'Qwentaur-14B', '14B'),
    ]

    # Llama family pairs
    llama_pairs = [
        ('Llama-3.2-1B', 'Llama-Centaur-1B', '1B'),
        ('Llama-3.2-3B', 'Llama-Centaur-3B', '3B'),
        ('Llama-3.1-8B', 'Llama-Centaur-8B', '8B'),
    ]

    def plot_family(ax, pairs, base_color, title, family_name, ft_name):
        n_pairs = len(pairs)
        pair_width = 0.8 / n_pairs
        bar_width = pair_width * 0.42

        # Create colour gradient for sizes
        colors = [tint(base_color, 0.55 * (1 - i / max(n_pairs - 1, 1))) for i in range(n_pairs)]

        for i, (base_key, ft_key, size_label) in enumerate(pairs):
            # Base values
            base_values = []
            base_errors = []
            for bench in BENCHMARKS:
                data = all_results.get(base_key, {}).get(bench, {})
                base_values.append(data.get('value', 0))
                base_errors.append(data.get('stderr', 0))
            base_mean, base_mean_se = compute_mean_with_se(base_values, base_errors)
            base_values.append(base_mean)
            base_errors.append(base_mean_se)

            # Fine-tuned values
            ft_values = []
            ft_errors = []
            for bench in BENCHMARKS:
                data = all_results.get(ft_key, {}).get(bench, {})
                ft_values.append(data.get('value', 0))
                ft_errors.append(data.get('stderr', 0))
            ft_mean, ft_mean_se = compute_mean_with_se(ft_values, ft_errors)
            ft_values.append(ft_mean)
            ft_errors.append(ft_mean_se)

            offset = (i - (n_pairs - 1) / 2) * pair_width
            color = colors[i]

            # Base: hollow
            ax.bar(x + offset - bar_width/2, base_values, bar_width, yerr=base_errors,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color='white', edgecolor=color, linewidth=1.0)

            # Fine-tuned: filled
            ax.bar(x + offset + bar_width/2, ft_values, bar_width, yerr=ft_errors,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color=color, edgecolor='white', linewidth=0.5)

        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks_with_mean)
        ax.set_ylim(0, 1.05)

        ax.axvline(x=len(BENCHMARKS) - 0.5, color='gray', linestyle='--',
                   linewidth=0.6, alpha=0.5)

        # Legend
        legend_elements = [
            Patch(facecolor='white', edgecolor=base_color, linewidth=1.0,
                  label=f'{family_name} (base)'),
            Patch(facecolor=base_color, edgecolor='white', linewidth=0.5,
                  label=ft_name),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5.5,
                  borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    # Plot both panels
    plot_family(axes[0], qwen_pairs, QWEN, 'a  Qwen Family', 'Qwen', 'Qwentaur')
    plot_family(axes[1], llama_pairs, LLAMA, 'b  Llama Family', 'Llama', 'Llama-Centaur')

    axes[0].set_ylabel('Performance')

    plt.tight_layout()
    save_figure(fig, output_dir, 'metabench_by_family')


# =============================================================================
# Figure 4: Heatmap of Z-scores with Significance
# =============================================================================

def create_heatmap_figure(all_results: dict, output_dir: Path):
    """
    Heatmap showing z-scores for each model-benchmark pair.
    Red = degradation, white = stable, green = improvement.
    Significance markers: * p<0.05, ** p<0.01, *** p<0.001
    """
    # Model pairs
    model_pairs = [
        ('Qwen3-0.6B', 'Qwentaur-0.6B', 'Qwentaur-0.6B'),
        ('Qwen3-1.7B', 'Qwentaur-1.7B', 'Qwentaur-1.7B'),
        ('Qwen3-4B', 'Qwentaur-4B', 'Qwentaur-4B'),
        ('Qwen3-8B', 'Qwentaur-8B', 'Qwentaur-8B'),
        ('Qwen3-14B', 'Qwentaur-14B', 'Qwentaur-14B'),
        ('Llama-3.2-1B', 'Llama-Centaur-1B', 'Llama-Centaur-1B'),
        ('Llama-3.2-3B', 'Llama-Centaur-3B', 'Llama-Centaur-3B'),
        ('Llama-3.1-8B', 'Llama-Centaur-8B', 'Llama-Centaur-8B'),
    ]

    benchmarks_with_mean = BENCHMARKS + ['Mean']

    # Compute z-scores matrix
    z_matrix = np.zeros((len(model_pairs), len(benchmarks_with_mean)))
    p_matrix = np.zeros((len(model_pairs), len(benchmarks_with_mean)))

    for i, (base_key, ft_key, label) in enumerate(model_pairs):
        z_vals = []
        for j, bench in enumerate(BENCHMARKS):
            base_data = all_results.get(base_key, {}).get(bench, {})
            ft_data = all_results.get(ft_key, {}).get(bench, {})

            z, p = compute_ztest(
                base_data.get('value', 0), base_data.get('stderr', 0),
                ft_data.get('value', 0), ft_data.get('stderr', 0)
            )
            z_matrix[i, j] = z if z is not None else 0
            p_matrix[i, j] = p if p is not None else 1
            if z is not None:
                z_vals.append(z)

        # Mean z-score
        z_matrix[i, -1] = np.mean(z_vals) if z_vals else 0
        # Combined z for mean p-value
        combined_z = np.sum(z_vals) / np.sqrt(len(z_vals)) if z_vals else 0
        p_matrix[i, -1] = 2 * (1 - stats.norm.cdf(abs(combined_z)))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Custom colourmap: red (negative) -> white (zero) -> green (positive)
    colors = ['#d62728', '#ffcccc', 'white', '#ccffcc', '#2ca02c']
    positions = [0, 0.35, 0.5, 0.65, 1.0]
    cmap = LinearSegmentedColormap.from_list('diverging', list(zip(positions, colors)))

    # Determine symmetric limits
    vmax = max(abs(z_matrix.min()), abs(z_matrix.max()))
    vmax = min(vmax, 6)  # Cap at 6 for visualisation

    im = ax.imshow(z_matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    # Add colourbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('z-score (fine-tuned - base)', fontsize=9)

    # Add text annotations
    for i in range(len(model_pairs)):
        for j in range(len(benchmarks_with_mean)):
            z = z_matrix[i, j]
            p = p_matrix[i, j]

            # Significance markers
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = ''

            text = f'{z:.1f}{sig}'

            # Text colour based on background
            text_color = 'white' if abs(z) > vmax * 0.6 else 'black'

            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                   color=text_color, fontweight='bold' if sig else 'normal')

    # Labels
    ax.set_xticks(np.arange(len(benchmarks_with_mean)))
    ax.set_xticklabels(benchmarks_with_mean)
    ax.set_yticks(np.arange(len(model_pairs)))
    ax.set_yticklabels([p[2] for p in model_pairs])

    # Add vertical line before Mean
    ax.axvline(x=len(BENCHMARKS) - 0.5, color='black', linestyle='-', linewidth=1.5)

    # Add horizontal line between Qwen and Llama families
    ax.axhline(y=4.5, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Model')

    # Significance legend
    ax.text(1.02, -0.15, '* p<0.05  ** p<0.01  *** p<0.001',
            transform=ax.transAxes, fontsize=7, va='top')

    plt.tight_layout()
    save_figure(fig, output_dir, 'metabench_heatmap')


# =============================================================================
# Figure 5: Combined (Delta on top, By-Family stacked below, shared legend)
# =============================================================================

def create_combined_figure(all_results: dict, output_dir: Path):
    """
    Three-panel stacked figure:
      a) Delta (fine-tuned - base) at top
      b) Qwen family (middle)
      c) Llama family (bottom)
    Single shared legend at the bottom.
    """
    benchmarks_with_mean = BENCHMARKS + ['Mean']
    n_benchmarks = len(benchmarks_with_mean)
    x = np.arange(n_benchmarks)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1]})
    ax_delta, ax_qwen, ax_llama = axes

    # --- Panel a: Delta ---
    delta_configs = [
        ('Qwen3-0.6B', 'Qwentaur-0.6B', 'Qwentaur-0.6B', tint(QWEN, 0.55)),
        ('Qwen3-1.7B', 'Qwentaur-1.7B', 'Qwentaur-1.7B', tint(QWEN, 0.40)),
        ('Qwen3-4B', 'Qwentaur-4B', 'Qwentaur-4B', tint(QWEN, 0.25)),
        ('Qwen3-8B', 'Qwentaur-8B', 'Qwentaur-8B', tint(QWEN, 0.12)),
        ('Qwen3-14B', 'Qwentaur-14B', 'Qwentaur-14B', QWEN),
        ('Llama-3.2-1B', 'Llama-Centaur-1B', 'Llama-Centaur-1B', tint(LLAMA, 0.45)),
        ('Llama-3.2-3B', 'Llama-Centaur-3B', 'Llama-Centaur-3B', tint(LLAMA, 0.25)),
        ('Llama-3.1-8B', 'Llama-Centaur-8B', 'Llama-Centaur-8B', LLAMA),
    ]

    n_models = len(delta_configs)
    bar_width = 0.75 / n_models

    for i, (base_key, ft_key, label, color) in enumerate(delta_configs):
        deltas = []
        errors = []
        for bench in BENCHMARKS:
            base_data = all_results.get(base_key, {}).get(bench, {})
            ft_data = all_results.get(ft_key, {}).get(bench, {})
            base_val = base_data.get('value', 0)
            ft_val = ft_data.get('value', 0)
            base_se = base_data.get('stderr', 0)
            ft_se = ft_data.get('stderr', 0)
            delta = ft_val - base_val
            pooled_se = np.sqrt(base_se**2 + ft_se**2) if (base_se and ft_se) else 0
            deltas.append(delta)
            errors.append(pooled_se)
        mean_delta, mean_se = compute_mean_with_se(deltas, errors)
        deltas.append(mean_delta)
        errors.append(mean_se)
        offset = (i - (n_models - 1) / 2) * bar_width
        ax_delta.bar(x + offset, deltas, bar_width, yerr=errors,
                     capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                     color=color, edgecolor='white', linewidth=0.5, label=label)

    ax_delta.axhline(y=0, color='black', linewidth=0.8)
    ax_delta.axvline(x=len(BENCHMARKS) - 0.5, color='gray', linestyle='--',
                     linewidth=0.6, alpha=0.5)
    ylim = ax_delta.get_ylim()
    ax_delta.axhspan(ylim[0], 0, alpha=0.03, color='red')
    ax_delta.axhspan(0, ylim[1], alpha=0.03, color='green')
    ax_delta.set_ylabel('Delta performance')
    ax_delta.set_title('a  Delta (fine-tuned - base)', fontweight='bold', fontsize=8)

    # --- Panels b/c: By Family ---
    fam_configs = [
        ('Qwen', ax_qwen, QWEN, 'Qwen', 'Qwentaur', [
            ('Qwen3-0.6B', 'Qwentaur-0.6B', '0.6B'),
            ('Qwen3-1.7B', 'Qwentaur-1.7B', '1.7B'),
            ('Qwen3-4B', 'Qwentaur-4B', '4B'),
            ('Qwen3-8B', 'Qwentaur-8B', '8B'),
            ('Qwen3-14B', 'Qwentaur-14B', '14B'),
        ]),
        ('Llama', ax_llama, LLAMA, 'Llama', 'Llama-Centaur', [
            ('Llama-3.2-1B', 'Llama-Centaur-1B', '1B'),
            ('Llama-3.2-3B', 'Llama-Centaur-3B', '3B'),
            ('Llama-3.1-8B', 'Llama-Centaur-8B', '8B'),
        ]),
    ]

    for fam_name, ax, base_color, base_label, ft_label, pairs in fam_configs:
        n_pairs = len(pairs)
        pw = 0.8 / n_pairs
        bw = pw * 0.42
        colors = [tint(base_color, 0.55 * (1 - i / max(n_pairs - 1, 1))) for i in range(n_pairs)]

        for i, (base_key, ft_key, size_label) in enumerate(pairs):
            base_values, base_errors = [], []
            ft_values, ft_errors = [], []
            for bench in BENCHMARKS:
                bd = all_results.get(base_key, {}).get(bench, {})
                fd = all_results.get(ft_key, {}).get(bench, {})
                base_values.append(bd.get('value', 0))
                base_errors.append(bd.get('stderr', 0))
                ft_values.append(fd.get('value', 0))
                ft_errors.append(fd.get('stderr', 0))
            bm, bm_se = compute_mean_with_se(base_values, base_errors)
            fm, fm_se = compute_mean_with_se(ft_values, ft_errors)
            base_values.append(bm); base_errors.append(bm_se)
            ft_values.append(fm); ft_errors.append(fm_se)

            offset = (i - (n_pairs - 1) / 2) * pw
            color = colors[i]
            ax.bar(x + offset - bw/2, base_values, bw, yerr=base_errors,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color='white', edgecolor=color, linewidth=1.0)
            ax.bar(x + offset + bw/2, ft_values, bw, yerr=ft_errors,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color=color, edgecolor='white', linewidth=0.5)

        ax.axvline(x=len(BENCHMARKS) - 0.5, color='gray', linestyle='--',
                   linewidth=0.6, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Performance')
        panel = 'b' if fam_name == 'Qwen' else 'c'
        ax.set_title(f'{panel}  {fam_name} family', fontweight='bold', fontsize=8)

    # X-tick labels only on bottom panel
    ax_llama.set_xticks(x)
    ax_llama.set_xticklabels(benchmarks_with_mean)

    # Shared legend at bottom
    legend_elements = [
        Patch(facecolor='white', edgecolor=QWEN, linewidth=1.0, label='Qwen (base)'),
        Patch(facecolor=QWEN, edgecolor='white', linewidth=0.5, label='Qwentaur'),
        Patch(facecolor='white', edgecolor=LLAMA, linewidth=1.0, label='Llama (base)'),
        Patch(facecolor=LLAMA, edgecolor='white', linewidth=0.5, label='Llama-Centaur'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=6,
               borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    save_figure(fig, output_dir, 'metabench_combined')


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all metabench figures."""
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check input directory exists
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please ensure JSON files are in the 'metabench' subdirectory.")
        return

    # List available files for debugging
    json_files = list(INPUT_DIR.glob('*.json'))
    if json_files:
        print(f"Found {len(json_files)} JSON files in {INPUT_DIR}:")
        for f in sorted(json_files)[:5]:  # Show first 5
            print(f"  {f.name}")
        if len(json_files) > 5:
            print(f"  ... and {len(json_files) - 5} more")
        print()

    # Load all results
    print("Loading results...")
    all_results = load_all_results(INPUT_DIR)

    if not all_results:
        print("Error: No results loaded. Check file paths and format.")
        return

    print(f"Loaded {len(all_results)} model results.\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("Generating figures...")
    create_finetuned_only_figure(all_results, OUTPUT_DIR)
    create_delta_figure(all_results, OUTPUT_DIR)
    create_family_panels_figure(all_results, OUTPUT_DIR)
    create_heatmap_figure(all_results, OUTPUT_DIR)
    create_combined_figure(all_results, OUTPUT_DIR)

    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
