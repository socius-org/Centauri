#!/usr/bin/env python3
"""
Generate control model comparison plots for Psych-101 experiments.

This script produces 2 figures comparing cognitive fine-tuned models
(Llama-Centaur, Qwentaur) vs non-cognitive control models (Hermes, Nemotron, Be.FM).

Usage:
    python generate_control_barplots.py

Output:
    figures/control_grouped_by_size.{png,pdf}
    figures/control_horizontal_delta.{png,pdf}
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

# =========================================================================
# STYLE
# =========================================================================

COLORS = {
    'llama': '#0082fb',   # Llama / Llama-Centaur primary
    'qwen':  '#7F6DEF',   # Qwen / Qwentaur
    'cog':   '#888888',   # Cognitive model reference
}

# Control model colors (kept distinct from project palette)
HERMES = '#2E8B57'       # Sea green
NEMOTRON = '#76B900'     # NVIDIA green
BEFM = '#E6A800'         # Golden yellow


def apply_style(fs=8, fl=5.5):
    plt.rcParams.update({
        'font.size': fs,
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': fs,
        'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1,
        'legend.fontsize': fl,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
    })


def tint(hex_color, amount=0.0):
    """Lighten a color by mixing with white."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])


def shade(hex_color, amount=0.0):
    """Darken a color by mixing with black."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c * (1 - amount) for c in rgb])


# Appended to output basenames; set to '_cognitive_matched' by the flag.
NAME_SUFFIX = ''


def save_figure(fig, outdir, basename, dpi=600):
    basename = f'{basename}{NAME_SUFFIX}'
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")


# =========================================================================
# DATA
# =========================================================================

COG_MODEL_NLL = 0.6851  # Domain-specific cognitive model baseline

# Cognitive fine-tuned models (name, size, NLL)
COGNITIVE = [
    ('Llama-Centaur-1B', 1, 0.6998),
    ('Llama-Centaur-3B', 3, 0.6827),
    ('Llama-Centaur-8B', 8, 0.6441),
    ('Qwentaur-0.6B', 0.6, 0.6926),
    ('Qwentaur-1.7B', 1.7, 0.6781),
    ('Qwentaur-4B', 4, 0.6605),
    ('Qwentaur-8B', 8, 0.6571),
    ('Qwentaur-14B', 14, 0.6410),
]

# Base (pretrained) models (name, size, NLL)
BASE = [
    ('Llama-3.2-3B', 3, 1.0199),
    ('Qwen3-4B', 4, 0.9040),
    ('Llama-3.1-8B', 8, 0.9157),
    ('Qwen3-8B', 8, 0.8916),
    ('Qwen3-14B', 14, 0.8663),
]

# Control models (name, size, NLL)
CONTROLS = [
    ('Hermes-3 3B', 3, 1.0186),
    ('Hermes-3 8B', 8, 0.9816),
    ('Hermes-4 14B', 14, 0.8719),
    ('Nemotron 4B', 4, 1.7227),
    ('Nemotron 8B', 8, 1.4285),
    ('Be.FM 8B', 8, 0.9887),
]

# Per-task data source for each bar, so its NLL can be recomputed over the
# cognitive-baseline subset. ('aggr', column) reads psych101_aggr.csv; ('flat',
# filename) reads a per-task <task, loss> CSV in results/psych101. The hardcoded
# NLLs above are exactly these sources' means over all 46 tasks.
SOURCES = {
    'Llama-Centaur-1B': ('aggr', 'Centaur-1B (bf16)'),
    'Llama-Centaur-3B': ('aggr', 'Centaur-3B (bf16)'),
    'Llama-Centaur-8B': ('aggr', 'Centaur-8B (bf16)'),
    'Qwentaur-0.6B':    ('aggr', 'Qwentaur-0.6B (bf16)'),
    'Qwentaur-1.7B':    ('aggr', 'Qwentaur-1.7B (bf16)'),
    'Qwentaur-4B':      ('aggr', 'Qwentaur-4B (bf16)'),
    'Qwentaur-8B':      ('aggr', 'Qwentaur-8B (bf16)'),
    'Qwentaur-14B':     ('aggr', 'Qwentaur-14B (bf16)'),
    'Llama-3.2-3B':     ('aggr', 'Llama-3.2-3B (base-bf16)'),
    'Qwen3-4B':         ('aggr', 'Qwen3-4B (base-bf16)'),
    'Llama-3.1-8B':     ('aggr', 'Llama-3.1-8B (base-bf16)'),
    'Qwen3-8B':         ('aggr', 'Qwen3-8B (base-bf16)'),
    'Qwen3-14B':        ('aggr', 'Qwen3-14B (base-bf16)'),
    'Hermes-3 3B':      ('flat', 'NousResearch-Hermes-3-Llama-3.2-3B.csv'),
    'Hermes-3 8B':      ('flat', 'NousResearch-Hermes-3-Llama-3.1-8B.csv'),
    'Hermes-4 14B':     ('flat', 'NousResearch-Hermes-4-14B.csv'),
    'Nemotron 4B':      ('flat', 'nvidia-Llama-3.1-Nemotron-Nano-4B-v1.1.csv'),
    'Nemotron 8B':      ('flat', 'nvidia-Llama-3.1-Nemotron-Nano-8B-v1.csv'),
    'Be.FM 8B':         ('flat', 'befm-Be.FM-8B.csv'),
}


def cognitive_matched_tasks(data_dir):
    """The 38 Experiment ids with a reported domain-specific cognitive model."""
    df = pd.read_csv(os.path.join(data_dir, 'psych101_aggr.csv'))
    df = df[df['Experiment'] != 'Mean']
    cog = pd.to_numeric(df['Cognitive model (reported)'].astype(str)
                        .str.replace('†', ''), errors='coerce')
    return set(df.loc[cog.notna(), 'Experiment'])


def recompute_over_subset(models, data_dir, keep):
    """Return `models` with each NLL replaced by its per-task mean over `keep`."""
    aggr = pd.read_csv(os.path.join(data_dir, 'psych101_aggr.csv'))
    aggr = aggr[aggr['Experiment'].isin(keep)]
    out = []
    for name, size, _nll in models:
        kind, ref = SOURCES[name]
        if kind == 'aggr':
            vals = pd.to_numeric(aggr[ref].astype(str).str.replace('†', ''),
                                 errors='coerce')
        else:
            d = pd.read_csv(os.path.join(data_dir, ref))
            vals = d[d['task'].isin(keep)]['loss']
        out.append((name, size, float(vals.mean())))
    return out


def _build_graded_colors():
    """Pre-compute size-graded colours for each model family.

    Smaller models get a lighter tint, larger ones are closer to the base colour.
    Gradient: amount = 0.55 * (1 - idx / max(n-1, 1)), idx=0 is smallest.
    """
    families = {
        'llama_ft':   ([m for m in COGNITIVE if 'Centaur' in m[0]], COLORS['llama']),
        'qwen_ft':    ([m for m in COGNITIVE if 'Qwentaur' in m[0]], COLORS['qwen']),
        'llama_base': ([m for m in BASE if 'Llama' in m[0]], COLORS['llama']),
        'qwen_base':  ([m for m in BASE if 'Qwen' in m[0]], COLORS['qwen']),
    }
    cmap = {}
    for _fam_key, (models, base_hex) in families.items():
        sorted_models = sorted(models, key=lambda m: m[1])  # sort by size
        n = len(sorted_models)
        for idx, (name, _size, _nll) in enumerate(sorted_models):
            amt = 0.55 * (1 - idx / max(n - 1, 1))
            cmap[name] = tint(base_hex, amt)
    return cmap


GRADED_COLORS = _build_graded_colors()


def get_color(name):
    """Get color based on model name, with size grading for project families."""
    if name in GRADED_COLORS:
        return GRADED_COLORS[name]
    if 'Hermes' in name:
        return HERMES
    elif 'Nemotron' in name:
        return NEMOTRON
    elif 'Be.FM' in name:
        return BEFM
    return COLORS['cog']


# =========================================================================
# PLOT FUNCTIONS
# =========================================================================

def plot_grouped_by_size(output_dir):
    """
    Plot 1: Grouped bars at each model size (3B, 4B, 8B, 14B).

    Cognitive fine-tuned models shown as solid bars, base models as hollow,
    control models as hatched.
    """
    sizes = [3, 4, 8, 14]
    size_data = {s: {'cognitive': [], 'base': [], 'control': []} for s in sizes}

    for name, size, nll in COGNITIVE:
        if size in sizes:
            size_data[size]['cognitive'].append((name, nll))

    for name, size, nll in BASE:
        if size in sizes:
            size_data[size]['base'].append((name, nll))

    for name, size, nll in CONTROLS:
        if size in sizes:
            size_data[size]['control'].append((name, nll))

    fig, ax = plt.subplots(figsize=(7, 3.2))

    x_pos = 0
    x_ticks = []
    x_labels = []
    bar_width = 0.7

    for size in sizes:
        cog = size_data[size]['cognitive']
        base = size_data[size]['base']
        ctrl = size_data[size]['control']

        group_start = x_pos

        # Cognitive fine-tuned models (solid)
        for name, nll in cog:
            ax.bar(x_pos, nll, width=bar_width, color=get_color(name),
                   edgecolor='white', linewidth=0.5)
            ax.text(x_pos, nll + 0.02, f'{nll:.2f}', ha='center', va='bottom',
                    fontsize=5, fontweight='bold')
            x_pos += 1

        # Base pretrained models (hollow -- facecolor white, colored edge)
        for name, nll in base:
            ax.bar(x_pos, nll, width=bar_width, facecolor='white',
                   edgecolor=get_color(name), linewidth=1.0)
            ax.text(x_pos, nll + 0.02, f'{nll:.2f}', ha='center', va='bottom',
                    fontsize=5, fontweight='bold')
            x_pos += 1

        # Control models (solid, own colours)
        for name, nll in ctrl:
            ax.bar(x_pos, nll, width=bar_width, color=get_color(name),
                   edgecolor='white', linewidth=0.5)
            ax.text(x_pos, nll + 0.02, f'{nll:.2f}', ha='center', va='bottom',
                    fontsize=5, fontweight='bold')
            x_pos += 1

        group_end = x_pos
        x_ticks.append((group_start + group_end - 1) / 2)
        x_labels.append(f'{size}B')
        x_pos += 0.8  # Gap between groups

    # Cognitive model baseline
    ax.axhline(COG_MODEL_NLL, color='gray', linestyle=':', linewidth=0.8,
               alpha=0.7, zorder=1)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Mean negative log-likelihood')
    ax.set_xlabel('Model size')
    ax.set_ylim(0, 1.85)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['llama'], edgecolor='white', label='Llama-Centaur'),
        Patch(facecolor=COLORS['qwen'], edgecolor='white', label='Qwentaur'),
        Patch(facecolor='white', edgecolor=COLORS['llama'], linewidth=1.0, label='Llama base'),
        Patch(facecolor='white', edgecolor=COLORS['qwen'], linewidth=1.0, label='Qwen base'),
        Patch(facecolor=HERMES, edgecolor='white', label='Hermes'),
        Patch(facecolor=NEMOTRON, edgecolor='white', label='Nemotron'),
        Patch(facecolor=BEFM, edgecolor='white', label='Be.FM'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=0.8, label='Cognitive model'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
              ncol=2)

    fig.tight_layout()
    save_figure(fig, output_dir, 'control_grouped_by_size')
    plt.close(fig)


def plot_horizontal_delta(output_dir):
    """
    Plot 2: Horizontal bar chart showing delta NLL relative to cognitive model baseline.

    Positive delta = better than cognitive model (lower NLL).
    Sorted by performance (best at top).
    """
    all_models = []
    for name, size, nll in COGNITIVE:
        delta = COG_MODEL_NLL - nll  # Positive = better
        all_models.append((name, size, nll, delta, 'cognitive'))
    for name, size, nll in CONTROLS:
        delta = COG_MODEL_NLL - nll  # Negative = worse
        all_models.append((name, size, nll, delta, 'control'))

    # Sort by delta (best first = most positive)
    all_models.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    y_pos = np.arange(len(all_models))
    bar_height = 0.7

    for i, (name, size, nll, delta, mtype) in enumerate(all_models):
        color = get_color(name)
        ax.barh(i, delta, height=bar_height, color=color,
                edgecolor='white', linewidth=0.5)

        # Add delta value annotation
        if delta >= 0:
            ax.text(delta + 0.01, i, f'+{delta:.3f}', ha='left', va='center',
                    fontsize=5, fontweight='bold')
        else:
            ax.text(delta - 0.01, i, f'{delta:.3f}', ha='right', va='center',
                    fontsize=5, fontweight='bold')

    # Vertical line at zero (cognitive model baseline)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, zorder=5)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in all_models])
    ax.set_xlabel('Delta NLL (vs cognitive model)')

    # Invert y-axis so best is at top
    ax.invert_yaxis()

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['llama'], edgecolor='white', label='Llama-Centaur'),
        Patch(facecolor=COLORS['qwen'], edgecolor='white', label='Qwentaur'),
        Patch(facecolor=HERMES, edgecolor='white', label='Hermes'),
        Patch(facecolor=NEMOTRON, edgecolor='white', label='Nemotron'),
        Patch(facecolor=BEFM, edgecolor='white', label='Be.FM'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=0.8, label='Cognitive model'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    # Set x limits with padding
    x_min = min(m[3] for m in all_models) - 0.15
    x_max = max(m[3] for m in all_models) + 0.15
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    save_figure(fig, output_dir, 'control_horizontal_delta')
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Control-model comparison bar plots.')
    parser.add_argument('--cognitive-matched', action='store_true',
                        help='Recompute every NLL over the 38 tasks with a '
                             'reported cognitive model, so the delta-vs-cognitive '
                             'comparison is on the same tasks; write '
                             '*_cognitive_matched figures.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    data_dir = os.path.join(script_dir, '..')   # results/psych101
    os.makedirs(output_dir, exist_ok=True)

    if args.cognitive_matched:
        global NAME_SUFFIX, COGNITIVE, BASE, CONTROLS, GRADED_COLORS
        NAME_SUFFIX = '_cognitive_matched'
        keep = cognitive_matched_tasks(data_dir)
        COGNITIVE = recompute_over_subset(COGNITIVE, data_dir, keep)
        BASE = recompute_over_subset(BASE, data_dir, keep)
        CONTROLS = recompute_over_subset(CONTROLS, data_dir, keep)
        GRADED_COLORS = _build_graded_colors()
        print(f"Cognitive-matched: recomputed NLLs over {len(keep)} tasks")

    with plt.style.context(['nature']):
        apply_style()

        print(f"Output directory: {output_dir}")
        print("Generating control model comparison plots...")

        print("  Plot 1: Grouped by Size")
        plot_grouped_by_size(output_dir)

        print("  Plot 2: Horizontal Delta")
        plot_horizontal_delta(output_dir)

    print(f"Done. Plots written to {output_dir}/")


if __name__ == '__main__':
    main()
