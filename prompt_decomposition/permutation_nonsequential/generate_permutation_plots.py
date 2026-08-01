#!/usr/bin/env python3
"""
Generate Permutation Test Figures
==================================

Reads pre-computed CSV files and generates publication-ready figures
for the order permutation test analysis.

Usage:
    cd "results/perturbation/permutation (non-sequential)"
    python generate_permutation_plots.py

Input:
    figures/permutation_aggregate.csv
    figures/permutation_per_participant.csv

Output:
    figures/permutation_combined.{png,pdf}
    figures/permutation_ecdf.{png,pdf}
    figures/permutation_violin.{png,pdf}
    figures/permutation_grid.{png,pdf}
"""

import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    'llama': '#0082fb',
    'qwen': '#7F6DEF',
}

EXP_TITLES = {
    'hebart2023things': ('THINGS odd-one-out', '(Hebart et al., 2023)'),
    'ruggeri2022globalizability': ('Intertemporal choice', '(Ruggeri et al., 2022)'),
}


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


def tint(hex_color, amount=0.0):
    """Lighten a color by mixing with white."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])


def save_figure(fig, outdir, basename, dpi=600):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")


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


def model_sort_key(name):
    return (get_model_family(name), 0 if is_finetuned(name) else 1, model_size(name))


def display_name(name):
    """Full display name for plot labels (strips suffixes only)."""
    return (name
            .replace('-LoRA', '')
            .replace('-Base', '')
            .replace('Meta-', ''))


def set_panel_title(ax, label, experiment, fontsize=7):
    """Add centered panel title with bold label+name and normal-weight citation."""
    name, cite = EXP_TITLES.get(experiment, (experiment, ''))
    bold_part = TextArea(
        f"{label}  {name} ",
        textprops=dict(fontweight='bold', fontsize=fontsize),
    )
    cite_part = TextArea(
        cite,
        textprops=dict(fontweight='normal', fontsize=fontsize),
    )
    title_box = HPacker(children=[bold_part, cite_part],
                        align="baseline", pad=0, sep=1)
    ab = AnchoredOffsetbox(
        loc='upper center', child=title_box,
        pad=0.3, frameon=False,
        bbox_to_anchor=(0.5, 1.14),
        bbox_transform=ax.transAxes,
    )
    ab.patch.set_visible(False)
    ax.add_artist(ab)


# ---------------------------------------------------------------------------
# Gradient colors
# ---------------------------------------------------------------------------


def compute_gradient_colors(models):
    """Assign gradient colors: lightest for smallest, full color for largest.

    Same physical size within the same family always gets the same shade,
    regardless of whether it is Finetuned or Base.
    """
    family_sizes = {'llama': set(), 'qwen': set()}
    for m in models:
        family_sizes[get_model_family(m)].add(model_size(m))
    family_sorted = {f: sorted(s) for f, s in family_sizes.items()}

    colors = {}
    for m in models:
        family = get_model_family(m)
        size = model_size(m)
        sizes = family_sorted[family]
        n = len(sizes)
        idx = sizes.index(size)
        amount = 0.55 * (1 - idx / max(n - 1, 1))
        colors[m] = tint(COLORS[family], amount)
    return colors


# ---------------------------------------------------------------------------
# Figure 1: Combined bar panel
# ---------------------------------------------------------------------------


def fig_combined_panel(agg, outdir):
    """Two-panel bar chart: Finetuned (solid gradient) | Base (hatched gradient)."""
    experiments = sorted(agg['experiment'].unique())
    if len(experiments) < 2:
        return

    all_models = agg['model'].unique()
    grad = compute_gradient_colors(all_models)

    with plt.style.context(['nature']):
        apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        panel_labels = ['a', 'b']

        for ax, experiment, label in zip(axes, experiments, panel_labels):
            df = agg[agg['experiment'] == experiment].copy()
            df['sort_key'] = df['model'].apply(model_sort_key)
            df = df.sort_values('sort_key')

            ft = df[df['type'] == 'Finetuned']
            base = df[df['type'] == 'Base']

            # Finetuned bars (solid gradient fill)
            x_ft = np.arange(len(ft))
            for i, (_, row) in enumerate(ft.iterrows()):
                color = grad[row['model']]
                yerr = row['mean_variance'] - row['ci_low']
                ax.bar(x_ft[i], row['mean_variance'], 0.7,
                       color=color, edgecolor='white', linewidth=0.5,
                       yerr=yerr, capsize=2, error_kw={'linewidth': 0.5})

            # Base bars (hatched, gradient edge)
            x_base = np.arange(len(base)) + len(ft) + 1
            for i, (_, row) in enumerate(base.iterrows()):
                color = grad[row['model']]
                yerr = row['mean_variance'] - row['ci_low']
                ax.bar(x_base[i], row['mean_variance'], 0.7,
                       color='white', edgecolor=color, linewidth=0.6,
                       hatch='///', yerr=yerr, capsize=2,
                       error_kw={'linewidth': 0.5})

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.3)

            # X-axis labels (full display names)
            all_x = np.concatenate([x_ft, x_base])
            all_labels = [display_name(m) for m in
                          list(ft['model']) + list(base['model'])]
            ax.set_xticks(all_x)
            ax.set_xticklabels(all_labels, rotation=50, ha='right', fontsize=5.5)
            ax.set_ylabel('Mean order variance')

            # Title (centered, bold experiment name + normal citation)
            set_panel_title(ax, label, experiment)

            # Group labels
            ymax = ax.get_ylim()[1]
            ax.text(np.mean(x_ft), ymax * 0.92, 'Finetuned',
                    ha='center', fontsize=5, fontstyle='italic', color='gray')
            ax.text(np.mean(x_base), ymax * 0.92, 'Base',
                    ha='center', fontsize=5, fontstyle='italic', color='gray')

        # Shared legend
        handles = [
            Patch(facecolor=COLORS['llama'], edgecolor=COLORS['llama'],
                  label='Llama / Llama-Centaur'),
            Patch(facecolor=COLORS['qwen'], edgecolor=COLORS['qwen'],
                  label='Qwen3 / Qwentaur'),
            Patch(facecolor='white', edgecolor='gray', hatch='///',
                  label='Base'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=5.5)

        fig.tight_layout(rect=[0, 0.04, 1, 0.92])
        save_figure(fig, outdir, 'permutation_combined')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: ECDF
# ---------------------------------------------------------------------------


def fig_ecdf(ppv, outdir):
    """ECDF of per-participant variance, two-panel by experiment."""
    with plt.style.context(['nature']):
        apply_style()

        experiments = sorted(ppv['experiment'].unique())
        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(3.5 * n_exp, 2.8),
                                 squeeze=False, sharey=True)
        panel_labels = 'abcdef'

        all_models = ppv['model'].unique()
        grad = compute_gradient_colors(all_models)

        for ax, experiment, label in zip(axes[0], experiments, panel_labels):
            df = ppv[ppv['experiment'] == experiment]
            models = sorted(df['model'].unique(), key=model_sort_key)

            for model_name in models:
                vals = np.sort(df[df['model'] == model_name]['mean_variance'].values)
                ecdf_y = np.arange(1, len(vals) + 1) / len(vals)

                ft = is_finetuned(model_name)
                color = grad[model_name]
                ls = '-' if ft else '--'
                alpha = 0.9 if ft else 0.5

                ax.step(vals, ecdf_y, where='post', color=color,
                        linestyle=ls, linewidth=0.8, alpha=alpha,
                        label=display_name(model_name))

            ax.set_xlabel('Per-participant order variance')
            if ax is axes[0][0]:
                ax.set_ylabel('Cumulative proportion')
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.3)

            set_panel_title(ax, label, experiment)
            ax.legend(fontsize=4, loc='lower right', frameon=False, ncol=2)

        fig.tight_layout(rect=[0, 0, 1, 0.91])
        save_figure(fig, outdir, 'permutation_ecdf')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Violin
# ---------------------------------------------------------------------------


def fig_violin(ppv, outdir):
    """Violin + box + strip plot, two-panel by experiment.

    Finetuned: gradient-colored boxes (solid).
    Base: white fill + gradient edge + hatching.
    """
    with plt.style.context(['nature']):
        apply_style()

        experiments = sorted(ppv['experiment'].unique())
        n_exp = len(experiments)
        fig, axes = plt.subplots(1, n_exp, figsize=(7, 3.2), squeeze=False)
        panel_labels = 'abcdef'

        all_models = ppv['model'].unique()
        grad = compute_gradient_colors(all_models)

        for ax, experiment, label in zip(axes[0], experiments, panel_labels):
            df = ppv[ppv['experiment'] == experiment].copy()

            # Sort: Finetuned first, then Base; within each by (family, size)
            models = sorted(df['model'].unique(), key=model_sort_key)
            ft_models = [m for m in models if is_finetuned(m)]
            base_models = [m for m in models if not is_finetuned(m)]
            ordered = ft_models + base_models

            data, colors, facecolors, labels, positions = [], [], [], [], []
            hatched = []
            pos = 0
            for i, m in enumerate(ordered):
                if i == len(ft_models) and len(ft_models) > 0:
                    pos += 1  # gap between groups
                vals = df[df['model'] == m]['mean_variance'].values
                data.append(vals)
                c = grad[m]
                ft = is_finetuned(m)
                colors.append(c)
                facecolors.append(c if ft else 'white')
                hatched.append(not ft)
                labels.append(display_name(m))
                positions.append(pos)
                pos += 1

            # Violin bodies
            parts = ax.violinplot(data, positions=positions,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.2)

            # Box plots
            bp = ax.boxplot(data, positions=positions, widths=0.25,
                            patch_artist=True, showfliers=False, zorder=3)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(facecolors[i])
                patch.set_edgecolor(colors[i])
                patch.set_linewidth(0.6)
                patch.set_alpha(0.7)
                if hatched[i]:
                    patch.set_hatch('///')
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color('black')
                    line.set_linewidth(0.5)

            # Jittered strip
            rng = np.random.default_rng(42)
            for i, vals in enumerate(data):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(positions[i] + jitter, vals, s=3, color=colors[i],
                           alpha=0.25, zorder=2, edgecolors='none')

            # Separator between groups
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.3)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=5.5)
            ax.set_ylabel('Per-participant order variance')

            set_panel_title(ax, label, experiment)

            # Group labels
            if ft_models and base_models:
                ft_pos = positions[:len(ft_models)]
                base_pos = positions[len(ft_models):]
                ymax = ax.get_ylim()[1]
                ax.text(np.mean(ft_pos), ymax * 0.95, 'Finetuned',
                        ha='center', fontsize=5, fontstyle='italic', color='gray')
                ax.text(np.mean(base_pos), ymax * 0.95, 'Base',
                        ha='center', fontsize=5, fontstyle='italic', color='gray')

        fig.tight_layout(rect=[0, 0, 1, 0.91])
        save_figure(fig, outdir, 'permutation_violin')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Combined violin + ECDF (2×2 grid)
# ---------------------------------------------------------------------------


def _draw_violin_on_ax(ax, ppv, experiment, grad, show_ylabel=True,
                        show_xlabel=False):
    """Draw violin + box + strip for one experiment on a given axes."""
    df = ppv[ppv['experiment'] == experiment].copy()
    models = sorted(df['model'].unique(), key=model_sort_key)
    ft_models = [m for m in models if is_finetuned(m)]
    base_models = [m for m in models if not is_finetuned(m)]
    ordered = ft_models + base_models

    data, colors, facecolors, labels, positions = [], [], [], [], []
    hatched = []
    pos = 0
    for i, m in enumerate(ordered):
        if i == len(ft_models) and len(ft_models) > 0:
            pos += 1
        vals = df[df['model'] == m]['mean_variance'].values
        data.append(vals)
        c = grad[m]
        ft = is_finetuned(m)
        colors.append(c)
        facecolors.append(c if ft else 'white')
        hatched.append(not ft)
        labels.append(display_name(m))
        positions.append(pos)
        pos += 1

    parts = ax.violinplot(data, positions=positions,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.2)

    bp = ax.boxplot(data, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=False, zorder=3)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(facecolors[i])
        patch.set_edgecolor(colors[i])
        patch.set_linewidth(0.6)
        patch.set_alpha(0.7)
        if hatched[i]:
            patch.set_hatch('///')
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(0.5)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, s=2, color=colors[i],
                   alpha=0.2, zorder=2, edgecolors='none')

    if ft_models and base_models:
        sep_x = positions[len(ft_models) - 1] + 0.75
        ax.axvline(x=sep_x, color='gray', linestyle=':', linewidth=0.4, alpha=0.5)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=6.5)
    if show_ylabel:
        ax.set_ylabel('Per-participant\norder variance', fontsize=11)

    if ft_models and base_models:
        ft_pos = positions[:len(ft_models)]
        base_pos = positions[len(ft_models):]
        ymax = ax.get_ylim()[1]
        ax.text(np.mean(ft_pos), ymax * 0.95, 'Finetuned',
                ha='center', fontsize=4.5, fontstyle='italic', color='gray')
        ax.text(np.mean(base_pos), ymax * 0.95, 'Base',
                ha='center', fontsize=4.5, fontstyle='italic', color='gray')


def _draw_ecdf_on_ax(ax, ppv, experiment, grad, show_ylabel=True):
    """Draw ECDF for one experiment on a given axes."""
    df = ppv[ppv['experiment'] == experiment]
    models = sorted(df['model'].unique(), key=model_sort_key)

    for model_name in models:
        vals = np.sort(df[df['model'] == model_name]['mean_variance'].values)
        ecdf_y = np.arange(1, len(vals) + 1) / len(vals)

        ft = is_finetuned(model_name)
        color = grad[model_name]
        ls = '-' if ft else '--'
        alpha = 0.9 if ft else 0.5

        ax.step(vals, ecdf_y, where='post', color=color,
                linestyle=ls, linewidth=0.8, alpha=alpha,
                label=display_name(model_name))

    ax.set_xlabel('Per-participant order variance', fontsize=11)
    if show_ylabel:
        ax.set_ylabel('Cumulative proportion', fontsize=11)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.3)
    ax.legend(fontsize=8, loc='lower right', frameon=False, ncol=2,
              borderpad=0.2, handlelength=1.2, handletextpad=0.3,
              labelspacing=0.25, columnspacing=0.8)


def fig_violin_ecdf_grid(ppv, outdir):
    """2×2 grid: violin (1/3) + ECDF (2/3), one row per experiment."""
    experiments = sorted(ppv['experiment'].unique())
    if len(experiments) < 2:
        return

    all_models = ppv['model'].unique()
    grad = compute_gradient_colors(all_models)

    with plt.style.context(['nature']):
        apply_style(fs=9, fl=7)
        fig = plt.figure(figsize=(10, 7.2))

        # Layout: 2 rows × 2 cols, violin=1/3, ecdf=2/3
        # Using fig.add_axes([left, bottom, width, height])
        violin_w = 0.30
        ecdf_w = 0.58
        row_h = 0.33
        left_v = 0.07
        left_e = 0.44
        top_bottom = 0.55     # top row bottom y
        bot_bottom = 0.07     # bottom row bottom y
        title_y_top = 0.93
        title_y_bot = 0.47

        ax_v_top = fig.add_axes([left_v, top_bottom, violin_w, row_h])
        ax_e_top = fig.add_axes([left_e, top_bottom, ecdf_w, row_h])
        ax_v_bot = fig.add_axes([left_v, bot_bottom, violin_w, row_h])
        ax_e_bot = fig.add_axes([left_e, bot_bottom, ecdf_w, row_h])

        # Top row: THINGS
        exp_top = experiments[0]  # hebart2023things
        _draw_violin_on_ax(ax_v_top, ppv, exp_top, grad)
        _draw_ecdf_on_ax(ax_e_top, ppv, exp_top, grad)

        # Bottom row: Intertemporal choice
        exp_bot = experiments[1]  # ruggeri2022globalizability
        _draw_violin_on_ax(ax_v_bot, ppv, exp_bot, grad)
        _draw_ecdf_on_ax(ax_e_bot, ppv, exp_bot, grad)

        # Row titles — bold label+name, normal citation, centered across row
        for exp, y in [(exp_top, title_y_top), (exp_bot, title_y_bot)]:
            name, cite = EXP_TITLES.get(exp, (exp, ''))
            label = 'a' if exp == exp_top else 'b'
            bold_part = TextArea(
                f"{label}  {name} ",
                textprops=dict(fontweight='bold', fontsize=12),
            )
            cite_part = TextArea(
                cite,
                textprops=dict(fontweight='normal', fontsize=12),
            )
            title_box = HPacker(children=[bold_part, cite_part],
                                align="baseline", pad=0, sep=1)
            ab = AnchoredOffsetbox(
                loc='upper center', child=title_box,
                pad=0, frameon=False,
                bbox_to_anchor=(0.5, y),
                bbox_transform=fig.transFigure,
            )
            ab.patch.set_visible(False)
            fig.add_artist(ab)

        save_figure(fig, outdir, 'permutation_grid')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    agg_path = os.path.join(FIGURES_DIR, 'permutation_aggregate.csv')
    ppv_path = os.path.join(FIGURES_DIR, 'permutation_per_participant.csv')

    if not os.path.exists(agg_path):
        print(f"Error: {agg_path} not found. Run analyse_permutation.py first.")
        return
    if not os.path.exists(ppv_path):
        print(f"Error: {ppv_path} not found. Run analyse_permutation.py first.")
        return

    print("Loading data...")
    agg = pd.read_csv(agg_path)
    ppv = pd.read_csv(ppv_path)
    print(f"  Aggregate: {len(agg)} rows")
    print(f"  Per-participant: {len(ppv)} rows")

    print("\nGenerating figures...")
    fig_combined_panel(agg, FIGURES_DIR)
    fig_ecdf(ppv, FIGURES_DIR)
    fig_violin(ppv, FIGURES_DIR)
    fig_violin_ecdf_grid(ppv, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
