#!/usr/bin/env python3
"""
Ablation figures.

Figure 1 (combined):  Panel a retention ratio lines + Panel b heatmap.
Figure 2 (appendix):  Full heatmap — main experiments (left) + excluded experiments (right).

Reads from:  ablation_results/
Saves to:    figures/
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import scienceplots  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'ablation_results')
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

COND_ORDER = ['original', 'instruction_ablated', 'content_masked', 'history_only']
COND_LABELS = ['Original', 'Instruction\nablated', 'Content\nmasked', 'History\nonly']

# Task-specific chance floor: ln(k) where k = number of response options
LN_K = {
    "badham2017deficits":          np.log(2),
    "bahrami2020four":             np.log(4),
    "collsiöö2023MCPL":            np.log(9),
    "feng2021dynamics":            np.log(2),
    "flesch2018comparing":         np.log(2),
    "frey2017cct":                 np.log(2),
    "gershman2018deconstructing":  np.log(2),
    "gershman2020reward":          np.log(3),
    "hilbig2014generalized":       np.log(2),
    "kool2016when":                np.log(2),
    "kool2017cost":                np.log(2),
    "lefebvre2017behavioural":     np.log(2),
    "peterson2021using":           np.log(2),
    "plonsky2018when":             np.log(2),
    "sadeghiyeh2020temporal":      np.log(2),
    "schulz2020finding":           np.log(8),
    "somerville2017charting":      np.log(2),
    "speekenbrink2008learning":    np.log(2),
    "steingroever2015data":        np.log(4),
    "tomov2020discovery":          np.log(5),
    "tomov2021multitask":          np.log(3),
    "waltz2020differential":       np.log(2),
    "wilson2014humans":            np.log(2),
    "wu2018generalisation":        np.log(30),
    "wulff2018description":        np.log(2),
    "xiong2023neural":             np.log(2),
    "zorowitz2023data":            np.log(2),
}

VALID_TASKS = set(LN_K.keys())

# Task type
TASK_TYPE = {
    "badham2017deficits":          "Supervised learning",
    "bahrami2020four":             "Multi-armed bandits",
    "collsiöö2023MCPL":            "Supervised learning",
    "feng2021dynamics":            "Multi-armed bandits",
    "flesch2018comparing":         "Decision-making",
    "frey2017cct":                 "Decision-making",
    "gershman2018deconstructing":  "Multi-armed bandits",
    "gershman2020reward":          "Memory",
    "hilbig2014generalized":       "Decision-making",
    "kool2016when":                "Markov decision processes",
    "kool2017cost":                "Markov decision processes",
    "lefebvre2017behavioural":     "Multi-armed bandits",
    "peterson2021using":           "Decision-making",
    "plonsky2018when":             "Decision-making",
    "sadeghiyeh2020temporal":      "Multi-armed bandits",
    "schulz2020finding":           "Multi-armed bandits",
    "somerville2017charting":      "Multi-armed bandits",
    "speekenbrink2008learning":    "Supervised learning",
    "steingroever2015data":        "Multi-armed bandits",
    "tomov2020discovery":          "Markov decision processes",
    "tomov2021multitask":          "Markov decision processes",
    "waltz2020differential":       "Multi-armed bandits",
    "wilson2014humans":            "Multi-armed bandits",
    "wu2018generalisation":        "Multi-armed bandits",
    "wulff2018description":        "Decision-making",
    "xiong2023neural":             "Multi-armed bandits",
    "zorowitz2023data":            "Markov decision processes",
}

TYPE_COLORS = {
    "Decision-making":           "#51DA4C",   # Primary Green
    "Markov decision processes": "#1C9418",   # Mid Green
    "Multi-armed bandits":       "#AEF2AC",   # Pale Green
    "Memory":                    "#3C46FF",   # Primary Blue
    "Supervised learning":       "#193718",   # Forest Green
}

# Color palette — matched to generate_scaling_plots.py
COLORS = {'Llama': '#0082fb', 'Qwen': '#7F6DEF'}

# Full display names (disambiguated horizon tasks and two-step tasks)
TASK_FULL = {
    "badham2017deficits":          "Shepard categorisation",
    "bahrami2020four":             "Drifting four-armed bandit",
    "collsiöö2023MCPL":            "Multiple-cue judgement",
    "feng2021dynamics":            "Horizon task (Feng)",
    "flesch2018comparing":         "Gardening task",
    "frey2017cct":                 "Columbia card task",
    "gershman2018deconstructing":  "Two-armed bandit",
    "gershman2020reward":          "Cond. assoc. learning",
    "hilbig2014generalized":       "Multi-attribute DM",
    "kool2016when":                "Two-step task (Kool '16)",
    "kool2017cost":                "Two-step task (Kool '17)",
    "lefebvre2017behavioural":     "Probabilistic instrumental learning",
    "peterson2021using":           "choices13k",
    "plonsky2018when":             "CPC18",
    "sadeghiyeh2020temporal":      "Horizon task (Sadeghiyeh)",
    "schulz2020finding":           "Structured bandit",
    "somerville2017charting":      "Horizon task (Somerville)",
    "speekenbrink2008learning":    "Weather prediction task",
    "steingroever2015data":        "Iowa gambling task",
    "tomov2020discovery":          "Virtual subway network",
    "tomov2021multitask":          "Multi-task RL",
    "waltz2020differential":       "Horizon task (Waltz)",
    "wilson2014humans":            "Horizon task (Wilson)",
    "wu2018generalisation":        "Spatially correlated MAB",
    "wulff2018description":        "Decisions from description",
    "xiong2023neural":             "Changing bandit",
    "zorowitz2023data":            "Two-step task (Zorowitz)",
}

MODEL_SIZES = {
    'Qwentaur-0.6B-LoRA': 0.6,
    'Llama-Centaur-1B-LoRA': 1.0,
    'Qwentaur-1.7B-LoRA': 1.7,
    'Llama-Centaur-3B-LoRA': 3.0,
    'Qwentaur-4B-LoRA': 4.0,
    'Qwentaur-8B-LoRA': 8.0,
    'Llama-Centaur-8B-LoRA': 8.0,
    'Qwentaur-14B-LoRA': 14.0,
}

MODEL_LABELS = {
    'Qwentaur-0.6B-LoRA': 'Qwentaur-0.6B',
    'Llama-Centaur-1B-LoRA': 'Llama-Centaur-1B',
    'Qwentaur-1.7B-LoRA': 'Qwentaur-1.7B',
    'Llama-Centaur-3B-LoRA': 'Llama-Centaur-3B',
    'Qwentaur-4B-LoRA': 'Qwentaur-4B',
    'Qwentaur-8B-LoRA': 'Qwentaur-8B',
    'Llama-Centaur-8B-LoRA': 'Llama-Centaur-8B',
    'Qwentaur-14B-LoRA': 'Qwentaur-14B',
}

CMAP = LinearSegmentedColormap.from_list(
    'slate_seq', ['#f8f9fa', '#d0d4da', '#8e97a4', '#4f5b6b', '#1e2a38'])


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_style(fs=8, fl=5.5):
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs,
        'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1,
        'legend.fontsize': fl, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
    })


def save_figure(fig, outdir, basename, dpi=600):
    os.makedirs(outdir, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")


def _family(model):
    return 'Qwen' if 'Qwen' in model else 'Llama'


def _marker(model):
    return 's' if 'Qwen' in model else 'o'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(filter_k=True):
    """Load all ablation CSVs. If filter_k=True, keep only tasks with known k."""
    data = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('_ablation.csv'):
            continue
        model = f.replace('_ablation.csv', '').replace('socius-', '')
        with open(os.path.join(DATA_DIR, f), encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if r['loss'].strip() == '' or r['condition'] not in COND_ORDER:
                    continue
                task = r['task']
                if filter_k and task not in VALID_TASKS:
                    continue
                data.setdefault(model, {}).setdefault(task, {})[r['condition']] = float(r['loss'])
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def retention(task, nll_orig, nll_cond):
    headroom = LN_K[task] - nll_orig
    if headroom < 1e-6:
        return None
    return (LN_K[task] - nll_cond) / headroom


def build_heatmap_matrix(data, tasks, models):
    """Build fraction-of-learned-info-lost matrix (n_tasks x 4 conditions)."""
    matrix, valid_tasks = [], []
    for task in tasks:
        ln_k = LN_K[task]
        row = []
        for c in COND_ORDER:
            fracs = []
            for m in models:
                tc = data[m].get(task, {})
                if 'original' in tc and c in tc:
                    headroom = ln_k - tc['original']
                    if headroom > 1e-6:
                        fracs.append((tc[c] - tc['original']) / headroom)
            row.append(np.mean(fracs) if fracs else 0)
        matrix.append(row)
        valid_tasks.append(task)
    matrix = np.array(matrix)
    # Sort by history-only column
    sort_idx = np.argsort(matrix[:, 3])
    matrix = matrix[sort_idx]
    valid_tasks = [valid_tasks[i] for i in sort_idx]
    return matrix, valid_tasks


def draw_heatmap(ax, matrix_t, valid_tasks, cond_labels, cmap, show_yticks=True,
                 fontsize_val=5.5, fontsize_label=5.5):
    """Draw heatmap on an axes. Returns the imshow artist."""
    im = ax.imshow(matrix_t, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(len(valid_tasks)))
    ax.set_xticklabels([TASK_FULL.get(t, t) for t in valid_tasks],
                       rotation=60, ha='right', fontsize=fontsize_label)
    ax.set_yticks(range(4))
    if show_yticks:
        ax.set_yticklabels(cond_labels)
    else:
        ax.set_yticklabels([])

    vmax = matrix_t.max()
    white_thresh = vmax * 0.4
    for i in range(matrix_t.shape[0]):
        for j in range(matrix_t.shape[1]):
            val = matrix_t[i, j]
            txt_color = 'white' if val > white_thresh else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=fontsize_val, color=txt_color)
    return im


def draw_type_strip(ax, valid_tasks):
    """Draw task-type colour strip on an axes."""
    for j, t in enumerate(valid_tasks):
        c = TYPE_COLORS.get(TASK_TYPE.get(t, ''), '#cccccc')
        ax.add_patch(plt.Rectangle((j - 0.5, 0), 1, 1,
                                   facecolor=c, edgecolor='white', linewidth=0.5))
    ax.set_xlim(-0.5, len(valid_tasks) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE (main paper)
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_ONLY_THRESH = 1.0

def fig_combined(data):
    """Panel a: retention ratio lines. Panel b: heatmap of fraction learned info lost."""
    ms, mew = 6, 0.8
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])
    tasks_all = sorted(set(t for m in data.values() for t in m.keys()))
    models = sorted(data.keys())
    cond_labels_short = ['Original', 'Instr. ablated', 'Content masked', 'History only']

    # Build full matrix then split main / excluded
    matrix_full, tasks_full = build_heatmap_matrix(data, tasks_all, models)

    keep = matrix_full[:, 3] <= HISTORY_ONLY_THRESH
    matrix = matrix_full[keep]
    valid_tasks = [t for t, k in zip(tasks_full, keep) if k]
    # Re-sort
    sort_idx = np.argsort(matrix[:, 3])
    matrix = matrix[sort_idx]
    valid_tasks = [valid_tasks[i] for i in sort_idx]
    matrix_t = matrix.T

    with plt.style.context(['nature']):
        apply_style(fs=11)
        fig = plt.figure(figsize=(14, 4.5))

        # ── Panel A: Retention ratio lines ──
        ax_left = fig.add_axes([0.05, 0.04, 0.26, 0.84])

        for model in models_sorted:
            means, sems = [], []
            for c in COND_ORDER:
                vals = []
                for t in tasks_all:
                    tc = data[model].get(t, {})
                    if 'original' in tc and c in tc:
                        v = retention(t, tc['original'], tc[c])
                        if v is not None:
                            vals.append(v)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))

            color = COLORS[_family(model)]
            alpha = 0.3 + 0.7 * (MODEL_SIZES[model] / 14.0)
            label = MODEL_LABELS[model]

            ax_left.errorbar(range(4), means, yerr=sems,
                             marker=_marker(model), markersize=ms,
                             markeredgecolor='white', markeredgewidth=mew,
                             linewidth=1.0, color=color, alpha=alpha, label=label,
                             capsize=2, capthick=0.5)

        ax_left.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax_left.set_xticks(range(4))
        ax_left.set_xticklabels(COND_LABELS)
        ax_left.set_ylabel(r'$\frac{\ln(k) - \mathrm{NLL_{ablation}}}{\ln(k) - \mathrm{NLL_{original}}}$',
                           fontsize=14)
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        ax_left.legend(ncol=2, loc='lower left', fontsize=6.5,
                       borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
        ax_left.set_title('a  Information retention\nacross ablation conditions',
                          fontsize=12, fontweight='bold', loc='center')

        # ── Panel B: Horizontal heatmap ──
        hm_bottom, hm_height = 0.31, 0.57
        strip_h = 0.02
        hm_left, hm_width = 0.38, 0.56
        ax_right = fig.add_axes([hm_left, hm_bottom, hm_width, hm_height])

        im = draw_heatmap(ax_right, matrix_t, valid_tasks, cond_labels_short, CMAP,
                          fontsize_val=8.5, fontsize_label=8)

        ax_right.set_title('b  Fraction of learned information lost by experiment',
                           fontsize=12, fontweight='bold', loc='center', pad=13)

        cax = fig.add_axes([hm_left + hm_width + 0.01, hm_bottom, 0.010, hm_height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label('')

        # ── Task-type colour strip below heatmap ──
        ax_strip = fig.add_axes([hm_left, hm_bottom - strip_h - 0.001, hm_width, strip_h])
        draw_type_strip(ax_strip, valid_tasks)
        ax_right.tick_params(axis='x', pad=7)

        type_legend = [Patch(facecolor=TYPE_COLORS[t], edgecolor='white', linewidth=0.5, label=t)
                       for t in TYPE_COLORS]
        fig.legend(handles=type_legend, loc='lower center',
                   bbox_to_anchor=(0.66, -0.06), ncol=5, fontsize=7.5,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

        save_figure(fig, OUT_DIR, 'fig_ablation_combined')
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX HEATMAP (stacked: main on top, excluded on bottom)
# ═══════════════════════════════════════════════════════════════════════════════

APPENDIX_EXCLUDE = {'lefebvre2017behavioural'}

def fig_heatmap_appendix(data):
    """Stacked heatmap: main experiments (top) + excluded experiments (bottom)."""
    tasks_all = sorted(set(t for m in data.values() for t in m.keys()
                           if t not in APPENDIX_EXCLUDE))
    models = sorted(data.keys())
    cond_labels_short = ['Original', 'Instr. ablated', 'Content masked', 'History only']

    # Build full matrix then split
    matrix_full, tasks_full = build_heatmap_matrix(data, tasks_all, models)

    keep = matrix_full[:, 3] <= HISTORY_ONLY_THRESH
    mat_main = matrix_full[keep]
    tasks_main = [t for t, k in zip(tasks_full, keep) if k]
    mat_excl = matrix_full[~keep]
    tasks_excl = [t for t, k in zip(tasks_full, keep) if not k]

    # Re-sort each by history-only
    idx_m = np.argsort(mat_main[:, 3])
    mat_main = mat_main[idx_m]
    tasks_main = [tasks_main[i] for i in idx_m]

    idx_e = np.argsort(mat_excl[:, 3])
    mat_excl = mat_excl[idx_e]
    tasks_excl = [tasks_excl[i] for i in idx_e]

    n_main, n_excl = len(tasks_main), len(tasks_excl)
    print(f"  Appendix: {n_main} main + {n_excl} excluded")

    with plt.style.context(['nature']):
        apply_style(fs=11)
        fig = plt.figure(figsize=(14, 8.5))

        strip_h = 0.015
        hm_left, hm_width = 0.08, 0.84
        cbar_gap, cbar_w = 0.01, 0.008

        # ── Top: main experiments ──
        top_bottom, top_height = 0.56, 0.28
        ax_top = fig.add_axes([hm_left, top_bottom, hm_width, top_height])
        im_top = draw_heatmap(ax_top, mat_main.T, tasks_main, cond_labels_short, CMAP,
                              fontsize_val=8.5, fontsize_label=8)

        cax_top = fig.add_axes([hm_left + hm_width + cbar_gap, top_bottom, cbar_w, top_height])
        fig.colorbar(im_top, cax=cax_top).ax.tick_params(labelsize=7)

        ax_strip_top = fig.add_axes([hm_left, top_bottom - strip_h - 0.001, hm_width, strip_h])
        draw_type_strip(ax_strip_top, tasks_main)
        ax_top.tick_params(axis='x', pad=7)

        # ── Bottom: excluded experiments ──
        bot_bottom, bot_height = 0.08, 0.28
        ax_bot = fig.add_axes([hm_left, bot_bottom, hm_width, bot_height])
        im_bot = draw_heatmap(ax_bot, mat_excl.T, tasks_excl, cond_labels_short, CMAP,
                              fontsize_val=8.5, fontsize_label=8)

        cax_bot = fig.add_axes([hm_left + hm_width + cbar_gap, bot_bottom, cbar_w, bot_height])
        fig.colorbar(im_bot, cax=cax_bot).ax.tick_params(labelsize=7)

        ax_strip_bot = fig.add_axes([hm_left, bot_bottom - strip_h - 0.001, hm_width, strip_h])
        draw_type_strip(ax_strip_bot, tasks_excl)
        ax_bot.tick_params(axis='x', pad=7)

        # Type legend
        type_legend = [Patch(facecolor=TYPE_COLORS[t], edgecolor='white', linewidth=0.5, label=t)
                       for t in TYPE_COLORS]
        fig.legend(handles=type_legend, loc='lower center',
                   bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=10,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

        save_figure(fig, OUT_DIR, 'fig_heatmap_appendix')
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_k = load_data(filter_k=True)
    tasks_k = sorted(set(t for m in data_k.values() for t in m.keys()))
    print(f"Loaded {len(data_k)} models — {len(tasks_k)} tasks (known k)")
    fig_combined(data_k)
    fig_heatmap_appendix(data_k)
    print("Done")
