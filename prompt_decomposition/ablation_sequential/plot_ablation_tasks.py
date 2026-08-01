#!/usr/bin/env python3
"""
Task-specific ablation figures for all experiments in the sequential ablation study.

For each task, produces a two-panel figure:
  a) Information retention ratio across ablation conditions (all 8 models)
  b) Bar plot of raw NLL for selected models under each condition,
     plus cognitive model baseline (where available) and chance line.

Reads from:  ablation_results/
Saves to:    figures/
"""

import csv
import math
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import scienceplots  # noqa: F401

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'ablation_results')
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

COND_ORDER = ['original', 'instruction_ablated', 'content_masked', 'history_only']
COND_LABELS = ['Original', 'Instruction\nablated', 'Content\nmasked', 'History\nonly']
COND_LABELS_BAR = ['Original', 'Instr.\nablated', 'Content\nmasked', 'History\nonly']

_ln = math.log
LN_K = {
    "badham2017deficits":          _ln(2),
    "bahrami2020four":             _ln(4),
    "collsiöö2023MCPL":            _ln(9),
    "feng2021dynamics":            _ln(2),
    "flesch2018comparing":         _ln(2),
    "frey2017cct":                 _ln(2),
    "garcia2023experiential":      None,      # mixed: binary + probability estimates
    "gershman2018deconstructing":  _ln(2),
    "gershman2020reward":          _ln(3),
    "hilbig2014generalized":       _ln(2),
    "krueger2022identifying":      None,      # mixed: multi-stage variable actions
    "kool2016when":                _ln(2),
    "kool2017cost":                _ln(2),
    "lefebvre2017behavioural":     _ln(2),
    "levering2020revisiting":      None,      # mixed: binary + 9-point rating
    "peterson2021using":           _ln(2),
    "plonsky2018when":             _ln(2),
    "sadeghiyeh2020temporal":      _ln(2),
    "schulz2020finding":           _ln(8),
    "somerville2017charting":      _ln(2),
    "speekenbrink2008learning":    _ln(2),
    "steingroever2015data":        _ln(4),
    "tomov2020discovery":          _ln(5),
    "tomov2021multitask":          _ln(3),
    "waltz2020differential":       _ln(2),
    "wilson2014humans":            _ln(2),
    "wise2019acomputational":      None,      # continuous probability estimates
    "wu2018generalisation":        _ln(30),
    "wulff2018description":        _ln(2),
    "wulff2018sampling":           None,      # mixed: sample/stop/choose phases
    "xiong2023neural":             _ln(2),
    "zorowitz2023data":            _ln(2),
}

# Cognitive model baselines (from Psych-101 reported values; None = unavailable)
COG_BASELINE = {
    "badham2017deficits":          0.6108,
    "bahrami2020four":             0.9043,
    "collsiöö2023MCPL":            1.9157,
    "feng2021dynamics":            0.3595,
    "flesch2018comparing":         0.9105,
    "frey2017cct":                 0.2629,
    "garcia2023experiential":      None,
    "gershman2018deconstructing":  0.4187,
    "gershman2020reward":          0.8575,
    "hilbig2014generalized":       0.1922,
    "krueger2022identifying":      None,
    "kool2016when":                0.6043,
    "kool2017cost":                0.6043,
    "lefebvre2017behavioural":     0.5047,
    "levering2020revisiting":      0.5313,
    "peterson2021using":           0.6563,
    "plonsky2018when":             0.6607,
    "sadeghiyeh2020temporal":      0.3595,
    "schulz2020finding":           1.0530,
    "somerville2017charting":      0.3595,
    "speekenbrink2008learning":    0.6267,
    "steingroever2015data":        1.1555,
    "tomov2020discovery":          None,
    "tomov2021multitask":          1.0424,
    "waltz2020differential":       0.3595,
    "wilson2014humans":            0.3595,
    "wise2019acomputational":      None,
    "wu2018generalisation":        2.7635,
    "wulff2018description":        0.6120,
    "wulff2018sampling":           0.5404,
    "xiong2023neural":             0.4378,
    "zorowitz2023data":            0.6043,
}

TASK_TYPE = {
    "badham2017deficits":          "Supervised Learning",
    "bahrami2020four":             "Multi-armed Bandits",
    "collsiöö2023MCPL":            "Supervised Learning",
    "feng2021dynamics":            "Multi-armed Bandits",
    "flesch2018comparing":         "Decision-making",
    "frey2017cct":                 "Decision-making",
    "garcia2023experiential":      "Decision-making",
    "gershman2018deconstructing":  "Multi-armed Bandits",
    "gershman2020reward":          "Memory",
    "hilbig2014generalized":       "Decision-making",
    "krueger2022identifying":      "Decision-making",
    "kool2016when":                "Markov Decision Processes",
    "kool2017cost":                "Markov Decision Processes",
    "lefebvre2017behavioural":     "Multi-armed Bandits",
    "levering2020revisiting":      "Supervised Learning",
    "peterson2021using":           "Decision-making",
    "plonsky2018when":             "Decision-making",
    "sadeghiyeh2020temporal":      "Multi-armed Bandits",
    "schulz2020finding":           "Multi-armed Bandits",
    "somerville2017charting":      "Multi-armed Bandits",
    "speekenbrink2008learning":    "Supervised Learning",
    "steingroever2015data":        "Multi-armed Bandits",
    "tomov2020discovery":          "Markov Decision Processes",
    "tomov2021multitask":          "Markov Decision Processes",
    "waltz2020differential":       "Multi-armed Bandits",
    "wilson2014humans":            "Multi-armed Bandits",
    "wise2019acomputational":      "Supervised Learning",
    "wu2018generalisation":        "Multi-armed Bandits",
    "wulff2018description":        "Decision-making",
    "wulff2018sampling":           "Multi-armed Bandits",
    "xiong2023neural":             "Multi-armed Bandits",
    "zorowitz2023data":            "Markov Decision Processes",
}

# Display name and citation for each experiment
# Format: "Task Type: Experiment name (Reference)"
TASK_DISPLAY = {
    "badham2017deficits":          ("Shepard categorisation",               "Badham et al., 2017"),
    "bahrami2020four":             ("Drifting four-armed bandit",           "Bahrami et al., 2020"),
    "collsiöö2023MCPL":            ("Multiple-cue judgement",               "Collsiöö et al., 2023"),
    "feng2021dynamics":            ("Horizon task",                         "Feng et al., 2021"),
    "flesch2018comparing":         ("Gardening task",                       "Flesch et al., 2018"),
    "frey2017cct":                 ("Columbia card task",                   "Frey et al., 2017"),
    "garcia2023experiential":      ("Experiential-symbolic task",           "Garcia et al., 2023"),
    "gershman2018deconstructing":  ("Two-armed bandit",                     "Gershman, 2018"),
    "gershman2020reward":          ("Conditional associative learning",     "Gershman, 2020"),
    "hilbig2014generalized":       ("Multi-attribute decision-making",      "Hilbig et al., 2014"),
    "krueger2022identifying":      ("Risky choice",                         "Krueger et al., 2022"),
    "kool2016when":                ("Two-step task",                        "Kool et al., 2016"),
    "kool2017cost":                ("Two-step task",                        "Kool et al., 2017"),
    "lefebvre2017behavioural":     ("Probabilistic instrumental learning",  "Lefebvre et al., 2017"),
    "levering2020revisiting":      ("Medin categorisation",                 "Levering et al., 2020"),
    "peterson2021using":           ("choices13k",                           "Peterson et al., 2021"),
    "plonsky2018when":             ("CPC18",                                "Plonsky et al., 2018"),
    "sadeghiyeh2020temporal":      ("Horizon task",                         "Sadeghiyeh et al., 2020"),
    "schulz2020finding":           ("Structured bandit",                    "Schulz et al., 2020"),
    "somerville2017charting":      ("Horizon task",                         "Somerville et al., 2017"),
    "speekenbrink2008learning":    ("Weather prediction task",              "Speekenbrink & Shanks, 2008"),
    "steingroever2015data":        ("Iowa gambling task",                   "Steingroever et al., 2015"),
    "tomov2020discovery":          ("Virtual subway network",               "Tomov et al., 2020"),
    "tomov2021multitask":          ("Multi-task reinforcement learning",    "Tomov et al., 2021"),
    "waltz2020differential":       ("Horizon task",                         "Waltz et al., 2020"),
    "wilson2014humans":            ("Horizon task",                         "Wilson et al., 2014"),
    "wise2019acomputational":      ("Aversive learning",                    "Wise et al., 2019"),
    "wu2018generalisation":        ("Spatially correlated MAB",             "Wu et al., 2018"),
    "wulff2018description":        ("Decisions from description",           "Wulff et al., 2018"),
    "wulff2018sampling":           ("Decisions from experience",            "Wulff et al., 2018"),
    "xiong2023neural":             ("Changing bandit",                      "Xiong et al., 2023"),
    "zorowitz2023data":            ("Two-step task",                        "Zorowitz et al., 2023"),
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

# Bar models for panel b (sorted by size within each family)
BAR_MODELS = [
    ('Qwentaur-0.6B-LoRA',   'Qwentaur-0.6B'),
    ('Qwentaur-4B-LoRA',     'Qwentaur-4B'),
    ('Qwentaur-14B-LoRA',    'Qwentaur-14B'),
    ('Llama-Centaur-1B-LoRA', 'Llama-Centaur-1B'),
    ('Llama-Centaur-3B-LoRA', 'Llama-Centaur-3B'),
    ('Llama-Centaur-8B-LoRA', 'Llama-Centaur-8B'),
]

COLORS = {'Llama': '#0082fb', 'Qwen': '#7F6DEF'}
COG_COLOR = '#E07B39'


# =============================================================================
# Helpers
# =============================================================================

def _family(model):
    return 'Qwen' if 'Qwen' in model else 'Llama'


def _marker(model):
    return 's' if 'Qwen' in model else 'o'


def tint(hex_color, amount=0.0):
    """Lighten a colour by mixing with white."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])


def compute_bar_colors(bar_models):
    """Assign gradient colours: lightest for smallest, full colour for largest."""
    family_sizes = {'Llama': [], 'Qwen': []}
    for key, _ in bar_models:
        family_sizes[_family(key)].append(MODEL_SIZES[key])
    family_sorted = {f: sorted(set(s)) for f, s in family_sizes.items()}

    colors = {}
    for key, _ in bar_models:
        fam = _family(key)
        sizes = family_sorted[fam]
        n = len(sizes)
        idx = sizes.index(MODEL_SIZES[key])
        amount = 0.55 * (1 - idx / max(n - 1, 1))
        colors[key] = tint(COLORS[fam], amount)
    return colors


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


def retention(task, nll_orig, nll_cond):
    headroom = LN_K[task] - nll_orig
    if headroom < 1e-6:
        return None
    return (LN_K[task] - nll_cond) / headroom


def make_basename(task_key):
    """Create a clean ASCII filename from a task key."""
    name = task_key.replace('ö', 'o').replace('ä', 'a').replace('ü', 'u')
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    return f"fig_ablation_{name}"


# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """Load ablation CSVs for all tasks."""
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
                if task not in LN_K:
                    continue
                data.setdefault(model, {}).setdefault(task, {})[r['condition']] = float(r['loss'])
    return data


# =============================================================================
# Figure generation
# =============================================================================

def make_task_figure(data, task_key):
    """Generate a two-panel figure for a single task."""
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])
    ms, mew = 6, 0.8
    lnk = LN_K[task_key]
    cog = COG_BASELINE.get(task_key)
    has_cog = cog is not None

    # Build title: "Task Type: Experiment name (Reference)"
    exp_name, ref = TASK_DISPLAY[task_key]
    task_type = TASK_TYPE[task_key]
    title = f"{task_type}: {exp_name} ({ref})"

    bar_colors = compute_bar_colors(BAR_MODELS)

    with plt.style.context(['nature']):
        apply_style()
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 3.5),
                                          gridspec_kw={'width_ratios': [1, 1.8]})

        # ── Panel A: Retention ratio lines ──
        for model in models_sorted:
            tc = data.get(model, {}).get(task_key, {})
            if 'original' not in tc:
                continue
            nll_orig = tc['original']
            means = []
            for c in COND_ORDER:
                if c in tc:
                    v = retention(task_key, nll_orig, tc[c])
                    means.append(v if v is not None else 0)
                else:
                    means.append(0)

            color = COLORS[_family(model)]
            alpha = 0.3 + 0.7 * (MODEL_SIZES[model] / 14.0)
            label = MODEL_LABELS[model]

            ax_a.plot(range(4), means,
                      marker=_marker(model), markersize=ms,
                      markeredgecolor='white', markeredgewidth=mew,
                      linewidth=1.0, color=color, alpha=alpha, label=label)

        ax_a.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax_a.set_xticks(range(4))
        ax_a.set_xticklabels(COND_LABELS)
        ax_a.set_ylabel(
            r'$\frac{\ln(k) - \mathrm{NLL_{ablation}}}{\ln(k) - \mathrm{NLL_{original}}}$',
            fontsize=11)
        ax_a.spines['top'].set_visible(False)
        ax_a.spines['right'].set_visible(False)
        ax_a.set_title('a  Information retention', fontsize=8,
                        fontweight='bold', loc='center')

        # ── Panel B: Bar plot ──
        n_cond = len(COND_ORDER)
        n_bar_models = len(BAR_MODELS)
        n_groups = n_cond + (1 if has_cog else 0)
        group_width = 0.75
        bar_width = group_width / n_bar_models

        x_all = np.arange(n_groups).astype(float)
        x_cond = x_all[:n_cond]
        x_cog = x_all[n_cond] if has_cog else None

        # Collect all bar values for y-axis limits
        all_vals = []
        for model_key, _ in BAR_MODELS:
            tc = data.get(model_key, {}).get(task_key, {})
            all_vals.extend(tc.get(c, 0) for c in COND_ORDER)
        if has_cog:
            all_vals.append(cog)
        all_vals.append(lnk)

        for i, (model_key, model_label) in enumerate(BAR_MODELS):
            tc = data.get(model_key, {}).get(task_key, {})
            vals = [tc.get(c, 0) for c in COND_ORDER]
            offset = (i - (n_bar_models - 1) / 2) * bar_width
            color = bar_colors[model_key]
            bars = ax_b.bar(x_cond + offset, vals, bar_width * 0.88,
                            color=color, alpha=0.92, label=model_label,
                            edgecolor='white', linewidth=0.4)
            # Value labels on top of bars
            for bar, v in zip(bars, vals):
                ax_b.text(bar.get_x() + bar.get_width() / 2, v,
                          f'{v:.2f}', ha='center', va='bottom',
                          fontsize=4.5, color=color, fontweight='bold')

        # Cognitive model bar (if available)
        if has_cog:
            cog_bar = ax_b.bar(x_cog, cog, bar_width * 0.88,
                               color=COG_COLOR, alpha=0.92,
                               label='Cognitive model',
                               edgecolor='white', linewidth=0.4)
            ax_b.text(cog_bar[0].get_x() + cog_bar[0].get_width() / 2, cog,
                      f'{cog:.2f}', ha='center', va='bottom',
                      fontsize=4.5, color=COG_COLOR, fontweight='bold')

            # Shaded band between cognitive model and chance
            band_lo, band_hi = min(cog, lnk), max(cog, lnk)
            ax_b.axhspan(band_lo, band_hi, color='gray', alpha=0.08, zorder=0)

            # Cognitive model reference line (light)
            ax_b.axhline(cog, color=COG_COLOR, linewidth=0.6, linestyle='--',
                          alpha=0.4, zorder=1)

        # Chance level line (grey dashed)
        ax_b.axhline(lnk, color='gray', linewidth=0.8, linestyle='--',
                      alpha=0.7, zorder=1, label='Chance')

        # Truncated y-axis: floor just below min, ceiling just above max
        y_min = min(all_vals) * 0.92
        y_max = max(all_vals) * 1.06
        ax_b.set_ylim(y_min, y_max)

        # X-axis
        all_labels = list(COND_LABELS_BAR)
        if has_cog:
            all_labels.append('Cognitive\nmodel')
        ax_b.set_xticks(x_all)
        ax_b.set_xticklabels(all_labels)
        ax_b.set_ylabel('NLL')
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.set_title('b  Raw NLL by condition', fontsize=8,
                        fontweight='bold', loc='center')

        plt.tight_layout(pad=0.8)

        # Shared legend at the bottom, centred
        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(handles), fontsize=7.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.06),
                   handlelength=1.5, handletextpad=0.4,
                   columnspacing=1.0)

        return fig


def make_task_figure_bar_only(data, task_key):
    """Generate a single-panel (bar-only) figure for tasks without a defined ln(k)."""
    cog = COG_BASELINE.get(task_key)
    has_cog = cog is not None

    bar_colors = compute_bar_colors(BAR_MODELS)

    with plt.style.context(['nature']):
        apply_style()
        fig, ax_b = plt.subplots(figsize=(7, 3.5))

        n_cond = len(COND_ORDER)
        n_bar_models = len(BAR_MODELS)
        n_groups = n_cond + (1 if has_cog else 0)
        group_width = 0.75
        bar_width = group_width / n_bar_models

        x_all = np.arange(n_groups).astype(float)
        x_cond = x_all[:n_cond]
        x_cog = x_all[n_cond] if has_cog else None

        all_vals = []
        for model_key, _ in BAR_MODELS:
            tc = data.get(model_key, {}).get(task_key, {})
            all_vals.extend(tc.get(c, 0) for c in COND_ORDER)
        if has_cog:
            all_vals.append(cog)

        for i, (model_key, model_label) in enumerate(BAR_MODELS):
            tc = data.get(model_key, {}).get(task_key, {})
            vals = [tc.get(c, 0) for c in COND_ORDER]
            offset = (i - (n_bar_models - 1) / 2) * bar_width
            color = bar_colors[model_key]
            bars = ax_b.bar(x_cond + offset, vals, bar_width * 0.88,
                            color=color, alpha=0.92, label=model_label,
                            edgecolor='white', linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax_b.text(bar.get_x() + bar.get_width() / 2, v,
                          f'{v:.2f}', ha='center', va='bottom',
                          fontsize=4.5, color=color, fontweight='bold')

        if has_cog:
            cog_bar = ax_b.bar(x_cog, cog, bar_width * 0.88,
                               color=COG_COLOR, alpha=0.92,
                               label='Cognitive model',
                               edgecolor='white', linewidth=0.4)
            ax_b.text(cog_bar[0].get_x() + cog_bar[0].get_width() / 2, cog,
                      f'{cog:.2f}', ha='center', va='bottom',
                      fontsize=4.5, color=COG_COLOR, fontweight='bold')

            ax_b.axhline(cog, color=COG_COLOR, linewidth=0.6, linestyle='--',
                          alpha=0.4, zorder=1)

        y_min = min(v for v in all_vals if v > 0) * 0.92
        y_max = max(all_vals) * 1.06
        ax_b.set_ylim(y_min, y_max)

        all_labels = list(COND_LABELS_BAR)
        if has_cog:
            all_labels.append('Cognitive\nmodel')
        ax_b.set_xticks(x_all)
        ax_b.set_xticklabels(all_labels)
        ax_b.set_ylabel('NLL')
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.set_title('Raw NLL by condition', fontsize=8,
                        fontweight='bold', loc='center')

        plt.tight_layout(pad=0.8)

        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(handles), fontsize=7.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.06),
                   handlelength=1.5, handletextpad=0.4,
                   columnspacing=1.0)

        return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    data = load_data()
    print(f"Loaded {len(data)} models")

    for task_key in sorted(LN_K.keys()):
        if LN_K[task_key] is None:
            print(f"\nGenerating bar-only figure for {task_key}...")
            fig = make_task_figure_bar_only(data, task_key)
            basename = make_basename(task_key)
            save_figure(fig, OUT_DIR, basename)
            plt.close(fig)
            continue
        print(f"\nGenerating figure for {task_key}...")
        fig = make_task_figure(data, task_key)
        basename = make_basename(task_key)
        save_figure(fig, OUT_DIR, basename)
        plt.close(fig)

    print("\nDone.")
