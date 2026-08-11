#!/usr/bin/env python3
"""
Project-page figures.

Re-renders three paper figures in the socius project page's typography
(Rethink Sans, vendored under fonts/rethink-sans/; swap in another face
with --font-dir, e.g. a local GeneralSans checkout):

  scaling_rank_sweep_with_panels_cognitive_matched
      As in the paper (results/psych101/figures/viz_r8_and_ranks.py),
      cognitive-matched variant.
  permutation_grid
      As in the paper (prompt_decomposition/permutation_nonsequential/
      generate_permutation_plots.py).
  ablation_retention
      Panel (a) of the paper's fig_ablation_combined
      (prompt_decomposition/ablation_sequential/plot_ablation.py), as a
      standalone panel with Centaur-70B added as a ninth series.

The source modules are imported and their `apply_style` is wrapped so every
rcParam they set still applies, with only the font family overridden — the
figures stay pixel-compatible with the paper versions apart from type.

Usage:
    python project_page/generate_figures.py
    python project_page/generate_figures.py --font-dir path/to/fonts --font-name GeneralSans
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, 'figures')

# Make the three source modules importable regardless of cwd.
for rel in ('', os.path.join('results', 'psych101', 'figures'),
            os.path.join('prompt_decomposition', 'ablation_sequential'),
            os.path.join('prompt_decomposition', 'permutation_nonsequential')):
    sys.path.insert(0, os.path.join(ROOT, rel))

DEFAULT_FONT_DIR = os.path.join(HERE, 'fonts', 'rethink-sans')

# Centaur-70B ablation results (evaluated alongside ours; the paper's panel
# omits it only because its combined figure iterates the small-model grid).
CENTAUR70_KEY = 'marcelbinz-Llama-3.1-Centaur-70B-adapter'
CENTAUR70_LABEL = 'Centaur-70B'
CENTAUR70_COLOR = '#005bb5'   # llama_dk — matches its diamond in the scaling figure


# ═══════════════════════════════════════════════════════════════════════════
# FONT HANDLING
# ═══════════════════════════════════════════════════════════════════════════

def register_fonts(font_dir, font_name=None):
    """Register every ttf/otf under font_dir; return the family name to use."""
    files = sorted(f for f in os.listdir(font_dir)
                   if f.lower().endswith(('.ttf', '.otf')))
    if not files:
        raise SystemExit(f'No font files in {font_dir}')
    family = font_name
    for f in files:
        path = os.path.join(font_dir, f)
        font_manager.fontManager.addfont(path)
        if family is None:
            family = font_manager.FontProperties(fname=path).get_name()
    print(f'Registered {len(files)} font file(s) from {font_dir} as "{family}"')
    return family


def font_rc(family):
    """rcParams that swap the source modules' serif setup for the page face."""
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': [family],
        'mathtext.fontset': 'custom',
        'mathtext.rm': family,
        'mathtext.it': family,          # no italic vendored; upright math
        'mathtext.bf': f'{family}:bold',
        'mathtext.default': 'rm',
        'axes.unicode_minus': False,
    }


def patch_style(module, family):
    """Wrap module.apply_style so its sizes apply, then our font overrides."""
    orig = module.apply_style

    def patched(*args, **kwargs):
        orig(*args, **kwargs)
        plt.rcParams.update(font_rc(family))

    module.apply_style = patched


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — rank-sweep composite (cognitive-matched), as in the paper
# ═══════════════════════════════════════════════════════════════════════════

def build_scaling(family):
    import viz_r8_and_ranks as viz
    patch_style(viz, family)
    viz.NAME_SUFFIX = '_cognitive_matched'
    viz.KEEP = viz.cognitive_matched_tasks()
    print(f'  Cognitive-matched: {len(viz.KEEP)} tasks')
    data = {k: viz.load_family(k) for k in viz.PLOT_FAMILIES}
    base, refs = viz.load_aggr()
    viz.make_composite_figure(data, refs, OUT_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — order-permutation grid, as in the paper
# ═══════════════════════════════════════════════════════════════════════════

def build_permutation(family):
    import generate_permutation_plots as gp
    patch_style(gp, family)
    gp.PANEL_LABELS = False   # standalone web figure; nothing references a/b
    ppv = pd.read_csv(os.path.join(gp.FIGURES_DIR,
                                   'permutation_per_participant.csv'))
    gp.fig_violin_ecdf_grid(ppv, OUT_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ablation retention panel, with Centaur-70B added
# ═══════════════════════════════════════════════════════════════════════════

def build_ablation(family):
    import plot_ablation as pa
    patch_style(pa, family)
    data = pa.load_data(filter_k=True)
    tasks_all = sorted({t for m in data.values() for t in m})
    print(f'  Ablation: {len(data)} models, {len(tasks_all)} tasks')

    def series(model):
        """Mean +/- SEM retention per condition, as in pa.fig_combined."""
        means, sems = [], []
        for c in pa.COND_ORDER:
            vals = []
            for t in tasks_all:
                tc = data[model].get(t, {})
                if 'original' in tc and c in tc:
                    v = pa.retention(t, tc['original'], tc[c])
                    if v is not None:
                        vals.append(v)
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        return means, sems

    models_sorted = sorted(pa.MODEL_SIZES, key=pa.MODEL_SIZES.get)
    ms, mew = 6, 0.8

    with plt.style.context(['nature']):
        pa.apply_style(fs=11)
        # Near-square panel: the page shows it beside its caption, where a
        # wide aspect reads as stretched.
        fig, ax = plt.subplots(figsize=(4.5, 4.4))

        for model in models_sorted:
            means, sems = series(model)
            color = pa.COLORS[pa._family(model)]
            alpha = 0.3 + 0.7 * (pa.MODEL_SIZES[model] / 14.0)
            ax.errorbar(range(4), means, yerr=sems,
                        marker=pa._marker(model), markersize=ms,
                        markeredgecolor='white', markeredgewidth=mew,
                        linewidth=1.0, color=color, alpha=alpha,
                        label=pa.MODEL_LABELS[model], capsize=2, capthick=0.5)

        # Centaur-70B on top, echoing its diamond styling in the scaling figure.
        means, sems = series(CENTAUR70_KEY)
        ax.errorbar(range(4), means, yerr=sems,
                    marker='D', markersize=ms + 0.5,
                    markeredgecolor='white', markeredgewidth=mew,
                    linewidth=1.2, color=CENTAUR70_COLOR,
                    label=CENTAUR70_LABEL, capsize=2, capthick=0.5, zorder=6)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(pa.COND_LABELS)
        ax.set_ylabel(
            r'$\frac{\ln(k) - \mathrm{NLL_{ablation}}}{\ln(k) - \mathrm{NLL_{original}}}$',
            fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(ncol=2, loc='lower left', fontsize=6.5, borderpad=0.3,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
        ax.set_title('Information retention\nacross ablation conditions',
                     fontsize=12, fontweight='bold', loc='center')

        plt.tight_layout(pad=0.4)
        pa.save_figure(fig, OUT_DIR, 'ablation_retention')
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--font-dir', default=DEFAULT_FONT_DIR,
                        help='Directory of ttf/otf files to register '
                             '(default: vendored Rethink Sans)')
    parser.add_argument('--font-name', default=None,
                        help='Family name to select (default: first file\'s)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    family = register_fonts(args.font_dir, args.font_name)

    print('\n-- Figure 1: rank-sweep composite (cognitive-matched) --')
    build_scaling(family)
    print('\n-- Figure 2: order-permutation grid --')
    build_permutation(family)
    print('\n-- Figure 3: ablation retention (+ Centaur-70B) --')
    build_ablation(family)
    print(f'\nDone. Figures in {OUT_DIR}')


if __name__ == '__main__':
    main()
