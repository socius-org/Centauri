#!/usr/bin/env python3
"""
Out-of-distribution (Psych-201) scaling figures, all four families.

  scaling_ood_psych201  standalone OOD scaling plot (mean NLL over the 18
                        held-out Psych-201 experiments vs model size, r=16).
  scaling_r16_and_ood   two panels: in-distribution Psych-101 r=16 (left)
                        next to the Psych-201 OOD panel (right).

Reads the flat per-task CSVs in results/psych201/ (this file lives in
results/psych201/figures/). The in-distribution panel reuses the loaders in
results/psych101/figures/viz_r8_and_ranks.py.

Context caveat: Smoltaur and Olmotaur-1B were evaluated at their native
contexts (8192 / 4096) vs 32768 for the other families. Psych-201 has no
published Centaur-70B number, so only the *reproduced* 70B point is shown on
the OOD panel.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
P201_DIR = os.path.join(HERE, '..')                       # flat OOD CSVs
_VR_DIR = os.path.join(HERE, '..', '..', 'psych101', 'figures')
sys.path.insert(0, _VR_DIR)
import viz_r8_and_ranks as vr  # noqa: E402
from viz_r8_and_ranks import COLORS  # noqa: E402

OUT_DIR = HERE

# Appended to the combined figure's basename by --cognitive-matched. Only the
# in-distribution (Psych-101) left panel changes: its cognitive comparison is
# restricted to the 38 tasks that have a reported cognitive model, driven by the
# matched machinery in viz_r8_and_ranks. Psych-201 has no cognitive baseline, so
# the OOD panel is identical.
NAME_SUFFIX = ''

# family -> series of (size_label, params_B, csv_path)
FAMILIES = {
    'llama': dict(label='Llama-Centaur', color=COLORS['llama'], marker='o',
                  cells=[(s, p, os.path.join(P201_DIR, f'socius-Llama-Centaur-{s}-LoRA.csv'))
                         for s, p in [('1B', 1.0), ('3B', 3.0), ('8B', 8.0)]]),
    'qwen': dict(label='Qwentaur', color=COLORS['qwen'], marker='s',
                 cells=[(s, p, os.path.join(P201_DIR, f'socius-Qwentaur-{s}-LoRA.csv'))
                        for s, p in [('0.6B', 0.6), ('1.7B', 1.7), ('4B', 4.0),
                                     ('8B', 8.0), ('14B', 14.0)]]),
    'smollm': dict(label='Smoltaur', color='#FFD21E', dark='#B8860B', marker='o',
                   cells=[(s, p, os.path.join(P201_DIR, f'socius-Smoltaur-{s}-LoRA-r16.csv'))
                          for s, p in [('0.1B', 0.1), ('0.4B', 0.4), ('1.7B', 1.7),
                                       ('3B', 3.0)]]),
    'olmo': dict(label='Olmotaur', color='#F0529C', marker='s',
                 cells=[(s, p, os.path.join(P201_DIR, f'socius-Olmotaur-{s}-LoRA-r16.csv'))
                        for s, p in [('1B', 1.0), ('7B', 7.0)]]),
}
PLOT_FAMILIES = ['llama', 'qwen', 'smollm', 'olmo']

BASES = {
    'llama': [(1.0, 'unsloth-Llama-3.2-1B'), (3.0, 'unsloth-Llama-3.2-3B'),
              (8.0, 'unsloth-Meta-Llama-3.1-8B')],
    'qwen': [(0.6, 'unsloth-Qwen3-0.6B-Base'), (1.7, 'unsloth-Qwen3-1.7B-Base'),
             (4.0, 'unsloth-Qwen3-4B-Base'), (8.0, 'unsloth-Qwen3-8B-Base'),
             (14.0, 'unsloth-Qwen3-14B-Base')],
}
CENTAUR_70B = os.path.join(P201_DIR, 'marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')

MS, MEW = vr.MS, vr.MEW
MS_70B, MS_70B_HOLLOW = vr.MS_70B, vr.MS_70B_HOLLOW
FS, FS_LEGEND, FS_SIZE_LABEL = vr.FS, vr.FS_LEGEND, vr.FS_SIZE_LABEL


def mean_nll(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)['loss'].mean()


def fit_line(x, y):
    return np.polyfit(np.log10(x), y, 1)


def param_axis(ax):
    ax.set_xscale('log')
    ax.set_xlim(0.07, 120)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    ax.set_xlabel('Parameters (billions)')


# ═══════════════════════════════════════════════════════════════════════════
# AXES CONTENT
# ═══════════════════════════════════════════════════════════════════════════

def draw_ood_axes(ax, gen_aware=False, families=None, extrapolate=True):
    """Psych-201 OOD scaling content (no title/legend). Psych-201 has only a
    reproduced Centaur-70B (no published/reported value). `gen_aware` fits only
    generation/context-matched size groups (Smoltaur SmolLM2, dotted to 3B;
    Olmotaur none) and shapes markers by context window. `families` restricts
    which families are drawn; `extrapolate` toggles the dotted extension."""
    fams = families or ['smollm', 'olmo', 'qwen', 'llama']
    zorders = {'llama': 5.1}
    for fam_key in fams:
        fam = FAMILIES[fam_key]
        pts = [(p, mean_nll(c), s) for s, p, c in fam['cells']]
        pts = [(p, v, s) for p, v, s in pts if v is not None]
        if not pts:
            continue
        accent = fam.get('dark', fam['color'])
        if gen_aware:
            for p, v, s in pts:
                ax.plot([p], [v], vr.ctx_marker(fam_key, s), linestyle='none',
                        markersize=MS, markerfacecolor=fam['color'],
                        markeredgecolor='white', markeredgewidth=MEW,
                        zorder=zorders.get(fam_key, 5))
            grp, extrap = vr.fit_group(fam_key, [s for _, _, s in pts])
        else:
            xs = np.array([p for p, _, _ in pts]); ys = np.array([v for _, v, _ in pts])
            ax.plot(xs, ys, fam['marker'], linestyle='none', markersize=MS,
                    markerfacecolor=fam['color'], markeredgecolor='white',
                    markeredgewidth=MEW, zorder=zorders.get(fam_key, 5))
            grp, extrap = [s for _, _, s in pts], None
        if grp is not None and len(grp) >= 2:
            gp = [(p, v) for p, v, s in pts if s in grp]
            gx = np.array([p for p, _ in gp]); gy = np.array([v for _, v in gp])
            slope, intercept = fit_line(gx, gy)
            x1 = gx.max() * (1.0 if extrap else 1.6)
            x_fit = np.linspace(gx.min() * 0.75, x1, 100)
            ax.plot(x_fit, slope * np.log10(x_fit) + intercept, '--',
                    color=accent, lw=0.7, alpha=0.5, zorder=2)
            pmap = {s: p for s, p, c in fam['cells']}
            if extrapolate and extrap and extrap in pmap:
                xx = np.linspace(gx.max(), pmap[extrap], 40)
                ax.plot(xx, slope * np.log10(xx) + intercept, ':',
                        color=accent, lw=0.7, alpha=0.4, zorder=2)
        for p, v, s in pts:
            if p == 8.0 and fam_key in ('llama', 'qwen'):
                dx, dy, ha, va = ((-4, 0, 'right', 'center') if fam_key == 'llama'
                                  else (4, 0, 'left', 'center'))
            else:
                dx, dy, ha, va = 0, -4, 'center', 'top'
            ax.annotate(s, (p, v), xytext=(dx, dy), textcoords='offset points',
                        fontsize=FS_SIZE_LABEL, ha=ha, va=va, color=accent)

    for fam_key, stems in BASES.items():
        if fam_key not in fams:
            continue
        fam = FAMILIES[fam_key]
        bs = [p for p, _ in stems]
        bl = [mean_nll(os.path.join(P201_DIR, f'{stem}.csv')) for _, stem in stems]
        bmk = 'o' if gen_aware else fam['marker']   # base models are all >=32k
        ax.plot(bs, bl, bmk, linestyle='none', markersize=MS,
                markerfacecolor='none', markeredgecolor=fam['color'],
                markeredgewidth=MEW, alpha=0.35, zorder=3)

    v70 = mean_nll(CENTAUR_70B)
    if v70 is not None:
        ax.plot([70], [v70], 'D', color=COLORS['llama_dk'], markersize=MS_70B,
                markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                markeredgewidth=MEW, zorder=6, clip_on=False)
        ax.annotate('reproduced', (70, v70), xytext=(4, 3),
                    textcoords='offset points', fontsize=vr.FS_70B_LABEL,
                    ha='left', va='bottom', color=COLORS['llama_dk'])
    param_axis(ax)


def draw_indist_r16_axes(ax, data, base, refs, gen_aware=False, families=None,
                         extrapolate=True):
    """In-distribution Psych-101 r=16 scaling (scaling_r8 layout at rank 16).
    `gen_aware` fits only generation/context-matched size groups (Smoltaur
    SmolLM2, dotted to 3B; Olmotaur none) and shapes markers by context.
    `families` restricts which families are drawn; `extrapolate` toggles the
    dotted extension."""
    ax.axhline(vr.COG_MODEL, color='gray', linewidth=0.8, linestyle=':',
               alpha=0.7, zorder=1)
    fams = families or ['smollm', 'olmo', 'qwen', 'llama']
    zorders = {'llama': 5.1}
    for fam_key in fams:
        fam = vr.FAMILIES[fam_key]
        r16 = vr.series_by_rank(data[fam_key], 16)
        accent = fam.get('dark', fam['color'])
        if gen_aware:
            for _, r in r16.iterrows():
                ax.plot([r['params']], [r['mean_nll']], vr.ctx_marker(fam_key, r['size']),
                        linestyle='none', markersize=MS, markerfacecolor=fam['color'],
                        markeredgecolor='white', markeredgewidth=MEW,
                        zorder=zorders.get(fam_key, 5))
            grp, extrap = vr.fit_group(fam_key, r16.sort_values('params')['size'])
        else:
            ax.plot(r16['params'], r16['mean_nll'], fam['marker'], linestyle='none',
                    markersize=MS, markerfacecolor=fam['color'],
                    markeredgecolor='white', markeredgewidth=MEW,
                    zorder=zorders.get(fam_key, 5))
            grp, extrap = list(r16['size']), None
        if grp is not None and len(grp) >= 2:
            g = r16[r16['size'].isin(grp)]
            slope, intercept = vr.fit_line(g['params'].values, g['mean_nll'].values)
            x1 = g['params'].max() * (1.0 if extrap else 1.6)
            x_fit = np.linspace(g['params'].min() * 0.75, x1, 100)
            ax.plot(x_fit, slope * np.log10(x_fit) + intercept, '--', color=accent,
                    lw=0.7, alpha=0.5, zorder=2)
            pmap = vr.FAMILIES[fam_key]['params']
            if extrapolate and extrap and extrap in pmap:
                xx = np.linspace(g['params'].max(), pmap[extrap], 40)
                ax.plot(xx, slope * np.log10(xx) + intercept, ':', color=accent,
                        lw=0.7, alpha=0.4, zorder=2)
        for x, y, s in zip(r16['params'], r16['mean_nll'], r16['size']):
            if x == 8.0 and fam_key in ('llama', 'qwen'):
                dx, dy, ha, va = ((-4, 0, 'right', 'center') if fam_key == 'llama'
                                  else (4, 0, 'left', 'center'))
            else:
                dx, dy, ha, va = 0, -4, 'center', 'top'
            ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points',
                        fontsize=FS_SIZE_LABEL, ha=ha, va=va, color=accent)

    for fam_key in ['llama', 'qwen']:
        if fam_key not in fams:
            continue
        fam = vr.FAMILIES[fam_key]
        bs, bl = base[fam_key]
        bmk = 'o' if gen_aware else fam['marker']   # base models are all >=32k
        ax.plot(bs, bl, bmk, linestyle='none', markersize=MS,
                markerfacecolor='none', markeredgecolor=fam['color'],
                markeredgewidth=MEW, alpha=0.35, zorder=3)

    vr.draw_70b(ax, refs)
    ax.plot([70], [refs['llama70b']], 'o', color=COLORS['llama'], markersize=MS,
            markerfacecolor='none', markeredgecolor=COLORS['llama'],
            markeredgewidth=MEW, alpha=0.35, zorder=3, clip_on=False)
    vr.draw_bracket(ax, refs['llama70b'], refs['binz'], FS)
    param_axis(ax)


# ═══════════════════════════════════════════════════════════════════════════
# LEGENDS + FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def mk(marker, mfc, mec, msize=MS, alpha=1.0):
    return plt.Line2D([], [], marker=marker, linestyle='none', markersize=msize,
                      markerfacecolor=mfc, markeredgecolor=mec,
                      markeredgewidth=MEW, alpha=alpha)


def family_row():
    return [(mk(FAMILIES[k]['marker'], FAMILIES[k]['color'], 'white'),
             FAMILIES[k]['label']) for k in PLOT_FAMILIES]


def add_legends(fig, row1, row2, y1, y2):
    fig.legend([h for h, _ in row1], [l for _, l in row1], loc='lower center',
               bbox_to_anchor=(0.5, y1), ncol=len(row1), fontsize=FS_LEGEND,
               frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
    fig.legend([h for h, _ in row2], [l for _, l in row2], loc='lower center',
               bbox_to_anchor=(0.5, y2), ncol=len(row2), fontsize=FS_LEGEND,
               frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)


def apply_style():
    vr.apply_style(FS, FS_LEGEND)


def make_figure(outdir):
    with plt.style.context(['nature']):
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 5.4))
        ax.set_title('Psych-201 (r = 16, bf16)', fontsize=12, fontweight='bold')
        draw_ood_axes(ax)
        ax.set_ylabel('Mean negative log-likelihood')
        row2 = [
            (mk('o', 'none', COLORS['llama'], alpha=0.35), 'Llama-3.1/3.2 (base)'),
            (mk('s', 'none', COLORS['qwen'], alpha=0.35), 'Qwen3 (base)'),
            (mk('D', COLORS['llama_dk'], 'white', MS_70B), 'Centaur-70B (reproduced)'),
        ]
        add_legends(fig, family_row(), row2, -0.020, -0.062)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(outdir, f'scaling_ood_psych201.{ext}'),
                        dpi=600, bbox_inches='tight', facecolor='white')
        print('  Saved scaling_ood_psych201')
        plt.close()


def make_combined_figure(outdir, families=None, basename='scaling_r16_and_ood'):
    fams = families or PLOT_FAMILIES
    data = {k: vr.load_family(k) for k in PLOT_FAMILIES}
    base, refs = vr.load_aggr()
    with plt.style.context(['nature']):
        apply_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.6))

        ax1.set_title('Psych-101 (r = 16, bf16)', fontsize=11, fontweight='bold')
        draw_indist_r16_axes(ax1, data, base, refs, gen_aware=True,
                             families=fams, extrapolate=False)
        ax1.set_ylabel('Mean negative log-likelihood')

        ax2.set_title('Psych-201 (r = 16, bf16)', fontsize=11, fontweight='bold')
        draw_ood_axes(ax2, gen_aware=True, families=fams, extrapolate=False)

        # Generation/context-aware legend: family (base model; context), split by
        # shape. The shape key only appears when a truncated (<32k) family is
        # shown; base models are circles.
        fam_h = vr.family_legend_handles(fams)
        has_square = any(f in ('smollm', 'olmo') for f in fams)
        row2 = (vr.shape_key_handles() if has_square else []) + [
            (mk('o', 'none', COLORS['llama'], alpha=0.35), 'Llama-3.1/3.2 (base)'),
            (mk('o', 'none', COLORS['qwen'], alpha=0.35), 'Qwen3 (base)'),
            (mk('D', COLORS['llama_dk'], 'white', MS_70B), 'Centaur-70B (reproduced)'),
            (mk('D', 'none', COLORS['reported'], MS_70B_HOLLOW), 'Centaur-70B (reported)'),
            (plt.Line2D([], [], color='gray', lw=0.8, linestyle=':', alpha=0.7),
             'Cognitive baseline'),
        ]
        add_legends(fig, fam_h, row2, -0.05, -0.15)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(outdir, f'{basename}{NAME_SUFFIX}.{ext}'),
                        dpi=600, bbox_inches='tight', facecolor='white')
        print(f'  Saved {basename}{NAME_SUFFIX}')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Psych-201 OOD scaling figures.')
    parser.add_argument('--outdir', default=OUT_DIR)
    parser.add_argument('--cognitive-matched', action='store_true',
                        help='Restrict the in-distribution (left) panel to the '
                             '38 tasks with a reported cognitive model, so its '
                             'cognitive comparison is honest; writes '
                             'scaling_r16_and_ood_cognitive_matched. The OOD '
                             'panel is unchanged.')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.cognitive_matched:
        global NAME_SUFFIX
        NAME_SUFFIX = '_cognitive_matched'
        vr.NAME_SUFFIX = '_cognitive_matched'    # -> matched psych101 summaries
        vr.KEEP = vr.cognitive_matched_tasks()   # -> aggr base/70B over 38 tasks
        print(f'Cognitive-matched: left panel over {len(vr.KEEP)} tasks')
        make_combined_figure(args.outdir)        # only panel with a cognitive line
        make_combined_figure(args.outdir, families=['llama', 'qwen'],
                             basename='scaling_r16_and_ood_llamaqwen')
        return

    make_figure(args.outdir)
    make_combined_figure(args.outdir)
    make_combined_figure(args.outdir, families=['llama', 'qwen'],
                         basename='scaling_r16_and_ood_llamaqwen')


if __name__ == '__main__':
    main()
