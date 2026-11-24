#!/usr/bin/env python3
"""
Out-of-distribution (Psych-201) LoRA-rank figures, all four families.

The Psych-201 counterparts of the Psych-101 rank figures in
results/psych101/figures/viz_r8_and_ranks.py, drawn with the same helpers:

  scaling_rank_sweep_ood              full rank sweep (r=4..64), per-family bands
  scaling_rank_panels_ood             1x5 grid, one scaling panel per rank
  scaling_rank_sweep_with_panels_ood  main sweep + five per-rank minis (vertical)
  scaling_r8_and_rank_sweep_ood       Llama+Qwen: r=8 panel | rank sweep + minis row
  scaling_rank_sweep_and_ood          Psych-101 rank sweep (left) next to the
                                      Psych-201 rank sweep (right)

Also writes psych201_rank_sweep_figure_data.csv (the plotted means).

Reads the flat per-task CSVs in results/psych201/ (this file lives in
results/psych201/figures/); mean NLL is the unweighted mean over the 18
held-out Psych-201 experiments. Psych-201 has no cognitive baseline and no
published Centaur-70B number, so the OOD panels show neither -- only the
reproduced 70B point.

--cognitive-matched re-draws scaling_rank_sweep_and_ood with the Psych-101
(left) panel restricted to the 38 tasks that have a reported cognitive model.

Run from anywhere: `python viz_ood_rank_sweep.py`.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator

import viz_ood_scaling as vo          # also puts psych101/figures on sys.path
import viz_r8_and_ranks as vr

HERE = os.path.dirname(os.path.abspath(__file__))
P201_DIR = os.path.join(HERE, '..')

COLORS = vr.COLORS
RANKS = vr.RANKS
PLOT_FAMILIES = vr.PLOT_FAMILIES
MS, MEW = vr.MS, vr.MEW
FS, FS_LEGEND = vr.FS, vr.FS_LEGEND

# Qwentaur-1.7B has only r=8 and r=16 (no full rank sweep): as in the Psych-101
# composites, it is dropped from the sweep panel and the per-rank minis there.
DROP = {'qwen': {'1.7B'}}

SWEEP_TITLE = 'r = 4$-$64, bf16'
YLABEL = 'Mean negative log-likelihood'


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_ood_family(fam_key):
    """Psych-201 means in the layout vr.series_by_rank expects (one row per
    evaluated (size, rank) cell; rank 16 is the 'baseline' axis)."""
    fam = vr.FAMILIES[fam_key]
    rows = []
    for size, params in fam['params'].items():
        for rank in RANKS:
            path = os.path.join(
                P201_DIR, f"socius-{fam['label']}-{size}-LoRA-r{rank}.csv")
            mean = vo.mean_nll(path)
            if mean is None:
                continue
            rows.append({'size': size, 'rank': rank,
                         'axis': 'baseline' if rank == 16 else 'rank_sweep',
                         'mean_nll': mean, 'params': params})
    return pd.DataFrame(rows).sort_values('params')


def load_ood():
    """(data, base, refs) for Psych-201, mirroring vr.load_family/load_aggr."""
    data = {k: load_ood_family(k) for k in PLOT_FAMILIES}
    base = {k: ([p for p, _ in stems],
                [vo.mean_nll(os.path.join(P201_DIR, f'{stem}.csv'))
                 for _, stem in stems])
            for k, stems in vo.BASES.items()}
    refs = {'repl': vo.mean_nll(vo.CENTAUR_70B)}   # no reported / base 70B
    return data, base, refs


def write_figure_data(data, outdir):
    rows = []
    for k in PLOT_FAMILIES:
        for _, r in data[k].sort_values(['params', 'rank']).iterrows():
            rows.append({'benchmark': 'Psych-201',
                         'family': vr.FAMILIES[k]['label'], 'size': r['size'],
                         'params_billions': r['params'], 'rank': int(r['rank']),
                         'mean_nll': round(r['mean_nll'], 4)})
    path = os.path.join(outdir, 'psych201_rank_sweep_figure_data.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print('  Saved psych201_rank_sweep_figure_data.csv')


# ═══════════════════════════════════════════════════════════════════════════
# LEGEND + LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def ref_handles(reported=False, cog=False, prefix='Centaur-70B'):
    """70B / cognitive-baseline legend entries; Psych-201 has only 'reproduced'."""
    out = [(plt.Line2D([], [], marker='D', linestyle='none', markersize=vr.MS_70B,
                       markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                       markeredgewidth=MEW), f'{prefix} (reproduced)')]
    if reported:
        out.append((plt.Line2D([], [], marker='D', linestyle='none',
                               markersize=vr.MS_70B_HOLLOW, markerfacecolor='none',
                               markeredgecolor=COLORS['reported'],
                               markeredgewidth=MEW), f'{prefix} (reported)'))
    if cog:
        out.append((plt.Line2D([], [], color='gray', lw=0.8, linestyle=':',
                               alpha=0.7), 'Cognitive baseline'))
    return out


def legend_row(fig, row, y, ncol=None, **kw):
    opts = dict(loc='lower center', bbox_to_anchor=(0.5, y),
                ncol=ncol or len(row), fontsize=FS_LEGEND, frameon=False,
                handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
    opts.update(kw)
    fig.legend([h for h, _ in row], [l for _, l in row], **opts)


def rank_legend_row(fig, rank_row, y):
    """The rank shade ramp (one tuple of family-coloured markers per rank)."""
    legend_row(fig, rank_row, y, ncol=len(RANKS),
               handlelength=0.7 * len(rank_row[0][0]) + 0.2, handletextpad=0.6,
               columnspacing=1.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})


def circle_rank_row(fams):
    """Rank ramp with circles (Llama and Qwen both run at 32k context)."""
    return [(tuple(plt.Line2D([], [], marker='o', linestyle='none', markersize=MS,
                              markerfacecolor=vr.fam_shade(vr.FAMILIES[k], t),
                              markeredgecolor='white', markeredgewidth=MEW)
                   for k in fams), f'r = {rank}')
            for rank, t in zip(RANKS, vr.RANK_TS)]


def mini_ylim(data, refs, families, drop):
    """Shared y-range for per-rank minis: every plotted mean plus the 70B point."""
    vals = [refs['repl']]
    for k in families:
        df = data[k]
        vals += list(df[~df['size'].isin(drop.get(k, set()))]['mean_nll'])
    pad = 0.07 * (max(vals) - min(vals))
    return min(vals) - pad, max(vals) + 2.2 * pad   # headroom for the r label


def style_mini(ax, ylim, xlim, xticks, xticklabels):
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.tick_params(labelsize=6, length=2)
    ax.minorticks_off()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=6)


def save_figure(fig, outdir, basename, suffix=''):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{basename}{suffix}.{ext}'),
                    dpi=600, bbox_inches='tight', facecolor='white')
    print(f'  Saved {basename}{suffix}')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def make_rank_figure(data, refs, outdir):
    """Psych-201 counterpart of scaling_rank_sweep."""
    with plt.style.context(['nature']):
        vr.apply_style(FS, FS_LEGEND)
        fig, ax = plt.subplots(figsize=vr.FIGSIZE)
        ax.set_title(f'Psych-201 LoRA rank sweep ({SWEEP_TITLE})', fontsize=12,
                     fontweight='bold')
        vr.draw_sweep_content(ax, data, refs, FS, cog=False)
        row1 = [(vr.fam_legend_marker(k), vr.FAMILIES[k]['label'])
                for k in PLOT_FAMILIES] + ref_handles()
        _, rank_row = vr.sweep_legend_rows()
        legend_row(fig, row1, -0.020, columnspacing=1.2)
        rank_legend_row(fig, rank_row, -0.062)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_sweep_ood')
        plt.close()


def make_rank_scaling_grid(data, refs, outdir):
    """Psych-201 counterpart of scaling_rank_panels."""
    with plt.style.context(['nature']):
        vr.apply_style(FS, FS_LEGEND)
        fig, axes = plt.subplots(1, len(RANKS), figsize=(13.5, 3.1), sharey=True)
        for ax, rank in zip(axes, RANKS):
            ax.set_title(f'r = {rank}', fontsize=10, fontweight='bold')
            vr.draw_rank_series(ax, data, refs, rank, gen_aware=True,
                                extrapolate=False, cog=False)
            vr.param_axis(ax)
            if ax is axes[0]:
                ax.set_ylabel(YLABEL)
        legend_row(fig, vr.family_legend_handles(PLOT_FAMILIES), -0.12)
        legend_row(fig, vr.shape_key_handles() + ref_handles(), -0.28)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_panels_ood')
        plt.close()


def make_composite_figure(data, refs, outdir):
    """Psych-201 counterpart of scaling_rank_sweep_with_panels: rank sweep
    (main, left) with five per-rank minis stacked on the right."""
    ylim = mini_ylim(data, refs, PLOT_FAMILIES, DROP)
    with plt.style.context(['nature']):
        vr.apply_style(FS, FS_LEGEND)
        fig = plt.figure(figsize=(10.2, 7.0))
        gs = fig.add_gridspec(len(RANKS), 2, width_ratios=[3.2, 1.0],
                              hspace=0.14, wspace=0.16)

        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.set_title(f'Psych-201 LoRA rank sweep ({SWEEP_TITLE})',
                          fontsize=12, fontweight='bold')
        vr.draw_sweep_content(ax_main, data, refs, FS, exclude_sizes=DROP,
                              gen_aware=True, extrapolate=False, cog=False)

        minis = []
        for i, rank in enumerate(RANKS):
            ax = fig.add_subplot(gs[i, 1], sharex=minis[0] if minis else None,
                                 sharey=minis[0] if minis else None)
            minis.append(ax)
            vr.draw_rank_series(ax, data, refs, rank, msize=MS * 0.62, lw=0.55,
                                exclude_sizes=DROP, gen_aware=True,
                                extrapolate=False, cog=False)
            ax.text(0.955, 0.86, f'r = {rank}', transform=ax.transAxes,
                    fontsize=7.5, fontweight='bold', ha='right', va='top')
            style_mini(ax, ylim, (0.07, 120), [0.1, 1, 10, 100],
                       ['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
            if i == len(RANKS) - 1:
                ax.set_xlabel('Parameters (billions)', fontsize=7.5)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

        _, rank_row = vr.sweep_legend_rows()
        legend_row(fig, vr.family_legend_handles(PLOT_FAMILIES), -0.035)
        legend_row(fig, vr.shape_key_handles() + ref_handles(), -0.105)
        rank_legend_row(fig, rank_row, -0.150)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_sweep_with_panels_ood')
        plt.close()


def make_r8_and_rank_sweep(data, base, refs, outdir):
    """Psych-201 counterpart of scaling_r8_and_rank_sweep (Llama + Qwen):
    r=8 panel | rank sweep, with five per-rank minis along the bottom."""
    fams = ['llama', 'qwen']
    ylim = mini_ylim(data, refs, fams, DROP)
    with plt.style.context(['nature']):
        vr.apply_style(FS, FS_LEGEND)
        fig = plt.figure(figsize=(10.2, 7.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.22)
        gs_top = outer[0].subgridspec(1, 2, wspace=0.28)
        gs_bot = outer[1].subgridspec(1, len(RANKS), wspace=0.12)

        ax_a = fig.add_subplot(gs_top[0, 0])
        ax_a.set_title('Psych-201 LoRA (r = 8, bf16)', fontsize=11,
                       fontweight='bold')
        vr.draw_r8_axes(ax_a, data, base, refs, FS, families=fams,
                        gen_aware=True, cog=False)

        ax_b = fig.add_subplot(gs_top[0, 1])
        ax_b.set_title(f'Psych-201 LoRA rank sweep ({SWEEP_TITLE})', fontsize=11,
                       fontweight='bold')
        vr.draw_sweep_content(ax_b, data, refs, FS, families=fams,
                              exclude_sizes=DROP, gen_aware=True, cog=False)

        minis = []
        for i, rank in enumerate(RANKS):
            ax = fig.add_subplot(gs_bot[0, i],
                                 sharex=minis[0] if minis else None,
                                 sharey=minis[0] if minis else None)
            minis.append(ax)
            vr.draw_rank_series(ax, data, refs, rank, msize=MS * 0.62, lw=0.55,
                                families=fams, exclude_sizes=DROP,
                                gen_aware=True, cog=False)
            ax.text(0.94, 0.88, f'r = {rank}', transform=ax.transAxes,
                    fontsize=7.5, fontweight='bold', ha='right', va='top')
            style_mini(ax, ylim, (0.4, 120), [1, 10, 100],
                       ['$10^0$', '$10^1$', '$10^2$'])
            if i == len(RANKS) // 2:
                ax.set_xlabel('Parameters (billions)', fontsize=7.5)
            if i > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        def circ(color, hollow=False):
            return plt.Line2D([], [], marker='o', linestyle='none', markersize=MS,
                              markerfacecolor='none' if hollow else color,
                              markeredgecolor=color if hollow else 'white',
                              markeredgewidth=MEW, alpha=0.35 if hollow else 1.0)
        row1 = [(circ(COLORS['llama']), 'Llama-Centaur'),
                (circ(COLORS['qwen']), 'Qwentaur'),
                (circ(COLORS['llama'], hollow=True), 'Llama-3.1/3.2 (base)'),
                (circ(COLORS['qwen'], hollow=True), 'Qwen3 (base)')] + ref_handles()
        legend_row(fig, row1, -0.013, columnspacing=1.2)
        rank_legend_row(fig, circle_rank_row(fams), -0.048)
        save_figure(fig, outdir, 'scaling_r8_and_rank_sweep_ood')
        plt.close()


def make_indist_and_ood_figure(ood_data, ood_refs, outdir, suffix=''):
    """Rank sweep in distribution (Psych-101, left) next to out of distribution
    (Psych-201, right). Separate y-axes: the two benchmarks are on different
    NLL scales. Only the Psych-101 panel has a cognitive baseline and a
    reported Centaur-70B."""
    in_data = {k: vr.load_family(k) for k in PLOT_FAMILIES}
    _, in_refs = vr.load_aggr()
    with plt.style.context(['nature']):
        vr.apply_style(FS, FS_LEGEND)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.9))

        ax1.set_title(f'Psych-101 ({SWEEP_TITLE})', fontsize=11, fontweight='bold')
        vr.draw_sweep_content(ax1, in_data, in_refs, FS, exclude_sizes=DROP,
                              gen_aware=True, extrapolate=False)

        ax2.set_title(f'Psych-201 ({SWEEP_TITLE})', fontsize=11, fontweight='bold')
        vr.draw_sweep_content(ax2, ood_data, ood_refs, FS, exclude_sizes=DROP,
                              gen_aware=True, extrapolate=False, cog=False,
                              ylabel=False)

        _, rank_row = vr.sweep_legend_rows()
        legend_row(fig, vr.family_legend_handles(PLOT_FAMILIES), -0.045)
        legend_row(fig, vr.shape_key_handles()
                   + ref_handles(reported=True, cog=True), -0.135)
        rank_legend_row(fig, rank_row, -0.190)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_sweep_and_ood', suffix)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Psych-201 OOD LoRA-rank figures.')
    parser.add_argument('--outdir', default=HERE)
    parser.add_argument('--cognitive-matched', action='store_true',
                        help='Only re-draw scaling_rank_sweep_and_ood, with the '
                             'Psych-101 (left) panel restricted to the 38 tasks '
                             'that have a reported cognitive model; writes '
                             'scaling_rank_sweep_and_ood_cognitive_matched. The '
                             'Psych-201 panel is unchanged.')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data, base, refs = load_ood()

    if args.cognitive_matched:
        vr.NAME_SUFFIX = '_cognitive_matched'    # -> matched psych101 summaries
        vr.KEEP = vr.cognitive_matched_tasks()   # -> aggr 70B over 38 tasks
        print(f'Cognitive-matched: left panel over {len(vr.KEEP)} tasks')
        make_indist_and_ood_figure(data, refs, args.outdir,
                                   suffix='_cognitive_matched')
        return

    write_figure_data(data, args.outdir)
    make_rank_figure(data, refs, args.outdir)
    make_rank_scaling_grid(data, refs, args.outdir)
    make_composite_figure(data, refs, args.outdir)
    make_r8_and_rank_sweep(data, base, refs, args.outdir)
    make_indist_and_ood_figure(data, refs, args.outdir)


if __name__ == '__main__':
    main()
