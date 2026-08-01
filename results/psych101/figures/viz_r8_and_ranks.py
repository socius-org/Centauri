#!/usr/bin/env python3
"""
LoRA-rank + r=8 scaling figures (Psych-101), all four families.

Reads the per-family <fam>_ablation_summary.csv files (written by
generate_ablation_plots.py, in this same directory) and the main aggregate
psych101_aggr.csv (one level up). Produces:

  scaling_r8                    r=8 scaling panel (rank matched to Centaur-70B)
  scaling_rank_sweep            full rank sweep (r=4..64), per-family bands
  scaling_rank_panels           1x5 grid, one scaling panel per rank
  scaling_rank_sweep_with_panels  main sweep + five per-rank minis (vertical)
  scaling_r8_and_rank_sweep     Llama+Qwen: r=8 panel | rank sweep + minis row

Run from anywhere: `python viz_r8_and_ranks.py`.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import scienceplots  # noqa: F401

# palette (formerly test_viz/_viz_common.COLORS)
COLORS = {
    'llama': '#0082fb', 'llama_dk': '#005bb5', 'llama_4b': '#66b3fd',
    'qwen': '#7F6DEF', 'qwen_4b': '#b3a8f5', 'reported': '#005bb5',
    'gray': '#888888',
}

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_DIR = HERE
AGGR_CSV = os.path.join(HERE, '..', 'psych101_aggr.csv')

COG_MODEL = 0.6851

# Appended to every output basename; set to '_cognitive_matched' by
# --cognitive-matched. KEEP, when set, restricts aggr means (base models + 70B)
# to the cognitive-baseline task set — the same subset the matched summary CSVs
# were built over.
NAME_SUFFIX = ''
KEEP = None


def cognitive_matched_tasks():
    """The 38 Experiment ids with a reported domain-specific cognitive model."""
    df = pd.read_csv(AGGR_CSV)
    df = df[df['Experiment'] != 'Mean']
    cog = pd.to_numeric(df['Cognitive model (reported)'].astype(str)
                        .str.replace('†', ''), errors='coerce')
    return set(df.loc[cog.notna(), 'Experiment'])


FAMILIES = {
    'llama': dict(label='Llama-Centaur', color=COLORS['llama'], marker='o',
                  csv='llama_ablation_summary.csv',
                  params={'1B': 1.0, '3B': 3.0, '8B': 8.0}),
    'qwen': dict(label='Qwentaur', color=COLORS['qwen'], marker='s',
                 csv='qwen_ablation_summary.csv',
                 params={'0.6B': 0.6, '1.7B': 1.7, '4B': 4.0, '8B': 8.0,
                         '14B': 14.0}),
    # HF brand yellow; ramp runs brand-yellow -> dark amber (yellow has no
    # usable white-tint range). Fine-tuned only in the r8 figure.
    'smollm': dict(label='Smoltaur', color='#FFD21E', dark='#B8860B',
                   marker='o', csv='smollm_ablation_summary.csv',
                   params={'0.1B': 0.1, '0.4B': 0.4, '1.7B': 1.7, '3B': 3.0}),
    # Ai2 brand pink.
    'olmo': dict(label='Olmotaur', color='#F0529C', marker='s',
                 csv='olmo_ablation_summary.csv',
                 params={'1B': 1.0, '7B': 7.0}),
}
PLOT_FAMILIES = ['llama', 'qwen', 'smollm', 'olmo']
RANKS = [4, 8, 16, 32, 64]

LLAMA_BASE = [(1, 'Llama-3.2-1B (base-bf16)'), (3, 'Llama-3.2-3B (base-bf16)'),
              (8, 'Llama-3.1-8B (base-bf16)')]
QWEN_BASE = [(0.6, 'Qwen3-0.6B (base-bf16)'), (1.7, 'Qwen3-1.7B (base-bf16)'),
             (4, 'Qwen3-4B (base-bf16)'), (8, 'Qwen3-8B (base-bf16)'),
             (14, 'Qwen3-14B (base-bf16)')]

MS, MEW = 5.2, 0.7
MS_70B = 5.8
MS_70B_HOLLOW = 4.9
FIGSIZE = (7, 5.4)
FS = 10
FS_LEGEND = 8
FS_SIZE_LABEL = 6.2
FS_70B_LABEL = 7


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION / CONTEXT-AWARE PLOTTING (Smoltaur & Olmotaur are not internally
# comparable: their largest model is a newer generation with a much larger
# context window, so a fitted trend across sizes conflates parameter count with
# generation and context). Used only where `gen_aware=True` is passed.
# ═══════════════════════════════════════════════════════════════════════════

# Run context window (thousands of tokens) per (family, size). Governs marker
# shape (circle >= 32k, square < 32k) and the legend annotation -- what the
# reader needs is whether the model saw the whole Psych-101 session.
CTX_K = {
    ('llama', '1B'): 32, ('llama', '3B'): 32, ('llama', '8B'): 32,
    ('qwen', '0.6B'): 32, ('qwen', '1.7B'): 32, ('qwen', '4B'): 32,
    ('qwen', '8B'): 32, ('qwen', '14B'): 32,
    ('smollm', '0.1B'): 8, ('smollm', '0.4B'): 8, ('smollm', '1.7B'): 8,
    ('smollm', '3B'): 32,
    ('olmo', '1B'): 4, ('olmo', '7B'): 32,
}


def ctx_marker(fam_key, size):
    """Circle if the model ran at >= 32k context, else square (truncated)."""
    return 'o' if CTX_K.get((fam_key, size), 32) >= 32 else 's'


# Sizes that may be joined by a fitted trend / envelope, per family. A family
# absent here is fit across all its sizes. `extrap` (optional) is a size the fit
# is extended to as a faint dotted line (no band) to expose a generation/context
# jump. olmo -> None: no fit at all (its two models differ in both).
FIT_GROUP = {
    'smollm': {'sizes': ['0.1B', '0.4B', '1.7B'], 'extrap': '3B'},
    'olmo': None,
}


def fit_group(fam_key, present_sizes):
    """(sizes_to_fit or None, extrap_size or None). None sizes => draw no fit."""
    if fam_key in FIT_GROUP:
        spec = FIT_GROUP[fam_key]
        if spec is None:
            return None, None
        sizes = [s for s in spec['sizes'] if s in list(present_sizes)]
        return (sizes if len(sizes) >= 2 else None), spec.get('extrap')
    return list(present_sizes), None


# Legend: (display name, base-model label, context-k, marker) per family. smollm
# and olmo split into two entries (same colour, different shape) because their
# sizes straddle the context boundary.
FAM_LEGEND = {
    'llama': [('Llama-Centaur', 'Llama-3.1/3.2', 32, 'o')],
    'qwen': [('Qwentaur', 'Qwen3-Base', 32, 'o')],
    'smollm': [('Smoltaur', 'SmolLM2', 8, 's'),
               ('Smoltaur', 'SmolLM3-Base', 32, 'o')],
    'olmo': [('Olmotaur', 'OLMo-2-0425-1B', 4, 's'),
             ('Olmotaur', 'OLMo-3-1025-7B', 32, 'o')],
}


def family_legend_handles(fam_keys, msize=MS):
    """Two-line (family / base-model; context) legend entries, split by context."""
    out = []
    for k in fam_keys:
        for name, basem, ctx, marker in FAM_LEGEND[k]:
            h = plt.Line2D([], [], marker=marker, linestyle='none', markersize=msize,
                           markerfacecolor=FAMILIES[k]['color'],
                           markeredgecolor='white', markeredgewidth=MEW)
            out.append((h, f'{name}\n({basem}; {ctx}k)'))
    return out


def shape_key_handles(msize=MS):
    """The context/shape key: circle >= 32k, square < 32k (truncated)."""
    def m(shape):
        return plt.Line2D([], [], marker=shape, linestyle='none', markersize=msize,
                          markerfacecolor='0.45', markeredgecolor='white',
                          markeredgewidth=MEW)
    return [(m('o'), r'$\geq$32k context'), (m('s'), '<32k (truncated)')]


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_family(fam_key):
    fam = FAMILIES[fam_key]
    csv = fam['csv'].replace('.csv', f'{NAME_SUFFIX}.csv')
    df = pd.read_csv(os.path.join(SUMMARY_DIR, csv))
    df['params'] = df['size'].map(fam['params'])
    return df.sort_values('params')


def series_by_rank(df, rank):
    """Rows for one LoRA rank across sizes (rank 16 = the baseline cells)."""
    if rank == 16:
        rows = df[df['axis'] == 'baseline']
    else:
        rows = df[(df['axis'] == 'rank_sweep') & (df['rank'] == rank)]
    return rows.sort_values('params')


def get_mean(df, col):
    vals = pd.to_numeric(df[col].astype(str).str.replace('†', ''), errors='coerce')
    return vals.mean()


def load_aggr():
    df = pd.read_csv(AGGR_CSV)
    if KEEP is not None:
        df = df[df['Experiment'].isin(KEEP)]
    base = {
        'llama': ([s for s, _ in LLAMA_BASE], [get_mean(df, c) for _, c in LLAMA_BASE]),
        'qwen': ([s for s, _ in QWEN_BASE], [get_mean(df, c) for _, c in QWEN_BASE]),
    }
    refs = {'repl': get_mean(df, 'Centaur-70B (4bit-reproduced)'),
            'binz': get_mean(df, 'Centaur-70B (4bit-reported)'),
            'llama70b': get_mean(df, 'Llama-70B (4bit-reported)')}
    return base, refs


# ═══════════════════════════════════════════════════════════════════════════
# STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def apply_style(fs=FS, fl=FS_LEGEND):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'font.size': fs, 'axes.labelsize': fs,
        'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1,
        'legend.fontsize': fl, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
    })


def _rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def shade(hex_color, t):
    """Blend a hex color toward white by t in [0, 1] (0 = full color)."""
    rgb = _rgb(hex_color)
    return tuple(rgb + (1.0 - rgb) * t)


RANK_TS = np.linspace(0.72, 0.0, len(RANKS))  # r=4 lightest -> r=64 darkest


def fam_shade(fam, t):
    """Rank-ramp colour at ramp position t. Families with a 'dark' anchor
    (HF yellow) run brand->dark; others tint toward white."""
    if 'dark' in fam:
        u = 1.0 - t / RANK_TS[0]
        c0, c1 = _rgb(fam['color']), _rgb(fam['dark'])
        return tuple(c0 + (c1 - c0) * u)
    return shade(fam['color'], t)


def fit_line(x, y):
    if len(x) < 2:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    slope, intercept = np.polyfit(np.log10(x), y, 1)
    return slope, intercept


def draw_bracket(ax, y_top, y_bot, fs):
    bx = 52
    ax.plot([bx, bx], [y_top, y_bot], color='gray', lw=0.7, clip_on=False)
    ax.plot([bx, bx * 1.04], [y_top, y_top], color='gray', lw=0.7, clip_on=False)
    ax.plot([bx, bx * 1.04], [y_bot, y_bot], color='gray', lw=0.7, clip_on=False)
    ax.text(bx / 1.08, (y_top + y_bot) / 2, 'load_in_4bit',
            fontsize=fs * 0.55, color='gray', ha='right', va='center',
            rotation=90, fontstyle='italic')


def draw_70b(ax, refs, annotate=True, scale=1.0):
    """70B diamonds (+ labels), always 4-bit, sized like the data markers."""
    ax.plot([70], [refs['repl']], 'D', color=COLORS['llama_dk'],
            markersize=MS_70B * scale, markerfacecolor=COLORS['llama_dk'],
            markeredgecolor='white', markeredgewidth=MEW, zorder=6, clip_on=False)
    ax.plot([70], [refs['binz']], 'D', color=COLORS['reported'],
            markersize=MS_70B_HOLLOW * scale, markerfacecolor='none',
            markeredgecolor=COLORS['reported'], markeredgewidth=MEW,
            zorder=6, clip_on=False)
    if not annotate:
        return
    ax.annotate('reproduced', (70, refs['repl']), xytext=(4, 3),
                textcoords='offset points', fontsize=FS_70B_LABEL, ha='left',
                va='bottom', color=COLORS['llama_dk'])
    ax.annotate('reported', (70, refs['binz']), xytext=(4, -3),
                textcoords='offset points', fontsize=FS_70B_LABEL, ha='left',
                va='top', color=COLORS['reported'])


def param_axis(ax, families=None):
    ax.set_xscale('log')
    smallest = min(p for k in (families or PLOT_FAMILIES)
                   for p in FAMILIES[k]['params'].values())
    if smallest < 0.4:
        ax.set_xlim(0.07, 120)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'])
    else:
        ax.set_xlim(0.4, 120)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$'])
    ax.set_xlabel('Parameters (billions)')


def fam_legend_marker(fam_key, mfc=None, alpha=1.0, msize=MS):
    fam = FAMILIES[fam_key]
    return plt.Line2D([], [], marker=fam['marker'], linestyle='none',
                      markersize=msize,
                      markerfacecolor=fam['color'] if mfc is None else mfc,
                      markeredgecolor='white' if mfc is None else fam['color'],
                      markeredgewidth=MEW, alpha=alpha)


def save_figure(fig, outdir, basename, dpi=600):
    basename = f'{basename}{NAME_SUFFIX}'
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'  Saved {basename}')


# ═══════════════════════════════════════════════════════════════════════════
# r = 8 SCALING
# ═══════════════════════════════════════════════════════════════════════════

def draw_r8_axes(ax, data, base, refs, fs, families=None, ylabel=True,
                 gen_aware=False):
    """r=8 scaling content: markers + fits + size labels + bases + 70B + bracket.
    `gen_aware` shapes markers by context window (circle >=32k, square <32k) and
    fits only generation-matched size groups (only affects smollm/olmo; llama and
    qwen fit across all sizes as before, but their markers become circles)."""
    fams = families or PLOT_FAMILIES
    ax.axhline(COG_MODEL, color='gray', linewidth=0.8, linestyle=':',
               alpha=0.7, zorder=1)

    # Llama and Qwen share an 8B model with near-identical r=8 NLLs; both sit at
    # x=8, blue circle drawn on top of the purple square (corners peek out).
    zorders = {'llama': 5.1}
    for fam_key in [k for k in ('smollm', 'olmo', 'qwen', 'llama') if k in fams]:
        fam = FAMILIES[fam_key]
        r8 = series_by_rank(data[fam_key], 8)
        accent = fam.get('dark', fam['color'])
        if gen_aware:
            for x, y, s in zip(r8['params'], r8['mean_nll'], r8['size']):
                ax.plot([x], [y], ctx_marker(fam_key, s), color=fam['color'],
                        markersize=MS, markerfacecolor=fam['color'],
                        markeredgecolor='white', markeredgewidth=MEW,
                        zorder=zorders.get(fam_key, 5))
            grp, extrap = fit_group(fam_key, r8.sort_values('params')['size'])
        else:
            ax.plot(r8['params'], r8['mean_nll'], fam['marker'], color=fam['color'],
                    markersize=MS, markerfacecolor=fam['color'],
                    markeredgecolor='white', markeredgewidth=MEW,
                    zorder=zorders.get(fam_key, 5))
            grp, extrap = list(r8['size']), None
        if grp is not None and len(grp) >= 2:
            g = r8[r8['size'].isin(grp)]
            slope, intercept = fit_line(g['params'].values, g['mean_nll'].values)
            x1 = g['params'].max() * (1.0 if extrap else 1.6)
            x_fit = np.linspace(g['params'].min() * 0.75, x1, 100)
            ax.plot(x_fit, slope * np.log10(x_fit) + intercept, '--',
                    color=accent, lw=0.7, alpha=0.5, zorder=2)
            if extrap and extrap in fam['params']:
                xx = np.linspace(g['params'].max(), fam['params'][extrap], 40)
                ax.plot(xx, slope * np.log10(xx) + intercept, ':',
                        color=accent, lw=0.7, alpha=0.4, zorder=2)
        for x, y, s in zip(r8['params'], r8['mean_nll'], r8['size']):
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
        fam = FAMILIES[fam_key]
        bs, bl = base[fam_key]
        bmk = 'o' if gen_aware else fam['marker']
        ax.plot(bs, bl, bmk, color=fam['color'], markersize=MS,
                markerfacecolor='none', markeredgecolor=fam['color'],
                markeredgewidth=MEW, alpha=0.35, zorder=3)

    draw_70b(ax, refs)
    ax.plot([70], [refs['llama70b']], 'o', color=COLORS['llama'], markersize=MS,
            markerfacecolor='none', markeredgecolor=COLORS['llama'],
            markeredgewidth=MEW, alpha=0.35, zorder=3, clip_on=False)
    draw_bracket(ax, refs['llama70b'], refs['binz'], fs)

    param_axis(ax, families)
    if ylabel:
        ax.set_ylabel('Mean negative log-likelihood')


def make_r8_figure(data, base, refs, outdir):
    with plt.style.context(['nature']):
        apply_style(FS, FS_LEGEND)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title('LoRA (r = 8, bf16)', fontsize=12, fontweight='bold')
        draw_r8_axes(ax, data, base, refs, FS)

        row1 = [(fam_legend_marker(k), FAMILIES[k]['label']) for k in PLOT_FAMILIES]
        row2 = [
            (fam_legend_marker('llama', mfc='none', alpha=0.35), 'Llama-3.1/3.2 (base)'),
            (fam_legend_marker('qwen', mfc='none', alpha=0.35), 'Qwen3 (base)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B,
                        markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                        markeredgewidth=MEW), 'Centaur-70B (reproduced)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B_HOLLOW,
                        markerfacecolor='none', markeredgecolor=COLORS['reported'],
                        markeredgewidth=MEW), 'Centaur-70B (reported)'),
            (plt.Line2D([], [], color='gray', lw=0.8, linestyle=':', alpha=0.7),
             'Cognitive baseline'),
        ]
        fig.legend([h for h, _ in row1], [l for _, l in row1], loc='lower center',
                   bbox_to_anchor=(0.5, -0.020), ncol=len(row1), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        fig.legend([h for h, _ in row2], [l for _, l in row2], loc='lower center',
                   bbox_to_anchor=(0.5, -0.062), ncol=len(row2), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_r8')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FULL RANK SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def draw_sweep_content(ax, data, refs, fs, families=None, ylabel=True,
                       exclude_sizes=None, cog_ax=None, gen_aware=False,
                       extrapolate=True):
    """Per-family envelope bands (straight fitted boundaries) + shade-ramp
    markers + cognitive line + 70B refs + bracket.

    `cog_ax`, if given, receives the cognitive line instead of `ax` — the thin
    upper segment of a broken y-axis, so `ax` can autoscale tight on the data.

    `gen_aware`: restrict each envelope to a generation/context-matched group of
    sizes (Smoltaur -> SmolLM2 only, with a faint dotted extrapolation of the
    band boundaries to SmolLM3-3B; Olmotaur -> no envelope) and shape markers by
    context window (circle >=32k, square <32k).
    """
    exclude_sizes = exclude_sizes or {}
    (cog_ax or ax).axhline(COG_MODEL, color='gray', linewidth=0.8, linestyle=':',
                           alpha=0.7, zorder=1)

    for fam_key in (families or PLOT_FAMILIES):
        fam, df = FAMILIES[fam_key], data[fam_key]
        drop = exclude_sizes.get(fam_key, set())

        worst = series_by_rank(df, 4)
        bests = series_by_rank(df, 64)

        # ---- envelope band, fitted only within a comparable size group ----
        if gen_aware:
            grp, extrap = fit_group(fam_key, worst.sort_values('params')['size'])
        else:
            grp, extrap = list(worst['size']), None
        if grp is not None:
            w, b = worst[worst['size'].isin(grp)], bests[bests['size'].isin(grp)]
            pf_hi = np.polyfit(np.log10(w['params']), w['mean_nll'], 1)
            pf_lo = np.polyfit(np.log10(b['params']), b['mean_nll'], 1)
            x0 = w['params'].min() * 0.85
            x1 = w['params'].max() * (1.0 if extrap else 1.2)
            x_fit = np.linspace(x0, x1, 100)
            y_hi, y_lo = np.polyval(pf_hi, np.log10(x_fit)), np.polyval(pf_lo, np.log10(x_fit))
            ax.fill_between(x_fit, y_lo, y_hi, color=fam['color'], alpha=0.15,
                            lw=0, zorder=1.5)
            for y_b in (y_hi, y_lo):
                ax.plot(x_fit, y_b, '-', color=fam.get('dark', fam['color']),
                        lw=0.6, alpha=0.45, zorder=1.6)
            # faint dotted extrapolation of both boundaries into the excluded
            # (newer-generation) size — the gap to its markers is the evidence.
            if extrapolate and extrap and extrap in fam['params']:
                xx = np.linspace(w['params'].max(), fam['params'][extrap], 40)
                for pf in (pf_hi, pf_lo):
                    ax.plot(xx, np.polyval(pf, np.log10(xx)), ':',
                            color=fam.get('dark', fam['color']), lw=0.7,
                            alpha=0.4, zorder=1.6)

        # ---- markers: shaded by rank, shaped by context when gen_aware ----
        for rank, t in zip(RANKS, RANK_TS):
            rows = series_by_rank(df, rank)
            if drop:
                rows = rows[~rows['size'].isin(drop)]
            if rows.empty:
                continue
            shade = fam_shade(fam, t)
            if gen_aware:
                for _, r in rows.iterrows():
                    ax.plot([r['params']], [r['mean_nll']],
                            ctx_marker(fam_key, r['size']), linestyle='none',
                            markersize=MS, markerfacecolor=shade,
                            markeredgecolor='white', markeredgewidth=MEW, zorder=4)
            else:
                ax.plot(rows['params'], rows['mean_nll'], fam['marker'],
                        linestyle='none', markersize=MS, markerfacecolor=shade,
                        markeredgecolor='white', markeredgewidth=MEW, zorder=4)

    draw_70b(ax, refs)
    draw_bracket(ax, refs['repl'], refs['binz'], fs)
    param_axis(ax, families)
    if ylabel:
        ax.set_ylabel('Mean negative log-likelihood')


def sweep_legend_rows(families=None, label_70b_full=False):
    """(row1, row2): families + 70B refs / rank shade ramp."""
    fams = families or PLOT_FAMILIES
    pfx = 'Centaur-70B' if label_70b_full else '70B'
    row1 = [(fam_legend_marker(k), FAMILIES[k]['label']) for k in fams]
    row1 += [
        (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B,
                    markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                    markeredgewidth=MEW), f'{pfx} (reproduced)'),
        (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B_HOLLOW,
                    markerfacecolor='none', markeredgecolor=COLORS['reported'],
                    markeredgewidth=MEW), f'{pfx} (reported)'),
    ]
    row2 = [(tuple(plt.Line2D([], [], marker=FAMILIES[k]['marker'], linestyle='none',
                              markersize=MS, markerfacecolor=fam_shade(FAMILIES[k], t),
                              markeredgecolor='white', markeredgewidth=MEW)
                   for k in fams), f'r = {rank}')
            for rank, t in zip(RANKS, RANK_TS)]
    return row1, row2


def add_sweep_legends(fig, row1, row2, y1, y2):
    fig.legend([h for h, _ in row1], [l for _, l in row1], loc='lower center',
               bbox_to_anchor=(0.5, y1), ncol=len(row1), fontsize=FS_LEGEND,
               frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    fig.legend([h for h, _ in row2], [l for _, l in row2], loc='lower center',
               bbox_to_anchor=(0.5, y2), ncol=5, fontsize=FS_LEGEND, frameon=False,
               handlelength=0.7 * len(row2[0][0]) + 0.2, handletextpad=0.6,
               columnspacing=1.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})


def make_rank_figure(data, refs, outdir):
    with plt.style.context(['nature']):
        apply_style(FS, FS_LEGEND)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title('LoRA rank sweep (r = 4$-$64, bf16)', fontsize=12,
                     fontweight='bold')
        draw_sweep_content(ax, data, refs, FS)
        row1, row2 = sweep_legend_rows(label_70b_full=True)
        add_sweep_legends(fig, row1, row2, -0.020, -0.062)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_sweep')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PER-RANK SCALING PANELS
# ═══════════════════════════════════════════════════════════════════════════

def draw_rank_series(ax, data, refs, rank, msize=None, lw=0.7, families=None,
                     exclude_sizes=None, cog_ax=None, gen_aware=False,
                     extrapolate=True):
    """One rank's scaling content: family series + dashed fits + cognitive
    line + 70B diamonds (no annotations). `cog_ax` routes the cognitive line to
    a broken-axis upper segment. `gen_aware` fits only generation/context-matched
    size groups (Smoltaur SmolLM2, dotted extrapolation to SmolLM3-3B; Olmotaur
    none) and shapes markers by context window."""
    exclude_sizes = exclude_sizes or {}
    msize = MS * 0.85 if msize is None else msize
    (cog_ax or ax).axhline(COG_MODEL, color='gray', linewidth=0.8, linestyle=':',
                           alpha=0.7, zorder=1)
    for fam_key in (families or PLOT_FAMILIES):
        fam = FAMILIES[fam_key]
        rows = series_by_rank(data[fam_key], rank)
        drop = exclude_sizes.get(fam_key, set())
        if drop:
            rows = rows[~rows['size'].isin(drop)]
        if rows.empty:
            continue
        if gen_aware:
            for _, r in rows.iterrows():
                ax.plot([r['params']], [r['mean_nll']], ctx_marker(fam_key, r['size']),
                        linestyle='none', markersize=msize, markerfacecolor=fam['color'],
                        markeredgecolor='white', markeredgewidth=MEW * 0.8, zorder=4)
            grp, extrap = fit_group(fam_key, rows.sort_values('params')['size'])
        else:
            ax.plot(rows['params'], rows['mean_nll'], fam['marker'], linestyle='none',
                    markersize=msize, markerfacecolor=fam['color'],
                    markeredgecolor='white', markeredgewidth=MEW * 0.8, zorder=4)
            grp, extrap = list(rows['size']), None
        if grp is not None and len(grp) >= 2:
            g = rows[rows['size'].isin(grp)]
            slope, intercept = fit_line(g['params'].values, g['mean_nll'].values)
            x1 = g['params'].max() * (1.0 if extrap else 1.6)
            x_fit = np.linspace(g['params'].min() * 0.75, x1, 100)
            ax.plot(x_fit, slope * np.log10(x_fit) + intercept, '--',
                    color=fam.get('dark', fam['color']), lw=lw, alpha=0.5, zorder=2)
            if extrapolate and extrap and extrap in fam['params']:
                xx = np.linspace(g['params'].max(), fam['params'][extrap], 40)
                ax.plot(xx, slope * np.log10(xx) + intercept, ':',
                        color=fam.get('dark', fam['color']), lw=lw, alpha=0.4, zorder=2)
    draw_70b(ax, refs, annotate=False, scale=msize / MS)


def make_rank_scaling_grid(data, refs, outdir):
    with plt.style.context(['nature']):
        apply_style(FS, FS_LEGEND)
        fig, axes = plt.subplots(1, len(RANKS), figsize=(13.5, 3.1), sharey=True)
        for ax, rank in zip(axes, RANKS):
            ax.set_title(f'r = {rank}', fontsize=10, fontweight='bold')
            draw_rank_series(ax, data, refs, rank, gen_aware=True, extrapolate=False)
            param_axis(ax)
            if ax is axes[0]:
                ax.set_ylabel('Mean negative log-likelihood')

        # Generation/context-aware legend: family (base model; context) split by
        # shape, a shape key, then the 70B/cognitive refs.
        fam_h = family_legend_handles(PLOT_FAMILIES)
        ref_h = shape_key_handles() + [
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B,
                        markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                        markeredgewidth=MEW), 'Centaur-70B (reproduced)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B_HOLLOW,
                        markerfacecolor='none', markeredgecolor=COLORS['reported'],
                        markeredgewidth=MEW), 'Centaur-70B (reported)'),
            (plt.Line2D([], [], color='gray', lw=0.8, linestyle=':', alpha=0.7),
             'Cognitive baseline'),
        ]
        fig.legend([h for h, _ in fam_h], [l for _, l in fam_h], loc='lower center',
                   bbox_to_anchor=(0.5, -0.12), ncol=len(fam_h), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        fig.legend([h for h, _ in ref_h], [l for _, l in ref_h], loc='lower center',
                   bbox_to_anchor=(0.5, -0.28), ncol=len(ref_h), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_panels')
        plt.close()


def make_composite_figure(data, refs, outdir):
    """Rank sweep (main, left ~75%) with five per-rank minis stacked vertically
    on the right, r=4 top -> r=64 bottom."""
    # Qwentaur-1.7B has only an r=16 baseline (no rank sweep); drop from the
    # sweep panel and the r=16 mini.
    drop = {'qwen': {'1.7B'}}
    with plt.style.context(['nature']):
        apply_style(FS, FS_LEGEND)
        fig = plt.figure(figsize=(10.2, 7.0))
        gs = fig.add_gridspec(len(RANKS), 2, width_ratios=[3.2, 1.0],
                              hspace=0.14, wspace=0.16)

        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.set_title('LoRA rank sweep (r = 4$-$64, bf16)', fontsize=12,
                          fontweight='bold')
        draw_sweep_content(ax_main, data, refs, FS, exclude_sizes=drop,
                           gen_aware=True, extrapolate=False)

        minis = []
        for i, rank in enumerate(RANKS):
            ax = fig.add_subplot(gs[i, 1], sharex=minis[0] if minis else None,
                                 sharey=minis[0] if minis else None)
            minis.append(ax)
            draw_rank_series(ax, data, refs, rank, msize=MS * 0.62, lw=0.55,
                             exclude_sizes=drop, gen_aware=True, extrapolate=False)
            ax.text(0.955, 0.86, f'r = {rank}', transform=ax.transAxes,
                    fontsize=7.5, fontweight='bold', ha='right', va='top')
            ax.set_xscale('log')
            ax.set_xlim(0.07, 120)
            if not NAME_SUFFIX:            # fixed grid tuned to the 46-task range;
                ax.set_ylim(0.595, 1.05)   # let the matched subset autoscale
                ax.set_yticks([0.7, 0.9])
            ax.tick_params(labelsize=6, length=2)
            ax.minorticks_off()
            if i == len(RANKS) - 1:
                ax.set_xticks([0.1, 1, 10, 100])
                ax.set_xticklabels(['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$'], fontsize=6)
                ax.set_xlabel('Parameters (billions)', fontsize=7.5)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

        # Generation/context-aware legend: family (base model; context) split by
        # shape, a shape key, the 70B/cognitive refs, then the rank shade ramp.
        fam_h = family_legend_handles(PLOT_FAMILIES)
        ref_h = shape_key_handles() + [
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B,
                        markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                        markeredgewidth=MEW), 'Centaur-70B (reproduced)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B_HOLLOW,
                        markerfacecolor='none', markeredgecolor=COLORS['reported'],
                        markeredgewidth=MEW), 'Centaur-70B (reported)'),
            (plt.Line2D([], [], color='gray', lw=0.8, linestyle=':', alpha=0.7),
             'Cognitive baseline'),
        ]
        _, rank_row = sweep_legend_rows()
        fig.legend([h for h, _ in fam_h], [l for _, l in fam_h], loc='lower center',
                   bbox_to_anchor=(0.5, -0.035), ncol=len(fam_h), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        fig.legend([h for h, _ in ref_h], [l for _, l in ref_h], loc='lower center',
                   bbox_to_anchor=(0.5, -0.105), ncol=len(ref_h), fontsize=FS_LEGEND,
                   frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
        fig.legend([h for h, _ in rank_row], [l for _, l in rank_row], loc='lower center',
                   bbox_to_anchor=(0.5, -0.150), ncol=5, fontsize=FS_LEGEND, frameon=False,
                   handlelength=0.7 * len(rank_row[0][0]) + 0.2, handletextpad=0.6,
                   columnspacing=1.5,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})
        plt.tight_layout()
        save_figure(fig, outdir, 'scaling_rank_sweep_with_panels')
        plt.close()


def _break_yaxis(ax_top, ax_bot, cogval, show_tick=True, amp=0.008, ticklabelsize=6):
    """Fuse a stacked (upper, lower) axes pair into one broken y-axis: the upper
    segment shows only the cognitive line in a thin band around `cogval`, the
    lower segment keeps its data-tight autoscale. Hides the facing spines and
    draws a double-wave (~) cut just above the lower axis. `ticklabelsize` should
    match the lower axis' tick font so the broken-off tick reads the same size."""
    ax_top.set_ylim(cogval - 0.004, cogval + 0.004)
    ax_top.set_yticks([cogval])
    ax_top.set_yticklabels([f'{cogval:.3f}'] if show_tick else [])
    ax_top.tick_params(labelsize=ticklabelsize, length=2, bottom=False, labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.minorticks_off()
    # small double-wave (~) break mark on the y-axis only: two short parallel
    # sine squiggles straddling the left spine, sitting in the gap above the
    # lower axis. Drawn in the lower axis' fraction coords.
    xw = np.linspace(-0.030, 0.040, 60)
    t = (xw - xw[0]) / (xw[-1] - xw[0])
    w = amp * np.sin(2 * np.pi * 1.5 * t)
    for y0 in (1.0 + amp * 1.3, 1.0 + amp * 3.7):
        ax_bot.plot(xw, y0 + w, transform=ax_bot.transAxes, color='0.45',
                    lw=0.8, clip_on=False, zorder=30, solid_capstyle='round')


def make_r8_and_rank_sweep(data, base, refs, outdir):
    """Llama+Qwen: r=8 panel | rank sweep, with five per-rank minis in a
    horizontal row along the bottom. In the cognitive-matched variant every
    model point sits far below the cognitive line, so the rank-sweep panel and
    the minis get a broken y-axis (double-wave cut) that parks the cognitive
    line in a thin top band and expands the data region to reveal the slopes."""
    fams = ['llama', 'qwen']
    # Qwentaur-1.7B has only an r=16 baseline + r=8 (no full rank sweep): keep
    # it in the r=8 panel, drop it from the sweep panel and the minis.
    drop = {'qwen': {'1.7B'}}
    broken = bool(NAME_SUFFIX)   # matched variant only
    with plt.style.context(['nature']):
        apply_style(FS, FS_LEGEND)
        fig = plt.figure(figsize=(10.2, 7.6))
        outer = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.22)
        gs_top = outer[0].subgridspec(1, 2, wspace=0.28)
        gs_bot = outer[1].subgridspec(1, len(RANKS), wspace=0.12)

        ax_a = fig.add_subplot(gs_top[0, 0])
        ax_a.set_title('LoRA (r = 8, bf16)', fontsize=11, fontweight='bold')
        draw_r8_axes(ax_a, data, base, refs, FS, families=fams, gen_aware=True)

        # ---- right panel: rank sweep (broken y-axis when matched) ----
        if broken:
            gsb = gs_top[0, 1].subgridspec(2, 1, height_ratios=[1, 9], hspace=0.05)
            ax_b_top = fig.add_subplot(gsb[0])
            ax_b = fig.add_subplot(gsb[1], sharex=ax_b_top)
            ax_b_top.set_title('LoRA rank sweep (r = 4$-$64, bf16)', fontsize=11,
                               fontweight='bold')
            draw_sweep_content(ax_b, data, refs, FS, families=fams,
                               exclude_sizes=drop, cog_ax=ax_b_top, gen_aware=True)
            plt.setp(ax_b_top.get_xticklabels(), visible=False)
            _break_yaxis(ax_b_top, ax_b, COG_MODEL, amp=0.007, ticklabelsize=FS - 1)
        else:
            ax_b = fig.add_subplot(gs_top[0, 1])
            ax_b.set_title('LoRA rank sweep (r = 4$-$64, bf16)', fontsize=11,
                           fontweight='bold')
            draw_sweep_content(ax_b, data, refs, FS, families=fams,
                               exclude_sizes=drop, gen_aware=True)

        # ---- bottom: per-rank minis (broken y-axis when matched) ----
        minis, minis_top = [], []
        for i, rank in enumerate(RANKS):
            if broken:
                gsm = gs_bot[0, i].subgridspec(2, 1, height_ratios=[1, 8],
                                               hspace=0.06)
                axt = fig.add_subplot(gsm[0],
                                      sharey=minis_top[0] if minis_top else None)
                ax = fig.add_subplot(gsm[1], sharex=axt,
                                     sharey=minis[0] if minis else None)
                minis_top.append(axt)
            else:
                ax = fig.add_subplot(gs_bot[0, i],
                                     sharex=minis[0] if minis else None,
                                     sharey=minis[0] if minis else None)
            minis.append(ax)
            draw_rank_series(ax, data, refs, rank, msize=MS * 0.62, lw=0.55,
                             families=fams, exclude_sizes=drop,
                             cog_ax=axt if broken else None, gen_aware=True)
            ax.text(0.94, 0.9 if broken else 0.88, f'r = {rank}',
                    transform=ax.transAxes, fontsize=7.5, fontweight='bold',
                    ha='right', va='top')
            ax.set_xscale('log')
            ax.set_xlim(0.4, 120)
            if not broken:
                ax.set_ylim(0.605, 0.79)
                ax.set_yticks([0.65, 0.75])
            ax.set_xticks([1, 10, 100])
            ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$'], fontsize=6)
            ax.tick_params(labelsize=6, length=2)
            ax.minorticks_off()
            if broken:
                _break_yaxis(axt, ax, COG_MODEL, show_tick=(i == 0), amp=0.009)
            if i == len(RANKS) // 2:
                ax.set_xlabel('Parameters (billions)', fontsize=7.5)
            if i > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        # Llama and Qwen both run at 32k context, so both use circles (family is
        # distinguished by colour); base models are hollow circles too.
        def _circ(color, hollow=False):
            return plt.Line2D([], [], marker='o', linestyle='none', markersize=MS,
                              markerfacecolor='none' if hollow else color,
                              markeredgecolor=color if hollow else 'white',
                              markeredgewidth=MEW, alpha=0.35 if hollow else 1.0)
        row1 = [
            (_circ(COLORS['llama']), 'Llama-Centaur'),
            (_circ(COLORS['qwen']), 'Qwentaur'),
            (_circ(COLORS['llama'], hollow=True), 'Llama-3.1/3.2 (base)'),
            (_circ(COLORS['qwen'], hollow=True), 'Qwen3 (base)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B,
                        markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                        markeredgewidth=MEW), 'Centaur-70B (reproduced)'),
            (plt.Line2D([], [], marker='D', linestyle='none', markersize=MS_70B_HOLLOW,
                        markerfacecolor='none', markeredgecolor=COLORS['reported'],
                        markeredgewidth=MEW), 'Centaur-70B (reported)'),
        ]
        # rank shade ramp, circles (both families are 32k)
        row2 = [(tuple(plt.Line2D([], [], marker='o', linestyle='none', markersize=MS,
                       markerfacecolor=fam_shade(FAMILIES[k], t), markeredgecolor='white',
                       markeredgewidth=MEW) for k in fams), f'r = {rank}')
                for rank, t in zip(RANKS, RANK_TS)]
        add_sweep_legends(fig, row1, row2, -0.013, -0.048)
        save_figure(fig, outdir, 'scaling_r8_and_rank_sweep')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='LoRA rank / r=8 scaling figures.')
    parser.add_argument('--outdir', default=HERE)
    parser.add_argument('--cognitive-matched', action='store_true',
                        help='Read the *_cognitive_matched summary CSVs and '
                             'restrict aggr means to the 38 cognitive-baseline '
                             'tasks; write *_cognitive_matched figures.')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    global NAME_SUFFIX, KEEP
    if args.cognitive_matched:
        NAME_SUFFIX = '_cognitive_matched'
        KEEP = cognitive_matched_tasks()
        print(f'Cognitive-matched: {len(KEEP)} tasks; '
              f'reading *_cognitive_matched summaries')

    data = {k: load_family(k) for k in PLOT_FAMILIES}
    base, refs = load_aggr()
    make_r8_figure(data, base, refs, args.outdir)
    make_rank_figure(data, refs, args.outdir)
    make_rank_scaling_grid(data, refs, args.outdir)
    make_composite_figure(data, refs, args.outdir)
    make_r8_and_rank_sweep(data, base, refs, args.outdir)


if __name__ == '__main__':
    main()
