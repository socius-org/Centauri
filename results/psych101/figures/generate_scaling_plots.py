#!/usr/bin/env python3
"""
Psych-101 Scaling Analysis: Figure Generation Script
=====================================================

Plots generated:
    scaling_finetuned_bf16   - finetuned model scaling at bf16 precision

Usage:
    python generate_scaling_plots.py [--csv PATH] [--outdir PATH]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Color palette
COLORS = {
    'llama': '#0082fb',       # Llama-Centaur - blue
    'llama_dk': '#005bb5',    # Centaur-70B - dark blue
    'llama_4b': '#66b3fd',    # Llama-Centaur 4-bit - light blue
    'qwen': '#7F6DEF',        # Qwentaur - purple
    'qwen_4b': '#b3a8f5',     # Qwentaur 4-bit - light purple
    'reported': '#005bb5',    # 70B reported - same dark blue (hollow marker)
}

# Reference values (from data)
COG_MODEL = 0.6851  # Domain-specific cognitive model baseline (mean over the
                    # 38 Psych-101 tasks for which Binz reports a domain model)

# Appended to every output basename. Set to '_cognitive_matched' by the
# --cognitive-matched flag so the matched-subset figures sit beside the originals.
NAME_SUFFIX = ''

# Model definitions: (name, param_size, bf16_col, 4bit_col)
QWENTAUR = [
    ('Qwentaur-0.6B', 0.6, 'Qwentaur-0.6B (bf16)', 'Qwentaur-0.6B (4bit)'),
    ('Qwentaur-1.7B', 1.7, 'Qwentaur-1.7B (bf16)', 'Qwentaur-1.7B (4bit)'),
    ('Qwentaur-4B',   4,   'Qwentaur-4B (bf16)',   'Qwentaur-4B (4bit)'),
    ('Qwentaur-8B',   8,   'Qwentaur-8B (bf16)',   'Qwentaur-8B (4bit)'),
    ('Qwentaur-14B', 14,   'Qwentaur-14B (bf16)',  'Qwentaur-14B (4bit)'),
]

CENTAUR = [
    ('Centaur-1B', 1, 'Centaur-1B (bf16)', 'Centaur-1B (4bit)'),
    ('Centaur-3B', 3, 'Centaur-3B (bf16)', 'Centaur-3B (4bit)'),
    ('Centaur-8B', 8, 'Centaur-8B (bf16)', 'Centaur-8B (4bit)'),
]

QWEN_BASE = [
    ('Qwen3-0.6B', 0.6, 'Qwen3-0.6B (base-bf16)', 'Qwen3-0.6B (base-4bit)'),
    ('Qwen3-1.7B', 1.7, 'Qwen3-1.7B (base-bf16)', 'Qwen3-1.7B (base-4bit)'),
    ('Qwen3-4B',   4,   'Qwen3-4B (base-bf16)',   'Qwen3-4B (base-4bit)'),
    ('Qwen3-8B',   8,   'Qwen3-8B (base-bf16)',   'Qwen3-8B (base-4bit)'),
    ('Qwen3-14B', 14,   'Qwen3-14B (base-bf16)',  'Qwen3-14B (base-4bit)'),
]

LLAMA_BASE = [
    ('Llama-3.2-1B', 1, 'Llama-3.2-1B (base-bf16)', 'Llama-3.2-1B (base-4bit)'),
    ('Llama-3.2-3B', 3, 'Llama-3.2-3B (base-bf16)', 'Llama-3.2-3B (base-4bit)'),
    ('Llama-3.1-8B', 8, 'Llama-3.1-8B (base-bf16)', 'Llama-3.1-8B (base-4bit)'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path):
    """Load and preprocess the Psych-101 evaluation data."""
    df = pd.read_csv(csv_path)
    return df


def get_mean(df, col):
    """Get mean NLL for a column, handling † markers."""
    if col is None or col not in df.columns:
        return np.nan
    vals = pd.to_numeric(df[col].astype(str).str.replace('†', ''), errors='coerce')
    return vals.mean()


def cognitive_matched_subset(df):
    """Rows for the 38 tasks that have a reported domain-specific cognitive
    model — the exact set the COG_MODEL dotted line is averaged over. Restricting
    every plotted mean to this set makes the model points and the cognitive line
    measure the same tasks (see cognitive_baseline_comparison.py)."""
    # Drop the aggregate 'Mean' row first: it carries a cognitive value but its
    # per-model cells are 46-task means, which would otherwise contaminate the
    # subset average (harmless on the full set, since a set's mean is unchanged
    # by re-adding its own mean, but not on this subset).
    df = df[df['Experiment'] != 'Mean']
    cog = pd.to_numeric(
        df['Cognitive model (reported)'].astype(str).str.replace('†', ''),
        errors='coerce')
    sub = df[cog.notna()].copy()
    print(f"  Cognitive-matched subset: {len(sub)} / {len(df)} tasks")
    return sub


def extract_model_data(df, model_list, precision):
    """Extract sizes, losses, and names for a model family at given precision."""
    sizes, losses, names = [], [], []
    col_idx = 2 if precision == 'bf16' else 3  # Index in model tuple
    
    for name, size, bf16_col, bit4_col in model_list:
        col = bf16_col if precision == 'bf16' else bit4_col
        if col is None:
            continue
        loss = get_mean(df, col)
        if not np.isnan(loss):
            sizes.append(size)
            losses.append(loss)
            names.append(name)
    
    return np.array(sizes), np.array(losses), names


def fit_line(sizes, losses):
    """Fit log-linear trend line: loss = slope * log10(size) + intercept."""
    if len(sizes) < 2:
        return 0, np.mean(losses) if len(losses) > 0 else 0
    log_sizes = np.log10(sizes)
    slope, intercept = np.polyfit(log_sizes, losses, 1)
    return slope, intercept


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_style(fs=8, fl=5.5):
    """Apply consistent styling to plots."""
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


def draw_bracket(ax, y_top, y_bot, fs):
    """Draw load_in_4bit bracket to the left of 70B points."""
    bx = 52
    ax.plot([bx, bx], [y_top, y_bot], color='gray', lw=0.7, clip_on=False)
    ax.plot([bx, bx * 1.04], [y_top, y_top], color='gray', lw=0.7, clip_on=False)
    ax.plot([bx, bx * 1.04], [y_bot, y_bot], color='gray', lw=0.7, clip_on=False)
    y_mid = (y_top + y_bot) / 2
    ax.text(bx / 1.08, y_mid, 'load_in_4bit',
            fontsize=fs * 0.55, color='gray', ha='right', va='center', rotation=90,
            fontstyle='italic')


def save_figure(fig, outdir, basename, dpi=600):
    """Save figure in both PNG and PDF formats."""
    basename = f'{basename}{NAME_SUFFIX}'
    for ext in ['png', 'pdf']:
        path = os.path.join(outdir, f'{basename}.{ext}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 & 2: FINE-TUNED ONLY (bf16 and 4-bit)
# ═══════════════════════════════════════════════════════════════════════════════

def make_finetuned_scaling(df, outdir, precision,
                           ct_s, ct_l, ct_n, qw_s, qw_l, qw_n,
                           repl_loss, binz_loss):
    """Generate fine-tuned-only scaling plot at given precision."""
    fs, fl, ms, mew, ms70, mew70, lw, lo = 8, 5.5, 6, 0.8, 8, 1.0, 1.0, 4
    
    # Include COG_MODEL so the dotted cognitive line stays in frame even when
    # every model point sits below it (as on the cognitive-matched subset).
    all_y = list(ct_l) + list(qw_l) + [repl_loss, binz_loss, COG_MODEL]
    y_min, y_max = min(all_y) - 0.015, max(all_y) + 0.02
    
    with plt.style.context(['nature']):
        apply_style(fs, fl)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        
        # Fine-tuned markers
        ax.plot(ct_s, ct_l, 'o', color=COLORS['llama'], markersize=ms,
                markerfacecolor=COLORS['llama'], markeredgecolor='white',
                markeredgewidth=mew, label='Llama-Centaur', zorder=5)
        ax.plot(qw_s, qw_l, 's', color=COLORS['qwen'], markersize=ms,
                markerfacecolor=COLORS['qwen'], markeredgecolor='white',
                markeredgewidth=mew, label='Qwentaur', zorder=5)
        
        # 70B diamonds
        ax.plot([70], [repl_loss], 'D', color=COLORS['llama_dk'], markersize=ms70,
                markerfacecolor=COLORS['llama_dk'], markeredgecolor='white',
                markeredgewidth=mew70, zorder=6, clip_on=False, label='70B (reproduced)')
        ax.plot([70], [binz_loss], 'D', color=COLORS['reported'], markersize=ms70 - 1.25,
                markerfacecolor='none', markeredgecolor=COLORS['reported'],
                markeredgewidth=mew70, zorder=6, clip_on=False, label='70B (reported)')
        
        # Best-fit lines
        x_fit = np.linspace(0.5, 20, 100)
        ls, li = fit_line(ct_s, ct_l)
        qs, qi = fit_line(qw_s, qw_l)
        ax.plot(x_fit, ls * np.log10(x_fit) + li, '--', color=COLORS['llama'], lw=lw, alpha=0.5, zorder=2)
        ax.plot(x_fit, qs * np.log10(x_fit) + qi, '--', color=COLORS['qwen'], lw=lw, alpha=0.5, zorder=2)
        
        # Size labels
        lkw = dict(textcoords='offset points', ha='center', va='top', fontsize=fs * 0.7)
        for n, s, l in zip(ct_n, ct_s, ct_l):
            ax.annotate(n.replace('Centaur-', ''), (s, l), xytext=(0, -lo), color=COLORS['llama'], **lkw)
        for n, s, l in zip(qw_n, qw_s, qw_l):
            dx = lo * 1.2 if '8B' in n else 0
            ax.annotate(n.replace('Qwentaur-', ''), (s, l), xytext=(dx, -lo), color=COLORS['qwen'], **lkw)
        
        # 70B labels
        ax.annotate('reproduced', (70, repl_loss), xytext=(lo * 1.5, lo * 0.5),
                    textcoords='offset points', fontsize=fs * 0.7, ha='left', va='bottom', color=COLORS['llama_dk'])
        ax.annotate('reported', (70, binz_loss), xytext=(lo * 1.5, -lo * 0.5),
                    textcoords='offset points', fontsize=fs * 0.7, ha='left', va='top', color=COLORS['reported'])
        
        # load_in_4bit bracket — only on bf16 plot
        if precision == 'bf16':
            draw_bracket(ax, repl_loss, binz_loss, fs)
        
        # Cognitive model reference line
        ax.axhline(COG_MODEL, color='gray', linewidth=0.8, linestyle=':', alpha=0.7, zorder=1)
        if precision == 'bf16':
            ax.text(0.50, COG_MODEL + 0.003, 'Domain-specific cognitive models',
                    fontsize=4.5, color='gray', va='bottom', ha='center',
                    transform=ax.get_yaxis_transform())
        else:
            ax.text(0.02, COG_MODEL - 0.004, 'Domain-specific cognitive models',
                    fontsize=4.5, color='gray', va='top', ha='left',
                    transform=ax.get_yaxis_transform())
        
        # Axes formatting
        ax.set_xscale('log')
        ax.set_xlim(0.4, 120)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$'])
        ax.set_xlabel('Parameters (billions)')
        ax.set_ylabel('Mean negative log-likelihood')
        
        # No title for these plots
        ax.legend(loc='upper left' if precision == 'bf16' else 'upper right',
                  borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
        
        plt.tight_layout(pad=0.4)
        suffix = 'bf16' if precision == 'bf16' else '4bit'
        save_figure(fig, outdir, f'scaling_finetuned_{suffix}')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(csv_path, outdir, cognitive_matched=False):
    """Generate all Psych-101 scaling figures."""
    global NAME_SUFFIX
    print("=" * 60)
    print("Psych-101 Scaling Analysis: Figure Generation")
    print("=" * 60)

    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    print(f"\nOutput directory: {outdir}")
    print(f"Input CSV: {csv_path}\n")

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} experiments\n")

    # Optionally restrict every model mean to the cognitive-baseline task set so
    # the plotted points and the dotted cognitive line cover the same tasks.
    if cognitive_matched:
        NAME_SUFFIX = '_cognitive_matched'
        df = cognitive_matched_subset(df)
        print()
    
    # Extract all model data
    print("Extracting model data...")
    
    # Fine-tuned
    ct_bf16_s, ct_bf16_l, ct_bf16_n = extract_model_data(df, CENTAUR, 'bf16')
    ct_4b_s, ct_4b_l, ct_4b_n = extract_model_data(df, CENTAUR, '4bit')
    qw_bf16_s, qw_bf16_l, qw_bf16_n = extract_model_data(df, QWENTAUR, 'bf16')
    qw_4b_s, qw_4b_l, qw_4b_n = extract_model_data(df, QWENTAUR, '4bit')
    
    # Base
    lb_bf16_s, lb_bf16_l, _ = extract_model_data(df, LLAMA_BASE, 'bf16')
    lb_4b_s, lb_4b_l, _ = extract_model_data(df, LLAMA_BASE, '4bit')
    qb_bf16_s, qb_bf16_l, _ = extract_model_data(df, QWEN_BASE, 'bf16')
    qb_4b_s, qb_4b_l, _ = extract_model_data(df, QWEN_BASE, '4bit')
    
    # 70B reference values
    repl_loss = get_mean(df, 'Centaur-70B (4bit-reproduced)')
    binz_loss = get_mean(df, 'Centaur-70B (4bit-reported)')
    llama70b = get_mean(df, 'Llama-70B (4bit-reported)')
    
    print(f"  Centaur-70B reproduced: {repl_loss:.4f}")
    print(f"  Centaur-70B reported:   {binz_loss:.4f}")
    print(f"  Llama-70B base:         {llama70b:.4f}")
    print(f"  Cognitive model:        {COG_MODEL:.4f}\n")
    
    # Generate plots
    print("Generating figures...")
    
    print("\n-- Finetuned scaling (bf16) --")
    make_finetuned_scaling(df, outdir, 'bf16',
                           ct_bf16_s, ct_bf16_l, ct_bf16_n,
                           qw_bf16_s, qw_bf16_l, qw_bf16_n,
                           repl_loss, binz_loss)

    print("\n" + "=" * 60)
    print("Done! All figures saved to:", outdir)
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Psych-101 scaling figures')
    parser.add_argument('--csv', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'psych101_aggr.csv'),
                        help='Path to the CSV file')
    parser.add_argument('--outdir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory for figures')
    parser.add_argument('--cognitive-matched', action='store_true',
                        help='Average every model mean over only the 38 tasks '
                             'with a reported cognitive model, and write '
                             '*_cognitive_matched figures.')
    args = parser.parse_args()

    main(args.csv, args.outdir, cognitive_matched=args.cognitive_matched)
