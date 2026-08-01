#!/usr/bin/env python3
"""
CogSoc Benchmark Analysis — Nature Journal Style

Reads:  lm_eval/cogsoc/*.json
Writes: lm_eval/figures/cogsoc/*.{png,pdf}  +  CSV + MD
"""

import json, csv, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats
import scienceplots  # noqa: F401
import warnings
warnings.filterwarnings('ignore')

plt.style.use(['nature'])

# ============================================================================
# PATHS — relative to script location
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'cogsoc')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures', 'cogsoc')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# COLOUR SCHEME
# ============================================================================
LLAMA = '#0082fb'
LLAMA_DARK = '#005bb5'
QWEN = '#7F6DEF'

def tint(hex_color, amount=0.4):
    """Lighten by blending toward white."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])

def shade(hex_color, amount=0.3):
    """Darken a color."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c * (1 - amount) for c in rgb])


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


def save_figure(fig, outdir, basename, dpi=600):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{basename}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved {basename}")
    plt.close(fig)


# ============================================================================
# FILE MAPPINGS
# ============================================================================
# Expected filenames in cogsoc/:
#   unsloth-Qwen3-0.6B-base_cogsoc.json   socius-Qwentaur-0.6B_cogsoc.json
#   unsloth-Qwen3-1.7B-base_cogsoc.json   socius-Qwentaur-1.7B_cogsoc.json
#   unsloth-Qwen3-4B-base_cogsoc.json     socius-Qwentaur-4B_cogsoc.json
#   unsloth-Qwen3-8B-base_cogsoc.json     socius-Qwentaur-8B_cogsoc.json
#   unsloth-Qwen3-14B-base_cogsoc.json    socius-Qwentaur-14B_cogsoc.json
#   unsloth-Llama-3.2-1B-base_cogsoc.json socius-Llama-Centaur-1B_cogsoc.json
#   unsloth-Llama-3.2-3B-base_cogsoc.json socius-Llama-Centaur-3B_cogsoc.json
#   unsloth-Llama-3.1-8B-base_cogsoc.json socius-Llama-Centaur-8B_cogsoc.json

MODEL_FILE_MAP = {
    "Qwen3-0.6B":    "unsloth-Qwen3-0.6B-base_cogsoc.json",
    "Qwen3-1.7B":    "unsloth-Qwen3-1.7B-base_cogsoc.json",
    "Qwen3-4B":      "unsloth-Qwen3-4B-base_cogsoc.json",
    "Qwen3-8B":      "unsloth-Qwen3-8B-base_cogsoc.json",
    "Qwen3-14B":     "unsloth-Qwen3-14B-base_cogsoc.json",
    "Qwentaur-0.6B": "socius-Qwentaur-0.6B_cogsoc.json",
    "Qwentaur-1.7B": "socius-Qwentaur-1.7B_cogsoc.json",
    "Qwentaur-4B":   "socius-Qwentaur-4B_cogsoc.json",
    "Qwentaur-8B":   "socius-Qwentaur-8B_cogsoc.json",
    "Qwentaur-14B":  "socius-Qwentaur-14B_cogsoc.json",
    "Llama-3.2-1B":  "unsloth-Llama-3.2-1B-base_cogsoc.json",
    "Llama-3.2-3B":  "unsloth-Llama-3.2-3B-base_cogsoc.json",
    "Llama-3.1-8B":  "unsloth-Llama-3.1-8B-base_cogsoc.json",
    "Centaur-1B":    "socius-Llama-Centaur-1B_cogsoc.json",
    "Centaur-3B":    "socius-Llama-Centaur-3B_cogsoc.json",
    "Centaur-8B":    "socius-Llama-Centaur-8B_cogsoc.json",
}

files = {k: os.path.join(DATA_DIR, v) for k, v in MODEL_FILE_MAP.items()}

# ============================================================================
# TASK DEFINITIONS
# ============================================================================
ALL_TASKS = [
    ('ethics_cm',             'acc,none',                          'acc_stderr,none',                          'CM',           'Ethics'),
    ('ethics_deontology',     'acc,none',                          'acc_stderr,none',                          'Deontology',   'Ethics'),
    ('ethics_justice',        'acc,none',                          'acc_stderr,none',                          'Justice',      'Ethics'),
    ('ethics_utilitarianism', 'acc,none',                          'acc_stderr,none',                          'Utilitarian',  'Ethics'),
    ('ethics_virtue',         'acc,none',                          'acc_stderr,none',                          'Virtue',       'Ethics'),
    ('logiqa',                'acc_norm,none',                     'acc_norm_stderr,none',                     'LogiQA',       'CogLang'),
    ('piqa',                  'acc_norm,none',                     'acc_norm_stderr,none',                     'PIQA',         'CogLang'),
    ('social_iqa',            'acc,none',                          'acc_stderr,none',                          'Social IQA',   'CogLang'),
    ('coqa',                  'f1,none',                           'f1_stderr,none',                           'CoQA (F1)',    'CogLang'),
    ('lambada_openai',        'acc,none',                          'acc_stderr,none',                          'LAMBADA (OpenAI)','CogLang'),
    ('lambada_standard',      'acc,none',                          'acc_stderr,none',                          'LAMBADA (standard)','CogLang'),
    ('acp_app_bool',          'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'App (B)',      'ACP'),
    ('acp_areach_bool',       'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Areach (B)',   'ACP'),
    ('acp_just_bool',         'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Just (B)',     'ACP'),
    ('acp_land_bool',         'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Land (B)',     'ACP'),
    ('acp_prog_bool',         'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Prog (B)',     'ACP'),
    ('acp_reach_bool',        'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Reach (B)',    'ACP'),
    ('acp_val_bool',          'exact_match,extract-yes-no',        'exact_match_stderr,extract-yes-no',        'Val (B)',      'ACP'),
    ('acp_app_mcq',           'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'App (M)',      'ACP'),
    ('acp_areach_mcq',        'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Areach (M)',   'ACP'),
    ('acp_just_mcq',          'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Just (M)',     'ACP'),
    ('acp_land_mcq',          'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Land (M)',     'ACP'),
    ('acp_prog_mcq',          'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Prog (M)',     'ACP'),
    ('acp_reach_mcq',         'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Reach (M)',    'ACP'),
    ('acp_val_mcq',           'exact_match,mcq-extract',           'exact_match_stderr,mcq-extract',           'Val (M)',      'ACP'),
]

GROUPS = {
    'Ethics':  [t for t in ALL_TASKS if t[4] == 'Ethics'],
    'CogLang': [t for t in ALL_TASKS if t[4] == 'CogLang'],
    'ACP':     [t for t in ALL_TASKS if t[4] == 'ACP'],
}
GROUP_TITLES = {'Ethics': 'Ethics', 'CogLang': 'Cognitive & Language', 'ACP': 'ACP Bench (Planning)'}

# ============================================================================
# LOAD DATA
# ============================================================================
all_data = {}
for model_name, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping {model_name}")
        continue
    with open(filepath) as f:
        d = json.load(f)
    all_data[model_name] = {}
    for task_id, metric, stderr_metric, display, grp in ALL_TASKS:
        if task_id in d['results']:
            all_data[model_name][task_id] = {
                'value': d['results'][task_id].get(metric, None),
                'stderr': d['results'][task_id].get(stderr_metric, None),
            }
    # Also load EQ-Bench for CSV
    if 'eq_bench' in d['results']:
        all_data[model_name]['eq_bench'] = {
            'value': d['results']['eq_bench'].get('eqbench,none', None),
            'stderr': d['results']['eq_bench'].get('eqbench_stderr,none', None),
        }

print(f"Loaded {len(all_data)} models from {DATA_DIR}")

# ============================================================================
# CSV AND MD TABLE
# ============================================================================
model_col_order = [
    'Qwen3-0.6B', 'Qwentaur-0.6B',
    'Llama-3.2-1B', 'Centaur-1B',
    'Qwen3-1.7B', 'Qwentaur-1.7B',
    'Llama-3.2-3B', 'Centaur-3B',
    'Qwen3-4B', 'Qwentaur-4B',
    'Qwen3-8B', 'Qwentaur-8B',
    'Llama-3.1-8B', 'Centaur-8B',
    'Qwen3-14B', 'Qwentaur-14B',
]

csv_rows = []
for task_id, metric, stderr_metric, display, grp in ALL_TASKS:
    row = {'Task': display, 'Group': grp}
    for model in model_col_order:
        d = all_data.get(model, {}).get(task_id, {})
        v = d.get('value', None)
        row[model] = f"{v:.4f}" if v is not None else ''
    csv_rows.append(row)

eq_row = {'Task': 'EQ-Bench', 'Group': 'EQ-Bench'}
for model in model_col_order:
    d = all_data.get(model, {}).get('eq_bench', {})
    v = d.get('value', None)
    eq_row[model] = f"{v:.4f}" if v is not None else ''
csv_rows.append(eq_row)

mean_row = {'Task': 'Mean (excl. EQ-Bench)', 'Group': ''}
for model in model_col_order:
    vals = []
    for task_id, metric, stderr_metric, display, grp in ALL_TASKS:
        d = all_data.get(model, {}).get(task_id, {})
        v = d.get('value', None)
        if v is not None: vals.append(v)
    mean_row[model] = f"{np.mean(vals):.4f}" if vals else ''
csv_rows.append(mean_row)

csv_path = os.path.join(FIG_DIR, 'cogsoc_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Task', 'Group'] + model_col_order)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"CSV: {csv_path}")

# ============================================================================
# HELPERS
# ============================================================================
def get_vals(model, task_list):
    vals, ses = [], []
    for tid, metric, se_metric, disp, grp in task_list:
        d = all_data.get(model, {}).get(tid, {})
        v = d.get('value', None)
        s = d.get('stderr', None)
        vals.append(v if v is not None else 0)
        ses.append(s if s is not None else 0)
    return np.array(vals), np.array(ses)

def compute_ztest(bv, bse, fv, fse):
    if bse is None or fse is None or bse == 0 or fse == 0:
        return None, None
    pooled = np.sqrt(bse**2 + fse**2)
    z = (fv - bv) / pooled
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

# ============================================================================
# GENERATE 15 PLOTS (3 groups x 5 types)
# ============================================================================
for grp_key, task_list in GROUPS.items():
    grp_title = GROUP_TITLES[grp_key]
    bench_names = [t[3] for t in task_list]
    bench_ids = [t[0] for t in task_list]
    n_bench = len(bench_names)
    benchmarks_with_mean = bench_names + ['Mean']
    n_benchmarks = len(benchmarks_with_mean)

    # Figure width scaling
    if n_bench <= 6: fw = 9
    elif n_bench <= 10: fw = 11
    else: fw = 14

    print(f"\n--- {grp_title} ({n_bench} benchmarks) ---")

    # ==================================================================
    # PLOT 1: BASE VS FINETUNED
    # ==================================================================
    print(f"  1/5 base_vs_finetuned")

    fig, ax = plt.subplots(figsize=(fw, 4))

    model_pairs_bvf = [
        ("Qwen3-0.6B", "Qwentaur-0.6B", "0.6B", tint(QWEN, 0.55), "Qwen"),
        ("Qwen3-4B",   "Qwentaur-4B",   "4B",   tint(QWEN, 0.25), "Qwen"),
        ("Qwen3-8B",   "Qwentaur-8B",   "8B",   tint(QWEN, 0.12), "Qwen"),
        ("Qwen3-14B",  "Qwentaur-14B",  "14B",  QWEN, "Qwen"),
        ("Llama-3.2-1B","Centaur-1B",   "1B",   tint(LLAMA, 0.45), "Llama"),
        ("Llama-3.2-3B","Centaur-3B",   "3B",   tint(LLAMA, 0.25), "Llama"),
        ("Llama-3.1-8B","Centaur-8B",   "8B",   LLAMA, "Llama"),
    ]

    n_pairs = len(model_pairs_bvf)
    pair_width = 0.8 / n_pairs
    bar_width = pair_width * 0.45
    x = np.arange(n_benchmarks)

    for i, (base_key, ft_key, size_label, color, family) in enumerate(model_pairs_bvf):
        base_values, base_errors = get_vals(base_key, task_list)
        ft_values, ft_errors = get_vals(ft_key, task_list)

        base_mean = np.mean(base_values)
        base_mean_se = np.sqrt(np.sum(base_errors**2)) / len(base_errors)
        ft_mean = np.mean(ft_values)
        ft_mean_se = np.sqrt(np.sum(ft_errors**2)) / len(ft_errors)

        bv = np.append(base_values, base_mean)
        be = np.append(base_errors, base_mean_se)
        fv = np.append(ft_values, ft_mean)
        fe = np.append(ft_errors, ft_mean_se)

        offset = (i - (n_pairs - 1) / 2) * pair_width

        ax.bar(x + offset - bar_width/2, bv, bar_width, yerr=be,
               capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
               color='white', edgecolor=color, linewidth=1.0)
        ax.bar(x + offset + bar_width/2, fv, bar_width, yerr=fe,
               capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
               color=color, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_with_mean)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    legend_elements = [
        Patch(facecolor='white', edgecolor=QWEN, linewidth=1.0, label='Qwen (base)'),
        Patch(facecolor=QWEN, edgecolor='white', linewidth=0.5, label='Qwentaur'),
        Patch(facecolor='white', edgecolor=LLAMA, linewidth=1.0, label='Llama (base)'),
        Patch(facecolor=LLAMA, edgecolor='white', linewidth=0.5, label='Llama-Centaur'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=5.5,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3, ncol=2)

    plt.tight_layout()
    save_figure(fig, FIG_DIR, f'cogsoc_{grp_key.lower()}_base_vs_finetuned')

    # ==================================================================
    # PLOT 2: DELTA
    # ==================================================================
    print(f"  2/5 delta")

    fig, ax = plt.subplots(figsize=(fw, 4))
    x = np.arange(n_benchmarks)

    delta_configs = [
        ("Qwen3-0.6B",  "Qwentaur-0.6B",  "Qwentaur-0.6B",      tint(QWEN, 0.55)),
        ("Qwen3-1.7B",  "Qwentaur-1.7B",  "Qwentaur-1.7B",      tint(QWEN, 0.40)),
        ("Qwen3-4B",    "Qwentaur-4B",     "Qwentaur-4B",        tint(QWEN, 0.25)),
        ("Qwen3-8B",    "Qwentaur-8B",     "Qwentaur-8B",        tint(QWEN, 0.12)),
        ("Qwen3-14B",   "Qwentaur-14B",    "Qwentaur-14B",       QWEN),
        ("Llama-3.2-1B","Centaur-1B",      "Llama-Centaur-1B",   tint(LLAMA, 0.45)),
        ("Llama-3.2-3B","Centaur-3B",      "Llama-Centaur-3B",   tint(LLAMA, 0.25)),
        ("Llama-3.1-8B","Centaur-8B",      "Llama-Centaur-8B",   LLAMA),
    ]

    n_models = len(delta_configs)
    bar_width_d = 0.75 / n_models

    for i, (bk, fk, label, color) in enumerate(delta_configs):
        bvals, _ = get_vals(bk, task_list)
        fvals, _ = get_vals(fk, task_list)
        deltas = fvals - bvals
        deltas_wm = np.append(deltas, np.mean(deltas))

        offset = (i - (n_models - 1) / 2) * bar_width_d
        ax.bar(x + offset, deltas_wm, bar_width_d, color=color, edgecolor='white',
               linewidth=0.5, label=label)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.03, color='red')
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.03, color='green')
    ax.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    ax.set_ylabel('Delta Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_with_mean)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=5.5,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, FIG_DIR, f'cogsoc_{grp_key.lower()}_delta')

    # ==================================================================
    # PLOT 3: FINETUNED ONLY
    # ==================================================================
    print(f"  3/5 finetuned_only")

    fig, ax = plt.subplots(figsize=(fw, 3.5))
    x = np.arange(n_benchmarks)

    ft_configs = [
        ("Qwentaur-0.6B", "Qwentaur-0.6B",      tint(QWEN, 0.55)),
        ("Qwentaur-1.7B", "Qwentaur-1.7B",      tint(QWEN, 0.40)),
        ("Qwentaur-4B",   "Qwentaur-4B",         tint(QWEN, 0.25)),
        ("Qwentaur-8B",   "Qwentaur-8B",         tint(QWEN, 0.12)),
        ("Qwentaur-14B",  "Qwentaur-14B",        QWEN),
        ("Centaur-1B",    "Llama-Centaur-1B",    tint(LLAMA, 0.45)),
        ("Centaur-3B",    "Llama-Centaur-3B",    tint(LLAMA, 0.25)),
        ("Centaur-8B",    "Llama-Centaur-8B",    LLAMA),
    ]

    n_models = len(ft_configs)
    bar_width_f = 0.75 / n_models

    for i, (mk, label, color) in enumerate(ft_configs):
        vals, errs = get_vals(mk, task_list)
        mean_val = np.mean(vals)
        mean_se = np.sqrt(np.sum(errs**2)) / len(errs)
        vals_wm = np.append(vals, mean_val)
        errs_wm = np.append(errs, mean_se)

        offset = (i - (n_models - 1) / 2) * bar_width_f
        ax.bar(x + offset, vals_wm, bar_width_f, yerr=errs_wm,
               capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
               color=color, edgecolor='white', linewidth=0.5, label=label)

    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_with_mean)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=5.5,
              borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, FIG_DIR, f'cogsoc_{grp_key.lower()}_finetuned_only')

    # ==================================================================
    # PLOT 4: BY FAMILY (two panels, a/b labels)
    # ==================================================================
    print(f"  4/5 by_family")

    fig, axes = plt.subplots(2, 1, figsize=(fw, 7), sharex=True)

    fam_data = {
        'Qwen': [
            ("Qwen3-0.6B", "Qwentaur-0.6B", "0.6B", tint(QWEN, 0.55)),
            ("Qwen3-1.7B", "Qwentaur-1.7B", "1.7B", tint(QWEN, 0.40)),
            ("Qwen3-4B",   "Qwentaur-4B",   "4B",   tint(QWEN, 0.25)),
            ("Qwen3-8B",   "Qwentaur-8B",   "8B",   tint(QWEN, 0.12)),
            ("Qwen3-14B",  "Qwentaur-14B",  "14B",  QWEN),
        ],
        'Llama': [
            ("Llama-3.2-1B", "Centaur-1B", "1B", tint(LLAMA, 0.45)),
            ("Llama-3.2-3B", "Centaur-3B", "3B", tint(LLAMA, 0.25)),
            ("Llama-3.1-8B", "Centaur-8B", "8B", LLAMA),
        ],
    }

    fam_base_names = {'Qwen': 'Qwen', 'Llama': 'Llama'}
    fam_ft_names = {'Qwen': 'Qwentaur', 'Llama': 'Llama-Centaur'}

    for ax_idx, (fam_name, pairs) in enumerate(fam_data.items()):
        ax = axes[ax_idx]
        x = np.arange(n_benchmarks)

        np_ = len(pairs)
        pw = 0.8 / np_
        bw = pw * 0.45
        base_color = QWEN if fam_name == 'Qwen' else LLAMA

        for i, (bk, fk, slabel, color) in enumerate(pairs):
            bvals, berrs = get_vals(bk, task_list)
            fvals, ferrs = get_vals(fk, task_list)

            bm = np.mean(bvals)
            fm = np.mean(fvals)
            bvals_wm = np.append(bvals, bm)
            fvals_wm = np.append(fvals, fm)
            berrs_wm = np.append(berrs, np.sqrt(np.sum(berrs**2))/len(berrs))
            ferrs_wm = np.append(ferrs, np.sqrt(np.sum(ferrs**2))/len(ferrs))

            offset = (i - (np_ - 1) / 2) * pw

            ax.bar(x + offset - bw/2, bvals_wm, bw, yerr=berrs_wm,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color='white', edgecolor=color, linewidth=1.0)
            ax.bar(x + offset + bw/2, fvals_wm, bw, yerr=ferrs_wm,
                   capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                   color=color, edgecolor='white', linewidth=0.5, label=slabel)

        ax.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.set_xticks(x)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Performance')

        # Centered panel label
        panel_label = chr(ord('a') + ax_idx)
        ax.set_title(f'{panel_label}  {fam_name} Family', fontweight='bold', fontsize=8)

        # Only show x-tick labels on the bottom panel
        if ax_idx == len(fam_data) - 1:
            ax.set_xticklabels(benchmarks_with_mean)
        else:
            ax.set_xticklabels([])

        # Legend with base/ft labels
        legend_elements = [
            Patch(facecolor='white', edgecolor=base_color, linewidth=1.0,
                  label=f'{fam_base_names[fam_name]} (base)'),
            Patch(facecolor=base_color, edgecolor='white', linewidth=0.5,
                  label=fam_ft_names[fam_name]),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=5.5,
                  borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    save_figure(fig, FIG_DIR, f'cogsoc_{grp_key.lower()}_by_family')

    # ==================================================================
    # PLOT 5: HEATMAP
    # ==================================================================
    print(f"  5/5 heatmap")

    hm_pairs = [
        ("Qwen3-0.6B",  "Qwentaur-0.6B",  "Qwentaur-0.6B"),
        ("Qwen3-1.7B",  "Qwentaur-1.7B",  "Qwentaur-1.7B"),
        ("Qwen3-4B",    "Qwentaur-4B",     "Qwentaur-4B"),
        ("Qwen3-8B",    "Qwentaur-8B",     "Qwentaur-8B"),
        ("Qwen3-14B",   "Qwentaur-14B",    "Qwentaur-14B"),
        ("Llama-3.2-1B","Centaur-1B",      "Llama-Centaur-1B"),
        ("Llama-3.2-3B","Centaur-3B",      "Llama-Centaur-3B"),
        ("Llama-3.1-8B","Centaur-8B",      "Llama-Centaur-8B"),
    ]

    bench_labels_wm = bench_names + ['Mean']
    n_mod = len(hm_pairs)
    n_bwm = len(bench_labels_wm)

    z_matrix = np.zeros((n_mod, n_bwm))
    p_matrix = np.ones((n_mod, n_bwm))

    for i, (bk, fk, label) in enumerate(hm_pairs):
        z_vals = []
        for j, tid in enumerate(bench_ids):
            bd = all_data.get(bk, {}).get(tid, {})
            fd = all_data.get(fk, {}).get(tid, {})
            z, p = compute_ztest(
                bd.get('value', 0), bd.get('stderr', 0),
                fd.get('value', 0), fd.get('stderr', 0)
            )
            z_matrix[i, j] = z if z is not None else 0
            p_matrix[i, j] = p if p is not None else 1
            if z is not None:
                z_vals.append(z)

        z_matrix[i, -1] = np.mean(z_vals) if z_vals else 0
        combined_z = np.sum(z_vals) / np.sqrt(len(z_vals)) if z_vals else 0
        p_matrix[i, -1] = 2 * (1 - stats.norm.cdf(abs(combined_z)))

    # Colormap
    hm_colors = ['#d62728', '#ffcccc', 'white', '#ccffcc', '#2ca02c']
    hm_positions = [0.0, 0.35, 0.5, 0.65, 1.0]
    cmap = LinearSegmentedColormap.from_list('diverging', list(zip(hm_positions, hm_colors)))

    vmax = max(abs(z_matrix.min()), abs(z_matrix.max()))
    vmax = min(vmax, 6)

    hm_w = max(8, n_bench + 2)
    fig, ax = plt.subplots(figsize=(hm_w, 5))

    im = ax.imshow(z_matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('z-score (fine-tuned - base)', fontsize=9)

    # Annotations
    for i in range(n_mod):
        for j in range(n_bwm):
            z_val = z_matrix[i, j]
            p_val = p_matrix[i, j]

            if p_val < 0.001: sig = '***'
            elif p_val < 0.01: sig = '**'
            elif p_val < 0.05: sig = '*'
            else: sig = ''

            text = f'{z_val:.1f}{sig}'
            text_color = 'white' if abs(z_val) > vmax * 0.6 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                    color=text_color, fontweight='bold' if sig else 'normal')

    # Labels
    ax.set_xticks(np.arange(n_bwm))
    ax.set_xticklabels(bench_labels_wm)
    ax.set_yticks(np.arange(n_mod))
    ax.set_yticklabels([p[2] for p in hm_pairs])

    # Separator lines
    ax.axvline(x=n_bench - 0.5, color='black', linestyle='-', linewidth=1.5)
    ax.axhline(y=4.5, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Model')

    # Significance legend
    ax.text(1.02, -0.15, '* p<0.05  ** p<0.01  *** p<0.001',
            transform=ax.transAxes, fontsize=7, va='top')

    plt.tight_layout()
    save_figure(fig, FIG_DIR, f'cogsoc_{grp_key.lower()}_heatmap')

    # ==================================================================
    # PLOT 6: COMBINED (delta top, Qwen middle, Llama bottom, shared legend)
    # ==================================================================
    print(f"  6/6 combined")

    fig_c, axes_c = plt.subplots(3, 1, figsize=(fw, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1, 1]})
    ax_delta_c, ax_qwen_c, ax_llama_c = axes_c

    # --- Panel a: Delta ---
    x_c = np.arange(n_benchmarks)

    for i, (bk, fk, label, color) in enumerate(delta_configs):
        bvals, _ = get_vals(bk, task_list)
        fvals, _ = get_vals(fk, task_list)
        deltas_c = fvals - bvals
        deltas_c_wm = np.append(deltas_c, np.mean(deltas_c))

        offset = (i - (n_models - 1) / 2) * bar_width_d
        ax_delta_c.bar(x_c + offset, deltas_c_wm, bar_width_d, color=color,
                       edgecolor='white', linewidth=0.5, label=label)

    ax_delta_c.axhline(y=0, color='black', linewidth=0.8)
    ax_delta_c.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--',
                       linewidth=0.6, alpha=0.5)
    ylim_c = ax_delta_c.get_ylim()
    ax_delta_c.axhspan(ylim_c[0], 0, alpha=0.03, color='red')
    ax_delta_c.axhspan(0, ylim_c[1], alpha=0.03, color='green')
    ax_delta_c.set_ylabel('Delta performance')
    ax_delta_c.set_title('a  Delta (fine-tuned - base)', fontweight='bold', fontsize=8)

    # --- Panels b/c: By Family ---
    fam_combined = {
        'Qwen': (ax_qwen_c, QWEN, 'Qwen', 'Qwentaur', [
            ("Qwen3-0.6B", "Qwentaur-0.6B", "0.6B", tint(QWEN, 0.55)),
            ("Qwen3-1.7B", "Qwentaur-1.7B", "1.7B", tint(QWEN, 0.40)),
            ("Qwen3-4B",   "Qwentaur-4B",   "4B",   tint(QWEN, 0.25)),
            ("Qwen3-8B",   "Qwentaur-8B",   "8B",   tint(QWEN, 0.12)),
            ("Qwen3-14B",  "Qwentaur-14B",  "14B",  QWEN),
        ]),
        'Llama': (ax_llama_c, LLAMA, 'Llama', 'Llama-Centaur', [
            ("Llama-3.2-1B", "Centaur-1B", "1B", tint(LLAMA, 0.45)),
            ("Llama-3.2-3B", "Centaur-3B", "3B", tint(LLAMA, 0.25)),
            ("Llama-3.1-8B", "Centaur-8B", "8B", LLAMA),
        ]),
    }

    for fam_idx, (fam_name, (ax_fam, base_color, base_label, ft_label, pairs_c)) in enumerate(fam_combined.items()):
        np_c = len(pairs_c)
        pw_c = 0.8 / np_c
        bw_c = pw_c * 0.45

        for i, (bk, fk, slabel, color) in enumerate(pairs_c):
            bvals, berrs = get_vals(bk, task_list)
            fvals, ferrs = get_vals(fk, task_list)
            bm = np.mean(bvals)
            fm = np.mean(fvals)
            bvals_wm = np.append(bvals, bm)
            fvals_wm = np.append(fvals, fm)
            berrs_wm = np.append(berrs, np.sqrt(np.sum(berrs**2)) / len(berrs))
            ferrs_wm = np.append(ferrs, np.sqrt(np.sum(ferrs**2)) / len(ferrs))

            offset = (i - (np_c - 1) / 2) * pw_c
            ax_fam.bar(x_c + offset - bw_c/2, bvals_wm, bw_c, yerr=berrs_wm,
                       capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                       color='white', edgecolor=color, linewidth=1.0)
            ax_fam.bar(x_c + offset + bw_c/2, fvals_wm, bw_c, yerr=ferrs_wm,
                       capsize=0, ecolor='#888888', error_kw={'linewidth': 0.6, 'alpha': 0.7},
                       color=color, edgecolor='white', linewidth=0.5)

        ax_fam.axvline(x=len(bench_names) - 0.5, color='gray', linestyle='--',
                       linewidth=0.6, alpha=0.5)
        ax_fam.set_ylim(0, 1.05)
        ax_fam.set_ylabel('Performance')
        panel = 'b' if fam_idx == 0 else 'c'
        ax_fam.set_title(f'{panel}  {fam_name} family', fontweight='bold', fontsize=8)

    # X-tick labels only on bottom panel
    ax_llama_c.set_xticks(x_c)
    ax_llama_c.set_xticklabels(benchmarks_with_mean)

    # Shared legend at bottom
    legend_elements_c = [
        Patch(facecolor='white', edgecolor=QWEN, linewidth=1.0, label='Qwen (base)'),
        Patch(facecolor=QWEN, edgecolor='white', linewidth=0.5, label='Qwentaur'),
        Patch(facecolor='white', edgecolor=LLAMA, linewidth=1.0, label='Llama (base)'),
        Patch(facecolor=LLAMA, edgecolor='white', linewidth=0.5, label='Llama-Centaur'),
    ]
    fig_c.legend(handles=legend_elements_c, loc='lower center', ncol=4,
                 bbox_to_anchor=(0.5, -0.02), fontsize=6,
                 borderpad=0.3, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)

    plt.tight_layout()
    fig_c.subplots_adjust(bottom=0.08)
    save_figure(fig_c, FIG_DIR, f'cogsoc_{grp_key.lower()}_combined')

print(f"\nAll plots (PNG + PDF) + CSV + MD generated in {FIG_DIR}")
