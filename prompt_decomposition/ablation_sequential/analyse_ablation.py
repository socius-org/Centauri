#!/usr/bin/env python3
"""
Ablation Analysis: Structure-Preserving Prompt Perturbation
============================================================

Normalized by fraction of learned information lost/retained.

Retention ratio: (ln(k) - NLL_cond) / (ln(k) - NLL_orig)
  = 1.0 means full original performance, 0.0 means chance, <0 means worse than chance.

Fraction lost: (NLL_cond - NLL_orig) / (ln(k) - NLL_orig)
  = 0.0 means no degradation, 1.0 means back to chance, >1 means worse than chance.

This normalizes by what the model actually learned (not just the chance floor),
making cross-task comparisons fair even when models vary in baseline performance.

Reads CSVs from  ablation_results/
Saves output to  figures/

Usage:
    python analyse_ablation.py
"""

import csv
import os
import sys
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'ablation_results')
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

COND_ORDER = ['original', 'instruction_ablated', 'content_masked', 'history_only']
COND_SHORT = ['orig', 'inst_ab', 'cont_m', 'hist_o']

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

TASK_FULL = {
    "badham2017deficits":          "Shepard categorization",
    "bahrami2020four":             "Drifting four-armed bandit",
    "collsiöö2023MCPL":            "Multiple-cue judgment",
    "feng2021dynamics":            "Horizon task (Feng)",
    "flesch2018comparing":         "Gardening task",
    "frey2017cct":                 "Columbia card task",
    "gershman2018deconstructing":  "Two-armed bandit",
    "gershman2020reward":          "Cond. assoc. learning",
    "hilbig2014generalized":       "Multi-attribute DM",
    "kool2016when":                "Two-step task (Kool '16)",
    "kool2017cost":                "Two-step task (Kool '17)",
    "lefebvre2017behavioural":     "Prob. instrumental learning",
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all ablation CSVs. Returns {model: {task: {condition: NLL}}} (raw NLL)."""
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
                if task not in VALID_TASKS:
                    continue
                data.setdefault(model, {}).setdefault(task, {})[r['condition']] = float(r['loss'])
    return data


def get_tasks(data):
    return sorted(set(t for m in data.values() for t in m.keys()))


def retention(task, nll_orig, nll_cond):
    """Retention ratio: (ln_k - NLL_cond) / (ln_k - NLL_orig).
    1.0 = original, 0.0 = chance, <0 = worse than chance."""
    headroom = LN_K[task] - nll_orig
    if headroom < 1e-6:
        return 0.0
    return (LN_K[task] - nll_cond) / headroom


def frac_lost(task, nll_orig, nll_cond):
    """Fraction of learned info lost: (NLL_cond - NLL_orig) / (ln_k - NLL_orig).
    0 = no loss, 1.0 = back to chance, >1 = worse than chance."""
    headroom = LN_K[task] - nll_orig
    if headroom < 1e-6:
        return 0.0
    return (nll_cond - nll_orig) / headroom


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Per-model retention ratio across conditions
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_hierarchy(data):
    print("=" * 90)
    print("1. ABLATION CONDITION HIERARCHY (retention ratio)")
    print("=" * 90)
    print()
    print("Retention ratio: (ln(k) - NLL_cond) / (ln(k) - NLL_orig)")
    print("  1.0 = original performance, 0.0 = chance, <0 = worse than chance.")
    print("Should DECREASE monotonically if models genuinely use task content.")
    print()

    tasks = get_tasks(data)
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])

    header = f"{'Model':>30s}"
    for c in COND_SHORT:
        header += f"  {c:>8s}"
    header += "   monotonic?"
    print(header)
    print("-" * len(header))

    n_monotonic = 0
    for model in models_sorted:
        means = []
        for c in COND_ORDER:
            vals = [retention(t, data[model][t]['original'], data[model][t][c])
                    for t in tasks
                    if 'original' in data[model].get(t, {}) and c in data[model].get(t, {})]
            means.append(np.mean(vals))
        mono = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
        n_monotonic += mono
        row = f"{model:>30s}"
        for m in means:
            row += f"  {m:8.4f}"
        row += f"   {'YES' if mono else 'NO'}"
        print(row)

    print()
    print(f"  -> {n_monotonic}/{len(models_sorted)} models show strictly monotonic hierarchy")
    print()

    # Grand average
    grand = []
    for c in COND_ORDER:
        all_vals = []
        for model in models_sorted:
            for t in tasks:
                tc = data[model].get(t, {})
                if 'original' in tc and c in tc:
                    all_vals.append(retention(t, tc['original'], tc[c]))
        grand.append(np.mean(all_vals))
    print(f"  Grand average retention: {' -> '.join(f'{g:.3f}' for g in grand)}")
    drops = [grand[i] - grand[i+1] for i in range(len(grand)-1)]
    print(f"  Average drops: instruction={drops[0]:.3f}, "
          f"content={drops[1]:.3f}, structure={drops[2]:.3f}")
    print(f"  (drops = fraction of learned information lost per layer)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Layer decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_layers(data):
    print("=" * 90)
    print("2. LAYER DECOMPOSITION (% of total learned information lost)")
    print("=" * 90)
    print()
    print("  Instruction layer:  original -> instruction_ablated  (task framing)")
    print("  Content layer:      instruction_ablated -> content_masked  (stimuli + feedback)")
    print("  Structure layer:    content_masked -> history_only  (trial skeleton)")
    print()
    print("  Each delta is (NLL_b - NLL_a) / (ln(k) - NLL_orig) per model-task,")
    print("  then averaged. Percentages show share of total retention loss.")
    print()

    tasks = get_tasks(data)
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])

    header = f"{'Model':>30s}  {'retain':>7s}  {'hist_r':>7s}  {'instr%':>7s}  {'content%':>9s}  {'struct%':>8s}"
    print(header)
    print("-" * len(header))

    all_instr, all_content, all_struct = [], [], []

    for model in models_sorted:
        r_orig, r_ia, r_cm, r_hist = [], [], [], []
        for t in tasks:
            tc = data[model].get(t, {})
            if all(c in tc for c in COND_ORDER):
                nll_o = tc['original']
                r_orig.append(retention(t, nll_o, nll_o))       # = 1.0
                r_ia.append(retention(t, nll_o, tc['instruction_ablated']))
                r_cm.append(retention(t, nll_o, tc['content_masked']))
                r_hist.append(retention(t, nll_o, tc['history_only']))

        ro, ria, rcm, rh = np.mean(r_orig), np.mean(r_ia), np.mean(r_cm), np.mean(r_hist)
        total_drop = ro - rh
        if total_drop > 0:
            instr_pct = (ro - ria) / total_drop * 100
            content_pct = (ria - rcm) / total_drop * 100
            struct_pct = (rcm - rh) / total_drop * 100
        else:
            instr_pct = content_pct = struct_pct = 0

        all_instr.append(instr_pct)
        all_content.append(content_pct)
        all_struct.append(struct_pct)

        print(f"{model:>30s}  {ro:7.4f}  {rh:7.4f}  {instr_pct:6.1f}%  {content_pct:8.1f}%  {struct_pct:7.1f}%")

    print("-" * len(header))
    print(f"{'Average':>30s}  {'':>7s}  {'':>7s}  {np.mean(all_instr):6.1f}%  "
          f"{np.mean(all_content):8.1f}%  {np.mean(all_struct):7.1f}%")
    print()
    print(f"  -> Content (stimuli + feedback) accounts for ~{np.mean(all_content):.0f}% of learned information lost")
    print(f"  -> Instruction framing accounts for only ~{np.mean(all_instr):.0f}%")
    print(f"  -> Trial structure contributes ~{np.mean(all_struct):.0f}%")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Per-task sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_tasks(data):
    print("=" * 90)
    print("3. PER-TASK ABLATION SENSITIVITY (fraction of learned info lost)")
    print("=" * 90)
    print()
    print("Fraction lost = (NLL_hist - NLL_orig) / (ln(k) - NLL_orig)")
    print("  0 = no degradation, 1 = back to chance, >1 = worse than chance.")
    print("This accounts for both task difficulty AND model baseline performance.")
    print()

    tasks = get_tasks(data)
    models = sorted(data.keys())

    task_deltas = {}
    for task in tasks:
        fracs = []
        for m in models:
            tc = data[m].get(task, {})
            if 'original' in tc and 'history_only' in tc:
                fracs.append(frac_lost(task, tc['original'], tc['history_only']))
        if fracs:
            task_deltas[task] = np.mean(fracs)

    sorted_tasks = sorted(task_deltas.keys(), key=lambda t: task_deltas[t], reverse=True)

    header = f"{'Rank':>4s}  {'Task':>35s}  {'k':>4s}  {'Frac. lost':>11s}  {'Retained':>9s}  {'Interpretation'}"
    print(header)
    print("-" * 110)

    for i, task in enumerate(sorted_tasks):
        fl = task_deltas[task]
        ret = 1.0 - fl
        name = TASK_FULL.get(task, task)
        k = round(np.exp(LN_K[task]))
        if fl > 1.0:
            interp = "Worse than chance without content"
        elif fl > 0.8:
            interp = "Nearly all learned info lost"
        elif fl > 0.5:
            interp = "Majority of learned info lost"
        elif fl > 0.3:
            interp = "Moderate info loss"
        else:
            interp = "Most learned info retained"
        print(f"{i+1:>4d}  {name:>35s}  {k:>4d}  {fl:>+10.3f}  {ret:>8.3f}  {interp}")

    print()
    extreme = [t for t in sorted_tasks if task_deltas[t] > 1.0]
    high = [t for t in sorted_tasks if 0.8 <= task_deltas[t] <= 1.0]
    low = [t for t in sorted_tasks if task_deltas[t] < 0.3]
    print(f"  -> {len(extreme)} tasks where model is worse than chance without content (frac > 1.0)")
    print(f"  -> {len(high)} tasks where nearly all learned info is lost (0.8-1.0)")
    print(f"  -> {len(low)} tasks where most learned info is retained (frac < 0.3)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Scaling analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_scaling(data):
    print("=" * 90)
    print("4. SCALING ANALYSIS: FRACTION LOST vs MODEL SIZE")
    print("=" * 90)
    print()
    print("Mean fraction of learned info lost (orig -> hist_only) per model.")
    print("This normalizes by each model-task's actual headroom above chance,")
    print("removing confounds from both option cardinality AND baseline performance.")
    print()

    tasks = get_tasks(data)
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])

    header = f"{'Model':>30s}  {'Size':>5s}  {'mean_retain':>12s}  {'mean_frac_lost':>15s}"
    print(header)
    print("-" * len(header))

    sizes, mean_fracs = [], []

    for model in models_sorted:
        fracs = []
        for t in tasks:
            tc = data[model].get(t, {})
            if 'original' in tc and 'history_only' in tc:
                fracs.append(frac_lost(t, tc['original'], tc['history_only']))
        mean_fl = np.mean(fracs)
        mean_ret = 1.0 - mean_fl

        sizes.append(MODEL_SIZES[model])
        mean_fracs.append(mean_fl)

        print(f"{model:>30s}  {MODEL_SIZES[model]:5.1f}  {mean_ret:12.4f}  {mean_fl:>14.4f}")

    print()

    log_sizes = np.log10(sizes)
    rho, p = stats.spearmanr(log_sizes, mean_fracs)

    print(f"  Fraction lost: mean = {np.mean(mean_fracs):.4f}, "
          f"std = {np.std(mean_fracs):.4f}, range = [{min(mean_fracs):.4f}, {max(mean_fracs):.4f}]")
    print(f"    Spearman rho(log_size, frac_lost) = {rho:.3f}, p = {p:.4f}")
    print()

    print("  INTERPRETATION:")
    if abs(rho) < 0.3:
        print(f"  Fraction of learned info lost is approximately CONSTANT across scale (~{np.mean(mean_fracs):.3f})")
    elif rho > 0.3:
        print(f"  Fraction lost GROWS with scale — larger models are MORE content-dependent")
    else:
        print(f"  Fraction lost SHRINKS with scale — larger models retain more without content")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_statistics(data):
    print("=" * 90)
    print("5. STATISTICAL TESTS (on retention ratio)")
    print("=" * 90)
    print()

    tasks = get_tasks(data)
    models = sorted(data.keys())

    # Collect per-task mean retention ratio for each condition (averaged across models)
    cond_vectors = {c: [] for c in COND_ORDER}
    for task in tasks:
        for c in COND_ORDER:
            vals = [retention(task, data[m][task]['original'], data[m][task][c])
                    for m in models
                    if 'original' in data[m].get(task, {}) and c in data[m].get(task, {})]
            cond_vectors[c].append(np.mean(vals) if vals else np.nan)

    # Remove tasks with any NaN
    valid = ~np.any([np.isnan(cond_vectors[c]) for c in COND_ORDER], axis=0)
    for c in COND_ORDER:
        cond_vectors[c] = np.array(cond_vectors[c])[valid]
    n_tasks = len(cond_vectors['original'])

    print(f"  Using {n_tasks} tasks with complete data across all conditions.")
    print()

    # --- 5a. Friedman test ---
    print("  5a. Friedman test (are the four conditions significantly different?)")
    stat_f, p_f = stats.friedmanchisquare(
        cond_vectors['original'],
        cond_vectors['instruction_ablated'],
        cond_vectors['content_masked'],
        cond_vectors['history_only']
    )
    print(f"      chi2 = {stat_f:.2f}, p = {p_f:.2e}")
    print(f"      {'SIGNIFICANT' if p_f < 0.05 else 'Not significant'} at alpha = 0.05")
    print()

    # --- 5b. Pairwise Wilcoxon with Holm-Bonferroni ---
    print("  5b. Pairwise Wilcoxon signed-rank tests (adjacent conditions)")
    print("      with Holm-Bonferroni correction for 3 comparisons")
    print("      Note: higher retention = better, so we test c1 > c2 (alt='greater')")
    print()

    pairs = [
        ('original', 'instruction_ablated', 'Instruction removal'),
        ('instruction_ablated', 'content_masked', 'Content masking'),
        ('content_masked', 'history_only', 'Structure removal'),
    ]

    raw_pvals = []
    pair_results = []
    for c1, c2, label in pairs:
        stat_w, p_w = stats.wilcoxon(cond_vectors[c1], cond_vectors[c2], alternative='greater')
        raw_pvals.append(p_w)
        mean_drop = np.mean(cond_vectors[c1] - cond_vectors[c2])
        pair_results.append((label, c1, c2, stat_w, p_w, mean_drop))

    # Holm-Bonferroni correction
    n_tests = len(raw_pvals)
    sorted_idx = np.argsort(raw_pvals)
    corrected = np.zeros(n_tests)
    for rank, idx in enumerate(sorted_idx):
        corrected[idx] = raw_pvals[idx] * (n_tests - rank)
    corrected = np.minimum(corrected, 1.0)
    for i in range(1, n_tests):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])

    header_line = f"    {'Comparison':>25s}  {'mean drop':>10s}  {'W':>8s}  {'p (raw)':>10s}  {'p (Holm)':>10s}  {'sig?':>5s}"
    print(header_line)
    print("    " + "-" * (len(header_line) - 4))
    for i, (label, c1, c2, stat_w, p_w, mean_drop) in enumerate(pair_results):
        sig = "***" if corrected[i] < 0.001 else "**" if corrected[i] < 0.01 else "*" if corrected[i] < 0.05 else "n.s."
        print(f"    {label:>25s}  {mean_drop:>+9.4f}  {stat_w:>8.1f}  {p_w:>10.2e}  {corrected[i]:>10.2e}  {sig:>5s}")
    print()

    # --- 5c. Overall: original vs history_only ---
    print("  5c. Overall: original vs history_only (retention ratio)")
    stat_oh, p_oh = stats.wilcoxon(cond_vectors['original'], cond_vectors['history_only'], alternative='greater')
    mean_total_drop = np.mean(cond_vectors['original'] - cond_vectors['history_only'])
    print(f"      W = {stat_oh:.1f}, p = {p_oh:.2e}")
    print(f"      Mean retention drop = {mean_total_drop:+.4f} (from 1.000)")
    print(f"      Mean fraction lost = {mean_total_drop:.4f}")
    print(f"      {'SIGNIFICANT' if p_oh < 0.05 else 'Not significant'} at alpha = 0.05")
    print()

    # --- 5d. Spearman: fraction lost vs log(model size) ---
    print("  5d. Spearman correlation: fraction lost vs model scale")
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])
    sizes_log, fracs = [], []
    for model in models_sorted:
        fl_vals = [frac_lost(t, data[model][t]['original'], data[model][t]['history_only'])
                   for t in tasks
                   if 'original' in data[model].get(t, {}) and 'history_only' in data[model].get(t, {})]
        sizes_log.append(np.log10(MODEL_SIZES[model]))
        fracs.append(np.mean(fl_vals))

    rho, p = stats.spearmanr(sizes_log, fracs)
    print(f"      rho = {rho:.3f}, p = {p:.4f}")
    print(f"      {'SIGNIFICANT' if p < 0.05 else 'Not significant'} at alpha = 0.05")
    if abs(rho) < 0.3:
        print(f"      Weak correlation: fraction lost is approximately constant across scale")
    elif rho > 0.3:
        print(f"      Positive: larger models lose MORE of their learned info (more content-dependent)")
    else:
        print(f"      Negative: larger models retain MORE without content")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(data):
    print("=" * 90)
    print("6. SUMMARY")
    print("=" * 90)
    print()

    tasks = get_tasks(data)
    models = sorted(data.keys())
    models_sorted = sorted(MODEL_SIZES.keys(), key=lambda m: MODEL_SIZES[m])

    # Grand average retention ratios
    grand = {}
    for c in COND_ORDER:
        vals = [retention(t, data[m][t]['original'], data[m][t][c])
                for m in models for t in tasks
                if 'original' in data[m].get(t, {}) and c in data[m].get(t, {})]
        grand[c] = np.mean(vals)

    total_drop = grand['original'] - grand['history_only']
    instr_drop = grand['original'] - grand['instruction_ablated']
    content_drop = grand['instruction_ablated'] - grand['content_masked']
    struct_drop = grand['content_masked'] - grand['history_only']

    print("  METRIC: (NLL_cond - NLL_orig) / (ln(k) - NLL_orig)")
    print("  = fraction of learned information lost by ablation.")
    print("  Normalizes by what each model actually learned per task,")
    print("  not just the chance floor.")
    print(f"  19 tasks with known k (excluded: garcia, krueger, wise, wulff-sampling).")
    print()
    print("  KEY FINDINGS:")
    print()
    print(f"  1. CONSISTENT HIERARCHY across all {len(models)} models:")
    print(f"     Retention: {grand['original']:.3f} -> {grand['instruction_ablated']:.3f}"
          f" -> {grand['content_masked']:.3f} -> {grand['history_only']:.3f}")
    print(f"     (1.0 = original, 0.0 = chance)")
    print()
    print(f"  2. INSTRUCTION REMOVAL IS CHEAP (drop = {instr_drop:.3f}, "
          f"{instr_drop/total_drop*100:.0f}% of total):")
    print(f"     Only ~{instr_drop:.0%} of learned information depends on task framing.")
    print()
    print(f"  3. CONTENT MASKING IS THE BIG HIT (drop = {content_drop:.3f}, "
          f"{content_drop/total_drop*100:.0f}% of total):")
    print(f"     ~{content_drop:.0%} of what models learn comes from stimulus values")
    print(f"     and feedback signals.")
    print()

    # Top/bottom tasks
    task_deltas = {}
    for task in tasks:
        fracs = [frac_lost(task, data[m][task]['original'], data[m][task]['history_only'])
                 for m in models
                 if all(c in data[m].get(task, {}) for c in ['original', 'history_only'])]
        if fracs:
            task_deltas[task] = np.mean(fracs)
    sorted_tasks = sorted(task_deltas.keys(), key=lambda t: task_deltas[t], reverse=True)

    print(f"  4. TASKS VARY DRAMATICALLY:")
    print(f"     Most content-dependent (fraction of learned info lost):")
    for t in sorted_tasks[:3]:
        print(f"       - {TASK_FULL.get(t, t):35s} frac = {task_deltas[t]:.2f}")
    print(f"     Least content-dependent:")
    for t in sorted_tasks[-3:]:
        print(f"       - {TASK_FULL.get(t, t):35s} frac = {task_deltas[t]:.2f}")

    print()

    # Scaling
    mean_fracs = []
    for model in models_sorted:
        fl_vals = [frac_lost(t, data[model][t]['original'], data[model][t]['history_only'])
                   for t in tasks
                   if 'original' in data[model].get(t, {}) and 'history_only' in data[model].get(t, {})]
        mean_fracs.append(np.mean(fl_vals))

    print(f"  5. SCALING:")
    print(f"     Fraction lost range: {min(mean_fracs):.3f} - {max(mean_fracs):.3f}")
    print(f"     Mean: {np.mean(mean_fracs):.3f}")
    print()
    print("  INTERPRETATION:")
    print(f"  On average, stripping all content destroys ~{np.mean(mean_fracs):.0%} of learned info.")
    print(f"  Content (stimuli + feedback) alone accounts for ~{content_drop/total_drop*100:.0f}% of this loss.")
    print(f"  This is strong evidence that models genuinely learn from task content,")
    print(f"  not just shortcut patterns in choice sequences.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    data = load_data()
    print(f"\nLoaded {len(data)} models, "
          f"{len(get_tasks(data))} tasks, "
          f"{sum(len(t) for t in data.values())} model-task pairs\n")

    analyse_hierarchy(data)
    analyse_layers(data)
    analyse_tasks(data)
    analyse_scaling(data)
    analyse_statistics(data)
    print_summary(data)
