# -*- coding: utf-8 -*-
"""
LM-Eval Significance Tables
============================

Generates LaTeX tables showing the impact of cognitive fine-tuning
(delta = fine-tuned - base) with two-sided z-test significance markers.

Tables generated:
  1. significance_full.tex   -- All tasks (MetaBench + Ethics + CogLang
                                + ACP + EQ-Bench) with per-section means
  2. combined_summary.tex    -- Summary: MetaBench benchmarks + group means
                                + EQ-Bench

Cells show delta (ft - base) with significance stars (* p<0.05, ** p<0.01,
*** p<0.001) and diverging green/red background colour.

Usage:
    python generate_heatmap_tables.py

Reads:  metabench/*.json, cogsoc/*.json
Writes: tables/*.tex
"""

import json
import math
import os

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METABENCH_DIR = os.path.join(SCRIPT_DIR, "metabench")
COGSOC_DIR = os.path.join(SCRIPT_DIR, "cogsoc")
OUT_DIR = os.path.join(SCRIPT_DIR, "tables")

# =============================================================================
# Model pairs: (size_label, base_file_stem, ft_file_stem)
# =============================================================================

QWEN_PAIRS = [
    ("0.6B", "unsloth-Qwen3-0.6B-base",  "socius-Qwentaur-0.6B"),
    ("1.7B", "unsloth-Qwen3-1.7B-base",  "socius-Qwentaur-1.7B"),
    ("4B",   "unsloth-Qwen3-4B-base",    "socius-Qwentaur-4B"),
    ("8B",   "unsloth-Qwen3-8B-base",    "socius-Qwentaur-8B"),
    ("14B",  "unsloth-Qwen3-14B-base",   "socius-Qwentaur-14B"),
]
LLAMA_PAIRS = [
    ("1B", "unsloth-Llama-3.2-1B-base",  "socius-Llama-Centaur-1B"),
    ("3B", "unsloth-Llama-3.2-3B-base",  "socius-Llama-Centaur-3B"),
    ("8B", "unsloth-Llama-3.1-8B-base",  "socius-Llama-Centaur-8B"),
]

ALL_PAIRS = QWEN_PAIRS + LLAMA_PAIRS

# =============================================================================
# Metabench task definitions: (task_key, metric_key, stderr_key)
# =============================================================================

METABENCH_TASKS = {
    "ARC":        ("metabench_arc",        "acc_norm,none",                    "acc_norm_stderr,none"),
    "GSM8K":      ("metabench_gsm8k",      "exact_match,flexible-extract",     "exact_match_stderr,flexible-extract"),
    "HellaSwag":  ("metabench_hellaswag",   "acc_norm,none",                    "acc_norm_stderr,none"),
    "MMLU":       ("metabench_mmlu",        "acc,none",                         "acc_stderr,none"),
    "TruthfulQA": ("metabench_truthfulqa",  "acc,none",                         "acc_stderr,none"),
    "Winogrande": ("metabench_winogrande",  "acc,none",                         "acc_stderr,none"),
}
METABENCH_ORDER = ["ARC", "GSM8K", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande"]

# =============================================================================
# CogSoc task definitions: (task_id, metric_key, stderr_key, display, group)
# =============================================================================

ALL_COGSOC_TASKS = [
    ("ethics_cm",             "acc,none",                    "acc_stderr,none",                    "CM",            "Ethics"),
    ("ethics_deontology",     "acc,none",                    "acc_stderr,none",                    "Deontology",    "Ethics"),
    ("ethics_justice",        "acc,none",                    "acc_stderr,none",                    "Justice",       "Ethics"),
    ("ethics_utilitarianism", "acc,none",                    "acc_stderr,none",                    "Utilitarian",   "Ethics"),
    ("ethics_virtue",         "acc,none",                    "acc_stderr,none",                    "Virtue",        "Ethics"),
    ("logiqa",                "acc_norm,none",               "acc_norm_stderr,none",               "LogiQA",        "CogLang"),
    ("piqa",                  "acc_norm,none",               "acc_norm_stderr,none",               "PIQA",          "CogLang"),
    ("social_iqa",            "acc,none",                    "acc_stderr,none",                    "Social IQA",    "CogLang"),
    ("coqa",                  "f1,none",                     "f1_stderr,none",                     "CoQA (F1)",     "CogLang"),
    ("lambada_openai",        "acc,none",                    "acc_stderr,none",                    "LAMBADA (OAI)", "CogLang"),
    ("lambada_standard",      "acc,none",                    "acc_stderr,none",                    "LAMBADA (Std)", "CogLang"),
    ("acp_app_bool",          "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "App (B)",       "ACP"),
    ("acp_areach_bool",       "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Areach (B)",    "ACP"),
    ("acp_just_bool",         "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Just (B)",      "ACP"),
    ("acp_land_bool",         "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Land (B)",      "ACP"),
    ("acp_prog_bool",         "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Prog (B)",      "ACP"),
    ("acp_reach_bool",        "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Reach (B)",     "ACP"),
    ("acp_val_bool",          "exact_match,extract-yes-no",  "exact_match_stderr,extract-yes-no",  "Val (B)",       "ACP"),
    ("acp_app_mcq",           "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "App (M)",       "ACP"),
    ("acp_areach_mcq",        "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Areach (M)",    "ACP"),
    ("acp_just_mcq",          "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Just (M)",      "ACP"),
    ("acp_land_mcq",          "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Land (M)",      "ACP"),
    ("acp_prog_mcq",          "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Prog (M)",      "ACP"),
    ("acp_reach_mcq",         "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Reach (M)",     "ACP"),
    ("acp_val_mcq",           "exact_match,mcq-extract",     "exact_match_stderr,mcq-extract",     "Val (M)",       "ACP"),
]

EQBENCH_TASK = ("eq_bench", "eqbench,none", "eqbench_stderr,none", "EQ-Bench", "EQ-Bench")

COGSOC_GROUP_ORDER = ["Ethics", "CogLang", "ACP"]
COGSOC_GROUP_DISPLAY = {
    "Ethics":  "Ethics",
    "CogLang": "Cog. \\& Lang.",
    "ACP":     "ACP (Planning)",
}

# =============================================================================
# Data loading
# =============================================================================


def load_json(path):
    """Load a JSON file, return None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_metric(data, task_key, metric_key):
    """Extract a metric value from lm-eval JSON results."""
    if data is None:
        return None
    results = data.get("results", {})
    task_data = results.get(task_key, {})
    return task_data.get(metric_key, None)


def load_metabench():
    """Load all metabench results with stderr."""
    data = {}
    for _, base_stem, ft_stem in ALL_PAIRS:
        for stem in [base_stem, ft_stem]:
            path = os.path.join(METABENCH_DIR, f"{stem}_metabench.json")
            raw = load_json(path)
            vals = {}
            for bench_name, (task_key, metric_key, stderr_key) in METABENCH_TASKS.items():
                v = get_metric(raw, task_key, metric_key)
                se = get_metric(raw, task_key, stderr_key)
                vals[bench_name] = {"value": v, "stderr": se}
            data[stem] = vals
    return data


def load_cogsoc():
    """Load all cogsoc results with stderr."""
    data = {}
    for _, base_stem, ft_stem in ALL_PAIRS:
        for stem in [base_stem, ft_stem]:
            path = os.path.join(COGSOC_DIR, f"{stem}_cogsoc.json")
            raw = load_json(path)
            vals = {}
            for task_id, metric_key, stderr_key, _, _ in ALL_COGSOC_TASKS:
                v = get_metric(raw, task_id, metric_key)
                se = get_metric(raw, task_id, stderr_key)
                vals[task_id] = {"value": v, "stderr": se}
            eq_v = get_metric(raw, EQBENCH_TASK[0], EQBENCH_TASK[1])
            eq_se = get_metric(raw, EQBENCH_TASK[0], EQBENCH_TASK[2])
            vals[EQBENCH_TASK[0]] = {"value": eq_v, "stderr": eq_se}
            data[stem] = vals
    return data


# =============================================================================
# Z-test computation
# =============================================================================


def ztest(base_val, base_se, ft_val, ft_se):
    """Two-sided z-test for difference between base and ft."""
    if base_val is None or ft_val is None:
        return None, None, None, ""
    delta = ft_val - base_val
    if base_se is None or ft_se is None or base_se == 0 or ft_se == 0:
        return delta, None, None, ""
    pooled_se = math.sqrt(base_se**2 + ft_se**2)
    z = delta / pooled_se
    p = math.erfc(abs(z) / math.sqrt(2))
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return delta, z, p, stars


def combine_ztests(z_list):
    """Stouffer's method: combine z-scores into a single z and p."""
    valid = [z for z in z_list if z is not None]
    if not valid:
        return None, None, ""
    combined_z = sum(valid) / math.sqrt(len(valid))
    p = math.erfc(abs(combined_z) / math.sqrt(2))
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    return combined_z, p, stars


# =============================================================================
# LaTeX preamble and cell formatters
# =============================================================================

PREAMBLE = r"""% ============================================================================
% Required packages and colour definitions for significance tables
% ============================================================================
% \usepackage{xcolor}
% \usepackage{colortbl}
% \usepackage{graphicx}
% \usepackage{booktabs}

% ---- Diverging colours for significance tables ----
\definecolor{cpos}{HTML}{2ca02c}   % green = improvement
\definecolor{cneg}{HTML}{d62728}   % red = degradation

% ---- Cell formatter: \zc{colour!tint}{content} ----
\newcommand{\zc}[2]{\cellcolor{#1}#2}
\newcommand{\zno}{\cellcolor{white}{---}}
"""

SIG_LEGEND = (
    r"\vspace{2pt}"
    "\n"
    r"{\tiny\centering"
    "\n"
    r"degrad. \colorbox{cneg!50}{\strut\,}"
    r" \colorbox{cneg!20}{\strut\,}"
    r" \colorbox{white}{\strut\,}"
    r" \colorbox{cpos!20}{\strut\,}"
    r" \colorbox{cpos!50}{\strut\,} improv."
    " \\quad $^{*}p{<}.05$ ~ $^{**}p{<}.01$ ~ $^{***}p{<}.001$"
    r"\par}"
)


def _tint_level(stars):
    """Map significance level to colour tint percentage."""
    if stars == "***":
        return 50
    if stars == "**":
        return 35
    if stars == "*":
        return 22
    return 8  # light tint for non-significant


def fmt_delta(delta, stars, decimals=2, min_tint=8):
    """Format a delta cell with significance stars and diverging colour."""
    if delta is None:
        return "\\zno"
    rounded = round(delta, decimals)
    if rounded == 0:
        colour = "cpos"
        text = f"+{0:.{decimals}f}"
    elif delta >= 0:
        colour = "cpos"
        text = f"+{delta:.{decimals}f}"
    else:
        colour = "cneg"
        text = f"{delta:.{decimals}f}"
    tint = max(_tint_level(stars), min_tint)
    star_tex = f"$^{{{stars}}}$" if stars else ""
    return f"\\zc{{{colour}!{tint}}}{{{text}{star_tex}}}"


# =============================================================================
# Helper: compute mean delta + combined significance for a task list
# =============================================================================


def _mean_delta_cells(data, task_keys, ALL_PAIRS_ref, min_tint=8):
    """Compute mean delta and Stouffer combined z for a list of task keys.

    data: dict mapping stem -> {task_key: {value, stderr}}
    task_keys: list of task key strings
    Returns: list of formatted cells (one per model pair)
    """
    cells = []
    for _, base_stem, ft_stem in ALL_PAIRS_ref:
        deltas = []
        z_vals = []
        for tk in task_keys:
            b = data.get(base_stem, {}).get(tk, {})
            f = data.get(ft_stem, {}).get(tk, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            if delta is not None:
                deltas.append(delta)
            if z is not None:
                z_vals.append(z)
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        _, _, mean_stars = combine_ztests(z_vals)
        cells.append(fmt_delta(mean_delta, mean_stars, min_tint=min_tint))
    return cells


# =============================================================================
# Table 1: significance_full.tex — All tasks in one table
# =============================================================================


def generate_significance_full(mb_data, cs_data):
    """All tasks combined: MetaBench + Ethics + CogLang + ACP + EQ-Bench."""
    n_qt = len(QWEN_PAIRS)
    n_lc = len(LLAMA_PAIRS)
    n_all = n_qt + n_lc

    colspec = "@{}l" + " r" * n_qt + "  " + " r" * n_lc + "@{}"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Family headers
    first_qt = 2
    last_qt = first_qt + n_qt - 1
    first_lc = last_qt + 1
    last_lc = first_lc + n_lc - 1

    lines.append(
        f"& \\multicolumn{{{n_qt}}}{{c}}"
        f"{{\\textbf{{\\texttt{{Qwentaur}}}}}} "
        f"& \\multicolumn{{{n_lc}}}{{c}}"
        f"{{\\textbf{{\\texttt{{Llama-Centaur}}}}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{first_qt}-{last_qt}}}"
        f"\\cmidrule(lr){{{first_lc}-{last_lc}}}"
    )

    # Size headers
    h2 = "\\textbf{Benchmark}"
    for size, _, _ in ALL_PAIRS:
        h2 += f" & \\texttt{{{size}}}"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # --- MetaBench section ---
    lines.append(f"\\multicolumn{{{1 + n_all}}}{{l}}"
                 "{\\textit{MetaBench}} \\\\")
    for bench_name in METABENCH_ORDER:
        cells = []
        for _, base_stem, ft_stem in ALL_PAIRS:
            b = mb_data.get(base_stem, {}).get(bench_name, {})
            f = mb_data.get(ft_stem, {}).get(bench_name, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            cells.append(fmt_delta(delta, stars))
        lines.append(f"{bench_name} & " + " & ".join(cells) + " \\\\")

    # MetaBench mean
    mb_keys = list(METABENCH_ORDER)
    mb_mean_cells = _mean_delta_cells(mb_data, mb_keys, ALL_PAIRS)
    lines.append(
        f"\\textbf{{Mean ({len(mb_keys)})}} & "
        + " & ".join(mb_mean_cells) + " \\\\"
    )

    # --- Ethics section ---
    lines.append("\\midrule")
    ethics_tasks = [t for t in ALL_COGSOC_TASKS if t[4] == "Ethics"]
    lines.append(f"\\multicolumn{{{1 + n_all}}}{{l}}"
                 "{\\textit{Ethics}} \\\\")
    for task_id, _, stderr_key, display_name, _ in ethics_tasks:
        cells = []
        for _, base_stem, ft_stem in ALL_PAIRS:
            b = cs_data.get(base_stem, {}).get(task_id, {})
            f = cs_data.get(ft_stem, {}).get(task_id, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            cells.append(fmt_delta(delta, stars))
        lines.append(f"{display_name} & " + " & ".join(cells) + " \\\\")

    # Ethics mean
    ethics_keys = [t[0] for t in ethics_tasks]
    ethics_mean_cells = _mean_delta_cells(cs_data, ethics_keys, ALL_PAIRS)
    lines.append(
        f"\\textbf{{Mean ({len(ethics_keys)})}} & "
        + " & ".join(ethics_mean_cells) + " \\\\"
    )

    # --- CogLang section ---
    lines.append("\\midrule")
    coglang_tasks = [t for t in ALL_COGSOC_TASKS if t[4] == "CogLang"]
    lines.append(f"\\multicolumn{{{1 + n_all}}}{{l}}"
                 "{\\textit{Cog. \\& Lang.}} \\\\")
    for task_id, _, stderr_key, display_name, _ in coglang_tasks:
        cells = []
        for _, base_stem, ft_stem in ALL_PAIRS:
            b = cs_data.get(base_stem, {}).get(task_id, {})
            f = cs_data.get(ft_stem, {}).get(task_id, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            cells.append(fmt_delta(delta, stars))
        lines.append(f"{display_name} & " + " & ".join(cells) + " \\\\")

    # EQ-Bench
    eq_task_id = EQBENCH_TASK[0]
    eq_cells = []
    for _, base_stem, ft_stem in ALL_PAIRS:
        b = cs_data.get(base_stem, {}).get(eq_task_id, {})
        f = cs_data.get(ft_stem, {}).get(eq_task_id, {})
        delta, z, p, stars = ztest(
            b.get("value"), b.get("stderr"),
            f.get("value"), f.get("stderr"),
        )
        eq_cells.append(fmt_delta(delta, stars, decimals=1))
    lines.append(f"EQ-Bench & " + " & ".join(eq_cells) + " \\\\")

    # CogLang mean (excluding EQ-Bench)
    coglang_keys = [t[0] for t in coglang_tasks]
    coglang_mean_cells = _mean_delta_cells(cs_data, coglang_keys, ALL_PAIRS)
    lines.append(
        f"\\textbf{{Mean ({len(coglang_keys)})}} & "
        + " & ".join(coglang_mean_cells) + " \\\\"
    )

    # --- ACP section ---
    lines.append("\\midrule")
    acp_tasks = [t for t in ALL_COGSOC_TASKS if t[4] == "ACP"]
    lines.append(f"\\multicolumn{{{1 + n_all}}}{{l}}"
                 "{\\textit{ACP (Planning)}} \\\\")
    for task_id, _, stderr_key, display_name, _ in acp_tasks:
        cells = []
        for _, base_stem, ft_stem in ALL_PAIRS:
            b = cs_data.get(base_stem, {}).get(task_id, {})
            f = cs_data.get(ft_stem, {}).get(task_id, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            cells.append(fmt_delta(delta, stars))
        lines.append(f"{display_name} & " + " & ".join(cells) + " \\\\")

    # ACP mean
    acp_keys = [t[0] for t in acp_tasks]
    acp_mean_cells = _mean_delta_cells(cs_data, acp_keys, ALL_PAIRS)
    n_acp = len(acp_keys)
    lines.append(
        f"\\textbf{{Mean ({n_acp})}} & "
        + " & ".join(acp_mean_cells) + " \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(SIG_LEGEND)
    lines.append(
        "\\caption{Impact of cognitive fine-tuning across all benchmarks. "
        "Each cell shows $\\Delta$ = fine-tuned $-$ base. "
        "Significance is assessed with a two-sided $z$-test using the "
        "standard errors reported by \\texttt{lm-eval}; "
        "group means use Stouffer's method to combine per-task $z$-scores. "
        "EQ-Bench is on a separate scale and excluded from means.}"
    )
    lines.append("\\label{tab:significance_full}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Table 2: combined_summary.tex — Combined main-text table
# =============================================================================


def generate_combined_summary(mb_data, cs_data):
    """Summary: MetaBench benchmarks + Ethics/CogLang groups + EQ-Bench."""
    n_qt = len(QWEN_PAIRS)
    n_lc = len(LLAMA_PAIRS)
    n_all = n_qt + n_lc

    group_tasks = {}
    for group in COGSOC_GROUP_ORDER:
        group_tasks[group] = [
            t for t in ALL_COGSOC_TASKS if t[4] == group
        ]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")

    colspec = "@{}l" + " r" * n_qt + "  " + " r" * n_lc + "@{}"
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Family headers
    first_qt = 2
    last_qt = first_qt + n_qt - 1
    first_lc = last_qt + 1
    last_lc = first_lc + n_lc - 1

    lines.append(
        f"& \\multicolumn{{{n_qt}}}{{c}}"
        f"{{\\textbf{{\\texttt{{Qwentaur}}}}}} "
        f"& \\multicolumn{{{n_lc}}}{{c}}"
        f"{{\\textbf{{\\texttt{{Llama-Centaur}}}}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{first_qt}-{last_qt}}}"
        f"\\cmidrule(lr){{{first_lc}-{last_lc}}}"
    )

    # Size headers
    h2 = "\\textbf{Benchmark}"
    for size, _, _ in ALL_PAIRS:
        h2 += f" & \\texttt{{{size}}}"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # --- MetaBench benchmarks (no section header) ---
    S_TINT = 15  # stronger minimum tint for summary table
    for bench_name in METABENCH_ORDER:
        cells = []
        for _, base_stem, ft_stem in ALL_PAIRS:
            b = mb_data.get(base_stem, {}).get(bench_name, {})
            f = mb_data.get(ft_stem, {}).get(bench_name, {})
            delta, z, p, stars = ztest(
                b.get("value"), b.get("stderr"),
                f.get("value"), f.get("stderr"),
            )
            cells.append(fmt_delta(delta, stars, min_tint=S_TINT))
        lines.append(f"{bench_name} & " + " & ".join(cells) + " \\\\")

    # MetaBench mean
    mb_keys = list(METABENCH_ORDER)
    mb_mean_cells = _mean_delta_cells(mb_data, mb_keys, ALL_PAIRS, min_tint=S_TINT)
    n_mb = len(mb_keys)
    lines.append(
        f"\\textbf{{Mean ({n_mb})}} & "
        + " & ".join(mb_mean_cells) + " \\\\"
    )

    # --- Ethics ---
    lines.append("\\midrule")
    ethics_tasks_s = group_tasks["Ethics"]
    ethics_keys_s = [t[0] for t in ethics_tasks_s]
    ethics_mean_cells_s = _mean_delta_cells(cs_data, ethics_keys_s, ALL_PAIRS, min_tint=S_TINT)
    lines.append(
        f"Ethics ({len(ethics_keys_s)}) & "
        + " & ".join(ethics_mean_cells_s) + " \\\\"
    )

    # --- Cog. & Lang. ---
    coglang_tasks_s = group_tasks["CogLang"]
    coglang_keys_s = [t[0] for t in coglang_tasks_s]
    coglang_mean_cells_s = _mean_delta_cells(cs_data, coglang_keys_s, ALL_PAIRS, min_tint=S_TINT)
    lines.append(
        f"Cog. \\& Lang. ({len(coglang_keys_s)}) & "
        + " & ".join(coglang_mean_cells_s) + " \\\\"
    )

    # --- EQ-Bench ---
    eq_task_id = EQBENCH_TASK[0]
    eq_cells = []
    for _, base_stem, ft_stem in ALL_PAIRS:
        b = cs_data.get(base_stem, {}).get(eq_task_id, {})
        f = cs_data.get(ft_stem, {}).get(eq_task_id, {})
        delta, z, p, stars = ztest(
            b.get("value"), b.get("stderr"),
            f.get("value"), f.get("stderr"),
        )
        eq_cells.append(fmt_delta(delta, stars, decimals=1, min_tint=S_TINT))
    lines.append(f"EQ-Bench & " + " & ".join(eq_cells) + " \\\\")

    # Overall mean of Ethics + CogLang
    all_ec_keys = ethics_keys_s + coglang_keys_s
    ec_mean_cells = _mean_delta_cells(cs_data, all_ec_keys, ALL_PAIRS, min_tint=S_TINT)
    n_ec = len(all_ec_keys)
    lines.append(
        f"\\textbf{{Mean ({n_ec})}} & "
        + " & ".join(ec_mean_cells) + " \\\\"
    )

    # --- ACP (Bool) and ACP (MCQ) aggregated rows + ACP mean ---
    lines.append("\\midrule")
    acp_tasks_s = group_tasks["ACP"]
    acp_bool_keys = [t[0] for t in acp_tasks_s if t[0].endswith("_bool")]
    acp_mcq_keys = [t[0] for t in acp_tasks_s if t[0].endswith("_mcq")]
    acp_bool_cells = _mean_delta_cells(cs_data, acp_bool_keys, ALL_PAIRS, min_tint=S_TINT)
    acp_mcq_cells = _mean_delta_cells(cs_data, acp_mcq_keys, ALL_PAIRS, min_tint=S_TINT)
    lines.append(
        f"ACP -- Bool ({len(acp_bool_keys)}) & "
        + " & ".join(acp_bool_cells) + " \\\\"
    )
    lines.append(
        f"ACP -- MCQ ({len(acp_mcq_keys)}) & "
        + " & ".join(acp_mcq_cells) + " \\\\"
    )
    acp_all_keys = [t[0] for t in acp_tasks_s]
    acp_mean_cells = _mean_delta_cells(cs_data, acp_all_keys, ALL_PAIRS, min_tint=S_TINT)
    lines.append(
        f"\\textbf{{Mean ({len(acp_all_keys)})}} & "
        + " & ".join(acp_mean_cells) + " \\\\"
    )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(SIG_LEGEND)
    lines.append(
        "\\caption{Impact of cognitive fine-tuning on standard, cognitive, and planning benchmarks. "
        "Each cell shows $\\Delta$ = fine-tuned $-$ base. "
        "Significance is assessed with a two-sided $z$-test using the "
        "standard errors reported by \\texttt{lm-eval}; "
        "group means use Stouffer's method to combine per-task $z$-scores. "
        "EQ-Bench is on a separate scale and excluded from means.}"
    )
    lines.append("\\label{tab:combined_summary}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading metabench data...")
    mb_data = load_metabench()
    n_mb = sum(
        1 for d in mb_data.values()
        if any(v.get("value") is not None for v in d.values())
    )
    print(f"  {n_mb} models with data")

    print("Loading cogsoc data...")
    cs_data = load_cogsoc()
    n_cs = sum(
        1 for d in cs_data.values()
        if any(v.get("value") is not None for v in d.values())
    )
    print(f"  {n_cs} models with data")

    print("\nGenerating tables...")

    # 1. Full significance table (all tasks combined)
    tex = generate_significance_full(mb_data, cs_data)
    path = os.path.join(OUT_DIR, "significance_full.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    # 2. Combined summary (main text)
    tex = generate_combined_summary(mb_data, cs_data)
    path = os.path.join(OUT_DIR, "combined_summary.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    print(f"\nDone. Two tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
