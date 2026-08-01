# -*- coding: utf-8 -*-
"""
Psych-201 Out-of-Distribution Evaluation: LaTeX Heatmap Tables
================================================================

Generates two LaTeX heatmap tables from per-model CSV results:

  Main-text (compact, portrait):
    1. psych201_main_normalised.tex  — (ln(k)-NLL)/ln(k) by task type, 15 discrete tasks

  Appendix (full, sideways):
    2. psych201_appendix_raw.tex     — raw NLL, all 18 experiments × 16 models

Usage:
    cd "results/Psych201-RT (NLL)"
    python generate_heatmap_tables.py

Reads:  results/*.csv  (per-model task-level NLL)
Writes: tables/*.tex
"""

import csv
import math
import os
import re
import glob
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..")
OUT_DIR = SCRIPT_DIR

# =============================================================================
# Model metadata
# =============================================================================


def get_model_family(name):
    n = name.lower()
    if 'llama' in n or 'centaur' in n:
        return 'llama'
    return 'qwen'


def is_finetuned(name):
    n = name.lower()
    return 'centaur' in n or 'qwentaur' in n or 'lora' in n


def model_size(name):
    m = re.search(r'(\d+\.?\d*)[Bb]', name)
    return float(m.group(1)) if m else 0


def display_name(name):
    return (name
            .replace('socius-', '')
            .replace('unsloth-', '')
            .replace('-LoRA', '')
            .replace('-Base', '')
            .replace('Meta-', ''))


def model_sort_key(name):
    return (get_model_family(name), 0 if is_finetuned(name) else 1,
            model_size(name))


def short_name(name):
    """Abbreviated name for column headers."""
    return (display_name(name)
            .replace('Llama-Centaur-', 'LC-')
            .replace('Qwentaur-', 'QT-')
            .replace('Llama-3.2-', 'L-')
            .replace('Llama-3.1-', 'L-')
            .replace('Qwen3-', 'Q-'))


# =============================================================================
# Experiment metadata
# =============================================================================

CITE_MAP = {
    "anllo2024weird":                  ("Experience-description choice",  "anllo2024weird"),
    "bavard2021range":                 ("Range adaptation RL",            "bavard2021range"),
    "busch2024navon":                  ("Navon task",                     "busch2024navon"),
    "busch2024stroop":                 ("Stroop task",                    "busch2024stroop"),
    "castrorodrigues2022twostep":      ("Two-step task",                  "castrorodrigues2022twostep"),
    "fan2022trait":                    ("Explore-exploit task",           "fan2022trait"),
    "franke2024bayesian":              ("Pragmatic language task",        "franke2024bayesian"),
    "frankedegen2016reasoning":        ("Reference game",                "frankedegen2016reasoning"),
    "guenther2020ts":                  ("Compound word processing",      "guenther2020ts"),
    "guenther2023grammaticality":      ("Grammaticality judgement",       "guenther2023grammaticality"),
    "palminteri2017confirmation":      ("Counterfactual learning",        "palminteri2017confirmation"),
    "rutledge2023happiness":           ("Risky decision and happiness",   "rutledge2023happiness"),
    "shahar2019twosteptask":           ("Two-step task",                  "shahar2019twosteptask"),
    "spektor2024lossaversion":         ("Loss aversion task",             "spektor2024lossaversion"),
    "tsvilodub2023xorsome":            ("Scalar inference",               "tsvilodub2023xorsome"),
    "vandendriessche2022depression":   ("Contextual reinforcement learning", "vandendriessche2022depression"),
    "xu2023augmenting":                ("Math arithmetic task",           "xu2023augmenting"),
    "zika2023traitanxiety":            ("Aversive reversal learning",     "zika2023traitanxiety"),
}

# Task type per experiment
TASK_TYPE = {
    "anllo2024weird":                  "Decision-making",
    "bavard2021range":                 "Multi-armed bandits",
    "busch2024navon":                  "Miscellaneous",
    "busch2024stroop":                 "Miscellaneous",
    "castrorodrigues2022twostep":      "Markov decision processes",
    "fan2022trait":                    "Multi-armed bandits",
    "franke2024bayesian":              "Miscellaneous",
    "frankedegen2016reasoning":        "Miscellaneous",
    "guenther2020ts":                  "Miscellaneous",
    "guenther2023grammaticality":      "Miscellaneous",
    "palminteri2017confirmation":      "Multi-armed bandits",
    "rutledge2023happiness":           "Decision-making",
    "shahar2019twosteptask":           "Markov decision processes",
    "spektor2024lossaversion":         "Decision-making",
    "tsvilodub2023xorsome":            "Miscellaneous",
    "vandendriessche2022depression":   "Multi-armed bandits",
    "xu2023augmenting":                "Miscellaneous",
    "zika2023traitanxiety":            "Multi-armed bandits",
}

TASK_ABBREV = {
    "Decision-making":          "Decision",
    "Markov decision processes": "MDP",
    "Multi-armed bandits":       "Bandit",
    "Miscellaneous":             "Misc.",
}

TASK_ORDER = [
    "Decision-making", "Markov decision processes",
    "Multi-armed bandits", "Miscellaneous",
]

_ln = math.log
LN_K = {
    "anllo2024weird":                  _ln(2),
    "bavard2021range":                 _ln(2),
    "busch2024navon":                  _ln(2),
    "busch2024stroop":                 _ln(3),
    "castrorodrigues2022twostep":      _ln(4),
    "fan2022trait":                    _ln(2),
    "franke2024bayesian":              _ln(4),
    "frankedegen2016reasoning":        _ln(4),
    "guenther2020ts":                  _ln(2),
    "guenther2023grammaticality":      _ln(2),
    "palminteri2017confirmation":      _ln(2),
    "rutledge2023happiness":           None,      # mixed: binary lottery + free rating
    "shahar2019twosteptask":           _ln(2),
    "spektor2024lossaversion":         _ln(2),
    "tsvilodub2023xorsome":            None,      # continuous slider 0-100
    "vandendriessche2022depression":   _ln(2),
    "xu2023augmenting":                _ln(2),
    "zika2023traitanxiety":            None,      # continuous probability
}

EXPERIMENT_ORDER = sorted(LN_K.keys())
DISCRETE_EXPS = sorted(e for e, v in LN_K.items() if v is not None and v > 0)


# =============================================================================
# Data loading
# =============================================================================


def load_all_results():
    """Load all per-model CSV files into a dict: {model: {task: loss}}."""
    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    data = {}
    for f in files:
        model = os.path.basename(f).replace('.csv', '')
        task_loss = {}
        with open(f, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                task_loss[row[0].strip()] = float(row[1])
        data[model] = task_loss
    return data


# =============================================================================
# Helpers
# =============================================================================


def mean_of(vals):
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def find_best_and_second(values):
    """Best = lowest."""
    numeric = sorted(set(round(v, 6) for v in values if v is not None))
    best = numeric[0] if len(numeric) >= 1 else None
    second = numeric[1] if len(numeric) >= 2 else None
    return best, second


def find_best_and_second_high(values):
    """Best = highest."""
    numeric = sorted(set(round(v, 6) for v in values if v is not None),
                     reverse=True)
    best = numeric[0] if len(numeric) >= 1 else None
    second = numeric[1] if len(numeric) >= 2 else None
    return best, second


def normalise(nll, lnk):
    """(ln(k) - NLL) / ln(k): fraction of information above chance."""
    if nll is None or lnk is None or lnk <= 0:
        return None
    return (lnk - nll) / lnk


# =============================================================================
# LaTeX preamble and cell formatters
# =============================================================================

_COLORBLOCK = r"""  \ifdim #1 pt < 0.20pt \cellcolor{c1!45}%
  \else\ifdim #1 pt < 0.40pt \cellcolor{c2!40}%
  \else\ifdim #1 pt < 0.60pt \cellcolor{c3!40}%
  \else\ifdim #1 pt < 0.80pt \cellcolor{c4!45}%
  \else\ifdim #1 pt < 1.00pt \cellcolor{c5!55}%
  \else\ifdim #1 pt < 1.20pt \cellcolor{c6!50}%
  \else\ifdim #1 pt < 1.50pt \cellcolor{c7!45}%
  \else\ifdim #1 pt < 2.00pt \cellcolor{c8!40}%
  \else \cellcolor{c9!35}%
  \fi\fi\fi\fi\fi\fi\fi\fi"""

PREAMBLE = r"""% ============================================================================
% Required packages and colour definitions for heatmap tables
% ============================================================================
% \usepackage{xcolor}
% \usepackage{colortbl}
% \usepackage{graphicx}
% \usepackage{booktabs}
% \usepackage{rotating}

% ---- Heatmap colour bins (green=good to red=bad) ----
\definecolor{c1}{HTML}{1a9850}
\definecolor{c2}{HTML}{66bd63}
\definecolor{c3}{HTML}{a6d96a}
\definecolor{c4}{HTML}{d9ef8b}
\definecolor{c5}{HTML}{fee08b}
\definecolor{c6}{HTML}{fdae61}
\definecolor{c7}{HTML}{f46d43}
\definecolor{c8}{HTML}{d73027}
\definecolor{c9}{HTML}{a50026}

% ---- Normal cell ----
\newcommand{\hc}[1]{%
""" + _COLORBLOCK + r"""
  #1%
}

% ---- Underline = 2nd best ----
\newcommand{\hcu}[1]{%
""" + _COLORBLOCK + r"""
  \underline{#1}%
}

% ---- Bold + underline = best ----
\newcommand{\hcbu}[1]{%
""" + _COLORBLOCK + r"""
  \underline{\textbf{#1}}%
}

% ---- No baseline ----
\newcommand{\hcn}{\cellcolor{white}{---}}

% ---- Reversed heatmap (higher = better = green) for normalised metric ----
\newcommand{\hcR}[1]{%
  \ifdim #1 pt < 0.05pt \cellcolor{c9!35}%
  \else\ifdim #1 pt < 0.12pt \cellcolor{c8!40}%
  \else\ifdim #1 pt < 0.18pt \cellcolor{c7!45}%
  \else\ifdim #1 pt < 0.25pt \cellcolor{c6!50}%
  \else\ifdim #1 pt < 0.32pt \cellcolor{c5!55}%
  \else\ifdim #1 pt < 0.40pt \cellcolor{c4!45}%
  \else\ifdim #1 pt < 0.48pt \cellcolor{c3!40}%
  \else\ifdim #1 pt < 0.55pt \cellcolor{c2!40}%
  \else \cellcolor{c1!45}%
  \fi\fi\fi\fi\fi\fi\fi\fi
  #1%
}
\newcommand{\hcRu}[1]{%
  \ifdim #1 pt < 0.05pt \cellcolor{c9!35}%
  \else\ifdim #1 pt < 0.12pt \cellcolor{c8!40}%
  \else\ifdim #1 pt < 0.18pt \cellcolor{c7!45}%
  \else\ifdim #1 pt < 0.25pt \cellcolor{c6!50}%
  \else\ifdim #1 pt < 0.32pt \cellcolor{c5!55}%
  \else\ifdim #1 pt < 0.40pt \cellcolor{c4!45}%
  \else\ifdim #1 pt < 0.48pt \cellcolor{c3!40}%
  \else\ifdim #1 pt < 0.55pt \cellcolor{c2!40}%
  \else \cellcolor{c1!45}%
  \fi\fi\fi\fi\fi\fi\fi\fi
  \underline{#1}%
}
\newcommand{\hcRbu}[1]{%
  \ifdim #1 pt < 0.05pt \cellcolor{c9!35}%
  \else\ifdim #1 pt < 0.12pt \cellcolor{c8!40}%
  \else\ifdim #1 pt < 0.18pt \cellcolor{c7!45}%
  \else\ifdim #1 pt < 0.25pt \cellcolor{c6!50}%
  \else\ifdim #1 pt < 0.32pt \cellcolor{c5!55}%
  \else\ifdim #1 pt < 0.40pt \cellcolor{c4!45}%
  \else\ifdim #1 pt < 0.48pt \cellcolor{c3!40}%
  \else\ifdim #1 pt < 0.55pt \cellcolor{c2!40}%
  \else \cellcolor{c1!45}%
  \fi\fi\fi\fi\fi\fi\fi\fi
  \underline{\textbf{#1}}%
}
"""

LEGEND_COMPACT = r"""\vspace{2pt}
{\tiny\centering
\colorbox{c1!45}{\strut\,} $<$0.20 ~
\colorbox{c2!40}{\strut\,} 0.20--0.40 ~
\colorbox{c3!40}{\strut\,} 0.40--0.60 ~
\colorbox{c4!45}{\strut\,} 0.60--0.80 ~
\colorbox{c5!55}{\strut\,} 0.80--1.00 ~
\colorbox{c6!50}{\strut\,} $>$1.00\par}"""

LEGEND_FULL = r"""\vspace{4pt}
\begin{center}\footnotesize
\colorbox{c1!45}{\strut\hspace{5pt}} \scriptsize$<$0.20 \quad
\colorbox{c2!40}{\strut\hspace{5pt}} \scriptsize 0.20--0.40 \quad
\colorbox{c3!40}{\strut\hspace{5pt}} \scriptsize 0.40--0.60 \quad
\colorbox{c4!45}{\strut\hspace{5pt}} \scriptsize 0.60--0.80 \quad
\colorbox{c5!55}{\strut\hspace{5pt}} \scriptsize 0.80--1.00 \quad
\colorbox{c6!50}{\strut\hspace{5pt}} \scriptsize 1.00--1.20 \quad
\colorbox{c7!45}{\strut\hspace{5pt}} \scriptsize 1.20--1.50 \quad
\colorbox{c8!40}{\strut\hspace{5pt}} \scriptsize 1.50--2.00 \quad
\colorbox{c9!35}{\strut\hspace{5pt}} \scriptsize$>$2.00
\end{center}"""

LEGEND_REVERSED_COMPACT = r"""\vspace{2pt}
{\tiny\centering
\colorbox{c9!35}{\strut\,} $<$0.05 ~
\colorbox{c8!40}{\strut\,} 0.05--0.12 ~
\colorbox{c7!45}{\strut\,} 0.12--0.18 ~
\colorbox{c6!50}{\strut\,} 0.18--0.25 ~
\colorbox{c5!55}{\strut\,} 0.25--0.32 ~
\colorbox{c4!45}{\strut\,} 0.32--0.40 ~
\colorbox{c3!40}{\strut\,} 0.40--0.48 ~
\colorbox{c2!40}{\strut\,} 0.48--0.55 ~
\colorbox{c1!45}{\strut\,} $\geq$0.55\par}"""


def fmt_hc(val, rank=None):
    if val is None:
        return "\\hcn"
    if rank == 1:
        return f"\\hcbu{{{val:.2f}}}"
    if rank == 2:
        return f"\\hcu{{{val:.2f}}}"
    return f"\\hc{{{val:.2f}}}"


def fmt_lnk(exp_code):
    val = LN_K.get(exp_code)
    if val is None:
        return "\\hcn"
    return f"\\hc{{{val:.2f}}}"


def fmt_norm(val, rank=None):
    if val is None:
        return "\\hcn"
    if rank == 1:
        return f"\\hcRbu{{{val:.2f}}}"
    if rank == 2:
        return f"\\hcRu{{{val:.2f}}}"
    return f"\\hcR{{{val:.2f}}}"


# =============================================================================
# Model column specifications
# =============================================================================

# Appendix: base|ft paired columns per size, grouped by family
QWEN_PAIRS = [
    ("0.6B", "unsloth-Qwen3-0.6B-Base",   "socius-Qwentaur-0.6B-LoRA"),
    ("1.7B", "unsloth-Qwen3-1.7B-Base",   "socius-Qwentaur-1.7B-LoRA"),
    ("4B",   "unsloth-Qwen3-4B-Base",     "socius-Qwentaur-4B-LoRA"),
    ("8B",   "unsloth-Qwen3-8B-Base",     "socius-Qwentaur-8B-LoRA"),
    ("14B",  "unsloth-Qwen3-14B-Base",    "socius-Qwentaur-14B-LoRA"),
]
LLAMA_PAIRS = [
    ("1B", "unsloth-Llama-3.2-1B",       "socius-Llama-Centaur-1B-LoRA"),
    ("3B", "unsloth-Llama-3.2-3B",       "socius-Llama-Centaur-3B-LoRA"),
    ("8B", "unsloth-Meta-Llama-3.1-8B",  "socius-Llama-Centaur-8B-LoRA"),
]
BINZ_MODEL = "marcelbinz-Llama-3.1-Centaur-70B-adapter"

# Main-text column specs: (label, model_key)
MAIN_QT_COLS = [
    ("\\texttt{0.6B}", "socius-Qwentaur-0.6B-LoRA"),
    ("\\texttt{1.7B}", "socius-Qwentaur-1.7B-LoRA"),
    ("\\texttt{4B}",   "socius-Qwentaur-4B-LoRA"),
    ("\\texttt{8B}",   "socius-Qwentaur-8B-LoRA"),
    ("\\texttt{14B}",  "socius-Qwentaur-14B-LoRA"),
]
MAIN_LC_COLS = [
    ("\\texttt{1B}",   "socius-Llama-Centaur-1B-LoRA"),
    ("\\texttt{3B}",   "socius-Llama-Centaur-3B-LoRA"),
    ("\\texttt{8B}",   "socius-Llama-Centaur-8B-LoRA"),
]
MAIN_SMOL_COLS = [
    ("\\texttt{0.1B}", "socius-Smoltaur-0.1B-LoRA-r16"),
    ("\\texttt{0.4B}", "socius-Smoltaur-0.4B-LoRA-r16"),
    ("\\texttt{1.7B}", "socius-Smoltaur-1.7B-LoRA-r16"),
    ("\\texttt{3B}",   "socius-Smoltaur-3B-LoRA-r16"),
]
MAIN_OLMO_COLS = [
    ("\\texttt{1B}", "socius-Olmotaur-1B-LoRA-r16"),
    ("\\texttt{7B}", "socius-Olmotaur-7B-LoRA-r16"),
]
MAIN_BASE_COLS = [
    ("\\texttt{Q-8B}",   "unsloth-Qwen3-8B-Base"),
    ("\\texttt{Q-14B}",  "unsloth-Qwen3-14B-Base"),
    ("\\texttt{L-8B}",   "unsloth-Meta-Llama-3.1-8B"),
]
MAIN_REF_COLS = [
    ("\\texttt{Centaur-70B}", BINZ_MODEL),
]


# =============================================================================
# Main-text Table: Normalised task-type summary
# =============================================================================


def generate_main_normalised(data):
    """Compact portrait table: normalised metric by task type.

    Mirrors Psych-101 main_tasktype_normalised.tex.
    Column groups: Qwentaur (bf16) | Llama-Centaur (bf16) | Base (bf16) | Binz et al.
    Filters to the 15 discrete experiments.
    """
    fam_groups = [
        ("\\textbf{\\texttt{Qwentaur}}", MAIN_QT_COLS),
        ("\\textbf{\\texttt{Llama-Centaur}}", MAIN_LC_COLS),
        ("\\textbf{\\texttt{Smoltaur}}", MAIN_SMOL_COLS),
        ("\\textbf{\\texttt{Olmotaur}}", MAIN_OLMO_COLS),
    ]
    all_cols = ([c for _, cols in fam_groups for c in cols]
                + MAIN_BASE_COLS + MAIN_REF_COLS)
    n_base = len(MAIN_BASE_COLS)
    n_ref = len(MAIN_REF_COLS)

    # Group discrete experiments by task type
    type_groups = defaultdict(list)
    for exp in DISCRETE_EXPS:
        type_groups[TASK_TYPE[exp]].append(exp)

    # Mean normalised value per task type per model
    task_means = {}
    for task_type in TASK_ORDER:
        exps = type_groups.get(task_type, [])
        means = {}
        for _, m in all_cols:
            vals = []
            for exp in exps:
                n = normalise(data.get(m, {}).get(exp), LN_K[exp])
                if n is not None:
                    vals.append(n)
            means[m] = sum(vals) / len(vals) if vals else None
        task_means[task_type] = means

    # Overall mean
    overall = {}
    for _, m in all_cols:
        vals = []
        for exp in DISCRETE_EXPS:
            n = normalise(data.get(m, {}).get(exp), LN_K[exp])
            if n is not None:
                vals.append(n)
        overall[m] = sum(vals) / len(vals) if vals else None

    # --- Build table ---
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")

    group_sizes = [len(cols) for _, cols in fam_groups] + [n_base, n_ref]
    colspec = "@{}l"
    for gs in group_sizes:
        colspec += "  " + " r" * gs
    colspec += "@{}"
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups + Base + Binz reference
    h1, col, cmids = "", 2, []
    for hdr, cols in fam_groups:
        n = len(cols)
        h1 += f"& \\multicolumn{{{n}}}{{c}}{{{hdr}}} "
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    h1 += f"& \\multicolumn{{{n_base}}}{{c}}{{\\textbf{{Base}}}} "
    cmids.append(f"\\cmidrule(lr){{{col}-{col + n_base - 1}}}")
    col += n_base
    h1 += f"& \\multicolumn{{{n_ref}}}{{c}}{{\\citet{{binz2025foundation}}}} \\\\"
    cmids.append(f"\\cmidrule(lr){{{col}-{col + n_ref - 1}}}")
    lines.append(h1)
    lines.append("".join(cmids))

    h2 = "\\textbf{Task type}"
    for label, _ in all_cols:
        h2 += f" & {label}"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # Task-type rows
    for task_type in TASK_ORDER:
        exps = type_groups.get(task_type, [])
        n_exps = len(exps)
        if n_exps == 0:
            continue
        label = TASK_ABBREV.get(task_type, task_type)
        label_with_n = f"{label} ({n_exps})"
        means = task_means[task_type]

        eligible = [means[m] for _, m in all_cols if means[m] is not None]
        best, second = find_best_and_second_high(eligible)

        cells = []
        for _, m in all_cols:
            v = means[m]
            if v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-4:
                rank = 1
            elif second is not None and abs(v - second) < 1e-4:
                rank = 2
            else:
                rank = None
            cells.append(fmt_norm(v, rank=rank))

        lines.append(f"{label_with_n} & " + " & ".join(cells) + " \\\\")

    # Mean row
    lines.append("\\midrule")
    eligible = [overall[m] for _, m in all_cols if overall[m] is not None]
    best, second = find_best_and_second_high(eligible)
    cells = []
    for _, m in all_cols:
        v = overall[m]
        if v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-4:
            rank = 1
        elif second is not None and abs(v - second) < 1e-4:
            rank = 2
        else:
            rank = None
        cells.append(fmt_norm(v, rank=rank))

    n_disc = len(DISCRETE_EXPS)
    lines.append(f"\\textbf{{Mean ({n_disc})}} & "
                 + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND_REVERSED_COMPACT)

    n_excluded = len(EXPERIMENT_ORDER) - n_disc
    lines.append(
        "\\caption{Fraction of available information captured above chance, "
        "$(\\ln k - \\mathrm{NLL}) \\,/\\, \\ln k$, "
        "by task type on Psych-201 (out-of-distribution). "
        "A value of 0 indicates chance-level performance; "
        "1 indicates perfect prediction. "
        f"Restricted to {n_disc} experiments (of {len(EXPERIMENT_ORDER)}) "
        "with a well-defined discrete response space ($\\ln k > 0$); "
        f"{n_excluded} experiments with continuous or mixed responses "
        "are excluded. "
        "\\underline{\\textbf{Bold+underline}} marks the best model, "
        "\\underline{underline} the second-best.}"
    )
    lines.append("\\label{tab:psych201_normalised}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Appendix Table: Raw NLL with base|ft paired columns by family
# =============================================================================


def generate_appendix_raw(data):
    """Full sideways table: raw NLL, all 18 experiments.

    Column groups: Qwen3 family (base|ft pairs) | Llama family (base|ft pairs)
                 | Binz et al. (2025) | Chance
    """
    all_pairs = QWEN_PAIRS + LLAMA_PAIRS
    n_qwen = len(QWEN_PAIRS)
    n_llama = len(LLAMA_PAIRS)
    n_qwen_cols = n_qwen * 2
    n_llama_cols = n_llama * 2
    n_ref = 1  # Binz C-70B

    # Column spec: paired base|ft columns
    colspec = "@{}l l"
    for _ in all_pairs:
        colspec += "  r@{\\;\\;}r"
    colspec += "  r"   # Binz
    colspec += "  r"   # ln(k)
    colspec += "@{}"

    lines = []
    lines.append("\\begin{sidewaystable}[p]")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # --- Header row 1: family groups ---
    first_qwen = 3
    last_qwen = first_qwen + n_qwen_cols - 1
    first_llama = last_qwen + 1
    last_llama = first_llama + n_llama_cols - 1
    col_binz = last_llama + 1

    lines.append(
        f"& & \\multicolumn{{{n_qwen_cols}}}{{c}}"
        f"{{\\texttt{{\\textbf{{Qwen3}}}} family}} "
        f"& \\multicolumn{{{n_llama_cols}}}{{c}}"
        f"{{\\texttt{{\\textbf{{Llama-3.1/3.2}}}} family}} "
        f"& \\multicolumn{{{n_ref}}}{{c}}"
        f"{{\\citet{{binz2025foundation}}}} "
        f"& \\textbf{{Chance}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{first_qwen}-{last_qwen}}}"
        f"\\cmidrule(lr){{{first_llama}-{last_llama}}}"
    )

    # --- Header row 2: size groups ---
    h2 = "& "
    for label, _, _ in all_pairs:
        h2 += f"& \\multicolumn{{2}}{{c}}{{{label}}} "
    h2 += "& & \\\\"
    lines.append(h2)

    # Cmidrules under each size pair
    rules = []
    col = first_qwen
    for _ in all_pairs:
        rules.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    lines.append(" ".join(rules))

    # --- Header row 3: base/ft labels ---
    h3 = "\\textbf{Experiment} & \\textbf{Type}"
    for _ in all_pairs:
        h3 += (" & {\\fontsize{4}{5}\\selectfont base}"
               " & {\\fontsize{4}{5}\\selectfont ft}")
    h3 += " & 70B"
    h3 += " & $\\ln(k)$"
    h3 += " \\\\"
    lines.append(h3)
    lines.append("\\midrule")

    # --- Data rows ---
    type_groups = defaultdict(list)
    for exp in EXPERIMENT_ORDER:
        type_groups[TASK_TYPE[exp]].append(exp)

    prev_type = None
    for task_type in TASK_ORDER:
        exps_in_type = type_groups.get(task_type, [])
        if not exps_in_type:
            continue
        if prev_type is not None:
            lines.append("\\midrule")
        prev_type = task_type

        for exp in exps_in_type:
            display, cite = CITE_MAP.get(exp, (exp, exp))
            cell_exp = (f"{display} "
                        f"{{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}")
            cell_type = TASK_ABBREV.get(task_type, task_type)

            # Collect all values for this row
            vals = []  # list of float|None
            for _, base_key, ft_key in all_pairs:
                vals.append(data.get(base_key, {}).get(exp))
                vals.append(data.get(ft_key, {}).get(exp))
            vals.append(data.get(BINZ_MODEL, {}).get(exp))

            eligible = [v for v in vals if v is not None]
            best, second = find_best_and_second(eligible)

            cells = []
            for v in vals:
                if v is None:
                    rank = None
                elif best is not None and abs(v - best) < 1e-6:
                    rank = 1
                elif second is not None and abs(v - second) < 1e-6:
                    rank = 2
                else:
                    rank = None
                cells.append(fmt_hc(v, rank=rank))

            lines.append(f"{cell_exp} & {cell_type} & "
                         + " & ".join(cells)
                         + f" & {fmt_lnk(exp)}" + " \\\\")

    # --- Mean row ---
    lines.append("\\midrule")
    mean_vals = []
    for _, base_key, ft_key in all_pairs:
        bv = [data.get(base_key, {}).get(exp) for exp in EXPERIMENT_ORDER]
        fv = [data.get(ft_key, {}).get(exp) for exp in EXPERIMENT_ORDER]
        mean_vals.append(mean_of(bv))
        mean_vals.append(mean_of(fv))
    rv = [data.get(BINZ_MODEL, {}).get(exp) for exp in EXPERIMENT_ORDER]
    mean_vals.append(mean_of(rv))

    eligible = [v for v in mean_vals if v is not None]
    best, second = find_best_and_second(eligible)
    cells = []
    for v in mean_vals:
        if v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-6:
            rank = 1
        elif second is not None and abs(v - second) < 1e-6:
            rank = 2
        else:
            rank = None
        cells.append(fmt_hc(v, rank=rank))

    lnk_nums = [LN_K[e] for e in EXPERIMENT_ORDER if LN_K[e] is not None]
    lnk_mean = sum(lnk_nums) / len(lnk_nums) if lnk_nums else None
    lnk_cell = f"\\hc{{{lnk_mean:.2f}}}" if lnk_mean is not None else "\\hcn"

    n_total = len(EXPERIMENT_ORDER)
    lines.append(f"\\textbf{{Mean ({n_total})}} & & "
                 + " & ".join(cells)
                 + f" & {lnk_cell}" + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND_FULL)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best. "
        "Lower is better. ``---''\\,=\\,continuous or mixed response space. "
        "Size labels in billions of parameters.}"
    )
    lines.append(
        "\\caption{Per-experiment NLL on Psych-201 (out-of-distribution). "
        "Psych-201 was not used during supervised fine-tuning; "
        "lower NLL indicates better generalisation. "
        "Within each model size, the left column (base) shows the "
        "pretrained model and the right column (ft) shows the "
        "cognitively finetuned variant. "
        "The $\\ln(k)$ column shows the random-guessing baseline where "
        "$k$ is the number of per-trial response options.}"
    )
    lines.append("\\label{tab:psych201_appendix_raw}")
    lines.append("\\end{sidewaystable}")

    return "\n".join(lines)


# Finetuned (one column per model) for the four families.
FAMILY_FT_COLS = [
    ("\\texttt{\\textbf{Qwentaur}}", [
        ("0.6B", "socius-Qwentaur-0.6B-LoRA"),
        ("1.7B", "socius-Qwentaur-1.7B-LoRA"),
        ("4B",   "socius-Qwentaur-4B-LoRA"),
        ("8B",   "socius-Qwentaur-8B-LoRA"),
        ("14B",  "socius-Qwentaur-14B-LoRA"),
    ]),
    ("\\texttt{\\textbf{Llama-Centaur}}", [
        ("1B", "socius-Llama-Centaur-1B-LoRA"),
        ("3B", "socius-Llama-Centaur-3B-LoRA"),
        ("8B", "socius-Llama-Centaur-8B-LoRA"),
    ]),
    ("\\texttt{\\textbf{Smoltaur}}", [
        ("0.1B", "socius-Smoltaur-0.1B-LoRA-r16"),
        ("0.4B", "socius-Smoltaur-0.4B-LoRA-r16"),
        ("1.7B", "socius-Smoltaur-1.7B-LoRA-r16"),
        ("3B",   "socius-Smoltaur-3B-LoRA-r16"),
    ]),
    ("\\texttt{\\textbf{Olmotaur}}", [
        ("1B", "socius-Olmotaur-1B-LoRA-r16"),
        ("7B", "socius-Olmotaur-7B-LoRA-r16"),
    ]),
]


def generate_appendix_families(data):
    """Full sideways table: raw NLL on Psych-201, all 18 experiments, the four
    finetuned families (one column per model; no base). Reference: reproduced
    Centaur-70B and the ln(k) chance level (both excluded from marking)."""
    model_cols = [c for _, cols in FAMILY_FT_COLS for c in cols]
    n_model = len(model_cols)

    colspec = "@{}l l" + "  r" * n_model + "  r" + "  r" + "@{}"  # +70B +ln(k)

    lines = []
    lines.append("\\begin{sidewaystable}[p]")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups + Binz reference
    h1, col, cmids = "& ", 3, []
    for fam_label, cols in FAMILY_FT_COLS:
        n = len(cols)
        h1 += f"& \\multicolumn{{{n}}}{{c}}{{{fam_label}}} "
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    h1 += ("& \\multicolumn{1}{c}{\\citet{binz2025foundation}} "
           "& \\textbf{Chance} \\\\")
    lines.append(h1)
    lines.append("".join(cmids))

    # Header row 2: sizes + 70B + ln(k)
    h2 = "\\textbf{Experiment} & \\textbf{Type}"
    for size_label, _ in model_cols:
        h2 += f" & {size_label}"
    h2 += " & 70B & $\\ln(k)$ \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    def render_row(label, cell_type, model_vals, ref_val, lnk_cell):
        best, second = find_best_and_second([v for v in model_vals if v is not None])
        cells = []
        for v in model_vals:
            if v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-6:
                rank = 1
            elif second is not None and abs(v - second) < 1e-6:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(v, rank=rank))
        cells.append(fmt_hc(ref_val))
        lines.append(f"{label} & {cell_type} & " + " & ".join(cells)
                     + f" & {lnk_cell}" + " \\\\")

    # Data rows, grouped by task type
    type_groups = defaultdict(list)
    for exp in EXPERIMENT_ORDER:
        type_groups[TASK_TYPE[exp]].append(exp)
    prev_type = None
    for task_type in TASK_ORDER:
        exps_in_type = type_groups.get(task_type, [])
        if not exps_in_type:
            continue
        if prev_type is not None:
            lines.append("\\midrule")
        prev_type = task_type
        for exp in exps_in_type:
            display, cite = CITE_MAP.get(exp, (exp, exp))
            cell_exp = (f"{display} "
                        f"{{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}")
            model_vals = [data.get(k, {}).get(exp) for _, k in model_cols]
            ref_val = data.get(BINZ_MODEL, {}).get(exp)
            render_row(cell_exp, TASK_ABBREV.get(task_type, task_type),
                       model_vals, ref_val, fmt_lnk(exp))

    # Mean row
    lines.append("\\midrule")
    model_means = [mean_of([data.get(k, {}).get(exp) for exp in EXPERIMENT_ORDER])
                   for _, k in model_cols]
    ref_mean = mean_of([data.get(BINZ_MODEL, {}).get(exp) for exp in EXPERIMENT_ORDER])
    lnk_nums = [LN_K[e] for e in EXPERIMENT_ORDER if LN_K[e] is not None]
    lnk_mean = sum(lnk_nums) / len(lnk_nums) if lnk_nums else None
    lnk_cell = f"\\hc{{{lnk_mean:.2f}}}" if lnk_mean is not None else "\\hcn"
    render_row(f"\\textbf{{Mean ({len(EXPERIMENT_ORDER)})}}", "",
               model_means, ref_mean, lnk_cell)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND_FULL)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best (across the four families). "
        "Lower is better. ``---''\\,=\\,continuous or mixed response space. "
        "Size labels in billions of parameters. 70B is the reproduced "
        "Centaur-70B (reference only; excluded from marking).}"
    )
    lines.append(
        "\\caption{Per-experiment NLL on Psych-201 (out-of-distribution) for the "
        "four finetuned families. Psych-201 was not used during fine-tuning; "
        "lower NLL indicates better generalisation. The $\\ln(k)$ column shows "
        "the random-guessing baseline where $k$ is the number of per-trial "
        "response options.}"
    )
    lines.append("\\label{tab:psych201_appendix_families}")
    lines.append("\\end{sidewaystable}")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_all_results()
    print(f"  {len(data)} models × {len(EXPERIMENT_ORDER)} experiments")

    # Check Binz model is present
    if BINZ_MODEL in data:
        print(f"  Binz model found: {BINZ_MODEL}")
    else:
        print(f"  WARNING: Binz model not found: {BINZ_MODEL}")

    print("\nGenerating tables...")

    # 1. Main-text: normalised task-type summary
    tex = generate_main_normalised(data)
    path = os.path.join(OUT_DIR, "psych201_main_normalised.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    # 2. Appendix: base vs ft raw NLL (sideways)
    tex = generate_appendix_raw(data)
    path = os.path.join(OUT_DIR, "psych201_appendix_raw.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    # 3. Appendix: four finetuned families, raw NLL (sideways)
    tex = generate_appendix_families(data)
    path = os.path.join(OUT_DIR, "psych201_appendix_families.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    print(f"\nDone. Three tables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
