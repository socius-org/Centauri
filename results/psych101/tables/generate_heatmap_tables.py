# -*- coding: utf-8 -*-
"""
Generate heatmap LaTeX tables from psych101_aggr.csv (+ flat Smoltaur/Olmotaur
r=16 eval CSVs for the all-families table):
  1. All four finetuned families, bf16, LoRA r=16 (one column per model)
  2. Base vs FT under bf16 inference
  3. Main-text summary: mean NLL by task type (+ filtered / chance / normalised)
  4. Non-cognitive control and cross-family comparisons

Usage:
    python generate_heatmap_tables.py

Reads:  psych101_aggr.csv (same dir) and ../socius-{Smoltaur,Olmotaur}-*-LoRA-r16.csv
Writes: tables/heatmap_ft_bf16_all.tex, tables/heatmap_baseft_bf16.tex,
        tables/main_tasktype*.tex, tables/non_cognitive_controls.tex,
        tables/family_comparison.tex
"""

import csv
import os
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psych101_aggr.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Column name -> 0-based index (from col 3 of CSV, i.e., first model column = 0)
COL = {
    "qwen_0.6b_base_bf16": 0,
    "qwen_0.6b_base_4bit": 1,
    "qwentaur_0.6b_bf16":  2,
    "qwentaur_0.6b_4bit":  3,
    "llama_1b_base_bf16":   4,
    "llama_1b_base_4bit":   5,
    "centaur_1b_bf16":      6,
    "centaur_1b_4bit":      7,
    "qwen_1.7b_base_bf16":  8,
    "qwen_1.7b_base_4bit":  9,
    "qwentaur_1.7b_bf16":  10,
    "qwentaur_1.7b_4bit":  11,
    "llama_3b_base_bf16":   12,
    "llama_3b_base_4bit":   13,
    "centaur_3b_bf16":      14,
    "centaur_3b_4bit":      15,
    "qwen_4b_base_bf16":    16,
    "qwen_4b_base_4bit":    17,
    "qwentaur_4b_bf16":     18,
    "qwentaur_4b_4bit":     19,
    "qwen_8b_base_bf16":    20,
    "qwen_8b_base_4bit":    21,
    "qwentaur_8b_bf16":     22,
    "qwentaur_8b_4bit":     23,
    "llama_8b_base_bf16":   24,
    "llama_8b_base_4bit":   25,
    "centaur_8b_bf16":      26,
    "centaur_8b_4bit":      27,
    "qwen_14b_base_bf16":   28,
    "qwen_14b_base_4bit":   29,
    "qwentaur_14b_bf16":    30,
    "qwentaur_14b_4bit":    31,
    "llama_70b_reported":   32,
    "centaur_70b_repr":     33,
    "centaur_70b_rep":      34,
    "cog_model":            35,
}

# Experiment display names and citation keys
CITE_MAP = {
    "frey2017risk":             ("Balloon analog risk task",    "frey2017risk"),
    "plonsky2018when":          ("CPC18",                       "plonsky2018when"),
    "frey2017cct":              ("Columbia card task",           "frey2017risk"),
    "wulff2018description":     ("Decisions from description",  "wulff2018sampling"),
    "garcia2023experiential":   ("Experiential-symbolic task",  "garcia2023experiential"),
    "flesch2018comparing":      ("Gardening task",              "flesch2018comparing"),
    "enkavi2019gonogo":         ("Go/no-go",                    "enkavi2019large"),
    "ruggeri2022globalizability": ("Intertemporal choice",      "ruggeri2022globalizability"),
    "hilbig2014generalized":    ("Multi-attribute DM",          "hilbig2014generalized"),
    "krueger2022identifying":   ("Risky choice",                "krueger2024identifying"),
    "peterson2021using":        ("choices13k",                   "peterson2021using"),
    "tomov2021multitask":       ("Multi-task RL",               "tomov2021multitask"),
    "kumar2023disentangling":   ("Tile-revealing task",         "kumar2023disentangling"),
    "kool2017cost":             ("Two-step task",               "kool2017cost"),
    "kool2016when":             ("Two-step task",               "kool2016when"),
    "zorowitz2023data":         ("Two-step task",               "zorowitz2023data"),
    "tomov2020discovery":       ("Virtual subway network",      "tomov2020discovery"),
    "ludwig2023human":          ("Zoopermarket",                "ludwig2023human"),
    "gershman2020reward":       ("Cond.\\ assoc.\\ learning",  "collins2014working"),
    "enkavi2019digitspan":      ("Digit span",                  "enkavi2019large"),
    "popov2023intent":          ("Episodic long-term memory",   "popov2023intent"),
    "enkavi2019adaptivenback":  ("N-back",                      "enkavi2019large"),
    "cox2017information":       ("Recall and recognition",      "cox2018information"),
    "enkavi2019recentprobes":   ("Recent probes",               "enkavi2019large"),
    "jansen2021dunningkruger":  ("Grammar judgement",           "jansen2021rational"),
    "zhu2020bayesian":          ("Probabilistic reasoning",     "zhu2020bayesian"),
    "wu2023chunking":           ("Serial reaction time task",   "wu2023chunking"),
    "hebart2023things":         ("THINGS odd-one-out",          "hebart2023things"),
    "xiong2023neural":          ("Changing bandit",             "xiong2023neural"),
    "wulff2018sampling":        ("Decisions from experience",   "wulff2018sampling"),
    "bahrami2020four":          ("Drifting four-armed bandit",  "bahrami2020four"),
    "feng2021dynamics":         ("Horizon task",                "feng2021dynamics"),
    "sadeghiyeh2020temporal":   ("Horizon task",                "sadeghiyeh2020temporal"),
    "somerville2017charting":   ("Horizon task",                "somerville2017charting"),
    "waltz2020differential":    ("Horizon task",                "waltz2020differential"),
    "wilson2014humans":         ("Horizon task",                "wilson2014humans"),
    "steingroever2015data":     ("Iowa gambling task",          "steingroever2015data"),
    "lefebvre2017behavioural":  ("Prob.\\ instrumental learning", "lefebvre2017behavioural"),
    "wu2018generalisation":     ("Spatially correlated MAB",    "wu2018generalization"),
    "schulz2020finding":        ("Structured bandit",           "schulz2020finding"),
    "gershman2018deconstructing": ("Two-armed bandit",          "gershman2018deconstructing"),
    "wise2019acomputational":   ("Aversive learning",           "wise2019computational"),
    "levering2020revisiting":   ("Medin categorization",        "levering2020revisiting"),
    "collsiöö2023MCPL":         ("Multiple-cue judgment",       "collsioo2023numerical"),
    "badham2017deficits":       ("Shepard categorization",      "badham2017deficits"),
    "speekenbrink2008learning": ("Weather prediction task",     "speekenbrink2008learning"),
}

TASK_ABBREV = {
    "Decision-making":          "Decision",
    "Markov decision processes": "MDP",
    "Memory":                    "Memory",
    "Miscellaneous":             "Misc.",
    "Multi-armed bandits":       "Bandit",
    "Supervised learning":       "Sup.\\ learn.",
}

TASK_ORDER = [
    "Decision-making", "Markov decision processes", "Multi-armed bandits",
    "Memory", "Miscellaneous", "Supervised learning",
]

# Experiments where Binz et al. report a single merged value
MERGED_EXPTS = {
    "kool2017cost", "kool2016when", "zorowitz2023data",
    "feng2021dynamics", "sadeghiyeh2020temporal", "somerville2017charting",
    "waltz2020differential", "wilson2014humans",
}

# Random-guessing baseline: ln(k) where k = number of response options.
# None = cannot compute (continuous, mixed, or variable action space).
import math as _math
_ln = _math.log
LN_K = {
    "badham2017deficits":       _ln(2),
    "bahrami2020four":          _ln(4),
    "collsiöö2023MCPL":         _ln(9),
    "cox2017information":       None,      # mixed: recognition + free recall
    "enkavi2019adaptivenback":  _ln(2),
    "enkavi2019digitspan":      _ln(10),
    "enkavi2019gonogo":         _ln(1),    # degenerate (only go trials tokenised)
    "enkavi2019recentprobes":   _ln(2),
    "feng2021dynamics":         _ln(2),
    "flesch2018comparing":      _ln(2),
    "frey2017cct":              _ln(2),
    "frey2017risk":             _ln(2),
    "garcia2023experiential":   None,      # mixed: binary + probability estimates
    "gershman2018deconstructing": _ln(2),
    "gershman2020reward":       _ln(3),
    "hebart2023things":         _ln(3),
    "hilbig2014generalized":    _ln(2),
    "jansen2021dunningkruger":  None,      # mixed: free numeric + 5-choice
    "kool2016when":             _ln(2),
    "kool2017cost":             _ln(2),
    "krueger2022identifying":   None,      # mixed: multi-stage variable actions
    "kumar2023disentangling":   None,      # variable: grid positions change per trial
    "lefebvre2017behavioural":  _ln(2),
    "levering2020revisiting":   None,      # mixed: binary + 9-point rating
    "ludwig2023human":          _ln(2),
    "peterson2021using":        _ln(2),
    "plonsky2018when":          _ln(2),
    "popov2023intent":          _ln(2),
    "ruggeri2022globalizability": _ln(2),
    "sadeghiyeh2020temporal":   _ln(2),
    "schulz2020finding":        _ln(8),
    "somerville2017charting":   _ln(2),
    "speekenbrink2008learning": _ln(2),
    "steingroever2015data":     _ln(4),
    "tomov2020discovery":       _ln(5),    # nominal ceiling (walls restrict actual k)
    "tomov2021multitask":       _ln(3),
    "waltz2020differential":    _ln(2),
    "wilson2014humans":         _ln(2),
    "wise2019acomputational":   None,      # continuous probability estimates
    "wu2018generalisation":     _ln(30),
    "wu2023chunking":           _ln(4),
    "wulff2018description":     _ln(2),
    "wulff2018sampling":        None,      # mixed: sample/stop/choose phases
    "xiong2023neural":          _ln(2),
    "zhu2020bayesian":          None,      # continuous probability estimates
    "zorowitz2023data":         _ln(2),
}


# =============================================================================
# Helpers
# =============================================================================

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip().replace("\r", "") for h in header]
        rows = []
        for row in reader:
            rows.append([c.strip().replace("\r", "") for c in row])
    return header, rows


def parse_val(v):
    """Return float or None."""
    v = v.strip().replace("\u2020", "").replace("\r", "")
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mean_of(vals):
    """Mean of non-None values."""
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def find_best_and_second(values, tol=1e-6):
    """
    Given a list of values (may contain None), find the best (min)
    and second-best (next distinct minimum) values.
    Returns (best_val, second_val) where either can be None.
    """
    numeric = sorted(set(round(v, 6) for v in values if v is not None))
    best = numeric[0] if len(numeric) >= 1 else None
    second = numeric[1] if len(numeric) >= 2 else None
    return best, second


def fmt_hc(val, dagger=False, rank=None):
    """
    Format heatmap cell.
    rank: 1 = best (bold+underline), 2 = second (underline), None = normal.
    dagger: True to append dagger superscript.
    """
    if val is None:
        return "\\hcn"
    if rank == 1 and dagger:
        return f"\\hcbud{{{val:.2f}}}"
    if rank == 1:
        return f"\\hcbu{{{val:.2f}}}"
    if rank == 2 and dagger:
        return f"\\hcud{{{val:.2f}}}"
    if rank == 2:
        return f"\\hcu{{{val:.2f}}}"
    if dagger:
        return f"\\hcd{{{val:.2f}}}"
    return f"\\hc{{{val:.2f}}}"


def fmt_lnk(exp_code):
    """Format ln(k) random baseline value for an experiment, with heatmap colour."""
    val = LN_K.get(exp_code)
    if val is None:
        return "\\hcn"
    return f"\\hc{{{val:.2f}}}"


# =============================================================================
# LaTeX preamble
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

% ---- Dagger variants ----
\newcommand{\hcd}[1]{\hc{#1}$^{\dagger}$}
\newcommand{\hcud}[1]{\hcu{#1}$^{\dagger}$}
\newcommand{\hcbud}[1]{\hcbu{#1}$^{\dagger}$}

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

LEGEND = r"""\vspace{4pt}
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

LEGEND_COMPACT = r"""\vspace{2pt}
{\tiny\centering
\colorbox{c1!45}{\strut\,} $<$0.20 ~
\colorbox{c2!40}{\strut\,} 0.20--0.40 ~
\colorbox{c3!40}{\strut\,} 0.40--0.60 ~
\colorbox{c4!45}{\strut\,} 0.60--0.80 ~
\colorbox{c5!55}{\strut\,} 0.80--1.00 ~
\colorbox{c6!50}{\strut\,} $>$1.00\par}"""

LEGEND_REVERSED = r"""\vspace{2pt}
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


# =============================================================================
# Model column specifications
# =============================================================================

# --- Base vs FT under bf16 ---
QWEN_BASEFT_BF16 = [
    ("0.6B", "qwen_0.6b_base_bf16",  "qwentaur_0.6b_bf16"),
    ("1.7B", "qwen_1.7b_base_bf16",  "qwentaur_1.7b_bf16"),
    ("4B",   "qwen_4b_base_bf16",    "qwentaur_4b_bf16"),
    ("8B",   "qwen_8b_base_bf16",    "qwentaur_8b_bf16"),
    ("14B",  "qwen_14b_base_bf16",   "qwentaur_14b_bf16"),
]
LLAMA_BASEFT_BF16 = [
    ("1B", "llama_1b_base_bf16",  "centaur_1b_bf16"),
    ("3B", "llama_3b_base_bf16",  "centaur_3b_bf16"),
    ("8B", "llama_8b_base_bf16",  "centaur_8b_bf16"),
]

# Reference columns: (col_key, can_have_dagger, exclude_from_bold)
# C-70B_p is excluded from best/second-best calculation because it was
# evaluated under different software conditions (package versions, CUDA).
REF_COLS = [
    ("centaur_70b_repr", False, False),
    ("centaur_70b_rep",  True,  True),    # exclude from bold
    ("cog_model",        True,  False),
]
REF_LABELS = ["C$_r$", "C$_p$", "Cog.$_p$"]

REF_COLS_BASEFT = [
    ("llama_70b_reported", False, False),
    ("centaur_70b_repr",   False, False),
    ("centaur_70b_rep",    True,  True),  # exclude from bold
    ("cog_model",          True,  False),
]
REF_LABELS_BASEFT = ["L-70B$_p$", "C$_r$", "C$_p$", "Cog.$_p$"]


# =============================================================================
# Appendix table generator (Tables 1-3)
# =============================================================================

def generate_heatmap_table(
    rows, qwen_models, llama_models, ref_cols, ref_labels,
    left_label, right_label, qwen_family_label, llama_family_label,
    footnote_text, caption_text, table_label,
):
    """Generate a single sidewaystable heatmap with best/2nd-best marking."""
    all_models = qwen_models + llama_models
    n_qwen = len(qwen_models)
    n_llama = len(llama_models)
    n_pairs = n_qwen + n_llama
    n_ref = len(ref_cols)

    colspec = "@{}l l"
    for _ in range(n_pairs):
        colspec += "  r@{\\;\\;}r"
    for _ in range(n_ref):
        colspec += "  r"
    colspec += "  r"  # ln(k) column
    colspec += "@{}"

    lines = []
    lines.append("\\begin{sidewaystable}[p]")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups
    n_qwen_cols = n_qwen * 2
    n_llama_cols = n_llama * 2
    first_qwen = 3
    last_qwen = first_qwen + n_qwen_cols - 1
    first_llama = last_qwen + 1
    last_llama = first_llama + n_llama_cols - 1
    first_ref = last_llama + 1
    last_ref = first_ref + n_ref - 1
    col_lnk = last_ref + 1

    lines.append(
        f"& & \\multicolumn{{{n_qwen_cols}}}{{c}}{{{qwen_family_label}}} "
        f"& \\multicolumn{{{n_llama_cols}}}{{c}}{{{llama_family_label}}} "
        f"& \\multicolumn{{{n_ref}}}{{c}}{{\\citet{{binz2025foundation}}}} "
        f"& \\textbf{{Chance}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{first_qwen}-{last_qwen}}}"
        f"\\cmidrule(lr){{{first_llama}-{last_llama}}}"
        f"\\cmidrule(lr){{{first_ref}-{last_ref}}}"
    )

    # Header row 2: size groups
    h2 = "& "
    for label, _, _ in all_models:
        h2 += f"& \\multicolumn{{2}}{{c}}{{{label}}} "
    h2 += "& " * n_ref + "& \\\\"   # extra & for ln(k)
    lines.append(h2)

    rules = []
    col = first_qwen
    for _ in all_models:
        rules.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
        col += 2
    lines.append(" ".join(rules))

    # Header row 3
    h3 = "\\textbf{Experiment} & \\textbf{Type}"
    for _ in all_models:
        h3 += f" & {{\\fontsize{{4}}{{5}}\\selectfont {left_label}}} & {{\\fontsize{{4}}{{5}}\\selectfont {right_label}}}"
    for rl in ref_labels:
        h3 += f" & {rl}"
    h3 += " & $\\ln(k)$"
    h3 += " \\\\"
    lines.append(h3)
    lines.append("\\midrule")

    # Data rows
    prev_task = None
    for row in rows:
        exp_code = row[0]
        task_name = row[1]
        task_type = row[2]
        is_merged = exp_code in MERGED_EXPTS

        if exp_code == "Mean":
            lines.append("\\midrule")
            cell_exp = "\\textbf{Mean (all 48)}"
            cell_task = ""
        else:
            display, cite = CITE_MAP.get(exp_code, (task_name, exp_code))
            cell_exp = f"{display} {{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}"
            cell_task = TASK_ABBREV.get(task_type, task_type)
            if prev_task is not None and task_type != prev_task and task_type != "":
                lines.append("\\midrule")

        if task_type:
            prev_task = task_type

        cell_info = []  # list of (value, dagger_flag, exclude_from_bold)
        for _, left_key, right_key in all_models:
            cell_info.append((parse_val(row[3 + COL[left_key]]), False, False))
            cell_info.append((parse_val(row[3 + COL[right_key]]), False, False))
        for col_key, can_dagger, exclude_bold in ref_cols:
            val = parse_val(row[3 + COL[col_key]])
            cell_info.append((val, is_merged and can_dagger, exclude_bold))

        # Find best/second only among non-excluded columns
        eligible_vals = [v for v, _, excl in cell_info if v is not None and not excl]
        best, second = find_best_and_second(eligible_vals)

        cells = []
        for val, dagger, excl in cell_info:
            if excl or val is None:
                rank = None
            elif best is not None and abs(val - best) < 1e-6:
                rank = 1
            elif second is not None and abs(val - second) < 1e-6:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(val, dagger=dagger, rank=rank))

        # Append ln(k) column
        if exp_code == "Mean":
            # Mean of ln(k) across all non-None experiments
            lnk_vals = [LN_K.get(r[0].strip()) for r in rows
                        if r[0].strip() != "Mean"]
            lnk_nums = [v for v in lnk_vals if v is not None]
            lnk_mean = sum(lnk_nums) / len(lnk_nums) if lnk_nums else None
            lnk_cell = f"\\hc{{{lnk_mean:.2f}}}" if lnk_mean is not None else "\\hcn"
        else:
            lnk_cell = fmt_lnk(exp_code)

        line = f"{cell_exp} & {cell_task} & " + " & ".join(cells) + f" & {lnk_cell}" + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best "
        "(C$_p$ excluded; see footnote). " + footnote_text + "}"
    )
    lines.append(f"\\caption{{{caption_text}}}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\end{sidewaystable}")

    return "\n".join(lines)


# =============================================================================
# All-families finetuned (bf16, LoRA r=16) per-experiment heatmap
# =============================================================================

# Smoltaur/Olmotaur are not in psych101_aggr.csv, so their bf16 r=16 per-task
# NLLs come from the flat eval CSVs (task, loss) in results/psych101/.
FAMILY_R16_CSVS = {
    "smoltaur_0.1b": "socius-Smoltaur-0.1B-LoRA-r16.csv",
    "smoltaur_0.4b": "socius-Smoltaur-0.4B-LoRA-r16.csv",
    "smoltaur_1.7b": "socius-Smoltaur-1.7B-LoRA-r16.csv",
    "smoltaur_3b":   "socius-Smoltaur-3B-LoRA-r16.csv",
    "olmotaur_1b":   "socius-Olmotaur-1B-LoRA-r16.csv",
    "olmotaur_7b":   "socius-Olmotaur-7B-LoRA-r16.csv",
}


def load_family_ext(data_dir):
    """{col_key: {experiment: nll}} for Smoltaur/Olmotaur (absent from aggr)."""
    ext = {}
    for key, fname in FAMILY_R16_CSVS.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            ext[key] = read_control_csv(fpath)
        else:
            print(f"  WARNING: {fname} not found, skipping {key}")
    return ext


# Finetuned family column groups for the task-type summary tables (bf16, r=16).
# (group header, [(\texttt label, source, key)]). Smoltaur/Olmotaur come from ext.
TASKTYPE_FT_GROUPS = [
    ("\\textbf{\\texttt{Qwentaur}}", [
        ("\\texttt{0.6B}", "aggr", "qwentaur_0.6b_bf16"),
        ("\\texttt{1.7B}", "aggr", "qwentaur_1.7b_bf16"),
        ("\\texttt{4B}",   "aggr", "qwentaur_4b_bf16"),
        ("\\texttt{8B}",   "aggr", "qwentaur_8b_bf16"),
        ("\\texttt{14B}",  "aggr", "qwentaur_14b_bf16"),
    ]),
    ("\\textbf{\\texttt{Llama-Centaur}}", [
        ("\\texttt{1B}", "aggr", "centaur_1b_bf16"),
        ("\\texttt{3B}", "aggr", "centaur_3b_bf16"),
        ("\\texttt{8B}", "aggr", "centaur_8b_bf16"),
    ]),
    ("\\textbf{\\texttt{Smoltaur}}", [
        ("\\texttt{0.1B}", "ext", "smoltaur_0.1b"),
        ("\\texttt{0.4B}", "ext", "smoltaur_0.4b"),
        ("\\texttt{1.7B}", "ext", "smoltaur_1.7b"),
        ("\\texttt{3B}",   "ext", "smoltaur_3b"),
    ]),
    ("\\textbf{\\texttt{Olmotaur}}", [
        ("\\texttt{1B}", "ext", "olmotaur_1b"),
        ("\\texttt{7B}", "ext", "olmotaur_7b"),
    ]),
]


def generate_ft_bf16_all_families(rows, data_dir):
    """Per-experiment NLL heatmap for all four finetuned families at bf16,
    LoRA r=16 (one column per model). Qwentaur/Llama-Centaur come from
    psych101_aggr.csv; Smoltaur/Olmotaur from their flat r=16 eval CSVs.
    Reference columns participate in best/second-best marking except C_p
    (different software conditions); ln(k) chance level is display-only."""

    ext = {}
    for key, fname in FAMILY_R16_CSVS.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            ext[key] = read_control_csv(fpath)
        else:
            print(f"  WARNING: {fname} not found, skipping {key}")

    # (family_label, [(size_label, source, key)])
    families = [
        ("\\texttt{\\textbf{Qwentaur}}", [
            ("0.6B", "aggr", "qwentaur_0.6b_bf16"),
            ("1.7B", "aggr", "qwentaur_1.7b_bf16"),
            ("4B",   "aggr", "qwentaur_4b_bf16"),
            ("8B",   "aggr", "qwentaur_8b_bf16"),
            ("14B",  "aggr", "qwentaur_14b_bf16"),
        ]),
        ("\\texttt{\\textbf{Llama-Centaur}}", [
            ("1B", "aggr", "centaur_1b_bf16"),
            ("3B", "aggr", "centaur_3b_bf16"),
            ("8B", "aggr", "centaur_8b_bf16"),
        ]),
        ("\\texttt{\\textbf{Smoltaur}}", [
            ("0.1B", "ext", "smoltaur_0.1b"),
            ("0.4B", "ext", "smoltaur_0.4b"),
            ("1.7B", "ext", "smoltaur_1.7b"),
            ("3B",   "ext", "smoltaur_3b"),
        ]),
        ("\\texttt{\\textbf{Olmotaur}}", [
            ("1B", "ext", "olmotaur_1b"),
            ("7B", "ext", "olmotaur_7b"),
        ]),
    ]
    # (label, source, key, can_dagger, exclude_from_marking) -- references.
    # C$_p$ is shown for reference only (different software conditions).
    ref_cols = [
        ("L-70B$_p$", "aggr", "llama_70b_reported", False, False),
        ("C$_r$",     "aggr", "centaur_70b_repr",   False, False),
        ("C$_p$",     "aggr", "centaur_70b_rep",    True,  True),
        ("Cog.$_p$",  "aggr", "cog_model",          True,  False),
    ]
    model_cols = [c for _, cols in families for c in cols]

    mean_row = next((r for r in rows if r[0].strip() == "Mean"), None)

    def get_val(col, row):
        source, key = col[1], col[2]
        if source == "aggr":
            return parse_val(row[3 + COL[key]])
        return ext.get(key, {}).get(row[0].strip())

    def get_mean(col):
        source, key = col[1], col[2]
        if source == "aggr":
            return parse_val(mean_row[3 + COL[key]]) if mean_row else None
        vals = [ext.get(key, {}).get(r[0].strip())
                for r in rows if r[0].strip() != "Mean"]
        return mean_of(vals)

    n_model, n_ref = len(model_cols), len(ref_cols)
    colspec = "@{}l l" + "  r" * n_model + "  " + "r" * n_ref + "  r@{}"

    lines = []
    lines.append("\\begin{sidewaystable}[p]")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups + references
    h1, col, cmids = "& ", 3, []
    for fam_label, cols in families:
        n = len(cols)
        h1 += f"& \\multicolumn{{{n}}}{{c}}{{{fam_label}}} "
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    h1 += (f"& \\multicolumn{{{n_ref}}}{{c}}{{\\citet{{binz2025foundation}}}} "
           "& \\textbf{Chance} \\\\")
    lines.append(h1)
    lines.append("".join(cmids))

    # Header row 2: sizes + reference labels + ln(k)
    h2 = "\\textbf{Experiment} & \\textbf{Type}"
    for size_label, _, _ in model_cols:
        h2 += f" & {size_label}"
    for rl, *_ in ref_cols:
        h2 += f" & {rl}"
    h2 += " & $\\ln(k)$ \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    prev_task = None
    for row in rows:
        exp_code = row[0].strip()
        is_merged = exp_code in MERGED_EXPTS

        if exp_code == "Mean":
            lines.append("\\midrule")
            cell_exp, cell_task = "\\textbf{Mean}", ""
            model_vals = [get_mean(c) for c in model_cols]
            ref_vals = [get_mean(c) for c in ref_cols]
            lnk_nums = [LN_K.get(r[0].strip()) for r in rows if r[0].strip() != "Mean"]
            lnk_nums = [v for v in lnk_nums if v is not None]
            lnk_val = sum(lnk_nums) / len(lnk_nums) if lnk_nums else None
        else:
            display, cite = CITE_MAP.get(exp_code, (row[1], exp_code))
            cell_exp = f"{display} {{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}"
            cell_task = TASK_ABBREV.get(row[2], row[2])
            if prev_task is not None and row[2] != prev_task and row[2] != "":
                lines.append("\\midrule")
            model_vals = [get_val(c, row) for c in model_cols]
            ref_vals = [get_val(c, row) for c in ref_cols]
            lnk_val = LN_K.get(exp_code)
        if row[2]:
            prev_task = row[2]

        # best/second among all columns except C$_p$
        eligible = [v for v in model_vals if v is not None]
        eligible += [v for j, v in enumerate(ref_vals)
                     if v is not None and not ref_cols[j][4]]
        best, second = find_best_and_second(eligible)
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
        for j, v in enumerate(ref_vals):
            _, _, _, can_d, excl = ref_cols[j]
            if excl or v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-6:
                rank = 1
            elif second is not None and abs(v - second) < 1e-6:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(v, dagger=(can_d and is_merged and exp_code != "Mean"),
                                rank=rank))
        lnk_cell = f"\\hc{{{lnk_val:.2f}}}" if lnk_val is not None else "\\hcn"

        lines.append(f"{cell_exp} & {cell_task} & " + " & ".join(cells)
                     + f" & {lnk_cell}" + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best "
        "(C$_p$ excluded; see below). Lower is better. ``---''\\,=\\,not available. "
        "All models finetuned at LoRA rank 16, bf16 inference. Reference columns "
        "(\\citet{binz2025foundation}): L-70B$_p$\\,=\\,Llama-3.1-70B base, "
        "C$_r$\\,=\\,our reproduced Centaur-70B, C$_p$\\,=\\,published Centaur-70B, "
        "Cog.$_p$\\,=\\,domain-specific cognitive baseline. "
        "C$_p$ is included for reference but excluded from best/second-best "
        "marking, as it was evaluated under different software conditions. "
        "$^\\dagger$Horizon and "
        "Two-step tasks are reported as single merged values by "
        "\\citet{binz2025foundation}. Rounded to 2\\,d.p.}"
    )
    lines.append(
        "\\caption{Per-experiment NLL on Psych-101 for all four finetuned "
        "families (bf16, LoRA rank 16). Cell colour encodes performance "
        "(green\\,=\\,better).}"
    )
    lines.append("\\label{tab:heatmap_ft_bf16_all}")
    lines.append("\\end{sidewaystable}")
    return "\n".join(lines)


# =============================================================================
# Main-text table (Option A: task-type heatmap)
# =============================================================================

def generate_main_tasktype(data_rows, task_groups, ext, filter_by=None):
    """Compact task-type summary table (mean NLL by task type) for the four
    finetuned families (Qwentaur, Llama-Centaur, Smoltaur, Olmotaur; bf16, r=16),
    plus base and Binz-et-al. reference columns.

    filter_by:
        None   — all 46 experiments
        "cog"  — only experiments with a reported cognitive model baseline
        "lnk"  — only experiments with a computable chance baseline ln(k)
    """
    # Flat columns: (label, source, key, exclude_from_bold)
    model_cols = [(lbl, src, key, False)
                  for _, cols in TASKTYPE_FT_GROUPS for lbl, src, key in cols]
    base_cols = [
        ("\\texttt{Q-8B}",  "aggr", "qwen_8b_base_bf16",  False),
        ("\\texttt{Q-14B}", "aggr", "qwen_14b_base_bf16", False),
        ("\\texttt{L-8B}",  "aggr", "llama_8b_base_bf16", False),
    ]
    ref_model_cols = [
        ("\\texttt{70B}$_p$", "aggr", "centaur_70b_rep",  True),   # ref only
        ("\\texttt{70B}$_r$", "aggr", "centaur_70b_repr", False),
        ("Cog.$_p$",          "aggr", "cog_model",        False),
    ]
    all_cols = model_cols + base_cols + ref_model_cols
    n_base = len(base_cols)
    n_ref = len(ref_model_cols)
    include_chance = (filter_by == "lnk")

    def gv(source, key, r):
        if source == "aggr":
            return parse_val(r[3 + COL[key]])
        return ext.get(key, {}).get(r[0].strip())

    # --- Filter rows if requested ---
    if filter_by == "cog":
        filt_data_rows = [r for r in data_rows
                          if parse_val(r[3 + COL["cog_model"]]) is not None]
    elif filter_by == "lnk":
        # Exclude experiments missing EITHER cognitive baseline OR ln(k)
        filt_data_rows = [r for r in data_rows
                          if (parse_val(r[3 + COL["cog_model"]]) is not None
                              and LN_K.get(r[0].strip()) is not None)]
    else:
        filt_data_rows = data_rows

    filt_task_groups = defaultdict(list)
    for row in filt_data_rows:
        filt_task_groups[row[2]].append(row)
    n_total = len(filt_data_rows)
    n_excluded = len(data_rows) - n_total

    # Mean NLL per task type per model
    task_means = {}
    # Also compute mean ln(k) per task type
    task_lnk_means = {}
    for task_type in TASK_ORDER:
        group = filt_task_groups.get(task_type, [])
        means = {}
        for label, source, key, _ in all_cols:
            means[key] = mean_of([gv(source, key, r) for r in group])
        task_means[task_type] = means
        lnk_vals = [LN_K.get(r[0].strip()) for r in group]
        lnk_nums = [v for v in lnk_vals if v is not None]
        task_lnk_means[task_type] = sum(lnk_nums) / len(lnk_nums) if lnk_nums else None

    # Overall mean
    overall = {}
    for label, source, key, _ in all_cols:
        overall[key] = mean_of([gv(source, key, r) for r in filt_data_rows])
    lnk_overall_vals = [LN_K.get(r[0].strip()) for r in filt_data_rows]
    lnk_overall_nums = [v for v in lnk_overall_vals if v is not None]
    overall_lnk = sum(lnk_overall_nums) / len(lnk_overall_nums) if lnk_overall_nums else None

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")

    # colspec: task-type col + per-family r-blocks + base + refs + optional chance
    group_sizes = [len(cols) for _, cols in TASKTYPE_FT_GROUPS] + [n_base, n_ref]
    colspec = "@{}l"
    for gs in group_sizes:
        colspec += "  " + " r" * gs
    if include_chance:
        colspec += "  r"
    colspec += "@{}"
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups + Base + Binz references
    h1, col, cmids = "", 2, []
    for hdr, cols in TASKTYPE_FT_GROUPS:
        n = len(cols)
        h1 += f"& \\multicolumn{{{n}}}{{c}}{{{hdr}}} "
        cmids.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    h1 += f"& \\multicolumn{{{n_base}}}{{c}}{{\\textbf{{Base}}}} "
    cmids.append(f"\\cmidrule(lr){{{col}-{col + n_base - 1}}}")
    col += n_base
    h1 += f"& \\multicolumn{{{n_ref}}}{{c}}{{\\citet{{binz2025foundation}}}}"
    cmids.append(f"\\cmidrule(lr){{{col}-{col + n_ref - 1}}}")
    col += n_ref
    if include_chance:
        h1 += " & \\textbf{Chance}"
    h1 += " \\\\"
    lines.append(h1)
    lines.append("".join(cmids))

    h2 = "\\textbf{Task type}"
    for label, _, _, _ in all_cols:
        h2 += f" & {label}"
    if include_chance:
        h2 += " & $\\ln(k)$"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # Task-type rows
    for task_type in TASK_ORDER:
        means = task_means[task_type]
        group = filt_task_groups.get(task_type, [])
        n_exps = len(group)
        if n_exps == 0:
            continue  # skip empty task types after filtering
        label = TASK_ABBREV.get(task_type, task_type)
        label_with_n = f"{label} ({n_exps})"

        eligible_vals = [means[key] for _, _, key, excl in all_cols
                         if not excl and means[key] is not None]
        best, second = find_best_and_second(eligible_vals)

        cells = []
        for _, _, key, excl in all_cols:
            v = means[key]
            if excl or v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-4:
                rank = 1
            elif second is not None and abs(v - second) < 1e-4:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(v, rank=rank))

        chance_cell = ""
        if include_chance:
            lnk_v = task_lnk_means[task_type]
            chance_cell = f" & \\hc{{{lnk_v:.2f}}}" if lnk_v is not None else " & \\hcn"
        lines.append(f"{label_with_n} & " + " & ".join(cells) + chance_cell + " \\\\")

    # Mean row
    lines.append("\\midrule")
    eligible_vals = [overall[key] for _, _, key, excl in all_cols
                     if not excl and overall[key] is not None]
    best, second = find_best_and_second(eligible_vals)
    cells = []
    for _, _, key, excl in all_cols:
        v = overall[key]
        if excl or v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-4:
            rank = 1
        elif second is not None and abs(v - second) < 1e-4:
            rank = 2
        else:
            rank = None
        cells.append(fmt_hc(v, rank=rank))
    chance_mean = ""
    if include_chance:
        chance_mean = f" & \\hc{{{overall_lnk:.2f}}}" if overall_lnk is not None else " & \\hcn"
    lines.append(f"\\textbf{{Mean ({n_total})}} & " + " & ".join(cells) + chance_mean + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND_COMPACT)

    # Caption below the table
    if filter_by == "cog":
        lines.append(
            "\\caption{Mean NLL by task type for finetuned models (bf16), "
            f"restricted to the {n_total} experiments (of 46) for which a "
            "cognitive model baseline is available. "
            f"{n_excluded} experiments without a reported cognitive baseline "
            "are excluded to enable direct comparison across all columns. "
            "Finetuned families: \\texttt{Qwentaur}, \\texttt{Llama-Centaur}, "
            "\\texttt{Smoltaur}, \\texttt{Olmotaur}. Base columns: "
            "\\texttt{Q-8B}/\\texttt{Q-14B} (Qwen3), \\texttt{L-8B} (Llama-3.1). "
            "Subscript $_r$ denotes our reproducing evaluation of the original "
            "Centaur model distributed by \\citet{binz2025foundation}, "
            "evaluated under identical python library and CUDA versions as our small "
            "foundation models for fair comparison. "
            "Subscript $_p$ denotes values published by \\citet{binz2025foundation}. "
            "\\texttt{70B}$_p$ is shown for reference but excluded from "
            "best/second-best marking, as it was evaluated under different "
            "software conditions. "
            "Cell colour encodes performance; "
            "\\underline{\\textbf{bold+underline}} marks the best model, "
            "\\underline{underline} the second-best. "
            "Full per-experiment results in Appendix~"
            "\\ref{app:full_experiment_results}.}"
        )
        lines.append("\\label{tab:main_tasktype_filtered}")
    elif filter_by == "lnk":
        lines.append(
            "\\caption{Mean NLL by task type for finetuned models (bf16), "
            f"restricted to the {n_total} experiments (of 46) that have both "
            "a reported cognitive model baseline and a well-defined discrete "
            "response space admitting a chance-level baseline "
            "$\\ln(k)$, where $k$ is the number of response options. "
            f"{n_excluded} experiments lacking either a cognitive baseline "
            "or a computable $\\ln(k)$ (continuous, mixed, or variable "
            "action spaces) are excluded. "
            "Finetuned families: \\texttt{Qwentaur}, \\texttt{Llama-Centaur}, "
            "\\texttt{Smoltaur}, \\texttt{Olmotaur}. Base columns: "
            "\\texttt{Q-8B}/\\texttt{Q-14B} (Qwen3), \\texttt{L-8B} (Llama-3.1). "
            "Subscript $_r$ denotes our reproducing evaluation of the original "
            "Centaur model distributed by \\citet{binz2025foundation}, "
            "evaluated under identical python library and CUDA versions as our small "
            "foundation models for fair comparison. "
            "Subscript $_p$ denotes values published by \\citet{binz2025foundation}. "
            "\\texttt{70B}$_p$ is shown for reference but excluded from "
            "best/second-best marking, as it was evaluated under different "
            "software conditions. "
            "Cell colour encodes performance; "
            "\\underline{\\textbf{bold+underline}} marks the best model, "
            "\\underline{underline} the second-best. "
            "Full per-experiment results in Appendix~"
            "\\ref{app:full_experiment_results}.}"
        )
        lines.append("\\label{tab:main_tasktype_chance}")
    else:
        lines.append(
            "\\caption{Mean NLL by task type for finetuned models (bf16). "
            "Finetuned families: \\texttt{Qwentaur}, \\texttt{Llama-Centaur}, "
            "\\texttt{Smoltaur}, \\texttt{Olmotaur}. Base columns: "
            "\\texttt{Q-8B}/\\texttt{Q-14B} (Qwen3), \\texttt{L-8B} (Llama-3.1). "
            "Subscript $_r$ denotes our reproducing evaluation of the original "
            "Centaur model distributed by \\citet{binz2025foundation}, "
            "evaluated under identical python library and CUDA versions as our small "
            "foundation models for fair comparison. "
            "Subscript $_p$ denotes values published by \\citet{binz2025foundation}. "
            "\\texttt{70B}$_p$ is shown for reference but excluded from "
            "best/second-best marking, as it was evaluated under different "
            "software conditions. "
            "Cell colour encodes performance; "
            "\\underline{\\textbf{bold+underline}} marks the best model, "
            "\\underline{underline} the second-best. "
            "Full per-experiment results in Appendix~"
            "\\ref{app:full_experiment_results}.}"
        )
        lines.append("\\label{tab:main_tasktype}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Main-text table: Normalised information captured
# =============================================================================

def generate_main_tasktype_normalised(data_rows, task_groups, ext):
    """Generate main table using normalised metric (ln(k) - NLL) / ln(k), for
    the four finetuned families (Qwentaur, Llama-Centaur, Smoltaur, Olmotaur)
    plus base and Binz-et-al. references.

    Filters to experiments with both a cognitive baseline and ln(k) > 0.
    Values range from 0 (chance) to 1 (perfect); higher is better.
    """
    # Flat columns: (label, source, key, exclude_from_marking)
    model_cols = [(lbl, src, key, False)
                  for _, cols in TASKTYPE_FT_GROUPS for lbl, src, key in cols]
    base_cols = [
        ("\\texttt{Q-8B}",  "aggr", "qwen_8b_base_bf16",  False),
        ("\\texttt{Q-14B}", "aggr", "qwen_14b_base_bf16", False),
        ("\\texttt{L-8B}",  "aggr", "llama_8b_base_bf16", False),
    ]
    ref_model_cols = [
        ("\\texttt{70B}$_p$", "aggr", "centaur_70b_rep",  True),   # ref only
        ("\\texttt{70B}$_r$", "aggr", "centaur_70b_repr", False),
        ("Cog.$_p$",          "aggr", "cog_model",        False),
    ]
    all_cols = model_cols + base_cols + ref_model_cols
    n_base = len(base_cols)
    n_ref = len(ref_model_cols)

    def gv(source, key, r):
        if source == "aggr":
            return parse_val(r[3 + COL[key]])
        return ext.get(key, {}).get(r[0].strip())

    # Filter: ln(k) > 0 AND cog baseline present
    filt_data_rows = [r for r in data_rows
                      if (LN_K.get(r[0].strip()) is not None
                          and LN_K.get(r[0].strip()) > 0
                          and parse_val(r[3 + COL["cog_model"]]) is not None)]
    n_total = len(filt_data_rows)
    n_excluded = len(data_rows) - n_total

    filt_task_groups = defaultdict(list)
    for row in filt_data_rows:
        filt_task_groups[row[2]].append(row)

    def normalise(nll, lnk):
        """(ln(k) - NLL) / ln(k): fraction of information above chance."""
        if nll is None or lnk is None or lnk <= 0:
            return None
        return (lnk - nll) / lnk

    # Normalised means per task type
    task_means = {}
    for task_type in TASK_ORDER:
        group = filt_task_groups.get(task_type, [])
        means = {}
        for label, source, key, _ in all_cols:
            vals = []
            for r in group:
                nll = gv(source, key, r)
                lnk = LN_K.get(r[0].strip())
                n = normalise(nll, lnk)
                if n is not None:
                    vals.append(n)
            means[key] = sum(vals) / len(vals) if vals else None
        task_means[task_type] = means

    # Overall mean
    overall = {}
    for label, source, key, _ in all_cols:
        vals = []
        for r in filt_data_rows:
            nll = gv(source, key, r)
            lnk = LN_K.get(r[0].strip())
            n = normalise(nll, lnk)
            if n is not None:
                vals.append(n)
        overall[key] = sum(vals) / len(vals) if vals else None

    def find_best_and_second_high(values, tol=1e-6):
        """Best = highest, second = next highest."""
        numeric = sorted(set(round(v, 6) for v in values if v is not None),
                         reverse=True)
        best = numeric[0] if len(numeric) >= 1 else None
        second = numeric[1] if len(numeric) >= 2 else None
        return best, second

    def fmt_norm(val, rank=None):
        """Format normalised cell with reversed heatmap colour."""
        if val is None:
            return "\\hcn"
        if rank == 1:
            return f"\\hcRbu{{{val:.2f}}}"
        if rank == 2:
            return f"\\hcRu{{{val:.2f}}}"
        return f"\\hcR{{{val:.2f}}}"

    # --- Build table ---
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")

    group_sizes = [len(cols) for _, cols in TASKTYPE_FT_GROUPS] + [n_base, n_ref]
    colspec = "@{}l"
    for gs in group_sizes:
        colspec += "  " + " r" * gs
    colspec += "@{}"
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header row 1: family groups + Base + Binz references
    h1, col, cmids = "", 2, []
    for hdr, cols in TASKTYPE_FT_GROUPS:
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
    for label, _, _, _ in all_cols:
        h2 += f" & {label}"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # Task-type rows
    for task_type in TASK_ORDER:
        means = task_means[task_type]
        group = filt_task_groups.get(task_type, [])
        n_exps = len(group)
        if n_exps == 0:
            continue
        label = TASK_ABBREV.get(task_type, task_type)
        label_with_n = f"{label} ({n_exps})"

        eligible_vals = [means[key] for _, _, key, excl in all_cols
                         if not excl and means[key] is not None]
        best, second = find_best_and_second_high(eligible_vals)

        cells = []
        for _, _, key, excl in all_cols:
            v = means[key]
            if excl or v is None:
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
    eligible_vals = [overall[key] for _, _, key, excl in all_cols
                     if not excl and overall[key] is not None]
    best, second = find_best_and_second_high(eligible_vals)
    cells = []
    for _, _, key, excl in all_cols:
        v = overall[key]
        if excl or v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-4:
            rank = 1
        elif second is not None and abs(v - second) < 1e-4:
            rank = 2
        else:
            rank = None
        cells.append(fmt_norm(v, rank=rank))
    lines.append(f"\\textbf{{Mean ({n_total})}} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND_REVERSED)

    # Caption below
    lines.append(
        "\\caption{Fraction of available information captured above chance, "
        "$(\\ln k - \\mathrm{NLL}) \\,/\\, \\ln k$, "
        "by task type for finetuned models (bf16). "
        "A value of 0 indicates chance-level performance; "
        "1 indicates perfect prediction. "
        f"Restricted to {n_total} experiments (of 46) with both a "
        "cognitive model baseline and a well-defined discrete response "
        "space ($\\ln k > 0$); "
        f"{n_excluded} experiments are excluded. "
        "Finetuned families: \\texttt{Qwentaur}, \\texttt{Llama-Centaur}, "
        "\\texttt{Smoltaur}, \\texttt{Olmotaur}. Base columns: "
        "\\texttt{Q-8B}/\\texttt{Q-14B} (Qwen3), \\texttt{L-8B} (Llama-3.1). "
        "Subscript $_r$ denotes our reproducing evaluation of the original "
        "Centaur model distributed by \\citet{binz2025foundation}, "
        "evaluated under identical python library and CUDA versions as our small "
        "foundation models for fair comparison. "
        "Subscript $_p$ denotes values published by \\citet{binz2025foundation}. "
        "\\texttt{70B}$_p$ is shown for reference but excluded from "
        "best/second-best marking, as it was evaluated under different "
        "software conditions. "
        "\\underline{\\textbf{Bold+underline}} marks the best model, "
        "\\underline{underline} the second-best. "
        "Full per-experiment results in Appendix~"
        "\\ref{app:full_experiment_results}.}"
    )
    lines.append("\\label{tab:main_tasktype_normalised}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Appendix table: Non-cognitive control comparison
# =============================================================================

# CSV files for external models (task,loss format from eval_model.py)
CONTROL_CSVS = {
    # Non-cognitive finetuned controls
    "hermes_3.2_3b":   "controls/NousResearch-Hermes-3-Llama-3.2-3B.csv",
    "hermes_3.1_8b":   "controls/NousResearch-Hermes-3-Llama-3.1-8B.csv",
    "hermes_4_14b":    "controls/NousResearch-Hermes-4-14B.csv",
    "nemotron_4b":     "controls/nvidia-Llama-3.1-Nemotron-Nano-4B-v1.1.csv",
    "nemotron_8b":     "controls/nvidia-Llama-3.1-Nemotron-Nano-8B-v1.csv",
    # Other cognitive/behavioural FMs
    "befm_8b":         "controls/befm-Be_FM-8B.csv",
    "socrates_l_8b":   "controls/socratesft-socrates-llama3-8b-sft.csv",
    "socrates_q_14b":  "controls/socratesft-socrates-qwen2.5-14b-sft.csv",
}


def read_control_csv(path):
    """Read a (task, loss) CSV into a dict {task_code: float}."""
    data = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data[row[0].strip()] = float(row[1])
    return data


def generate_control_comparison(rows, control_dir):
    """Generate appendix table comparing cognitive SFT models with
    non-cognitive controls and other behavioural FMs on Psych-101."""

    # --- Load external CSVs ---
    ext = {}
    for key, relpath in CONTROL_CSVS.items():
        fpath = os.path.join(control_dir, os.path.basename(relpath))
        if os.path.exists(fpath):
            ext[key] = read_control_csv(fpath)
        else:
            # Try the full relative path
            if os.path.exists(relpath):
                ext[key] = read_control_csv(relpath)
            else:
                print(f"  WARNING: {relpath} not found, skipping {key}")

    # --- Column specification ---
    # (label, source, key/col_key, exclude_from_bold)
    #   source="aggr" -> read from psych101_aggr.csv via COL
    #   source="ext"  -> read from external CSV dict
    cols_cognitive = [
        ("\\texttt{Q-8B}",    "aggr", "qwentaur_8b_bf16",  False),
        ("\\texttt{Q-14B}",   "aggr", "qwentaur_14b_bf16", False),
        ("\\texttt{LC-8B}",   "aggr", "centaur_8b_bf16",   False),
        ("\\texttt{C-70B}$_r$", "aggr", "centaur_70b_repr", False),
        ("Cog.$_p$",          "aggr", "cog_model",          False),
    ]
    cols_other_cog = [
        ("\\texttt{Be.FM-8B}",      "ext", "befm_8b",        False),
        ("\\texttt{Socr.-8B}",    "ext", "socrates_l_8b",  False),
        ("\\texttt{Socr.-14B}",   "ext", "socrates_q_14b", False),
    ]
    cols_controls = [
        ("\\texttt{Herm.-3B}",  "ext", "hermes_3.2_3b",  False),
        ("\\texttt{Herm.-8B}",  "ext", "hermes_3.1_8b",  False),
        ("\\texttt{Herm.-14B}", "ext", "hermes_4_14b",   False),
        ("\\texttt{Nemo.-4B}",  "ext", "nemotron_4b",    False),
        ("\\texttt{Nemo.-8B}",  "ext", "nemotron_8b",    False),
    ]
    all_cols = cols_cognitive + cols_other_cog + cols_controls
    n_cog = len(cols_cognitive)
    n_other = len(cols_other_cog)
    n_ctrl = len(cols_controls)

    def get_val(row, col_spec):
        """Get value for a row given (label, source, key, excl) spec."""
        _, source, key, _ = col_spec
        if source == "aggr":
            return parse_val(row[3 + COL[key]])
        else:  # "ext"
            task_code = row[0].strip()
            if key in ext and task_code in ext[key]:
                return ext[key][task_code]
            return None

    # --- Separate data rows ---
    data_rows = [r for r in rows if r[0].strip() != "Mean"]

    # --- Build table ---
    lines = []
    lines.append("\\begin{sidewaystable}")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")

    n_total = n_cog + n_other + n_ctrl
    colspec = "@{}l l " + " r" * n_cog + "  " + " r" * n_other + "  " + " r" * n_ctrl + "  r" + "@{}"
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Group headers
    c_cog_start = 3
    c_cog_end = c_cog_start + n_cog - 1
    c_other_start = c_cog_end + 1
    c_other_end = c_other_start + n_other - 1
    c_ctrl_start = c_other_end + 1
    c_ctrl_end = c_ctrl_start + n_ctrl - 1

    lines.append(
        f"& & \\multicolumn{{{n_cog}}}{{c}}{{\\textbf{{Cognitive SFT}}}} "
        f"& \\multicolumn{{{n_other}}}{{c}}{{\\textbf{{Other behavioural FMs}}}} "
        f"& \\multicolumn{{{n_ctrl}}}{{c}}{{\\textbf{{Non-cognitive finetuned}}}} "
        f"& \\textbf{{Chance}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{c_cog_start}-{c_cog_end}}}"
        f"\\cmidrule(lr){{{c_other_start}-{c_other_end}}}"
        f"\\cmidrule(lr){{{c_ctrl_start}-{c_ctrl_end}}}"
    )

    # Column labels
    h2 = "\\textbf{Experiment} & \\textbf{Type}"
    for label, _, _, _ in all_cols:
        h2 += f" & {label}"
    h2 += " & $\\ln(k)$"
    h2 += " \\\\"
    lines.append(h2)
    lines.append("\\midrule")

    # Data rows
    prev_task = None
    for row in data_rows:
        exp_code = row[0].strip()
        task_name = row[1].strip()
        task_type = row[2].strip()

        display, cite = CITE_MAP.get(exp_code, (task_name, exp_code))
        cell_exp = f"{display} {{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}"
        cell_task = TASK_ABBREV.get(task_type, task_type)
        if prev_task is not None and task_type != prev_task and task_type != "":
            lines.append("\\midrule")

        if task_type:
            prev_task = task_type

        # Collect values
        cell_info = []  # (value, exclude_from_bold)
        for col_spec in all_cols:
            v = get_val(row, col_spec)
            cell_info.append((v, col_spec[3]))

        # Best/second among non-excluded
        eligible = [v for v, excl in cell_info if v is not None and not excl]
        best, second = find_best_and_second(eligible)

        cells = []
        for v, excl in cell_info:
            if excl or v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-6:
                rank = 1
            elif second is not None and abs(v - second) < 1e-6:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(v, rank=rank))

        line = f"{cell_exp} & {cell_task} & " + " & ".join(cells) + f" & {fmt_lnk(exp_code)}" + " \\\\"
        lines.append(line)

    # Mean row
    lines.append("\\midrule")
    mean_vals = []
    for col_spec in all_cols:
        vals = [get_val(row, col_spec) for row in data_rows]
        mean_vals.append((mean_of(vals), col_spec[3]))

    eligible = [v for v, excl in mean_vals if v is not None and not excl]
    best, second = find_best_and_second(eligible)
    cells = []
    for v, excl in mean_vals:
        if excl or v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-4:
            rank = 1
        elif second is not None and abs(v - second) < 1e-4:
            rank = 2
        else:
            rank = None
        cells.append(fmt_hc(v, rank=rank))
    # Mean ln(k)
    lnk_vals = [LN_K.get(r[0].strip()) for r in data_rows]
    lnk_nums = [v for v in lnk_vals if v is not None]
    lnk_mean = f"\\hc{{{sum(lnk_nums)/len(lnk_nums):.2f}}}" if lnk_nums else "\\hcn"
    lines.append(f"\\textbf{{Mean (all 46)}} & & " + " & ".join(cells) + f" & {lnk_mean}" + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best. "
        "Lower is better. ``---''\\,=\\,not available. "
        "\\texttt{Q} = \\texttt{Qwentaur}; "
        "\\texttt{LC} = \\texttt{Llama-Centaur}; "
        "\\texttt{C-70B}$_r$ = Centaur-70B (reproduced); "
        "Cog.$_p$ = best cognitive model \\citep{binz2025foundation}; "
        "\\texttt{Herm.} = Hermes \\citep{teknium2024hermes}; "
        "\\texttt{Nemo.} = Nemotron \\citep{nvidia2025nemotron}; "
        "\\texttt{Socr.} = Socrates \\citep{kolluri2025finetuning}. "
        "Cognitive SFT models are finetuned on behavioural data "
        "(Psych-101 or domain-specific); "
        "non-cognitive models are finetuned for general instruction-following "
        "or reasoning tasks.}"
    )
    lines.append(
        "\\caption{Comparison of cognitively finetuned models with "
        "other behavioural foundation models and non-cognitive controls "
        "on Psych-101. Non-cognitive finetuned models (Hermes, Nemotron) "
        "are instruction-tuned or reasoning-optimised variants of the "
        "same base model families. Their substantially higher NLL "
        "demonstrates that finetuning on non-cognitive data does not "
        "align models with human behavioural patterns, ruling out the "
        "hypothesis that any finetuning improves cognitive prediction.}"
    )
    lines.append("\\label{tab:non_cognitive_controls}")
    lines.append("\\end{sidewaystable}")

    return "\n".join(lines)


# =============================================================================
# Appendix table: Family-based control comparison
# =============================================================================

def generate_family_comparison(rows, control_dir):
    """Generate appendix table grouping models by base-model family
    (Qwen 2.5/3 vs Llama 3/3.1/3.2) with sub-groups by training purpose."""

    # --- Load external CSVs ---
    ext = {}
    for key, relpath in CONTROL_CSVS.items():
        fpath = os.path.join(control_dir, os.path.basename(relpath))
        if os.path.exists(fpath):
            ext[key] = read_control_csv(fpath)
        elif os.path.exists(relpath):
            ext[key] = read_control_csv(relpath)
        else:
            print(f"  WARNING: {relpath} not found, skipping {key}")

    # --- Column groups (label, source, key, exclude_from_bold) ---
    # Qwen 2.5/3 family
    qwen_base = [
        ("8B",  "aggr", "qwen_8b_base_bf16",  False),
        ("14B", "aggr", "qwen_14b_base_bf16", False),
    ]
    qwen_ours = [
        ("8B",  "aggr", "qwentaur_8b_bf16",  False),
        ("14B", "aggr", "qwentaur_14b_bf16", False),
    ]
    qwen_other_cog = [
        ("\\texttt{Socr.-14B}",  "ext", "socrates_q_14b", False),
    ]
    qwen_noncog = [
        ("\\texttt{Herm.-14B}",  "ext", "hermes_4_14b",   False),
    ]

    # Llama 3/3.1/3.2 family
    llama_base = [
        ("3B",  "aggr", "llama_3b_base_bf16", False),
        ("8B",  "aggr", "llama_8b_base_bf16", False),
    ]
    llama_ours = [
        ("3B",  "aggr", "centaur_3b_bf16", False),
        ("8B",  "aggr", "centaur_8b_bf16", False),
    ]
    llama_other_cog = [
        ("\\texttt{Be.FM-8B}",   "ext", "befm_8b",       False),
        ("\\texttt{Socr.-8B}", "ext", "socrates_l_8b",  False),
    ]
    llama_noncog = [
        ("\\texttt{Herm.-3B}",  "ext", "hermes_3.2_3b", False),
        ("\\texttt{Nemo.-4B}",  "ext", "nemotron_4b",   False),
        ("\\texttt{Herm.-8B}",  "ext", "hermes_3.1_8b", False),
        ("\\texttt{Nemo.-8B}",  "ext", "nemotron_8b",   False),
    ]

    # Binz et al. reference
    binz_ref = [
        ("\\texttt{C-70B}$_r$", "aggr", "centaur_70b_repr", False),
        ("Cog.$_p$",            "aggr", "cog_model",         False),
    ]

    # Flatten for iteration
    all_cols = (qwen_base + qwen_ours + qwen_other_cog + qwen_noncog
                + llama_base + llama_ours + llama_other_cog + llama_noncog
                + binz_ref)

    # Group sizes
    n_qb  = len(qwen_base)
    n_qo  = len(qwen_ours)
    n_qoc = len(qwen_other_cog)
    n_qnc = len(qwen_noncog)
    n_qwen = n_qb + n_qo + n_qoc + n_qnc

    n_lb  = len(llama_base)
    n_lo  = len(llama_ours)
    n_loc = len(llama_other_cog)
    n_lnc = len(llama_noncog)
    n_llama = n_lb + n_lo + n_loc + n_lnc

    n_binz = len(binz_ref)

    def get_val(row, col_spec):
        _, source, key, _ = col_spec
        if source == "aggr":
            return parse_val(row[3 + COL[key]])
        else:
            task_code = row[0].strip()
            if key in ext and task_code in ext[key]:
                return ext[key][task_code]
            return None

    data_rows = [r for r in rows if r[0].strip() != "Mean"]

    # --- Build table ---
    lines = []
    lines.append("\\begin{sidewaystable}")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")

    # Column spec with small gaps between families
    colspec = ("@{}l l "
               + " r" * n_qwen + "  "
               + " r" * n_llama + "  "
               + " r" * n_binz + "  r" + "@{}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # --- Header level 1: Family groups ---
    # Column positions (1-indexed, col 1 = Experiment, col 2 = Type)
    q_start = 3
    q_end = q_start + n_qwen - 1
    l_start = q_end + 1
    l_end = l_start + n_llama - 1
    b_start = l_end + 1
    b_end = b_start + n_binz - 1

    lines.append(
        f"& & \\multicolumn{{{n_qwen}}}{{c}}{{\\texttt{{\\textbf{{Qwen 2.5/3}}}} family}} "
        f"& \\multicolumn{{{n_llama}}}{{c}}{{\\texttt{{\\textbf{{Llama 3/3.1/3.2}}}} family}} "
        f"& \\multicolumn{{{n_binz}}}{{c}}{{\\citet{{binz2025foundation}}}} "
        f"& \\textbf{{Chance}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{q_start}-{q_end}}}"
        f"\\cmidrule(lr){{{l_start}-{l_end}}}"
    )

    # --- Header level 2: Sub-groups ---
    # Qwen sub-groups
    qb_s = q_start
    qb_e = qb_s + n_qb - 1
    qo_s = qb_e + 1
    qo_e = qo_s + n_qo - 1
    qoc_s = qo_e + 1
    qoc_e = qoc_s + n_qoc - 1
    qnc_s = qoc_e + 1
    qnc_e = qnc_s + n_qnc - 1

    # Llama sub-groups
    lb_s = l_start
    lb_e = lb_s + n_lb - 1
    lo_s = lb_e + 1
    lo_e = lo_s + n_lo - 1
    loc_s = lo_e + 1
    loc_e = loc_s + n_loc - 1
    lnc_s = loc_e + 1
    lnc_e = lnc_s + n_lnc - 1

    lines.append(
        f"& & \\multicolumn{{{n_qb}}}{{c}}{{Base}} "
        f"& \\multicolumn{{{n_qo}}}{{c}}{{\\texttt{{Qwentaur}}}} "
        f"& \\multicolumn{{{n_qoc}}}{{c}}{{Other cog.}} "
        f"& \\multicolumn{{{n_qnc}}}{{c}}{{Non-cog.}} "
        f"& \\multicolumn{{{n_lb}}}{{c}}{{Base}} "
        f"& \\multicolumn{{{n_lo}}}{{c}}{{\\texttt{{Llama-Centaur}}}} "
        f"& \\multicolumn{{{n_loc}}}{{c}}{{Other cog.}} "
        f"& \\multicolumn{{{n_lnc}}}{{c}}{{Non-cog.}} "
        f"& & \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{qb_s}-{qb_e}}}"
        f"\\cmidrule(lr){{{qo_s}-{qo_e}}}"
        f"\\cmidrule(lr){{{qoc_s}-{qoc_e}}}"
        f"\\cmidrule(lr){{{qnc_s}-{qnc_e}}}"
        f"\\cmidrule(lr){{{lb_s}-{lb_e}}}"
        f"\\cmidrule(lr){{{lo_s}-{lo_e}}}"
        f"\\cmidrule(lr){{{loc_s}-{loc_e}}}"
        f"\\cmidrule(lr){{{lnc_s}-{lnc_e}}}"
    )

    # --- Header level 3: Model labels ---
    h3 = "\\textbf{Experiment} & \\textbf{Type}"
    for label, _, _, _ in all_cols:
        h3 += f" & {label}"
    h3 += " & $\\ln(k)$"
    h3 += " \\\\"
    lines.append(h3)
    lines.append("\\midrule")

    # --- Data rows ---
    prev_task = None
    for row in data_rows:
        exp_code = row[0].strip()
        task_name = row[1].strip()
        task_type = row[2].strip()

        display, cite = CITE_MAP.get(exp_code, (task_name, exp_code))
        cell_exp = f"{display} {{\\fontsize{{4}}{{5}}\\selectfont\\citep{{{cite}}}}}"
        cell_task = TASK_ABBREV.get(task_type, task_type)
        if prev_task is not None and task_type != prev_task and task_type != "":
            lines.append("\\midrule")
        if task_type:
            prev_task = task_type

        cell_info = []
        for col_spec in all_cols:
            v = get_val(row, col_spec)
            cell_info.append((v, col_spec[3]))

        eligible = [v for v, excl in cell_info if v is not None and not excl]
        best, second = find_best_and_second(eligible)

        cells = []
        for v, excl in cell_info:
            if excl or v is None:
                rank = None
            elif best is not None and abs(v - best) < 1e-6:
                rank = 1
            elif second is not None and abs(v - second) < 1e-6:
                rank = 2
            else:
                rank = None
            cells.append(fmt_hc(v, rank=rank))

        lines.append(f"{cell_exp} & {cell_task} & " + " & ".join(cells) + f" & {fmt_lnk(exp_code)}" + " \\\\")

    # --- Mean row ---
    lines.append("\\midrule")
    mean_vals = []
    for col_spec in all_cols:
        vals = [get_val(row, col_spec) for row in data_rows]
        mean_vals.append((mean_of(vals), col_spec[3]))

    eligible = [v for v, excl in mean_vals if v is not None and not excl]
    best, second = find_best_and_second(eligible)
    cells = []
    for v, excl in mean_vals:
        if excl or v is None:
            rank = None
        elif best is not None and abs(v - best) < 1e-4:
            rank = 1
        elif second is not None and abs(v - second) < 1e-4:
            rank = 2
        else:
            rank = None
        cells.append(fmt_hc(v, rank=rank))
    # Mean ln(k)
    lnk_vals = [LN_K.get(r[0].strip()) for r in data_rows]
    lnk_nums = [v for v in lnk_vals if v is not None]
    lnk_mean = f"\\hc{{{sum(lnk_nums)/len(lnk_nums):.2f}}}" if lnk_nums else "\\hcn"
    lines.append(f"\\textbf{{Mean (all 46)}} & & " + " & ".join(cells) + f" & {lnk_mean}" + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best. "
        "Lower is better. ``---''\\,=\\,not available. "
        "Models are grouped by base-model family. "
        "``Base'' = pretrained model without finetuning (bf16); "
        "``Other cog.'' = other cognitive/behavioural foundation models "
        "finetuned on non-Psych-101 behavioural data; "
        "``Non-cog.'' = models finetuned for general instruction-following "
        "or reasoning. "
        "\\texttt{Herm.} = Hermes \\citep{teknium2024hermes}; "
        "\\texttt{Nemo.} = Nemotron \\citep{nvidia2025nemotron}; "
        "\\texttt{Socr.} = Socrates \\citep{kolluri2025finetuning}; "
        "\\texttt{Be.FM} = \\citet{xie2025fm}. "
        "Base model versions differ within families: "
        "\\texttt{Qwentaur} uses Qwen3; \\texttt{Socr.-14B} uses Qwen2.5; "
        "\\texttt{Herm.-14B} uses Qwen2.5. "
        "\\texttt{Llama-Centaur} uses Llama-3.1/3.2; "
        "\\texttt{Socr.-8B} uses Llama-3; "
        "\\texttt{Be.FM} uses Llama-3.1.}"
    )
    lines.append(
        "\\caption{Comparison of cognitively finetuned models with "
        "other behavioural foundation models and non-cognitive controls "
        "on Psych-101, grouped by base-model family. "
        "Within each family, \\texttt{Qwentaur}/\\texttt{Llama-Centaur} "
        "(cognitive SFT on Psych-101) consistently outperform both "
        "other behavioural FMs and non-cognitive controls, "
        "demonstrating that alignment with human behavioural patterns "
        "requires cognitive finetuning on structured experimental data, "
        "not merely finetuning per se.}"
    )
    lines.append("\\label{tab:family_comparison}")
    lines.append("\\end{sidewaystable}")

    return "\n".join(lines)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    header, rows = read_csv(CSV_PATH)

    data_rows = [r for r in rows if r[0] != "Mean"]
    task_groups = defaultdict(list)
    for row in data_rows:
        task_groups[row[2]].append(row)

    fn_common = (
        "Lower is better. ``---''\\,=\\,no cognitive baseline reported. "
        "Size labels in billions of parameters. "
        "Subscript $_r$ denotes our reproducing evaluation of the original "
        "Centaur model under identical library and CUDA versions as our "
        "models; $_p$ denotes values published by \\citet{binz2025foundation}. "
        "C$_p$ is included for reference but excluded from best/second-best "
        "marking, as it was evaluated under different software conditions. "
        "$^\\dagger$Horizon task (5 experiments) and Two-step task "
        "(3 experiments) are reported as single merged values by "
        "\\citet{binz2025foundation}. Rounded to 2\\,d.p."
    )

    fn_common_baseft = (
        "Lower is better. ``---''\\,=\\,no cognitive baseline reported. "
        "Size labels in billions of parameters. "
        "Subscript $_r$ denotes our reproducing evaluation of the original "
        "Centaur model under identical library and CUDA versions as our "
        "models; $_p$ denotes values published by \\citet{binz2025foundation}. "
        "C$_p$ is included for reference but excluded from best/second-best "
        "marking, as it was evaluated under different software conditions. "
        "$^\\dagger$Horizon task (5 experiments) and Two-step task "
        "(3 experiments) are reported as single merged values by "
        "\\citet{binz2025foundation}. Rounded to 2\\,d.p."
    )

    # Smoltaur/Olmotaur per-task NLLs (not in aggr) for the family tables.
    ext = load_family_ext(os.path.join(OUT_DIR, ".."))

    # ---- Table 1: All four finetuned families (bf16, LoRA r=16) ----
    tex1 = generate_ft_bf16_all_families(rows, os.path.join(OUT_DIR, ".."))
    with open(os.path.join(OUT_DIR, "heatmap_ft_bf16_all.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex1)
    print(f"  Wrote {OUT_DIR}/heatmap_ft_bf16_all.tex")

    # ---- Table 3: Base vs FT bf16 ----
    tex3 = generate_heatmap_table(
        rows,
        QWEN_BASEFT_BF16, LLAMA_BASEFT_BF16,
        REF_COLS_BASEFT, REF_LABELS_BASEFT,
        "base", "ft",
        "\\texttt{\\textbf{Qwen3}} family", "\\texttt{\\textbf{Llama-3.1/3.2}} family",
        "NLL under bf16 (half-precision) inference. "
        "Reference columns report 4-bit values as originally published. "
        + fn_common_baseft,
        "Base vs.\\ finetuned NLL on Psych-101 under bf16 inference. "
        "Within each model size, the left column (base) shows the "
        "pretrained model and the right column (ft) shows the "
        "cognitively finetuned variant. "
        "Cell colour encodes performance; the consistent colour shift "
        "from base to ft demonstrates the effect of cognitive "
        "finetuning across all scales.",
        "tab:heatmap_baseft_bf16",
    )
    with open(os.path.join(OUT_DIR, "heatmap_baseft_bf16.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex3)
    print(f"  Wrote {OUT_DIR}/heatmap_baseft_bf16.tex")

    # ---- Table 4: Main-text task-type summary ----
    tex4 = generate_main_tasktype(data_rows, task_groups, ext)
    with open(os.path.join(OUT_DIR, "main_tasktype.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex4)
    print(f"  Wrote {OUT_DIR}/main_tasktype.tex")

    # ---- Table 5: Filtered main-text task-type summary (cog) ----
    tex5 = generate_main_tasktype(data_rows, task_groups, ext, filter_by="cog")
    with open(os.path.join(OUT_DIR, "main_tasktype_filtered.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex5)
    print(f"  Wrote {OUT_DIR}/main_tasktype_filtered.tex")

    # ---- Table 5b: Filtered main-text task-type with chance column ----
    tex5b = generate_main_tasktype(data_rows, task_groups, ext, filter_by="lnk")
    with open(os.path.join(OUT_DIR, "main_tasktype_chance.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex5b)
    print(f"  Wrote {OUT_DIR}/main_tasktype_chance.tex")

    # ---- Table 7 (non-cognitive control comparison) -- COMMENTED OUT ----
    # tex6 = generate_control_comparison(rows, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    # with open(os.path.join(OUT_DIR, "non_cognitive_controls.tex"), "w", encoding="utf-8") as f:
    #     f.write(PREAMBLE + "\n" + tex6)
    # print(f"  Wrote {OUT_DIR}/non_cognitive_controls.tex")

    # ---- Table 7: Family-based comparison ----
    tex7 = generate_family_comparison(rows, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    with open(os.path.join(OUT_DIR, "family_comparison.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex7)
    print(f"  Wrote {OUT_DIR}/family_comparison.tex")

    # ---- Table 8: Normalised main-text task-type ----
    tex8 = generate_main_tasktype_normalised(data_rows, task_groups, ext)
    with open(os.path.join(OUT_DIR, "main_tasktype_normalised.tex"), "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n" + tex8)
    print(f"  Wrote {OUT_DIR}/main_tasktype_normalised.tex")

    print("\nDone. All seven tables written to", OUT_DIR)


if __name__ == "__main__":
    main()
