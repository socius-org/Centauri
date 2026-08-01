# -*- coding: utf-8 -*-
"""
Ablation (Sequential) Heatmap Table
====================================

Generates a sideways appendix LaTeX table showing raw NLL under each
ablation condition for every model–task pair.

Each model size has 4 subcolumns:
    orig | inst | cont | hist
    (original, instruction_ablated, content_masked, history_only)

Usage:
    cd "results/perturbation/ablation (sequential)"
    python generate_heatmap_table.py

Reads:  ablation_results/*.csv
Writes: tables/ablation_appendix.tex
"""

import csv
import math
import os
import glob
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "ablation_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "tables")

# =============================================================================
# Ablation conditions (4 with data; choice_only always empty)
# =============================================================================

CONDITIONS = ["original", "instruction_ablated", "content_masked", "history_only"]
COND_LABELS = ["orig", "inst", "cont", "hist"]

# =============================================================================
# Model metadata
# =============================================================================

QWEN_SIZES = [
    ("\\texttt{0.6B}", "Qwentaur-0.6B-LoRA"),
    ("\\texttt{1.7B}", "Qwentaur-1.7B-LoRA"),
    ("\\texttt{4B}",   "Qwentaur-4B-LoRA"),
    ("\\texttt{8B}",   "Qwentaur-8B-LoRA"),
    ("\\texttt{14B}",  "Qwentaur-14B-LoRA"),
]
LLAMA_SIZES = [
    ("\\texttt{1B}",   "Llama-Centaur-1B-LoRA"),
    ("\\texttt{3B}",   "Llama-Centaur-3B-LoRA"),
    ("\\texttt{8B}",   "Llama-Centaur-8B-LoRA"),
]

ALL_MODELS = QWEN_SIZES + LLAMA_SIZES

CENTAUR_70B_KEY = "Llama-3.1-Centaur-70B-adapter"

# =============================================================================
# Experiment metadata
# =============================================================================

CITE_MAP = {
    "badham2017deficits":          ("Shepard categorization",          "badham2017deficits"),
    "bahrami2020four":             ("Drifting four-armed bandit",      "bahrami2020four"),
    "collsiöö2023MCPL":            ("Multiple-cue judgment",           "collsioo2023numerical"),
    "feng2021dynamics":            ("Horizon task",                    "feng2021dynamics"),
    "flesch2018comparing":         ("Gardening task",                  "flesch2018comparing"),
    "frey2017cct":                 ("Columbia card task",              "frey2017risk"),
    "garcia2023experiential":      ("Experiential-symbolic task",      "garcia2023experiential"),
    "gershman2018deconstructing":  ("Two-armed bandit",                "gershman2018deconstructing"),
    "gershman2020reward":          ("Cond.\\ assoc.\\ learning",      "collins2014working"),
    "hilbig2014generalized":       ("Multi-attribute DM",             "hilbig2014generalized"),
    "krueger2022identifying":      ("Risky choice",                    "krueger2024identifying"),
    "kool2016when":                ("Two-step task",                   "kool2016when"),
    "kool2017cost":                ("Two-step task",                   "kool2017cost"),
    "lefebvre2017behavioural":     ("Prob.\\ instrumental learning",   "lefebvre2017behavioural"),
    "levering2020revisiting":      ("Medin categorization",            "levering2020revisiting"),
    "peterson2021using":           ("choices13k",                       "peterson2021using"),
    "plonsky2018when":             ("CPC18",                           "plonsky2018when"),
    "sadeghiyeh2020temporal":      ("Horizon task",                    "sadeghiyeh2020temporal"),
    "schulz2020finding":           ("Structured bandit",               "schulz2020finding"),
    "somerville2017charting":      ("Horizon task",                    "somerville2017charting"),
    "speekenbrink2008learning":    ("Weather prediction task",         "speekenbrink2008learning"),
    "steingroever2015data":        ("Iowa gambling task",              "steingroever2015data"),
    "tomov2020discovery":          ("Virtual subway network",          "tomov2020discovery"),
    "tomov2021multitask":          ("Multi-task RL",                   "tomov2021multitask"),
    "waltz2020differential":       ("Horizon task",                    "waltz2020differential"),
    "wilson2014humans":            ("Horizon task",                    "wilson2014humans"),
    "wise2019acomputational":      ("Aversive learning",               "wise2019computational"),
    "wu2018generalisation":        ("Spatially correlated MAB",        "wu2018generalization"),
    "wulff2018description":        ("Decisions from description",      "wulff2018sampling"),
    "wulff2018sampling":           ("Decisions from experience",       "wulff2018sampling"),
    "xiong2023neural":             ("Changing bandit",                 "xiong2023neural"),
    "zorowitz2023data":            ("Two-step task",                   "zorowitz2023data"),
}

TASK_TYPE = {
    "badham2017deficits":          "Supervised learning",
    "bahrami2020four":             "Multi-armed bandits",
    "collsiöö2023MCPL":            "Supervised learning",
    "feng2021dynamics":            "Multi-armed bandits",
    "flesch2018comparing":         "Decision-making",
    "frey2017cct":                 "Decision-making",
    "garcia2023experiential":      "Decision-making",
    "gershman2018deconstructing":  "Multi-armed bandits",
    "gershman2020reward":          "Memory",
    "hilbig2014generalized":       "Decision-making",
    "krueger2022identifying":      "Decision-making",
    "kool2016when":                "Markov decision processes",
    "kool2017cost":                "Markov decision processes",
    "lefebvre2017behavioural":     "Multi-armed bandits",
    "levering2020revisiting":      "Supervised learning",
    "peterson2021using":           "Decision-making",
    "plonsky2018when":             "Decision-making",
    "sadeghiyeh2020temporal":      "Multi-armed bandits",
    "schulz2020finding":           "Multi-armed bandits",
    "somerville2017charting":      "Multi-armed bandits",
    "speekenbrink2008learning":    "Supervised learning",
    "steingroever2015data":        "Multi-armed bandits",
    "tomov2020discovery":          "Markov decision processes",
    "tomov2021multitask":          "Markov decision processes",
    "waltz2020differential":       "Multi-armed bandits",
    "wilson2014humans":            "Multi-armed bandits",
    "wise2019acomputational":      "Supervised learning",
    "wu2018generalisation":        "Multi-armed bandits",
    "wulff2018description":        "Decision-making",
    "wulff2018sampling":           "Multi-armed bandits",
    "xiong2023neural":             "Multi-armed bandits",
    "zorowitz2023data":            "Markov decision processes",
}

TASK_ABBREV = {
    "Decision-making":           "Decision",
    "Markov decision processes": "MDP",
    "Multi-armed bandits":       "Bandit",
    "Memory":                    "Memory",
    "Supervised learning":       "Sup.\\ learn.",
}

TASK_ORDER = [
    "Decision-making", "Markov decision processes",
    "Multi-armed bandits", "Memory", "Supervised learning",
]

_ln = math.log
LN_K = {
    "badham2017deficits":          _ln(2),
    "bahrami2020four":             _ln(4),
    "collsiöö2023MCPL":            _ln(9),
    "feng2021dynamics":            _ln(2),
    "flesch2018comparing":         _ln(2),
    "frey2017cct":                 _ln(2),
    "garcia2023experiential":      None,      # mixed: binary + probability estimates
    "gershman2018deconstructing":  _ln(2),
    "gershman2020reward":          _ln(3),
    "hilbig2014generalized":       _ln(2),
    "krueger2022identifying":      None,      # mixed: multi-stage variable actions
    "kool2016when":                _ln(2),
    "kool2017cost":                _ln(2),
    "lefebvre2017behavioural":     _ln(2),
    "levering2020revisiting":      None,      # mixed: binary + 9-point rating
    "peterson2021using":           _ln(2),
    "plonsky2018when":             _ln(2),
    "sadeghiyeh2020temporal":      _ln(2),
    "schulz2020finding":           _ln(8),
    "somerville2017charting":      _ln(2),
    "speekenbrink2008learning":    _ln(2),
    "steingroever2015data":        _ln(4),
    "tomov2020discovery":          _ln(5),
    "tomov2021multitask":          _ln(3),
    "waltz2020differential":       _ln(2),
    "wilson2014humans":            _ln(2),
    "wise2019acomputational":      None,      # continuous probability estimates
    "wu2018generalisation":        _ln(30),
    "wulff2018description":        _ln(2),
    "wulff2018sampling":           None,      # mixed: sample/stop/choose phases
    "xiong2023neural":             _ln(2),
    "zorowitz2023data":            _ln(2),
}

EXPERIMENT_ORDER = sorted(LN_K.keys())  # includes tasks with LN_K=None


# =============================================================================
# Data loading
# =============================================================================


def load_all_results():
    """Load all ablation CSVs.

    Returns: {model_key: {task: {condition: loss}}}
    """
    files = glob.glob(os.path.join(RESULTS_DIR, "*_ablation.csv"))
    data = {}
    for f in files:
        basename = os.path.basename(f).replace("_ablation.csv", "")
        model_key = basename.replace("socius-", "").replace("marcelbinz-", "")
        task_cond = {}
        with open(f, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cond = row["condition"]
                loss = row["loss"].strip()
                task = row["task"]
                if cond not in CONDITIONS or loss == "" or task not in LN_K:
                    continue
                task_cond.setdefault(task, {})[cond] = float(loss)
        data[model_key] = task_cond
    return data


# =============================================================================
# Helpers
# =============================================================================


def mean_of(vals):
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def find_best_and_second(values):
    """Best = lowest NLL."""
    numeric = sorted(set(round(v, 6) for v in values if v is not None))
    best = numeric[0] if len(numeric) >= 1 else None
    second = numeric[1] if len(numeric) >= 2 else None
    return best, second


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


# =============================================================================
# Table generator
# =============================================================================


def generate_ablation_table(data):
    """Sideways appendix table: raw NLL under each ablation condition.

    Column layout per model size: orig | inst | cont | hist
    Grouped by family: Qwentaur (5 sizes) | Llama-Centaur (3 sizes) | Chance
    """
    n_qwen = len(QWEN_SIZES)
    n_llama = len(LLAMA_SIZES)
    n_cond = len(CONDITIONS)
    n_qwen_cols = n_qwen * n_cond     # 5 × 4 = 20
    n_llama_cols = n_llama * n_cond    # 3 × 4 = 12

    # Column spec: grouped 4-columns per model size
    colspec = "@{}l l"
    for _ in ALL_MODELS:
        colspec += "  " + " ".join(["r"] * n_cond)
    colspec += "  r"   # ln(k)
    colspec += "@{}"

    lines = []
    lines.append("\\begin{sidewaystable}[p]")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # --- Header row 1: family groups ---
    first_qwen = 3
    last_qwen = first_qwen + n_qwen_cols - 1
    first_llama = last_qwen + 1
    last_llama = first_llama + n_llama_cols - 1

    lines.append(
        f"& & \\multicolumn{{{n_qwen_cols}}}{{c}}"
        f"{{\\texttt{{\\textbf{{Qwentaur}}}}}} "
        f"& \\multicolumn{{{n_llama_cols}}}{{c}}"
        f"{{\\texttt{{\\textbf{{Llama-Centaur}}}}}} "
        f"& \\textbf{{Chance}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{first_qwen}-{last_qwen}}}"
        f"\\cmidrule(lr){{{first_llama}-{last_llama}}}"
    )

    # --- Header row 2: size groups ---
    h2 = "& "
    for label, _ in ALL_MODELS:
        h2 += f"& \\multicolumn{{{n_cond}}}{{c}}{{{label}}} "
    h2 += "& \\\\"
    lines.append(h2)

    # Cmidrules under each size group
    rules = []
    col = first_qwen
    for _ in ALL_MODELS:
        rules.append(f"\\cmidrule(lr){{{col}-{col + n_cond - 1}}}")
        col += n_cond
    lines.append(" ".join(rules))

    # --- Header row 3: condition labels ---
    h3 = "\\textbf{Experiment} & \\textbf{Type}"
    for _ in ALL_MODELS:
        for cl in COND_LABELS:
            h3 += f" & {{\\fontsize{{4}}{{5}}\\selectfont {cl}}}"
    h3 += " & $\\ln(k)$"
    h3 += " \\\\"
    lines.append(h3)
    lines.append("\\midrule")

    # --- Data rows grouped by task type ---
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

            # Collect all values for this row (model × condition)
            vals = []
            for _, model_key in ALL_MODELS:
                for cond in CONDITIONS:
                    v = data.get(model_key, {}).get(exp, {}).get(cond)
                    vals.append(v)

            # Best/second across ALL cells in the row (lower = better)
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
    for _, model_key in ALL_MODELS:
        for cond in CONDITIONS:
            cond_vals = [data.get(model_key, {}).get(exp, {}).get(cond)
                         for exp in EXPERIMENT_ORDER]
            mean_vals.append(mean_of(cond_vals))

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
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny \\underline{\\textbf{Bold+underline}}\\,=\\,best; "
        "\\underline{underline}\\,=\\,second-best. "
        "Lower is better. "
        "Condition labels: "
        "\\textbf{orig}\\,=\\,original prompt, "
        "\\textbf{inst}\\,=\\,instruction ablated, "
        "\\textbf{cont}\\,=\\,content masked, "
        "\\textbf{hist}\\,=\\,history only.}"
    )
    lines.append(
        "\\caption{Per-experiment NLL under sequential ablation conditions "
        "on Psych-101 (in-distribution). "
        "Each model size shows four columns corresponding to progressive "
        "prompt degradation: "
        "\\emph{orig} retains the full prompt; "
        "\\emph{inst} removes task instructions; "
        "\\emph{cont} additionally masks stimulus values and feedback; "
        "\\emph{hist} further removes trial structure, "
        "leaving only the response history. "
        "The $\\ln(k)$ column shows the random-guessing baseline where "
        "$k$ is the number of per-trial response options. "
        "Experiments with mixed or continuous response formats "
        "have no well-defined $k$ and are shown as {---}.}"
    )
    lines.append("\\label{tab:ablation_appendix}")
    lines.append("\\end{sidewaystable}")

    return "\n".join(lines)


# =============================================================================
# Centaur-70B table (single model, normal table orientation)
# =============================================================================


def generate_centaur70b_table(data):
    """Normal (non-sideways) table for Centaur-70B ablation results.

    Rows = experiments grouped by task type.
    Columns = orig | inst | cont | hist | ln(k)
    """
    model_key = CENTAUR_70B_KEY
    n_cond = len(CONDITIONS)

    colspec = "@{}l l " + " ".join(["r"] * n_cond) + " r@{}"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")

    # Header
    h = ("\\textbf{Experiment} & \\textbf{Type}"
         " & \\textbf{orig} & \\textbf{inst}"
         " & \\textbf{cont} & \\textbf{hist}"
         " & $\\ln(k)$ \\\\")
    lines.append(h)
    lines.append("\\midrule")

    # Data rows grouped by task type
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
                        f"{{\\fontsize{{6}}{{7}}\\selectfont\\citep{{{cite}}}}}")
            cell_type = TASK_ABBREV.get(task_type, task_type)

            vals = [data.get(model_key, {}).get(exp, {}).get(c) for c in CONDITIONS]
            cells = [fmt_hc(v) for v in vals]

            lines.append(f"{cell_exp} & {cell_type} & "
                         + " & ".join(cells)
                         + f" & {fmt_lnk(exp)}" + " \\\\")

    # Mean row
    lines.append("\\midrule")
    mean_vals = []
    for cond in CONDITIONS:
        cond_vals = [data.get(model_key, {}).get(exp, {}).get(cond)
                     for exp in EXPERIMENT_ORDER]
        mean_vals.append(mean_of(cond_vals))

    cells = [fmt_hc(v) for v in mean_vals]

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
    lines.append(LEGEND)
    lines.append("\\vspace{-2pt}")
    lines.append(
        "{\\tiny Lower is better. "
        "Condition labels: "
        "\\textbf{orig}\\,=\\,original prompt, "
        "\\textbf{inst}\\,=\\,instruction ablated, "
        "\\textbf{cont}\\,=\\,content masked, "
        "\\textbf{hist}\\,=\\,history only.}"
    )
    lines.append(
        "\\caption{Per-experiment NLL under sequential ablation conditions "
        "for Centaur-70B \\citep{binz2025foundation} "
        "on Psych-101 (in-distribution). "
        "Columns correspond to progressive prompt degradation: "
        "\\emph{orig} retains the full prompt; "
        "\\emph{inst} removes task instructions; "
        "\\emph{cont} additionally masks stimulus values and feedback; "
        "\\emph{hist} further removes trial structure, "
        "leaving only the response history. "
        "The $\\ln(k)$ column shows the random-guessing baseline where "
        "$k$ is the number of per-trial response options. "
        "Experiments with mixed or continuous response formats "
        "have no well-defined $k$ and are shown as {---}.}"
    )
    lines.append("\\label{tab:ablation_centaur70b}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_all_results()
    print(f"  {len(data)} models × {len(EXPERIMENT_ORDER)} experiments "
          f"× {len(CONDITIONS)} conditions")

    for label, key in ALL_MODELS:
        status = "OK" if key in data else "MISSING"
        print(f"  {key}: {status}")
    status_70b = "OK" if CENTAUR_70B_KEY in data else "MISSING"
    print(f"  {CENTAUR_70B_KEY}: {status_70b}")

    print("\nGenerating ablation appendix table...")
    tex = generate_ablation_table(data)
    path = os.path.join(OUT_DIR, "ablation_appendix.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex + "\n")
    print(f"  Wrote {path}")

    print("\nGenerating Centaur-70B table...")
    tex_70b = generate_centaur70b_table(data)
    path_70b = os.path.join(OUT_DIR, "ablation_centaur70b.tex")
    with open(path_70b, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + "\n\n" + tex_70b + "\n")
    print(f"  Wrote {path_70b}")

    print("\nDone.")


if __name__ == "__main__":
    main()
