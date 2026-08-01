#!/usr/bin/env python3
"""
Build the cognitive-matched companion data CSVs (38-task subset), in exactly the
same shape as the committed 46-task exports:

  psych101_rank_datasize_figure_data_cognitive_matched.csv   (this dir)
  r16_psych101_vs_psych201_figure_data_cognitive_matched.csv (../../psych201/figures)

Why: the dotted cognitive-model line (NLL = 0.6851) is a mean over only the 38 of
46 Psych-101 tasks for which Binz et al. report a domain-specific model. These
tables recompute every Psych-101 mean over that same 38-task subset so slopes,
crossing thresholds, and size/rank substitution pairs can be read off the numbers
rather than inferred from pixels (the effects of interest are 0.005-0.02 nats).

Sources per row:
  - baseline / rank_sweep / datasize / finetuned_r16 : the *_ablation_summary
    CSVs (46-task) and *_ablation_summary_cognitive_matched CSVs (38-task) --
    the canonical outputs of generate_ablation_plots.py.
  - base / Centaur-70B (reproduced|reported)         : psych101_aggr.csv columns.
  - Cognitive baseline                               : 0.6851 (already the
    38-task cognitive mean, so identical in both) -- copied verbatim.
  - All Psych-201 rows                               : copied verbatim (no
    cognitive baseline exists for OOD, so there is no subset to restrict to).

Verification: every recomputed row's 46-task value is checked against the
committed figure-data file before the 38-task value is written. The script aborts
if any 46-task value disagrees, so a passing run means the 38-task numbers were
produced by a pipeline that exactly reproduces the originals.

Run generate_ablation_plots.py (with and without --cognitive-matched) for all four
families first, so both sets of *_ablation_summary CSVs exist.
"""

import os
import sys
import pandas as pd

FIG101 = os.path.dirname(os.path.abspath(__file__))
P101 = os.path.join(FIG101, "..")                              # results/psych101
FIG201 = os.path.join(P101, "..", "psych201", "figures")
AGGR = os.path.join(P101, "psych101_aggr.csv")

RANKDATA = os.path.join(FIG101, "psych101_rank_datasize_figure_data.csv")
R16DATA = os.path.join(FIG201, "r16_psych101_vs_psych201_figure_data.csv")

# figure-data family label -> ablation-summary file stem
FAM_STEM = {
    "Llama-Centaur": "llama_ablation_summary",
    "Qwentaur": "qwen_ablation_summary",
    "Smoltaur": "smollm_ablation_summary",
    "Olmotaur": "olmo_ablation_summary",
}
# (base family label, size) -> psych101_aggr column
BASE_COL = {
    ("Llama-3.1/3.2 (base)", "1B"): "Llama-3.2-1B (base-bf16)",
    ("Llama-3.1/3.2 (base)", "3B"): "Llama-3.2-3B (base-bf16)",
    ("Llama-3.1/3.2 (base)", "8B"): "Llama-3.1-8B (base-bf16)",
    ("Qwen3 (base)", "0.6B"): "Qwen3-0.6B (base-bf16)",
    ("Qwen3 (base)", "1.7B"): "Qwen3-1.7B (base-bf16)",
    ("Qwen3 (base)", "4B"): "Qwen3-4B (base-bf16)",
    ("Qwen3 (base)", "8B"): "Qwen3-8B (base-bf16)",
    ("Qwen3 (base)", "14B"): "Qwen3-14B (base-bf16)",
}
CEN_COL = {
    "centaur70b_reproduced": "Centaur-70B (4bit-reproduced)",
    "centaur70b_reported": "Centaur-70B (4bit-reported)",
}
CEN_REF = {  # rank_datasize 'reference' family label -> aggr column
    "Centaur-70B (reproduced)": "Centaur-70B (4bit-reproduced)",
    "Centaur-70B (reported)": "Centaur-70B (4bit-reported)",
}
ATOL = 1.5e-3  # committed files are rounded to 3-4 dp; tolerate that
OUT_DP = 6     # write matched means at 1e-6 nat resolution


def load_aggr():
    df = pd.read_csv(AGGR)
    df = df[df["Experiment"] != "Mean"]
    cog = pd.to_numeric(df["Cognitive model (reported)"].astype(str)
                        .str.replace("†", ""), errors="coerce")
    keep = set(df.loc[cog.notna(), "Experiment"])
    return df, keep


def aggr_mean(df, col, keep=None):
    sub = df if keep is None else df[df["Experiment"].isin(keep)]
    return pd.to_numeric(sub[col].astype(str).str.replace("†", ""),
                         errors="coerce").mean()


def load_summaries(matched):
    """dict[(family, size, axis, rank:int, frac:float)] -> mean_nll."""
    suffix = "_cognitive_matched" if matched else ""
    out = {}
    for fam, stem in FAM_STEM.items():
        s = pd.read_csv(os.path.join(FIG101, f"{stem}{suffix}.csv"))
        for _, r in s.iterrows():
            out[(fam, str(r["size"]), str(r["axis"]),
                 int(r["rank"]), round(float(r["fraction"]), 6))] = float(r["mean_nll"])
    return out


def rewrite(in_path, out_path, value_fn):
    """Copy in_path to out_path, replacing only the last (mean_nll) field via
    value_fn(fields) -> (v46_or_None, v38_or_None). None,None means copy verbatim.
    Returns list of (row, committed, recomputed_46) mismatches."""
    lines = open(in_path).read().splitlines()
    out, bad = [lines[0]], []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(",")
        v46, v38 = value_fn(f)
        if v46 is None and v38 is None:            # copy verbatim
            out.append(line)
            continue
        committed = float(f[-1])
        if abs(v46 - committed) > ATOL:
            bad.append((",".join(f[:-1]), committed, round(v46, 6)))
        f[-1] = f"{v38:.{OUT_DP}f}"
        out.append(",".join(f))
    with open(out_path, "w", newline="") as fh:
        fh.write("\n".join(out) + "\n")
    return bad


def main():
    aggr, keep = load_aggr()
    print(f"Cognitive-matched subset: {len(keep)} / "
          f"{(aggr['Experiment'] != 'Mean').sum()} tasks\n")
    orig, matched = load_summaries(False), load_summaries(True)
    all_bad = []

    # ---- 1) rank sweep + data fractions (all Psych-101) ----
    def rankdata_val(f):
        family, size, _params, axis, rank, frac, _mean = f
        if axis in ("baseline", "rank_sweep", "datasize"):
            k = (family, size, axis, int(float(rank)), round(float(frac), 6))
            return orig[k], matched[k]
        if axis == "reference":
            if family in CEN_REF:
                col = CEN_REF[family]
                return aggr_mean(aggr, col), aggr_mean(aggr, col, keep)
            return None, None                        # cognitive baseline: verbatim
        raise ValueError(f"unknown axis {axis!r}")

    out1 = os.path.join(FIG101,
                        "psych101_rank_datasize_figure_data_cognitive_matched.csv")
    all_bad += rewrite(RANKDATA, out1, rankdata_val)
    print(f"wrote {os.path.basename(out1)}")

    # ---- 2) r=16 in-distribution + OOD ----
    def r16_val(f):
        bench, family, size, _params, mtype, _mean = f
        if bench == "Psych-201":
            return None, None                        # OOD unchanged: verbatim
        if mtype == "finetuned_r16":
            k = (family, size, "baseline", 16, 1.0)
            return orig[k], matched[k]
        if mtype == "base":
            col = BASE_COL[(family, size)]
            return aggr_mean(aggr, col), aggr_mean(aggr, col, keep)
        if mtype in CEN_COL:
            col = CEN_COL[mtype]
            return aggr_mean(aggr, col), aggr_mean(aggr, col, keep)
        if mtype == "cognitive_baseline":
            return None, None                        # 0.6851: verbatim
        raise ValueError(f"unknown model_type {mtype!r}")

    out2 = os.path.join(FIG201,
                        "r16_psych101_vs_psych201_figure_data_cognitive_matched.csv")
    all_bad += rewrite(R16DATA, out2, r16_val)
    print(f"wrote {os.path.basename(out2)}")

    # ---- verification gate ----
    if all_bad:
        print("\nFAIL: recomputed 46-task values disagree with committed files:")
        for key, committed, got in all_bad:
            print(f"  {key}: committed {committed} vs recomputed {got}")
        sys.exit(1)
    print("\nOK: every recomputed 46-task value matches the committed originals "
          f"(within {ATOL} nats); matched 38-task values written.")


if __name__ == "__main__":
    main()
