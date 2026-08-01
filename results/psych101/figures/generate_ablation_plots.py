#!/usr/bin/env python3
"""
generate_ablation_plots.py

Visualise the LoRA-rank sweep and the dataset-size ablation (rebuttal
experiments for "Small Foundation Models of Human Cognition and Behaviour")
for either model family.

Two-panel figure, shared y-axis so the axes are directly comparable:
    a) LoRA-rank sweep at full data  -- capacity axis
    b) Training-data fraction at rank 16 -- data axis

The shared baseline cell (rank 16, full data) is the existing main-paper run.
Its per-task losses are read from the main results table
(results/Psych-101 (NLL)/psych101_aggr.csv) and averaged over exactly the same
46 tasks present in the ablation eval CSVs, so the baseline point is directly
comparable to the swept points and anchors BOTH panels.

Follows results/STYLE_GUIDE.md (nature style, family colour with a size
gradient, family marker, 600-dpi PNG+PDF). Partial grids are fine: only the
cells whose eval CSVs exist are plotted.

Usage
-----
    python generate_ablation_plots.py --family llama
    python generate_ablation_plots.py --family qwen
    python generate_ablation_plots.py --family qwen \
        --abl_dir ./eval_results/qwentaur --outdir ./eval_results/figures
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots  # noqa: F401

# Resolve default paths against this script's location, not the CWD, so the
# script runs identically from anywhere (e.g. /workspace on a cloud box).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COG_MODEL = 0.6851              # domain-specific cognitive-model baseline
BASELINE_RANK = 16
BASELINE_FRACTION = 1.0

# Appended to every output basename (figures + summary CSVs). Set to
# '_cognitive_matched' by --cognitive-matched so matched-subset outputs sit
# beside the originals.
NAME_SUFFIX = ""


def cognitive_matched_tasks(aggr_path):
    """The Experiment ids (38 of the 46) for which Binz reports a domain-specific
    cognitive model — the exact task set the COG_MODEL line averages over.
    Restricting every ablation/baseline mean to these makes the plotted points
    and the cognitive line cover the same tasks."""
    aggr = pd.read_csv(aggr_path)
    aggr = aggr[aggr["Experiment"] != "Mean"]
    cog = pd.to_numeric(aggr["Cognitive model (reported)"].astype(str)
                        .str.replace("†", ""), errors="coerce")
    return set(aggr.loc[cog.notna(), "Experiment"])

# ---------------------------------------------------------------- families
# colour/marker per results/STYLE_GUIDE.md: Llama blue circles, Qwen purple
# squares. baseline_col maps each size to its main-paper column in
# psych101_aggr.csv (full data, rank 16, bf16). size_order drives the colour
# gradient (lighter = smaller); sizes absent from the ablation dir are skipped.
FAMILIES = {
    "llama": {
        "label": "Llama-Centaur",
        "color": "#0082fb",
        "marker": "o",
        "size_order": ["1B", "3B", "8B"],
        "baseline_col": {
            "1B": "Centaur-1B (bf16)",
            "3B": "Centaur-3B (bf16)",
            "8B": "Centaur-8B (bf16)",
        },
        "abl_subdir": "llama-centaur",
        "basename": "llama_ablation_rank_and_datasize",
        "summary": "llama_ablation_summary.csv",
    },
    "qwen": {
        "label": "Qwentaur",
        "color": "#7F6DEF",
        "marker": "s",
        "size_order": ["0.6B", "1.7B", "4B", "8B", "14B"],
        "baseline_col": {
            "0.6B": "Qwentaur-0.6B (bf16)",
            "1.7B": "Qwentaur-1.7B (bf16)",
            "4B":   "Qwentaur-4B (bf16)",
            "8B":   "Qwentaur-8B (bf16)",
            "14B":  "Qwentaur-14B (bf16)",
        },
        "abl_subdir": "qwentaur",
        "basename": "qwen_ablation_rank_and_datasize",
        "summary": "qwen_ablation_summary.csv",
    },
    "olmo": {
        "label": "Olmotaur",
        "color": "#F0529C",   # Ai2 brand pink; square (matches scaling_rank_panels)
        "marker": "s",
        "size_order": ["1B", "7B"],
        # No main-paper columns (new family); baselines come from the eval
        # CSVs via the fallback in main().
        "baseline_col": {},
        "abl_subdir": "olmotaur",
        "basename": "olmo_ablation_rank_and_datasize",
        "summary": "olmo_ablation_summary.csv",
    },
    "smollm": {
        "label": "Smoltaur",
        "color": "#FFD21E",   # HF brand yellow; circle (matches scaling_rank_panels)
        "marker": "o",
        "size_order": ["0.1B", "0.4B", "1.7B", "3B"],
        # No main-paper columns in psych101_aggr.csv: the rank-16 full-data
        # baselines were evaluated in the ablation run itself and are picked
        # up from the eval CSVs by the fallback in main().
        "baseline_col": {},
        "abl_subdir": "smoltaur",
        "basename": "smollm_ablation_rank_and_datasize",
        "summary": "smollm_ablation_summary.csv",
    },
}


def apply_style(fs=8, fl=5.5):
    plt.rcParams.update({
        "font.size": fs,
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": fs,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "legend.fontsize": fl,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
    })


def tint(hex_color, amount=0.0):
    """Lighten by mixing with white. amount=0 original, larger = lighter."""
    rgb = mcolors.hex2color(hex_color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])


def size_color(base_color, idx, n):
    """Size gradient: smallest (idx 0) lightest, largest darkest."""
    return tint(base_color, amount=0.55 * (1 - idx / max(n - 1, 1)))


def save_figure(fig, outdir, basename, dpi=600):
    basename = f"{basename}{NAME_SUFFIX}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{basename}.{ext}"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved {basename}.png / .pdf")


# ---------------------------------------------------------------- data
def parse_cell(fname):
    """Extract (size, rank, fraction) from an ablation result filename."""
    m = re.search(r"-(\d+(?:\.\d+)?)B-LoRA-r(\d+)(?:-f([0-9.]+))?\.csv$", fname)
    if not m:
        return None
    size = m.group(1) + "B"
    rank = int(m.group(2))
    frac = float(m.group(3)) if m.group(3) else 1.0
    return size, rank, frac


def load_ablation(abl_dir, token, keep=None):
    """Return DataFrame [size, rank, frac, mean] over one family's ablation
    CSVs in the flat results/psych101 dir, plus the shared task list.

    `token` is the family's repo display name (e.g. "Qwentaur"); only files
    named socius-<token>-...csv are read, so the flat directory's other
    families / base models / controls are ignored.

    `keep`, if given, restricts each cell's per-task mean to that task set (the
    cognitive-matched subset); the returned task list is the intersection.
    """
    rows, tasks = [], None
    prefix = f"socius-{token}-"
    for fn in sorted(os.listdir(abl_dir)):
        if not (fn.startswith(prefix) and fn.endswith(".csv")):
            continue
        cell = parse_cell(fn)
        if cell is None:
            continue
        df = pd.read_csv(os.path.join(abl_dir, fn))
        if keep is not None:
            df = df[df["task"].isin(keep)]
        if tasks is None:
            tasks = list(df["task"])
        rows.append((*cell, df["loss"].mean()))
    return pd.DataFrame(rows, columns=["size", "rank", "frac", "mean"]), tasks


def load_baselines(aggr_path, tasks, fam, sizes):
    """Mean NLL of the rank-16 full-data main-paper run per size, averaged over
    exactly the ablation task set."""
    aggr = pd.read_csv(aggr_path)
    sub = aggr[aggr["Experiment"].isin(tasks)]
    out = {}
    for size in sizes:
        col = fam["baseline_col"].get(size)
        if col and col in sub.columns:
            vals = pd.to_numeric(sub[col].astype(str).str.replace("†", ""),
                                 errors="coerce")
            out[size] = vals.mean()
    return out


# ---------------------------------------------------------------- plot
def make_figure(abl, baselines, fam, outdir):
    fs, fl, ms, mew, lw = 8, 5.5, 6, 0.8, 1.1
    mk = fam["marker"]
    sizes = [s for s in fam["size_order"] if s in baselines]
    n = len(sizes)

    with plt.style.context(["nature"]):
        apply_style(fs, fl)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

        # ---- panel a: rank sweep (full data) ----
        axA.set_title("a  LoRA rank sweep (full data)",
                      fontsize=9, fontweight="bold")
        for idx, size in enumerate(sizes):
            color = size_color(fam["color"], idx, n)
            sub = abl[(abl["size"] == size) & (abl["frac"] == BASELINE_FRACTION)]
            pts = {int(r["rank"]): r["mean"] for _, r in sub.iterrows()}
            pts[BASELINE_RANK] = baselines[size]            # anchor with baseline
            ranks = sorted(pts)
            ys = [pts[r] for r in ranks]
            axA.plot(ranks, ys, "-" + mk, color=color, markersize=ms,
                     markerfacecolor=color, markeredgecolor="white",
                     markeredgewidth=mew, lw=lw,
                     label=f"{fam['label']}-{size}", zorder=4)

        axA.set_xscale("log", base=2)
        axA.set_xticks([4, 8, 16, 32, 64])
        axA.set_xticklabels(["4", "8", "16", "32", "64"])
        axA.set_xlim(3.4, 75)
        axA.set_xlabel("LoRA rank")
        axA.set_ylabel("Mean negative log-likelihood")

        # ---- panel b: data-size ablation (rank 16) ----
        axB.set_title("b  Training-data fraction (rank 16)",
                      fontsize=9, fontweight="bold")
        for idx, size in enumerate(sizes):
            color = size_color(fam["color"], idx, n)
            sub = abl[(abl["size"] == size) & (abl["rank"] == BASELINE_RANK)]
            pts = {r["frac"]: r["mean"] for _, r in sub.iterrows()}
            pts[BASELINE_FRACTION] = baselines[size]        # anchor with baseline
            fracs = sorted(pts)
            ys = [pts[f] for f in fracs]
            axB.plot(fracs, ys, "-" + mk, color=color, markersize=ms,
                     markerfacecolor=color, markeredgecolor="white",
                     markeredgewidth=mew, lw=lw,
                     label=f"{fam['label']}-{size}", zorder=4)

        axB.set_xscale("log", base=2)
        fracs_ticks = [0.0625, 0.125, 0.25, 0.5, 1.0]
        axB.set_xticks(fracs_ticks)
        axB.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1"])
        axB.set_xlim(0.054, 1.16)
        axB.set_xlabel("Training-data fraction")

        # ---- shared reference + baseline-marker legend handle ----
        for ax in (axA, axB):
            ax.axhline(COG_MODEL, color="gray", linewidth=0.8, linestyle=":",
                       alpha=0.7, zorder=1)
        axB.text(0.0625, COG_MODEL + 0.004, "cognitive models", fontsize=4.5,
                 color="gray", va="bottom", ha="left")

        handles, labels = axA.get_legend_handles_labels()
        axA.legend(handles, labels, loc="upper right", borderpad=0.3,
                   handlelength=1.5, handletextpad=0.4, labelspacing=0.4,
                   frameon=False)

        plt.tight_layout(pad=0.5)
        save_figure(fig, outdir, fam["basename"])
        plt.close()


def write_summary(abl, baselines, fam, outdir):
    """Emit a tidy summary CSV of every cell (incl. baseline) for the write-up."""
    recs = []
    for size in [s for s in fam["size_order"] if s in baselines]:
        recs.append({"size": size, "axis": "baseline", "rank": BASELINE_RANK,
                     "fraction": BASELINE_FRACTION, "mean_nll": baselines[size]})
        for _, r in abl[(abl["size"] == size)
                        & (abl["frac"] == BASELINE_FRACTION)].iterrows():
            if int(r["rank"]) == BASELINE_RANK:
                continue
            recs.append({"size": size, "axis": "rank_sweep", "rank": int(r["rank"]),
                         "fraction": 1.0, "mean_nll": r["mean"]})
        for _, r in abl[(abl["size"] == size)
                        & (abl["rank"] == BASELINE_RANK)].iterrows():
            if abs(r["frac"] - BASELINE_FRACTION) < 1e-9:
                continue
            recs.append({"size": size, "axis": "datasize", "rank": BASELINE_RANK,
                         "fraction": r["frac"], "mean_nll": r["mean"]})
    out = pd.DataFrame(recs).sort_values(["size", "axis", "rank", "fraction"])
    name = fam["summary"].replace(".csv", f"{NAME_SUFFIX}.csv")
    path = os.path.join(outdir, name)
    out.to_csv(path, index=False)
    print(f"  Saved {name} ({len(out)} rows)")
    return out


def load_family_data(family, abl_dir, aggr_path, keep=None):
    """Load ablation cells + rank-16 baselines for one family (mirrors main)."""
    fam = FAMILIES[family]
    abl, tasks = load_ablation(abl_dir, fam["label"], keep=keep)
    swept = [s for s in fam["size_order"]
             if s in set(abl["size"]) or s in fam["baseline_col"]]
    baselines = load_baselines(aggr_path, tasks, fam, swept)
    for s in swept:
        if s not in baselines:
            rows = abl[(abl["size"] == s) & (abl["rank"] == BASELINE_RANK)
                       & ((abl["frac"] - BASELINE_FRACTION).abs() < 1e-9)]
            if len(rows):
                baselines[s] = rows["mean"].iloc[0]
    return abl, baselines, tasks


def get_70b_refs(aggr_path, tasks):
    """Mean NLL of the Centaur-70B reproduced / reported runs over the ablation
    task set (matching how the per-size baselines are averaged)."""
    aggr = pd.read_csv(aggr_path)
    sub = aggr[aggr["Experiment"].isin(tasks)]

    def gm(col):
        return pd.to_numeric(sub[col].astype(str).str.replace("†", ""),
                             errors="coerce").mean()
    return {"repl": gm("Centaur-70B (4bit-reproduced)"),
            "binz": gm("Centaur-70B (4bit-reported)")}


# Context window: circle if the model ran at >=32k tokens, else square
# (truncated). Only the smaller Smoltaur/Olmotaur models fall below 32k.
SMALL_CTX = {("smollm", "0.1B"), ("smollm", "0.4B"), ("smollm", "1.7B"),
             ("olmo", "1B")}


def ctx_marker(fam_key, size):
    return "s" if (fam_key, size) in SMALL_CTX else "o"


def _draw_ablation_panel(ax, abl, baselines, fam, fam_key, axis,
                         ms=6, mew=0.8, lw=1.1):
    """Draw one family's rank-sweep ('rank') or datasize ('frac') panel. Marker
    shape encodes context window (circle >=32k, square <32k)."""
    sizes = [s for s in fam["size_order"] if s in baselines]
    n = len(sizes)
    for idx, size in enumerate(sizes):
        color = size_color(fam["color"], idx, n)
        mk = ctx_marker(fam_key, size)
        if axis == "rank":
            sub = abl[(abl["size"] == size) & (abl["frac"] == BASELINE_FRACTION)]
            pts = {int(r["rank"]): r["mean"] for _, r in sub.iterrows()}
            pts[BASELINE_RANK] = baselines[size]
        else:
            sub = abl[(abl["size"] == size) & (abl["rank"] == BASELINE_RANK)]
            pts = {r["frac"]: r["mean"] for _, r in sub.iterrows()}
            pts[BASELINE_FRACTION] = baselines[size]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, "-" + mk, color=color, markersize=ms,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=mew, lw=lw, label=f"{fam['label']}-{size}",
                zorder=4)
    ax.axhline(COG_MODEL, color="gray", linewidth=0.8, linestyle=":",
               alpha=0.7, zorder=1)


DK70B = "#005bb5"   # Centaur-70B reference colour


def _draw_70b_margin(ax, refs, axis):
    """Centaur-70B reproduced (filled) / reported (hollow) diamonds in the
    right margin (beyond the swept range). The 70B has no natural rank/fraction
    position, so it sits as a y-reference in the margin, exactly as it sits
    beyond the family sizes in the scaling figures."""
    if axis == "rank":
        x_d, xr = 96, (3.4, 130)
    else:
        x_d, xr = 1.58, (0.054, 2.05)
    ax.set_xlim(*xr)
    ax.plot([x_d], [refs["repl"]], "D", color=DK70B, markersize=5,
            markerfacecolor=DK70B, markeredgecolor="white", markeredgewidth=0.7,
            zorder=6, clip_on=False)
    ax.plot([x_d], [refs["binz"]], "D", color=DK70B, markersize=4.2,
            markerfacecolor="none", markeredgecolor=DK70B, markeredgewidth=0.7,
            zorder=6, clip_on=False)


def make_combined_figure(fams_data, refs, outdir):
    """Four families (rows) x two panels (rank sweep | datasize), with the
    Centaur-70B reproduced/reported reference on every panel."""
    fs, fl = 8, 5.5
    order = ["llama", "qwen", "smollm", "olmo"]
    with plt.style.context(["nature"]):
        apply_style(fs, fl)
        fig, axes = plt.subplots(len(order), 2, figsize=(7, 10.5), sharey="row")
        for row, fkey in enumerate(order):
            fam = FAMILIES[fkey]
            abl, baselines = fams_data[fkey]
            axA, axB = axes[row]
            _draw_ablation_panel(axA, abl, baselines, fam, fkey, "rank")
            _draw_ablation_panel(axB, abl, baselines, fam, fkey, "frac")
            _draw_70b_margin(axA, refs, "rank")
            _draw_70b_margin(axB, refs, "frac")

            for ax in (axA, axB):
                ax.set_xscale("log", base=2)
                ax.minorticks_off()
            axA.set_xticks([4, 8, 16, 32, 64])
            axA.set_xticklabels(["4", "8", "16", "32", "64"])
            axB.set_xticks([0.0625, 0.125, 0.25, 0.5, 1.0])
            axB.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1"])
            axA.set_ylabel("Mean NLL")
            # fixed axes-fraction position (identical across rows) -- matches
            # where it lands just below the cognitive line in the Llama row.
            axA.text(0.03, 0.93, fam["label"], transform=axA.transAxes,
                     fontsize=9, fontweight="bold", color="black",
                     va="top", ha="left")
            axB.legend(loc="upper right", fontsize=fl + 0.5, frameon=False,
                       borderpad=0.2, handlelength=1.2, handletextpad=0.3,
                       labelspacing=0.25)
            if row == 0:
                axA.set_title("LoRA rank sweep (full data; r = 4$-$64)",
                              fontsize=9, fontweight="bold")
                axB.set_title("Training data fraction (r = 16)",
                              fontsize=9, fontweight="bold")
            if row == len(order) - 1:
                axA.set_xlabel("LoRA rank")
                axB.set_xlabel("Training-data fraction")
            else:
                plt.setp(axA.get_xticklabels(), visible=False)
                plt.setp(axB.get_xticklabels(), visible=False)

        # Shared reference legend (below the grid): context-shape key +
        # Centaur-70B + cognitive line
        ref_handles = [
            plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor="0.45", markeredgecolor="white",
                       markeredgewidth=0.7),
            plt.Line2D([], [], marker="s", linestyle="none", markersize=5,
                       markerfacecolor="0.45", markeredgecolor="white",
                       markeredgewidth=0.7),
            plt.Line2D([], [], marker="D", linestyle="none", markersize=5,
                       markerfacecolor=DK70B, markeredgecolor="white",
                       markeredgewidth=0.7),
            plt.Line2D([], [], marker="D", linestyle="none", markersize=4.2,
                       markerfacecolor="none", markeredgecolor=DK70B,
                       markeredgewidth=0.7),
            plt.Line2D([], [], color="gray", lw=0.8, linestyle=":", alpha=0.7),
        ]
        fig.legend(ref_handles,
                   [r"$\geq$32k context", "<32k (truncated)",
                    "Centaur-70B (reproduced)", "Centaur-70B (reported)",
                    "Cognitive baseline"],
                   loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=5,
                   fontsize=fl + 1.5, frameon=False, handlelength=1.4,
                   handletextpad=0.4, columnspacing=1.6)

        plt.tight_layout(rect=(0, 0.02, 1, 1))
        save_figure(fig, outdir, "ablation_all_families_rank_and_datasize")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot rank/data ablations for one model family.")
    parser.add_argument("--family", default="llama",
                        choices=sorted(FAMILIES.keys()))
    parser.add_argument("--combined", action="store_true",
                        help="Produce the 4-family combined figure "
                             "(rows = families) instead of one family.")
    parser.add_argument("--abl_dir", default=None,
                        help="Flat dir of per-task eval CSVs "
                             "(default: results/psych101).")
    parser.add_argument("--aggr",
                        default=os.path.join(SCRIPT_DIR, "..", "psych101_aggr.csv"))
    parser.add_argument("--outdir", default=SCRIPT_DIR)
    parser.add_argument("--cognitive-matched", action="store_true",
                        help="Average every ablation/baseline mean over only "
                             "the 38 tasks with a reported cognitive model, and "
                             "write *_cognitive_matched figures + summaries.")
    args = parser.parse_args()

    if args.abl_dir is None:
        args.abl_dir = os.path.join(SCRIPT_DIR, "..")   # results/psych101
    os.makedirs(args.outdir, exist_ok=True)

    global NAME_SUFFIX
    keep = None
    if args.cognitive_matched:
        NAME_SUFFIX = "_cognitive_matched"
        keep = cognitive_matched_tasks(args.aggr)
        print(f"Cognitive-matched: restricting to {len(keep)} tasks\n")

    if args.combined:
        print("Combined 4-family ablation figure")
        fams_data, tasks_ref = {}, None
        for fk in ["llama", "qwen", "smollm", "olmo"]:
            abl, baselines, tasks = load_family_data(fk, args.abl_dir, args.aggr,
                                                     keep=keep)
            fams_data[fk] = (abl, baselines)
            tasks_ref = tasks
            print(f"  {FAMILIES[fk]['label']}: {len(abl)} cells, "
                  f"{len([s for s in FAMILIES[fk]['size_order'] if s in baselines])} sizes")
        refs = get_70b_refs(args.aggr, tasks_ref)
        print(f"  Centaur-70B: reproduced {refs['repl']:.4f}, reported {refs['binz']:.4f}")
        make_combined_figure(fams_data, refs, args.outdir)
        print("\nDone (combined).")
        return

    fam = FAMILIES[args.family]

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 60)
    print(f"{fam['label']} rank-sweep + dataset-size ablation figures")
    print("=" * 60)
    print(f"  Ablation dir : {args.abl_dir}")
    print(f"  Baseline csv : {args.aggr}")
    print(f"  Output dir   : {args.outdir}\n")

    abl, tasks = load_ablation(args.abl_dir, fam["label"], keep=keep)
    print(f"Loaded {len(abl)} ablation cells over {len(tasks)} tasks.")
    # Include sizes with a main-paper baseline column even when the ablation
    # dir has no cells for them (e.g. Qwentaur-1.7B: never swept, but its
    # r=16 baseline exists in psych101_aggr.csv and belongs in the summary).
    swept_sizes = [s for s in fam["size_order"]
                   if s in set(abl["size"]) or s in fam["baseline_col"]]
    baselines = load_baselines(args.aggr, tasks, fam, swept_sizes)
    # New families (smollm/olmo) have no main-paper columns: their rank-16
    # full-data baselines were evaluated in this run, so read them from the
    # ablation cells themselves.
    for s in swept_sizes:
        if s not in baselines:
            rows = abl[(abl["size"] == s) & (abl["rank"] == BASELINE_RANK)
                       & ((abl["frac"] - BASELINE_FRACTION).abs() < 1e-9)]
            if len(rows):
                baselines[s] = rows["mean"].iloc[0]
    print("Baselines (rank 16, full data, over ablation tasks):")
    for s in swept_sizes:
        if s in baselines:
            print(f"  {fam['label']}-{s}: {baselines[s]:.4f}")
    # Report grid completeness so partial runs are obvious.
    full_ranks = {4, 8, 32, 64}
    full_fracs = {0.0625, 0.125, 0.25, 0.5}
    for s in swept_sizes:
        have_r = set(abl[(abl["size"] == s)
                         & (abl["frac"] == BASELINE_FRACTION)]["rank"])
        have_f = set(abl[(abl["size"] == s)
                         & (abl["rank"] == BASELINE_RANK)
                         & (abl["frac"] < 1.0)]["frac"])
        miss = []
        if full_ranks - have_r:
            miss.append("ranks " + ",".join(
                f"r{r}" for r in sorted(full_ranks - have_r)))
        if full_fracs - have_f:
            miss.append("fractions " + ",".join(
                f"{f:g}" for f in sorted(full_fracs - have_f)))
        if miss:
            print(f"  NOTE {fam['label']}-{s}: missing {('; '.join(miss))}")
    print()

    summary = write_summary(abl, baselines, fam, args.outdir)
    make_figure(abl, baselines, fam, args.outdir)

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
