#!/usr/bin/env python
"""
build_dataset_fractions.py

Construct nested, experiment-stratified subsets of Psych-101 for the
dataset-size ablation.

Design guarantees
-----------------
1. Stratified by experiment. Every experiment contributes the same
   *fraction* of its own participant-sequences to each subset, so task
   coverage is preserved at every size. Small fractions never silently
   drop whole experiments.

2. Whole-sequence sampling. Sampling is at the level of dataset rows.
   Each row in Psych-101 is one participant's complete trial-by-trial
   session, so we never split a sequence; we only include or exclude
   whole participants. This protects the within-participant sequential
   structure that the paper is about.

3. Nested subsets. The 6.25% subset is contained in 12.5%, which is
   contained in 25%, and so on. Within each experiment we shuffle once
   with a stable per-experiment seed and take growing prefixes. The only
   thing that changes between fractions is how many participants are
   added, never which ones are swapped, which makes the data-size axis
   clean rather than five independent random draws.

4. Reproducible and order-independent. Each experiment's shuffle seed is
   derived by hashing (global_seed, experiment_name), so a given
   experiment's ordering is identical regardless of dataset row order or
   how many other experiments are present.

A note on realised vs nominal fraction
---------------------------------------
To guarantee every experiment is represented at every size, each
experiment contributes at least `min_per_experiment` rows (default 1),
and counts are rounded up (ceil). For experiments with very few
participants this slightly inflates the realised fraction above the
nominal one at the smallest sizes. The manifest reports both so the
inflation is transparent; for the typical Psych-101 experiment (hundreds
of participants) it is negligible.

Outputs
-------
- fractions_indices.json : {fraction: [row indices into the split]}
- manifest.json          : per-fraction summary (nominal vs realised
                           fraction, row counts, experiment coverage,
                           per-experiment counts)
- subset_config.json     : the exact arguments used, for reproducibility
- (optional) materialised HF datasets, one per fraction

Usage
-----
    python build_dataset_fractions.py
    python build_dataset_fractions.py --fractions 0.125 0.5 1.0 --seed 3407
    python build_dataset_fractions.py --materialise
"""

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np

# Candidate names for the column that tags each row with its experiment.
# Auto-detection tries these in order; override with --experiment_col.
EXPERIMENT_COL_CANDIDATES = ["experiment", "experiment_name", "study", "task", "dataset"]


def detect_experiment_column(column_names, override=None):
    if override:
        if override not in column_names:
            raise ValueError(
                f"--experiment_col '{override}' not found. "
                f"Available columns: {column_names}"
            )
        return override
    for candidate in EXPERIMENT_COL_CANDIDATES:
        if candidate in column_names:
            return candidate
    raise ValueError(
        "Could not auto-detect the experiment column. "
        f"Available columns: {column_names}. "
        "Pass one explicitly with --experiment_col."
    )


def stable_experiment_seed(global_seed, experiment_name):
    """Deterministic per-experiment seed, independent of dataset ordering."""
    key = f"{global_seed}::{experiment_name}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:16], 16) % (2**32)


def build_nested_subsets(experiment_labels, fractions, global_seed,
                         min_per_experiment=1):
    """
    Build nested, stratified subsets.

    Parameters
    ----------
    experiment_labels : sequence
        experiment_labels[i] is the experiment tag of row i.
    fractions : iterable of float in (0, 1]
    global_seed : int
    min_per_experiment : int
        Minimum rows each experiment contributes to every subset.

    Returns
    -------
    selected : dict {fraction: sorted list of row indices}
    groups   : dict {experiment: list of row indices}
    """
    fractions = sorted(set(fractions))
    if any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("Fractions must lie in (0, 1].")

    # Group row indices by experiment.
    groups = defaultdict(list)
    for row_idx, exp in enumerate(experiment_labels):
        groups[exp].append(row_idx)

    selected = {f: [] for f in fractions}

    # Iterate experiments in sorted order for determinism. The per-experiment
    # seed makes each experiment's permutation independent of this order.
    for exp in sorted(groups.keys()):
        idxs = np.asarray(groups[exp])
        rng = np.random.default_rng(stable_experiment_seed(global_seed, exp))
        perm = idxs[rng.permutation(len(idxs))]
        n = len(perm)
        for f in fractions:
            k = math.ceil(n * f)
            k = max(min_per_experiment, k)
            k = min(n, k)
            selected[f].extend(perm[:k].tolist())

    for f in fractions:
        selected[f] = sorted(int(i) for i in selected[f])

    return selected, groups


def verify_nesting(selected, fractions):
    """Assert that each subset is contained in the next-larger one."""
    fractions = sorted(fractions)
    for small, large in zip(fractions[:-1], fractions[1:]):
        if not set(selected[small]).issubset(set(selected[large])):
            raise AssertionError(
                f"Nesting violated: fraction {small} is not a subset of {large}"
            )
    return True


def build_manifest(selected, groups, experiment_labels, total_rows, fractions):
    fractions = sorted(fractions)
    manifest = {
        "total_rows": total_rows,
        "n_experiments": len(groups),
        "fractions": {},
    }
    for f in fractions:
        rows = selected[f]
        per_exp = defaultdict(int)
        for i in rows:
            per_exp[experiment_labels[i]] += 1
        manifest["fractions"][str(f)] = {
            "nominal_fraction": f,
            "realised_fraction": round(len(rows) / total_rows, 6),
            "n_rows": len(rows),
            "n_experiments_covered": len(per_exp),
            "per_experiment_counts": dict(sorted(per_exp.items())),
        }
    return manifest


def print_summary(manifest):
    print("\n" + "=" * 72)
    print("  Psych-101 dataset-size ablation subsets")
    print("=" * 72)
    print(f"  Total rows in split : {manifest['total_rows']:,}")
    print(f"  Experiments         : {manifest['n_experiments']}")
    print("-" * 72)
    print(f"  {'nominal':>8}  {'realised':>9}  {'rows':>12}  {'experiments':>12}")
    print("-" * 72)
    for info in manifest["fractions"].values():
        print(
            f"  {info['nominal_fraction']:>8.4f}  "
            f"{info['realised_fraction']:>9.4f}  "
            f"{info['n_rows']:>12,}  "
            f"{info['n_experiments_covered']:>12}"
        )
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Build nested, experiment-stratified Psych-101 subsets."
    )
    parser.add_argument("--dataset", default="marcelbinz/Psych-101")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.0625, 0.125, 0.25, 0.5, 1.0],
        help="Nominal fractions in (0, 1]. 1.0 reproduces the full set.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--experiment_col", default=None)
    parser.add_argument("--min_per_experiment", type=int, default=1)
    parser.add_argument("--output_dir", default="./psych101_fractions")
    parser.add_argument(
        "--materialise", action="store_true",
        help="Also save each subset to disk as a HF dataset (more storage).",
    )
    args = parser.parse_args()

    # Imported here (not at module top) so the pure functions above can be
    # imported and unit-tested without the datasets dependency installed.
    from datasets import load_dataset

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset)[args.split]
    total_rows = len(ds)
    print(f"  {total_rows:,} rows")

    exp_col = detect_experiment_column(ds.column_names, args.experiment_col)
    print(f"  Using experiment column: '{exp_col}'")
    experiment_labels = list(ds[exp_col])

    selected, groups = build_nested_subsets(
        experiment_labels, args.fractions, args.seed, args.min_per_experiment
    )
    verify_nesting(selected, args.fractions)
    print("  Nesting verified: each subset is contained in the next.")

    manifest = build_manifest(
        selected, groups, experiment_labels, total_rows, args.fractions
    )
    print_summary(manifest)

    indices_path = os.path.join(args.output_dir, "fractions_indices.json")
    with open(indices_path, "w") as fh:
        json.dump({str(f): selected[f] for f in selected}, fh)
    print(f"\nSaved indices  -> {indices_path}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved manifest -> {manifest_path}")

    config_path = os.path.join(args.output_dir, "subset_config.json")
    with open(config_path, "w") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "split": args.split,
                "seed": args.seed,
                "experiment_col": exp_col,
                "min_per_experiment": args.min_per_experiment,
                "fractions": sorted(set(args.fractions)),
            },
            fh, indent=2,
        )
    print(f"Saved config   -> {config_path}")

    if args.materialise:
        for f in sorted(selected):
            sub = ds.select(selected[f])
            out = os.path.join(args.output_dir, f"fraction_{f}")
            sub.save_to_disk(out)
            print(f"Materialised {f:.4f} -> {out}")


if __name__ == "__main__":
    main()
