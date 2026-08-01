#!/usr/bin/env python
"""
schedule_training_runs.py

Schedule the rebuttal experiments for the paper
"Small Foundation Models of Human Cognition and Behaviour":
  (1) LoRA-rank sweep   -- vary rank at full data, to test whether the
                           in-distribution plateau and the steeper OOD
                           scaling gradient are artefacts of adapter capacity.
  (2) Dataset-size ablation -- vary the Psych-101 fraction at fixed rank 16,
                           to test whether the plateau is a data ceiling
                           rather than a capacity ceiling.

Families are scheduled SEPARATELY. The model families have different size
ladders (Qwen: 0.6B/1.7B/4B/8B/14B; Llama: 1B/3B/8B; OLMo: 1B/7B;
SmolLM: 0.1B/0.4B/1.7B/3B) and different training scripts, so each --family
run uses its own size list, per-family sweep design, training script,
run-name prefix, output subtree, and wandb project. Launch each family as
a separate invocation.

The new families (olmo, smollm — ADDITIONAL_EXPERIMENTS.md §7) run the
same L-shape as qwen/llama: the full rank sweep {4,8,16,32,64} at every
size plus the fraction ladder {0.0625..0.5} at rank 16. Their rank-16/
full-data baselines do not exist from the main paper and are therefore
launched by default (no --include_baseline needed); baselines already
trained by invoking the training script directly are recognised by
--resume via their default output layout (e.g. ./outputs/olmotaur-1b-bf16).

Design: an L-shape, NOT a full grid. Each axis is swept while the other is
held at its baseline (rank 16, fraction 1.0). The shared baseline cell
(size, rank=16, fraction=1.0) is the existing main-paper run and is not
re-launched. Crossing both axes fully would confound the two questions;
the L-shape answers each cleanly with far fewer runs.

This wraps the per-family training script (train_qwentaur.py /
train_centaur.py). It builds the exact command line per cell, tags
each wandb run with family / size / rank / fraction / axis, writes a plan,
and (unless --dry_run) launches the runs sequentially, logging status.

Prerequisites
-------------
- The training script must accept --data_fraction (see integration notes)
  in addition to --size, --load_in_4bit, --num_gpus, and the wandb args.
- For any fraction < 1.0, run build_dataset_fractions.py FIRST so the
  subset indices exist. This scheduler verifies that and refuses to launch
  fraction runs if the indices are missing. The subsets are family-agnostic
  (they index Psych-101 rows), so they are built once and shared.

Usage
-----
    # Inspect the Qwen plan without running anything:
    python schedule_training_runs.py --family qwen --dry_run

    # Run only the Llama dataset-size ablation:
    python schedule_training_runs.py --family llama --axes datasize

    # Run all Qwen cells, 8B+ in 4-bit, resuming a partial schedule:
    python schedule_training_runs.py --family qwen --quantise_from 8B --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


from utils import (
    BASELINE_RANK, BASELINE_FRACTION, DEFAULT_MAX_SEQ_LENGTH, FAMILIES,
    build_cells, cell_run_name, cell_output_dir, is_quantised,
    fractions_indices_ok,
)


def build_command(fam, cell, args, quantised):
    """Construct the training command line for one cell."""
    run_name = cell_run_name(fam, cell)
    out_dir = cell_output_dir(args.output_root, fam, cell, quantised)
    cmd = [
        sys.executable, args.train_script,
        "--size", cell["size"],
        "--data_fraction", f"{cell['fraction']:g}",
        "--indices_dir", args.indices_dir,
        "--lora_rank", str(cell["rank"]),
        "--output_dir", out_dir,
        "--num_gpus", str(args.num_gpus),
    ]
    if quantised:
        cmd.append("--load_in_4bit")
    if args.wandb:
        cmd += [
            "--wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_run", run_name,
            "--wandb_tags",
            f"family:{args.family}",
            f"size:{cell['size']}",
            f"rank:{cell['rank']}",
            f"fraction:{cell['fraction']:g}",
            f"axis:{cell['axis']}",
            "ablation",
        ]
        if args.wandb_entity:
            cmd += ["--wandb_entity", args.wandb_entity]
    return cmd, out_dir, run_name


def has_saved_model(out_dir):
    """True if out_dir already holds a saved adapter."""
    if not os.path.isdir(out_dir):
        return False
    return any(f.startswith("adapter") or f.endswith(".safetensors")
               for f in os.listdir(out_dir))


def already_trained(entry, fam):
    """The --resume predicate for one plan entry.

    Used both for the pre-launch summary and the launch loop so the reported
    'will train' count is exactly what actually runs. The scheduler's own
    output layout always counts; baseline cells additionally accept the
    training script's default layout (no -r/-f coordinates, e.g.
    ./outputs/olmotaur-1b-bf16), so a baseline trained by invoking the
    script directly is not re-trained. upload_adapters.resolve_local_dir
    probes the same fallback, so such adapters still get uploaded.
    """
    dirs = [entry["output_dir"]]
    if entry["axis"] == "baseline":
        s = entry["size"].lower()
        for q in ("bf16", "4bit"):
            dirs.append(os.path.join("./outputs", f"{fam['run_prefix']}-{s}-{q}"))
    return any(has_saved_model(d) for d in dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Schedule rank-sweep + dataset-size ablation runs for "
                    "Small Foundation Models of Human Cognition and Behaviour."
    )
    parser.add_argument(
        "--family", required=True, choices=sorted(FAMILIES.keys()),
        help="Model family to schedule. Run families as separate invocations.",
    )
    parser.add_argument(
        "--train_script", default=None,
        help="Override the family's default training script.",
    )
    parser.add_argument(
        "--axes", nargs="+", default=["rank", "datasize"],
        choices=["rank", "datasize"],
        help="Which sweeps to schedule.",
    )
    parser.add_argument("--output_root", default=None,
                        help="Default: ./outputs/<run_prefix>, e.g. "
                             "./outputs/qwentaur or ./outputs/olmotaur.")
    parser.add_argument(
        "--indices_dir", default="./psych101_fractions",
        help="Where build_dataset_fractions.py wrote fractions_indices.json.",
    )
    parser.add_argument(
        "--quantise_from", default=None,
        help="Train this size and larger in 4-bit (e.g. 8B). Family-aware. "
             "Default: all bf16.",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--include_baseline", action="store_true",
        help="Also (re)launch the rank=16, fraction=1.0 cells. "
             "Off by default since these are the existing main-paper runs.",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="Only schedule cells with these sizes (e.g. --sizes 1.7B). "
             "Default: all sizes in the family grid.",
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=None,
        help="Only schedule cells with these LoRA ranks (e.g. --ranks 8). "
             "Default: all ranks in the family grid.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells whose output_dir already contains a saved model.",
    )
    parser.add_argument("--dry_run", action="store_true")
    # wandb
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb_project", default=None,
                        help="Default: the family's project name.")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    fam = FAMILIES[args.family]

    # Fill family-derived defaults.
    if args.train_script is None:
        args.train_script = fam["train_script"]
    if args.output_root is None:
        args.output_root = os.path.join("./outputs", fam["run_prefix"])
    if args.wandb_project is None:
        args.wandb_project = fam["wandb_project"]

    cells = build_cells(fam, args.axes)

    # Optional cell filters (applied before the indices check so a
    # rank-only selection never demands fraction indices it won't use).
    if args.sizes:
        unknown = set(args.sizes) - set(fam["size_order"])
        if unknown:
            print(f"ERROR: --sizes {sorted(unknown)} not in this family's "
                  f"ladder {fam['size_order']}.")
            sys.exit(1)
        cells = [c for c in cells if c["size"] in args.sizes]
    if args.ranks:
        cells = [c for c in cells if c["rank"] in args.ranks]
    if not cells:
        print("ERROR: the --sizes/--ranks filter matched no cells in the grid.")
        sys.exit(1)

    # Verify subset indices are present for every non-trivial fraction needed.
    if "datasize" in args.axes:
        needed = sorted({c["fraction"] for c in cells})
        ok, msg = fractions_indices_ok(args.indices_dir, needed)
        if not ok:
            print(f"ERROR: dataset-size cells requested but {msg}.")
            print("Run build_dataset_fractions.py first (matching --indices_dir).")
            sys.exit(1)

    # Order: cheapest first (small sizes, then small fractions, low ranks),
    # so failures surface fast and partial schedules still yield small-model
    # curves early. size_order is family-specific.
    size_rank = {s: i for i, s in enumerate(fam["size_order"])}
    cells.sort(key=lambda c: (size_rank[c["size"]], c["fraction"], c["rank"]))

    os.makedirs(args.output_root, exist_ok=True)
    plan_path = os.path.join(args.output_root, "run_plan.json")

    # Baseline cells are skipped only when they already exist from the main
    # paper (baseline_exists=True: qwen/llama). For the new families the
    # baseline IS the experiment, so it is launchable by default.
    skip_baseline = fam.get("baseline_exists", True) and not args.include_baseline
    launchable = [c for c in cells
                  if not (skip_baseline and c["axis"] == "baseline")]

    print("\n" + "=" * 78)
    print("  Small Foundation Models of Human Cognition and Behaviour")
    print(f"  ablation schedule -- family: {args.family}")
    print("=" * 78)
    print(f"  Axes              : {', '.join(args.axes)}")
    print(f"  Sizes              : {', '.join(fam['size_order'])}")
    print(f"  Train script       : {args.train_script}")
    print(f"  Total cells        : {len(cells)} "
          f"({len(cells) - len(launchable)} baseline cells skipped)")
    print(f"  Launchable runs    : {len(launchable)}")
    print(f"  Quantise from      : {args.quantise_from or 'none (all bf16)'}")
    print(f"  Output root        : {args.output_root}")
    print(f"  Wandb project      : {args.wandb_project if args.wandb else 'disabled'}")
    print("-" * 78)
    print(f"  {'#':>3}  {'size':>5}  {'rank':>4}  {'frac':>7}  "
          f"{'prec':>4}  {'axis':<18}  run name")
    print("-" * 78)

    plan = []
    for i, cell in enumerate(launchable, 1):
        quantised = is_quantised(fam, cell["size"], args.quantise_from)
        cmd, out_dir, run_name = build_command(fam, cell, args, quantised)
        plan.append({
            "index": i,
            "family": args.family,
            **cell,
            "quantised": quantised,
            "output_dir": out_dir,
            "run_name": run_name,
            "command": cmd,
        })
        print(f"  {i:>3}  {cell['size']:>5}  {cell['rank']:>4}  "
              f"{cell['fraction']:>7g}  {'4bit' if quantised else 'bf16':>4}  "
              f"{cell['axis']:<18}  {run_name}")
    print("=" * 78)

    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    print(f"\nWrote plan -> {plan_path}")

    if args.dry_run:
        print("\nDry run: nothing launched.")
        return

    status_path = os.path.join(args.output_root, "run_status.jsonl")
    if args.resume:
        # Pre-scan with the SAME predicate the loop uses, so the count is exact.
        to_run = [e for e in plan if not already_trained(e, fam)]
        already = len(plan) - len(to_run)
        print(f"Resume: {already}/{len(plan)} cells already trained (will skip);"
              f" {len(to_run)} will actually train. Status -> {status_path}")
        if to_run:
            for e in to_run:
                print(f"    will train: {e['run_name']}")
        else:
            print("    (nothing to do -- every planned cell is already trained)")
        print()
    else:
        print(f"Launching {len(plan)} runs sequentially. "
              f"Status -> {status_path}\n")

    for entry in plan:
        out_dir = entry["output_dir"]
        # Resume: skip if a saved model already exists for this cell.
        if args.resume and already_trained(entry, fam):
            print(f"[skip] {entry['run_name']} (already has a saved model)")
            continue

        os.makedirs(out_dir, exist_ok=True)
        start = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        print(f"[run ] {entry['index']}/{len(plan)}  {entry['run_name']}")
        print("       " + " ".join(entry["command"]))

        result = subprocess.run(entry["command"])
        elapsed = round(time.time() - t0, 1)
        record = {
            "family": entry["family"],
            "run_name": entry["run_name"],
            "size": entry["size"],
            "rank": entry["rank"],
            "fraction": entry["fraction"],
            "axis": entry["axis"],
            "quantised": entry["quantised"],
            "returncode": result.returncode,
            "started_utc": start,
            "elapsed_sec": elapsed,
        }
        with open(status_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"       -> {status}, {elapsed:.0f}s\n")

        if result.returncode != 0 and not args.resume:
            print("Stopping on first failure (re-run with --resume to continue "
                  "past completed cells).")
            sys.exit(result.returncode)

    print("Schedule complete.")


if __name__ == "__main__":
    main()
