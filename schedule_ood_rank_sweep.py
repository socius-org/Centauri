#!/usr/bin/env python
"""
schedule_ood_rank_sweep.py

Run the Psych-201-RT (OOD) evaluation over the full LoRA-rank sweep of every
family (Smoltaur, Olmotaur, Llama-Centaur, Qwentaur). So far only the rank-16
adapters were evaluated OOD; this fills in the remaining ranks.

Thin wrapper: for each family it calls schedule_eval_runs.py with
``--axes rank --eval_script ./eval_model_ood.py --resume``, so the cells, repo
names and per-size context lengths all come from utils.FAMILIES. Cells whose
CSV already exists in the output dir are skipped, and a failed cell does not
stop the sweep (it is logged in <output_dir>/eval_status.jsonl).

The rank-16 cells of qwen/llama are not re-run: they are the main-paper runs,
whose OOD results are already in the output dir (socius-<Name>-LoRA-r16.csv).
Use utils.result_csv_path to locate any cell's CSV.

Usage
-----
    python schedule_ood_rank_sweep.py --dry_run            # inspect the plan
    python schedule_ood_rank_sweep.py                      # run everything
    python schedule_ood_rank_sweep.py --families smollm olmo
    python schedule_ood_rank_sweep.py --quantise_from 8    # >= 8B in 4-bit
    python schedule_ood_rank_sweep.py --ranks 4 64 --device_map auto

Any argument not listed here is forwarded to schedule_eval_runs.py
(e.g. --ranks, --device_map, --max_seq_length).
"""

import argparse
import json
import os
import subprocess
import sys

from utils import FAMILIES, build_cells, hf_repo_id, result_csv_path

# Cheapest family first.
FAMILY_ORDER = ["smollm", "olmo", "llama", "qwen"]


def size_in_billions(size):
    return float(size.rstrip("Bb"))


def family_quantise_from(fam, threshold):
    """Smallest size label in this family at or above ``threshold`` billion
    parameters (None if the family has no such size)."""
    for size in fam["size_order"]:
        if size_in_billions(size) >= threshold:
            return size
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Psych-201-RT (OOD) eval across the rank sweep of all families.")
    parser.add_argument("--families", nargs="+", default=FAMILY_ORDER,
                        choices=FAMILY_ORDER)
    parser.add_argument("--output_dir", default="./results/psych201")
    parser.add_argument("--namespace", default="socius")
    parser.add_argument(
        "--quantise_from", type=size_in_billions, default=None,
        help="Evaluate models of at least this many billion parameters in "
             "4-bit (e.g. 8 or 8B). Applied per family, so it need not be a "
             "size label of every family. Default: all bf16.")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    plan_path = os.path.join(args.output_dir, "eval_plan.json")
    combined_plan = []

    for family in args.families:
        cmd = [
            sys.executable, "./schedule_eval_runs.py",
            "--family", family,
            "--axes", "rank",
            "--eval_script", "./eval_model_ood.py",
            "--output_dir", args.output_dir,
            "--namespace", args.namespace,
            "--resume",
        ] + passthrough
        if args.quantise_from is not None:
            label = family_quantise_from(FAMILIES[family], args.quantise_from)
            if label:
                cmd += ["--quantise_from", label]
        if args.dry_run:
            cmd.append("--dry_run")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Scheduler failed for family '{family}' "
                  f"(exit {result.returncode}); stopping.")
            sys.exit(result.returncode)
        # The scheduler rewrites eval_plan.json on every call, so collect
        # each family's plan before the next one overwrites it.
        with open(plan_path) as fh:
            combined_plan += json.load(fh)

    with open(plan_path, "w") as fh:
        json.dump(combined_plan, fh, indent=2)
    print(f"\nWrote combined plan ({len(combined_plan)} evals, "
          f"{len(args.families)} families) -> {plan_path}")

    if args.dry_run:
        return

    # Summary: which rank-sweep CSVs are still missing (failed cells).
    missing = []
    for family in args.families:
        for cell in build_cells(FAMILIES[family], ["rank"]):
            csv = result_csv_path(args.output_dir, args.namespace, family, cell)
            if not os.path.exists(csv):
                missing.append(hf_repo_id(args.namespace, family, cell))

    print("\n" + "=" * 78)
    if missing:
        print(f"  {len(missing)} OOD result(s) still missing "
              "(filtered out or failed; re-run to retry):")
        for repo_id in missing:
            print(f"    {repo_id}")
    else:
        print("  All rank-sweep OOD results present.")
    print("=" * 78)


if __name__ == "__main__":
    main()
