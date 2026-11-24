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

The rank-16 cells are listed for every family but not re-run: their OOD
results are already in the output dir (socius-<Name>-LoRA-r16.csv), so
--resume skips them. Use utils.result_csv_path to locate any cell's CSV.

Usage
-----
    python schedule_ood_rank_sweep.py --dry_run            # inspect the plan
    python schedule_ood_rank_sweep.py                      # run everything
    python schedule_ood_rank_sweep.py --families smollm olmo
    python schedule_ood_rank_sweep.py --quantise_from 8    # >= 8B in 4-bit
    python schedule_ood_rank_sweep.py --ranks 4 64 --device_map auto
    python schedule_ood_rank_sweep.py --purge_model_cache  # small disks

Disk space. Every size pulls its own base model (~2.5 GB per billion
parameters in bf16), and the HuggingFace cache keeps them all, which fills a
small pod disk part-way through the sweep. --purge_model_cache deletes the
cached model weights after each (family, size) group -- a base model is only
needed by the ranks of its own size -- so the disk only ever has to hold the
largest single model. The sweep also stops early, before launching anything,
when the disk is nearly full, rather than failing every remaining cell.

Any argument not listed here is forwarded to schedule_eval_runs.py
(e.g. --device_map, --max_seq_length).
"""

import argparse
import json
import glob
import os
import shutil
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


def hf_hub_cache():
    """The HuggingFace hub cache directory (same resolution as huggingface_hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(hf_home, "hub")


def free_gb(path):
    """Free space (GB) on the disk holding ``path`` (nearest existing parent)."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


def purge_model_cache():
    """Delete cached model weights (models--* only; datasets are kept)."""
    for d in glob.glob(os.path.join(hf_hub_cache(), "models--*")):
        shutil.rmtree(d, ignore_errors=True)
    print(f"Purged cached models -> {free_gb(hf_hub_cache()):.0f} GB free "
          f"in {hf_hub_cache()}")


def check_disk(family, size, output_dir, quantised):
    """Stop before a group that cannot succeed; warn when space looks tight."""
    out_free, cache_free = free_gb(output_dir), free_gb(hf_hub_cache())
    if min(out_free, cache_free) < 1:
        print(f"ERROR: disk nearly full before {family} {size} "
              f"(results dir {out_free:.1f} GB free, model cache "
              f"{cache_free:.1f} GB free). Free space (or re-run with "
              "--purge_model_cache) and run the same command again.")
        sys.exit(1)
    need = size_in_billions(size) * (1.0 if quantised else 2.5) + 2
    if cache_free < need:
        print(f"WARNING: {family} {size} needs roughly {need:.0f} GB for its "
              f"base model if not already cached; only {cache_free:.0f} GB "
              f"free in {hf_hub_cache()}.")


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
    parser.add_argument("--sizes", nargs="+", default=None,
                        help="Only these sizes (e.g. --sizes 8B 14B).")
    parser.add_argument("--ranks", nargs="+", type=int, default=None,
                        help="Only these LoRA ranks (e.g. --ranks 4 64).")
    parser.add_argument(
        "--purge_model_cache", action="store_true",
        help="Delete the cached HuggingFace model weights after each "
             "(family, size) group, so the disk only holds one base model at "
             "a time. Deletes ALL models--* entries in the hub cache.")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    # One scheduler call per (family, size) that has cells to evaluate.
    groups = []
    for family in args.families:
        fam = FAMILIES[family]
        for size in fam["size_order"]:
            if args.sizes and size not in args.sizes:
                continue
            ranks = [c["rank"] for c in build_cells(fam, ["rank"])
                     if c["size"] == size
                     and (not args.ranks or c["rank"] in args.ranks)]
            if ranks:
                groups.append((family, size, ranks))
    if not groups:
        print("ERROR: the --families/--sizes/--ranks filter matched no cells.")
        sys.exit(1)

    plan_path = os.path.join(args.output_dir, "eval_plan.json")
    combined_plan = []

    for family, size, ranks in groups:
        label = None
        if args.quantise_from is not None:
            label = family_quantise_from(FAMILIES[family], args.quantise_from)
        if not args.dry_run:
            quantised = (label is not None and
                         size_in_billions(size) >= size_in_billions(label))
            check_disk(family, size, args.output_dir, quantised)
        cmd = [
            sys.executable, "./schedule_eval_runs.py",
            "--family", family,
            "--sizes", size,
            "--ranks", *map(str, ranks),
            "--axes", "rank",
            "--eval_script", "./eval_model_ood.py",
            "--output_dir", args.output_dir,
            "--namespace", args.namespace,
            "--resume",
            # List the qwen/llama r16 cells too (their CSVs already exist, so
            # --resume skips them) -- same treatment as olmo/smollm.
            "--include_baseline",
        ] + passthrough
        if label:
            cmd += ["--quantise_from", label]
        if args.dry_run:
            cmd.append("--dry_run")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Scheduler failed for {family} {size} "
                  f"(exit {result.returncode}); stopping.")
            sys.exit(result.returncode)
        # The scheduler rewrites eval_plan.json on every call, so collect
        # each group's plan before the next one overwrites it.
        with open(plan_path) as fh:
            combined_plan += json.load(fh)
        if args.purge_model_cache and not args.dry_run:
            purge_model_cache()

    with open(plan_path, "w") as fh:
        json.dump(combined_plan, fh, indent=2)
    print(f"\nWrote combined plan ({len(combined_plan)} evals, "
          f"{len({g[0] for g in groups})} families) -> {plan_path}")

    if args.dry_run:
        return

    # Summary: which rank-sweep CSVs are still missing (failed cells).
    missing = []
    for family, size, ranks in groups:
        for cell in build_cells(FAMILIES[family], ["rank"]):
            if cell["size"] != size or cell["rank"] not in ranks:
                continue
            csv = result_csv_path(args.output_dir, args.namespace, family, cell)
            if not (os.path.exists(csv) and os.path.getsize(csv) > 0):
                missing.append(hf_repo_id(args.namespace, family, cell))

    print("\n" + "=" * 78)
    if missing:
        print(f"  {len(missing)} OOD result(s) still missing "
              "(failed; re-run to retry):")
        for repo_id in missing:
            print(f"    {repo_id}")
    else:
        print("  All rank-sweep OOD results present.")
    print("=" * 78)


if __name__ == "__main__":
    main()
