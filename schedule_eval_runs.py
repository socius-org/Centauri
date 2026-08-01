#!/usr/bin/env python
"""
schedule_eval_runs.py

Schedule Psych-101 evaluation of every rank-sweep + dataset-size ablation
adapter. The evaluator companion to utils.py: same families,
same cells, same ordering — but instead of launching a training run per cell it
launches ``eval_model.py`` per cell against the adapter's socius HuggingFace
repo.

Covers all four families: qwen, llama, and the new families olmo/smollm
(ADDITIONAL_EXPERIMENTS.md §7). For the new families the only cells are the
rank-16/full-data baselines, which are evaluated by default (they are new
adapters, not main-paper runs). Short-context models are evaluated at their
native context automatically (OLMo-2-1B: 4096, SmolLM2: 8192; see
cell_max_seq_length) so eval truncation matches training.

This is step (2)+(3) of the close-out workflow:

    1. upload_adapters.py    -> push adapters to socius/<RepoName>
    2. eval_model.py --model socius/<RepoName>   (this scheduler)
    3. run it over ALL cells, collecting one results CSV each.

Pulling from the Hub (rather than the local output dir) is deliberate: it
round-trips the upload, so a successful eval also confirms each adapter was
uploaded and downloads correctly.

IMPORTANT — backend.
    eval_model.py's auto-backend picks `unsloth` only when the model name ENDS
    in "lora"/"adapter". The ablation repos end in "-r64" / "-f0.5", so auto
    would wrongly choose the `transformers` backend, which cannot load a bare
    LoRA adapter. This scheduler therefore passes `--backend unsloth`
    explicitly (override with --backend if you ever eval merged models).

The grid is imported from utils.py and the repo names from
utils.py, so the repos evaluated here are exactly those trained and
uploaded.

Usage
-----
    # Inspect the Qwen eval plan (no eval launched):
    python schedule_eval_runs.py --family qwen --dry_run

    # Evaluate every Llama ablation adapter, resuming a partial run:
    python schedule_eval_runs.py --family llama --resume

    # Evaluate only the dataset-size cells, 8B+ in 4-bit:
    python schedule_eval_runs.py --family qwen --axes datasize --quantise_from 8B
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# Single source of truth for the experiment grid + repo names.
from utils import (FAMILIES, DEFAULT_MAX_SEQ_LENGTH, build_cells,
                   is_quantised, hf_repo_name)


def result_csv_name(model_id):
    """eval_model.py writes <model_id with '/'->'-'>.csv for Hub models."""
    return model_id.replace("/", "-") + ".csv"


def cell_max_seq_length(fam, cell, args):
    """Per-cell eval context length.

    An explicit --max_seq_length wins; otherwise the family's per-size
    override applies (OLMo-2-1B: 4096, SmolLM2: 8192 — short-context models
    must be evaluated at their native context, matching training), falling
    back to the 32768 default used by qwen/llama.
    """
    if args.max_seq_length is not None:
        return args.max_seq_length
    return fam.get("max_seq_length", {}).get(cell["size"], DEFAULT_MAX_SEQ_LENGTH)


def build_command(fam, cell, args):
    """Construct the eval_model.py command line for one cell."""
    repo_id = f"{args.namespace}/{hf_repo_name(args.family, cell)}"
    cmd = [
        sys.executable, args.eval_script,
        "--model", repo_id,
        "--backend", args.backend,
        "--output_dir", args.output_dir,
        "--max_seq_length", str(cell_max_seq_length(fam, cell, args)),
    ]
    if is_quantised(fam, cell["size"], args.quantise_from):
        cmd.append("--load_in_4bit")
    if args.device_map:
        cmd += ["--device_map", args.device_map]
    return cmd, repo_id


def main():
    parser = argparse.ArgumentParser(
        description="Schedule Psych-101 evaluation of the ablation adapters "
                    "(from their socius HuggingFace repos)."
    )
    parser.add_argument(
        "--family", required=True, choices=sorted(FAMILIES.keys()),
        help="Model family to evaluate. Run families as separate invocations.",
    )
    parser.add_argument(
        "--axes", nargs="+", default=["rank", "datasize"],
        choices=["rank", "datasize"],
        help="Which sweep's adapters to evaluate (default: both).",
    )
    parser.add_argument("--eval_script", default="./eval_model.py")
    parser.add_argument(
        "--namespace", default="socius",
        help="HuggingFace org/user the adapters live under (default: socius).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where eval_model.py writes result CSVs "
             "(default: ./results/psych101; use ./results/psych201 for OOD).",
    )
    parser.add_argument(
        "--backend", default="unsloth",
        choices=["auto", "unsloth", "transformers"],
        help="eval_model.py backend. Default 'unsloth' because these adapter "
             "repo names defeat auto-detection (see module docstring).",
    )
    parser.add_argument(
        "--quantise_from", default=None,
        help="Evaluate this size and larger in 4-bit (e.g. 8B). Family-aware. "
             "Default: all bf16.",
    )
    parser.add_argument("--device_map", default=None,
                        help="Pass through to eval_model.py (e.g. 'auto').")
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help="Force one eval context length for every cell. Default: the "
             "family's per-size value (OLMo-2-1B 4096, SmolLM2 8192, "
             "otherwise 32768).")
    parser.add_argument(
        "--include_baseline", action="store_true",
        help="Also evaluate the rank=16, fraction=1.0 baseline cells "
             "(off by default — those are the existing main-paper runs).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells whose result CSV already exists in --output_dir.",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=None,
        help="Only evaluate cells with these sizes (e.g. --sizes 1.7B). "
             "Default: all sizes in the family grid.",
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=None,
        help="Only evaluate cells with these LoRA ranks (e.g. --ranks 16). "
             "Default: all ranks in the family grid.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    fam = FAMILIES[args.family]
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", "psych101")

    cells = build_cells(fam, args.axes)

    # Optional cell filters (mirrors utils.py).
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

    # Match the training scheduler's order: cheapest first.
    size_rank = {s: i for i, s in enumerate(fam["size_order"])}
    cells.sort(key=lambda c: (size_rank[c["size"]], c["fraction"], c["rank"]))

    # Baseline cells are skipped only when they already exist from the main
    # paper (baseline_exists=True: qwen/llama). The new families' (olmo/
    # smollm) baselines are the adapters this folder trains, so they are
    # evaluated by default.
    skip_baseline = fam.get("baseline_exists", True) and not args.include_baseline
    launchable = [c for c in cells
                  if not (skip_baseline and c["axis"] == "baseline")]

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"  Schedule Psych-101 eval -- family: {args.family}")
    print("=" * 78)
    print(f"  Axes            : {', '.join(args.axes)}")
    print(f"  Eval script     : {args.eval_script}")
    print(f"  Backend         : {args.backend}")
    print(f"  Namespace       : {args.namespace}")
    print(f"  Quantise from   : {args.quantise_from or 'none (all bf16)'}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Launchable evals: {len(launchable)}")
    print("-" * 78)
    print(f"  {'#':>3}  {'size':>5}  {'rank':>4}  {'frac':>7}  {'prec':>4}  model")
    print("-" * 78)

    plan = []
    for i, cell in enumerate(launchable, 1):
        cmd, repo_id = build_command(fam, cell, args)
        quantised = is_quantised(fam, cell["size"], args.quantise_from)
        plan.append({
            "index": i, "family": args.family,
            "size": cell["size"], "rank": cell["rank"],
            "fraction": cell["fraction"], "axis": cell["axis"],
            "quantised": quantised, "model": repo_id,
            "max_seq_length": cell_max_seq_length(fam, cell, args),
            "result_csv": os.path.join(args.output_dir, result_csv_name(repo_id)),
            "command": cmd,
        })
        print(f"  {i:>3}  {cell['size']:>5}  {cell['rank']:>4}  "
              f"{cell['fraction']:>7g}  {'4bit' if quantised else 'bf16':>4}  "
              f"{repo_id}")
    print("=" * 78)

    plan_path = os.path.join(args.output_dir, "eval_plan.json")
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    print(f"\nWrote plan -> {plan_path}")

    if args.dry_run:
        print("\nDry run: nothing evaluated.")
        return

    status_path = os.path.join(args.output_dir, "eval_status.jsonl")
    print(f"Launching {len(plan)} evals sequentially. Status -> {status_path}\n")

    for entry in plan:
        # Resume: skip if the result CSV is already present.
        if args.resume and os.path.exists(entry["result_csv"]):
            print(f"[skip] {entry['model']} (result CSV exists)")
            continue

        start = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        print(f"[eval] {entry['index']}/{len(plan)}  {entry['model']}")
        print("       " + " ".join(entry["command"]))

        result = subprocess.run(entry["command"])
        elapsed = round(time.time() - t0, 1)
        record = {
            "family": entry["family"], "model": entry["model"],
            "size": entry["size"], "rank": entry["rank"],
            "fraction": entry["fraction"], "axis": entry["axis"],
            "quantised": entry["quantised"],
            "returncode": result.returncode,
            "started_utc": start, "elapsed_sec": elapsed,
        }
        with open(status_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"       -> {status}, {elapsed:.0f}s\n")

        if result.returncode != 0 and not args.resume:
            print("Stopping on first failure (re-run with --resume to continue "
                  "past completed evals).")
            sys.exit(result.returncode)

    print("Eval schedule complete.")


if __name__ == "__main__":
    main()
