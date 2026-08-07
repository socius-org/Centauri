#!/usr/bin/env python
"""
upload_adapters.py

Upload the LoRA adapters from the rank-sweep + dataset-size ablation to the
``socius`` HuggingFace account.

This is step (1) of the close-out workflow:

    1. upload_adapters.py    -> push every trained adapter to socius/<RepoName>
    2. schedule_eval_runs.py -> eval_model.py pulls those SAME socius repos
                                (round-trips upload+download as a correctness check)
    3. eval over all cells, then plots / tables / write-up.

The experiment grid (which size/rank/fraction cells exist) is imported from
utils.py, and the repo names from utils.py, so the
adapter an run trained, the repo it uploads to, and the repo eval downloads
are guaranteed identical.

Repo names (see ablation_naming.hf_repo_name):
    socius/Qwentaur-0.6B-LoRA-r16-f0.0625
    socius/Llama-Centaur-1B-LoRA-r32-f0.5
    socius/Qwentaur-8B-LoRA-r64            (full-data: no -f suffix)
    socius/Olmotaur-1B-LoRA-r16            (new-family baselines)
    socius/Smoltaur-0.1B-LoRA-r16

Covers all four families: qwen, llama, and the new families olmo/smollm
(ADDITIONAL_EXPERIMENTS.md §7). For the new families the only cells are the
rank-16/full-data baselines, which are uploaded by default (they are new
adapters, not main-paper runs).

Local adapter directories are resolved with the same cell_output_dir() the
scheduler used to write them. Precision (bf16 vs 4bit) is auto-detected: for
each cell whichever of the two candidate dirs actually exists on disk is used,
so you do not have to remember --quantise_from here. Baseline cells trained by
invoking the training script directly (whose default output naming carries no
-r/-f coordinates, e.g. ./outputs/olmotaur-1b-bf16) are found via a fallback
probe of that layout.

Auth: requires a write token for the socius org. Either run
``huggingface-cli login`` once, or pass --token, or set HF_TOKEN.

Usage
-----
    # Inspect what WOULD be uploaded (no network calls):
    python upload_adapters.py --family qwen --dry_run

    # Upload every Qwen ablation adapter to socius (private by default):
    python upload_adapters.py --family qwen

    # Upload Llama, public, skipping repos already populated on the Hub:
    python upload_adapters.py --family llama --public --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Single source of truth for the experiment grid + repo names.
from utils import FAMILIES, build_cells, cell_output_dir, hf_repo_name


def resolve_local_dir(output_root, fam, cell):
    """Return the on-disk adapter dir for a cell, auto-detecting bf16/4bit.

    The scheduler always writes an explicit -bf16 or -4bit suffix, so we try
    both and return whichever exists (preferring bf16). Baseline cells may
    instead have been trained by invoking the training script directly, whose
    default output dir carries no -r/-f coordinates and lives directly under
    ./outputs (e.g. ./outputs/olmotaur-1b-bf16), so those candidates are
    probed as a fallback. Returns (path, quantised) or (None, None) if
    nothing is present yet.
    """
    candidates = []
    for quantised in (False, True):          # prefer bf16 if both somehow exist
        candidates.append((cell_output_dir(output_root, fam, cell, quantised),
                           quantised))
    if cell["axis"] == "baseline":
        s = cell["size"].lower()
        for quantised, q in ((False, "bf16"), (True, "4bit")):
            candidates.append(
                (os.path.join("./outputs", f"{fam['run_prefix']}-{s}-{q}"),
                 quantised))
    for path, quantised in candidates:
        if os.path.isdir(path):
            return path, quantised
    return None, None


def dir_has_adapter(path):
    """True if the dir looks like a saved LoRA adapter (config + weights)."""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    has_cfg = "adapter_config.json" in files
    has_wts = any(f.startswith("adapter_model.") for f in files)
    return has_cfg and has_wts


def make_model_card(repo_name, args, cell, quantised):
    """A minimal, machine-readable model card so the repo is self-describing."""
    frac = cell["fraction"]
    prec = "4-bit QLoRA" if quantised else "bf16 LoRA"
    data_desc = ("the full" if abs(frac - 1.0) < 1e-9
                 else f"a stratified `{frac:g}` subset of")
    return f"""---
library_name: peft
tags:
- lora
- centaur
- psych-101
- cognitive-modeling
- ablation
base_model_relation: adapter
---

# {repo_name}

LoRA adapter from the additional experiments (rank sweep / dataset-size
ablation / new model families) for
[*Small Foundation Models of Human Cognition and Behaviour*](https://arxiv.org/abs/2608.05224).

| field          | value |
|----------------|-------|
| family         | {args.family} |
| model size     | {cell['size']} |
| LoRA rank      | {cell['rank']} (alpha = rank, rsLoRA) |
| data fraction  | {frac:g} of Psych-101 |
| axis           | {cell['axis']} |
| precision      | {prec} |

Trained for 1 epoch on {data_desc}
[`marcelbinz/Psych-101`](https://huggingface.co/datasets/marcelbinz/Psych-101).
Evaluate with `eval_model.py --backend unsloth`.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Upload rank-sweep + dataset-size LoRA adapters to socius."
    )
    parser.add_argument(
        "--family", required=True, choices=sorted(FAMILIES.keys()),
        help="Model family to upload. Run families as separate invocations.",
    )
    parser.add_argument(
        "--axes", nargs="+", default=["rank", "datasize"],
        choices=["rank", "datasize"],
        help="Which sweep's adapters to upload (default: both).",
    )
    parser.add_argument(
        "--namespace", default="socius",
        help="HuggingFace org/user to upload under (default: socius).",
    )
    parser.add_argument("--output_root", default=None,
                        help="Default: ./outputs/<run_prefix> (matches scheduler).")
    parser.add_argument(
        "--include_baseline", action="store_true",
        help="Also upload the rank=16, fraction=1.0 baseline cells "
             "(off by default — those are the existing main-paper runs).",
    )
    parser.add_argument(
        "--private", dest="private", action="store_true", default=True,
        help="Create repos as private (default).",
    )
    parser.add_argument(
        "--public", dest="private", action="store_false",
        help="Create repos as public instead of private.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip cells whose target repo already contains an adapter on the Hub.",
    )
    parser.add_argument("--token", default=None,
                        help="HF write token (else uses login cache / HF_TOKEN).")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    fam = FAMILIES[args.family]
    if args.output_root is None:
        args.output_root = os.path.join("./outputs", fam["run_prefix"])

    cells = build_cells(fam, args.axes)
    # Baseline cells are skipped only when they already exist from the main
    # paper (baseline_exists=True: qwen/llama). The new families' (olmo/
    # smollm) baselines are the adapters this folder trains, so they are
    # uploaded by default.
    skip_baseline = fam.get("baseline_exists", True) and not args.include_baseline
    launchable = [c for c in cells
                  if not (skip_baseline and c["axis"] == "baseline")]

    # Resolve local dirs up front so the plan reflects reality on disk.
    plan = []
    missing = []
    for cell in launchable:
        local_dir, quantised = resolve_local_dir(args.output_root, fam, cell)
        repo = f"{args.namespace}/{hf_repo_name(args.family, cell)}"
        if local_dir is None or not dir_has_adapter(local_dir):
            missing.append((cell, repo, local_dir))
            continue
        plan.append({
            "size": cell["size"], "rank": cell["rank"],
            "fraction": cell["fraction"], "axis": cell["axis"],
            "quantised": quantised, "local_dir": local_dir,
            "repo_id": repo,
            "_cell": cell,
        })

    print("\n" + "=" * 78)
    print(f"  Upload ablation adapters -> {args.namespace}  (family: {args.family})")
    print("=" * 78)
    print(f"  Axes            : {', '.join(args.axes)}")
    print(f"  Visibility      : {'private' if args.private else 'PUBLIC'}")
    print(f"  Output root     : {args.output_root}")
    print(f"  Ready to upload : {len(plan)}")
    print(f"  Missing on disk : {len(missing)}")
    print("-" * 78)
    print(f"  {'size':>5}  {'rank':>4}  {'frac':>7}  {'prec':>4}  repo_id")
    print("-" * 78)
    for e in plan:
        print(f"  {e['size']:>5}  {e['rank']:>4}  {e['fraction']:>7g}  "
              f"{'4bit' if e['quantised'] else 'bf16':>4}  {e['repo_id']}")
    if missing:
        print("-" * 78)
        print("  NOT FOUND ON DISK (skipped -- train these or check --output_root):")
        for cell, repo, _ in missing:
            print(f"  {cell['size']:>5}  {cell['rank']:>4}  {cell['fraction']:>7g}  "
                  f"      -> {repo}")
    print("=" * 78)

    os.makedirs(args.output_root, exist_ok=True)
    plan_path = os.path.join(args.output_root, "upload_plan.json")
    with open(plan_path, "w") as fh:
        json.dump([{k: v for k, v in e.items() if k != "_cell"} for e in plan],
                  fh, indent=2)
    print(f"\nWrote plan -> {plan_path}")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return
    if not plan:
        print("\nNothing to upload. Exiting.")
        return

    # Import the Hub client only when we actually upload (keeps --dry_run light).
    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    status_path = os.path.join(args.output_root, "upload_status.jsonl")
    print(f"Uploading {len(plan)} adapters. Status -> {status_path}\n")

    for i, entry in enumerate(plan, 1):
        repo_id, local_dir, cell = entry["repo_id"], entry["local_dir"], entry["_cell"]

        if args.resume and dir_has_adapter_on_hub(api, repo_id):
            print(f"[skip] {repo_id} (already populated on the Hub)")
            continue

        print(f"[push] {i}/{len(plan)}  {repo_id}")
        print(f"       from {local_dir}")
        t0 = time.time()
        record = {
            "repo_id": repo_id, "local_dir": local_dir,
            "size": cell["size"], "rank": cell["rank"],
            "fraction": cell["fraction"], "axis": cell["axis"],
            "quantised": entry["quantised"], "private": args.private,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            api.create_repo(repo_id, private=args.private, exist_ok=True,
                            repo_type="model")
            # Write a small model card alongside the adapter weights.
            card = make_model_card(hf_repo_name(args.family, cell), args,
                                   cell, entry["quantised"])
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id, repo_type="model",
            )
            api.upload_folder(
                repo_id=repo_id, repo_type="model", folder_path=local_dir,
                commit_message="Upload ablation LoRA adapter",
            )
            record["status"] = "ok"
        except Exception as exc:                       # noqa: BLE001 — log & continue
            record["status"] = "FAILED"
            record["error"] = repr(exc)
            print(f"       -> FAILED: {exc}")
        record["elapsed_sec"] = round(time.time() - t0, 1)
        with open(status_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if record["status"] == "ok":
            print(f"       -> ok ({record['elapsed_sec']:.0f}s) "
                  f"https://huggingface.co/{repo_id}\n")

    print("Upload complete.")


def dir_has_adapter_on_hub(api, repo_id):
    """True if the Hub repo already contains adapter weights (for --resume)."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:                                  # repo doesn't exist yet
        return False
    return any(f.startswith("adapter_model.") for f in files)


if __name__ == "__main__":
    main()
