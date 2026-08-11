"""
utils.py — shared library for the Centauri training / evaluation pipeline.

Single source of truth imported by schedule_training_runs.py,
schedule_eval_runs.py, upload_adapters.py and the per-family train scripts.
Consolidates what used to live in schedule_ablation_runs.py (the experiment
grid + cell helpers), ablation_naming.py (HuggingFace repo names), and
load_fraction.py (the dataset-size loader), so a fresh clone has one place
to edit the design and every script stays in sync by construction.

Contents:
  * FAMILIES / build_cells / is_quantised / cell_run_name / cell_output_dir
    / fractions_indices_ok  — the per-family experiment design and cell math.
  * hf_repo_name / hf_repo_id / DISPLAY_NAME  — adapter repo naming.
  * load_psych101_fraction  — fraction-aware Psych-101 loader for training.
"""

import json
import os


# ══════════════════════════════════════════════════════════════════════════
# Experiment design (LoRA-rank sweep + dataset-size ablation)
# ══════════════════════════════════════════════════════════════════════════
# Each family specifies:
#   size_order  : the family's size ladder, smallest -> largest. Drives both
#                 run ordering and the --quantise_from threshold.
#   rank_sweep  : {size: [ranks]} at full data.
#   datasize    : {size: [fractions]} at fixed rank 16 (1.0 is the main-paper
#                 run and is skipped automatically).
#   train_script: default training entry point (path relative to repo root).
#   run_prefix  : wandb run-name / output-dir prefix.
#   wandb_project: default wandb project for the family.
#   baseline_exists: True if the rank-16/full-data baseline already exists
#                 from the main paper (qwen/llama) and should be skipped
#                 unless --include_baseline. False for the new families
#                 (olmo/smollm), whose baseline IS the experiment.
#   max_seq_length (optional): {size: int} overrides of the 32768 default for
#                 short-context models. Consumed by schedule_eval_runs.
BASELINE_RANK = 16
BASELINE_FRACTION = 1.0
DEFAULT_MAX_SEQ_LENGTH = 32768

FAMILIES = {
    "qwen": {
        "size_order": ["0.6B", "1.7B", "4B", "8B", "14B"],
        "rank_sweep": {
            "0.6B": [4, 8, 16, 32, 64],
            # 1.7B carries only r=8: it completes the r=8 curve (the rank
            # matched to Centaur-70B) without adding a full sweep column.
            "1.7B": [8],
            "4B":   [4, 8, 16, 32, 64],
            "8B":   [4, 8, 16, 32, 64],
        },
        "datasize": {
            "0.6B": [0.0625, 0.125, 0.25, 0.5],
            "4B":   [0.0625, 0.125, 0.25, 0.5],
            "8B":   [0.0625, 0.125, 0.25, 0.5],
        },
        "train_script": "./qwentaur/train_qwentaur.py",
        "run_prefix": "qwentaur",
        "wandb_project": "qwentaur-rank-and-datasize",
        "baseline_exists": True,   # main-paper run; skip unless --include_baseline
    },
    "llama": {
        # Llama has only three sizes, so the smallest (1B) and largest (8B)
        # get the dense rank sweep, with 3B as the mid anchor.
        "size_order": ["1B", "3B", "8B"],
        "rank_sweep": {
            "1B": [4, 8, 16, 32, 64],
            "3B": [8, 16, 32],
            "8B": [4, 8, 16, 32, 64],
        },
        "datasize": {
            "1B": [0.0625, 0.125, 0.25, 0.5],
            "3B": [0.0625, 0.125, 0.25, 0.5],
            "8B": [0.0625, 0.125, 0.25, 0.5],
        },
        "train_script": "./llama-centaur/train_centaur.py",
        "run_prefix": "llama-centaur",
        "wandb_project": "llama-centaur-rank-and-datasize",
        "baseline_exists": True,   # main-paper run; skip unless --include_baseline
    },
    # ------------------------------------------------------------------
    # New model families (misc/ADDITIONAL_EXPERIMENTS.md §7). Same L-shape as
    # qwen/llama, but the rank-16/full-data baseline does NOT exist from the
    # main paper, so it is a launchable cell (baseline_exists=False).
    # NOTE: olmo has only two sizes, so its datasize interaction read is
    # weaker than the other families'.
    # ------------------------------------------------------------------
    "olmo": {
        "size_order": ["1B", "7B"],
        "rank_sweep": {
            "1B": [4, 8, 16, 32, 64],
            "7B": [4, 8, 16, 32, 64],
        },
        "datasize": {
            "1B": [0.0625, 0.125, 0.25, 0.5],
            "7B": [0.0625, 0.125, 0.25, 0.5],
        },
        "train_script": "./olmotaur/train_olmotaur.py",
        "run_prefix": "olmotaur",
        "wandb_project": "olmotaur",
        "baseline_exists": False,
        # OLMo-2-0425-1B was pretrained at a 4096-token context.
        "max_seq_length": {"1B": 4096, "7B": 32768},
    },
    "smollm": {
        # 0.1B = SmolLM2-135M, 0.4B = SmolLM2-360M (B-suffix labels for
        # consistency with the other families' size ladders).
        "size_order": ["0.1B", "0.4B", "1.7B", "3B"],
        "rank_sweep": {
            "0.1B": [4, 8, 16, 32, 64],
            "0.4B": [4, 8, 16, 32, 64],
            "1.7B": [4, 8, 16, 32, 64],
            "3B":   [4, 8, 16, 32, 64],
        },
        "datasize": {
            "0.1B": [0.0625, 0.125, 0.25, 0.5],
            "0.4B": [0.0625, 0.125, 0.25, 0.5],
            "1.7B": [0.0625, 0.125, 0.25, 0.5],
            "3B":   [0.0625, 0.125, 0.25, 0.5],
        },
        "train_script": "./smoltaur/train_smoltaur.py",
        "run_prefix": "smoltaur",
        "wandb_project": "smoltaur",
        "baseline_exists": False,
        # SmolLM2 models were pretrained at an 8192-token context.
        "max_seq_length": {"0.1B": 8192, "0.4B": 8192, "1.7B": 8192,
                           "3B": 32768},
    },
}


def build_cells(fam, axes):
    """Enumerate (size, rank, fraction, axis) cells for one family.

    The shared baseline (rank=BASELINE_RANK, fraction=BASELINE_FRACTION) is
    emitted at most once per size and labelled 'baseline', so it is never
    duplicated between the two sweeps. It is skipped at launch as an existing
    run unless --include_baseline is set.
    """
    seen = set()
    cells = []

    def add(size, rank, fraction, axis):
        key = (size, rank, round(fraction, 6))
        if key in seen:
            return
        seen.add(key)
        is_baseline = (rank == BASELINE_RANK
                       and abs(fraction - BASELINE_FRACTION) < 1e-9)
        cells.append({
            "size": size,
            "rank": rank,
            "fraction": round(fraction, 6),
            "axis": "baseline" if is_baseline else axis,
        })

    if "rank" in axes:
        for size, ranks in fam["rank_sweep"].items():
            for r in ranks:
                add(size, r, BASELINE_FRACTION, "rank_sweep")

    if "datasize" in axes:
        for size, fracs in fam["datasize"].items():
            for f in fracs:
                add(size, BASELINE_RANK, f, "datasize_ablation")

    return cells


def cell_run_name(fam, cell):
    """Stable, human-readable wandb run name encoding the full coordinate."""
    s = cell["size"].lower()
    return f"{fam['run_prefix']}-{s}-r{cell['rank']}-f{cell['fraction']:g}"


def cell_output_dir(base, fam, cell, quantised):
    s = cell["size"].lower()
    q = "4bit" if quantised else "bf16"
    return os.path.join(
        base,
        f"{fam['run_prefix']}-{s}-r{cell['rank']}-f{cell['fraction']:g}-{q}",
    )


def is_quantised(fam, size, quantise_from):
    """Decide whether this size trains in 4-bit, based on --quantise_from."""
    if quantise_from is None:
        return False
    order = fam["size_order"]
    if quantise_from not in order:
        raise ValueError(
            f"--quantise_from '{quantise_from}' is not a valid size for this "
            f"family. Valid sizes: {order}"
        )
    return order.index(size) >= order.index(quantise_from)


def fractions_indices_ok(indices_dir, needed_fractions):
    """Confirm the subset-index file exists and covers all needed fractions.

    Families whose cells are all full-data (e.g. the olmo/smollm baselines)
    need no indices file at all, so full-data fractions are excluded before
    the existence check.
    """
    real = [f for f in needed_fractions if abs(f - 1.0) > 1e-9]
    if not real:
        return True, "ok (only full-data cells; no indices needed)"
    idx_path = os.path.join(indices_dir, "fractions_indices.json")
    if not os.path.exists(idx_path):
        return False, f"missing {idx_path}"
    with open(idx_path) as fh:
        have = set(json.load(fh).keys())
    missing = [f for f in real if str(f) not in have]
    if missing:
        return False, f"indices file lacks fractions {missing}"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════
# HuggingFace adapter repo naming (was ablation_naming.py)
# ══════════════════════════════════════════════════════════════════════════
# Human-facing model-family display name. The local run_prefix is lowercase
# and hyphenated for directories; the published repo uses the cased name.
DISPLAY_NAME = {
    "qwen": "Qwentaur",
    "llama": "Llama-Centaur",
    "olmo": "Olmotaur",
    "smollm": "Smoltaur",
}


def hf_repo_name(family, cell):
    """Map a cell dict {size, rank, fraction, ...} -> HF repo name.

    The fraction suffix is omitted for full-data (1.0) runs, so a full-data
    rank-sweep adapter reads e.g. ``Qwentaur-8B-LoRA-r64`` rather than
    ``...-r64-f1``. Every other coordinate is always present.
    """
    name = f"{DISPLAY_NAME[family]}-{cell['size']}-LoRA-r{cell['rank']}"
    if abs(cell["fraction"] - BASELINE_FRACTION) > 1e-9:
        name += f"-f{cell['fraction']:g}"
    return name


def hf_repo_id(namespace, family, cell):
    """Full ``namespace/RepoName`` id, e.g. ``socius/Qwentaur-8B-LoRA-r64``."""
    return f"{namespace}/{hf_repo_name(family, cell)}"


# ══════════════════════════════════════════════════════════════════════════
# Fraction-aware Psych-101 loader for training (was load_fraction.py)
# ══════════════════════════════════════════════════════════════════════════

def load_psych101_fraction(
    fraction=1.0,
    seed=3407,
    indices_dir="./psych101_fractions",
    dataset_name="marcelbinz/Psych-101",
    split="train",
):
    """Load the Psych-101 split, optionally restricted to a saved fraction.

    The returned dataset is shuffled with `seed`, matching the original
    training script's behaviour. Subsetting is applied to the *unshuffled*
    split first (so the saved indices are stable), then shuffled. Full-data
    runs with no precomputed indices fall back to the full split.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name)[split]
    idx_path = os.path.join(indices_dir, "fractions_indices.json")

    if abs(fraction - 1.0) < 1e-9 and not os.path.exists(idx_path):
        return ds.shuffle(seed=seed)

    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"No subset indices at {idx_path}. "
            "Run build_dataset_fractions.py first."
        )

    with open(idx_path) as fh:
        all_idx = json.load(fh)

    key = str(fraction)
    if key not in all_idx:
        raise KeyError(
            f"Fraction {fraction} not found in {idx_path}. "
            f"Available: {sorted(all_idx.keys())}. "
            "Re-run build_dataset_fractions.py including this fraction."
        )

    indices = [int(i) for i in all_idx[key]]
    return ds.select(indices).shuffle(seed=seed)
