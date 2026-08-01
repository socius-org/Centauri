#!/usr/bin/env python3
"""
Order Permutation Test for Cognitive Foundation Models
=======================================================

Tests whether LLM-based cognitive foundation models produce predictions
invariant to the ordering of exchangeable context trials from Psych-101.

Under exchangeability, shuffling the order of context trials should not
change the predictive distribution for a held-out target trial.

Usage:
    python order_permutation_test.py \
        --model unsloth/Qwen3-8B \
        --experiments hebart2023things ruggeri2022globalizability \
        --n-permutations 50 \
        --output-dir results/martingale/Qwen3-8B
"""

import unsloth  # noqa: F401 — must precede HF imports
from unsloth import FastLanguageModel

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

SEED = 3407

EXPERIMENTS = ["hebart2023things", "ruggeri2022globalizability"]

DEFAULT_MAX_TRIALS = {
    "hebart2023things": 60,
    "ruggeri2022globalizability": None,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    stimulus: str
    response: str
    full_text: str


@dataclass
class Participant:
    instruction: str
    trials: List[Trial]
    response_tokens: List[str]
    participant_id: int


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------

def extract_response_tokens_from_instruction(instruction: str) -> Optional[List[str]]:
    """Extract response token letters from the instruction header.

    Handles patterns like:
        "assigned to the keys S, U, and Q" -> ['S', 'U', 'Q']
        "two options T and P"              -> ['T', 'P']
        "two options, labeled E and K"     -> ['E', 'K']
    """
    m = re.search(r'keys?\s+([A-Z]),\s*([A-Z]),?\s*and\s+([A-Z])', instruction)
    if m:
        return [m.group(1), m.group(2), m.group(3)]

    m = re.search(r'options?,?\s*(?:labeled\s+)?([A-Z])\s+and\s+([A-Z])', instruction)
    if m:
        return [m.group(1), m.group(2)]

    return None


def extract_response_tokens_fallback(text: str) -> List[str]:
    """Fallback: scan trial data for unique <<X>> response tokens."""
    return sorted({m.group(1) for m in re.finditer(r"<<([^>]+)>>", text)})


def parse_participant(text: str, participant_id: int) -> Participant:
    """Parse a Psych-101 text entry into a Participant."""
    lines = text.strip().split('\n')

    trial_lines = []
    instruction_lines = []
    found_first_trial = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not found_first_trial:
                instruction_lines.append(stripped)
            continue
        if '<<' in stripped and '>>' in stripped:
            found_first_trial = True
            trial_lines.append(stripped)
        elif not found_first_trial:
            instruction_lines.append(stripped)

    instruction = '\n'.join(instruction_lines).strip()

    response_tokens = extract_response_tokens_from_instruction(instruction)
    if response_tokens is None:
        response_tokens = extract_response_tokens_fallback(text)

    trials = []
    for line in trial_lines:
        resp_match = re.search(r'<<([^>]+)>>', line)
        if resp_match is None:
            continue
        response = resp_match.group(1)

        stim_match = re.search(r'^(.+?)\s*You press\s*<<', line)
        stimulus = stim_match.group(1).strip() if stim_match else line[:line.index('<<')].strip()

        trials.append(Trial(stimulus=stimulus, response=response, full_text=line.strip()))

    return Participant(
        instruction=instruction,
        trials=trials,
        response_tokens=response_tokens,
        participant_id=participant_id,
    )


def load_participants(
    experiment: str, tokenizer, args, dataset=None
) -> List[Participant]:
    """Load and filter participants for one experiment.

    Filters:
        1. Trial count (min_context + n_target_trials .. max_trials)
        2. Token length (must fit within max_seq_length)
        3. max_participants cap
    """
    if dataset is None:
        if args.data_path:
            dataset = load_dataset("json", data_files=args.data_path, split="train")
        else:
            dataset = load_dataset("marcelbinz/Psych-101-test", split="test")

    filtered = [ex for ex in dataset if ex["experiment"].startswith(experiment)]
    print(f"Found {len(filtered)} participants for '{experiment}'")
    if not filtered:
        raise ValueError(f"No data found for experiment '{experiment}'")

    max_trials = args.max_trials if args.max_trials is not None else DEFAULT_MAX_TRIALS.get(experiment)
    min_trials = args.min_context + args.n_target_trials

    participants = []
    n_too_few = n_too_many = 0
    for i, ex in enumerate(filtered):
        p = parse_participant(ex["text"], participant_id=i)
        if len(p.trials) < min_trials:
            n_too_few += 1
        elif max_trials and len(p.trials) > max_trials:
            n_too_many += 1
        else:
            participants.append(p)

    print(f"Trial count filtering:")
    print(f"  {n_too_few} excluded (fewer than {min_trials} trials)")
    if max_trials:
        print(f"  {n_too_many} excluded (more than {max_trials} trials)")
    print(f"  {len(participants)} passed")

    # Filter by token length
    before = len(participants)
    kept = []
    for p in participants:
        full_text = p.instruction + "\n" + "\n".join(t.full_text for t in p.trials)
        n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        if n_tokens <= args.max_seq_length:
            kept.append(p)
    participants = kept
    print(f"Token length filtering: {before - len(participants)} excluded "
          f"(limit={args.max_seq_length})")

    if args.max_participants and len(participants) > args.max_participants:
        participants = participants[:args.max_participants]

    print(f"Retained {len(participants)} participants")

    # Log response-token diversity
    token_sets: Dict[str, int] = {}
    for p in participants:
        key = ','.join(p.response_tokens)
        token_sets[key] = token_sets.get(key, 0) + 1
    for ts, count in sorted(token_sets.items(), key=lambda x: -x[1])[:10]:
        print(f"  response tokens {{{ts}}}: {count} participants")

    return participants


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(instruction: str, context_trials: List[Trial], target_stimulus: str) -> str:
    """Build a prompt ending with 'You press <<' for the model to complete."""
    parts = [instruction, ""]
    for trial in context_trials:
        parts.append(trial.full_text)
    parts.append(f"{target_stimulus} You press <<")
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def resolve_token_id(tokenizer, char: str) -> int:
    """Resolve a single-character response token to its token ID."""
    ids = tokenizer.encode(char, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    ids = tokenizer.encode(f" {char}", add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    if len(ids) == 2:
        return ids[-1]

    return tokenizer.encode(char, add_special_tokens=False)[0]


def build_token_id_map(tokenizer, response_tokens: List[str]) -> Dict[str, int]:
    """Map response token characters to their token IDs."""
    return {tok: resolve_token_id(tokenizer, tok) for tok in response_tokens}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictive_distribution(
    model, tokenizer, prompt: str, token_id_map: Dict[str, int], device: str
) -> Dict[str, float]:
    """Get softmax distribution over valid response tokens at the last position."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    logits = model(input_ids).logits[0, -1, :]

    token_ids = list(token_id_map.values())
    token_names = list(token_id_map.keys())
    probs = torch.softmax(logits[token_ids].float(), dim=0).cpu().numpy()

    return {name: float(prob) for name, prob in zip(token_names, probs)}


# ---------------------------------------------------------------------------
# Order Permutation Test
# ---------------------------------------------------------------------------

def run_order_permutation_test(
    model, tokenizer, participants: List[Participant],
    experiment: str, args, device: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Shuffle context order and record predictions.

    For each participant, takes the last n_target_trials as targets.
    For each target, shuffles the preceding context n_permutations times
    and records the full predictive distribution each time.

    Under exchangeability, predictions should be invariant to context order.
    """
    rows = []
    model_name = os.path.basename(args.model.rstrip('/'))

    for p in tqdm(participants, desc=f"Order permutation — {experiment}"):
        token_id_map = build_token_id_map(tokenizer, p.response_tokens)
        n_trials = len(p.trials)

        for target_offset in range(args.n_target_trials):
            target_idx = n_trials - args.n_target_trials + target_offset
            if target_idx < args.min_context:
                continue

            context_trials = p.trials[:target_idx]
            target_trial = p.trials[target_idx]

            for perm_i in range(args.n_permutations):
                shuffled = [context_trials[j] for j in rng.permutation(len(context_trials))]
                prompt = build_prompt(p.instruction, shuffled, target_trial.stimulus)
                dist = get_predictive_distribution(model, tokenizer, prompt, token_id_map, device)

                row = {
                    'model': model_name,
                    'experiment': experiment,
                    'participant_id': p.participant_id,
                    'target_idx': target_idx,
                    'permutation': perm_i,
                    'human_response': target_trial.response,
                    'response_tokens': ','.join(p.response_tokens),
                }
                for idx, tok in enumerate(p.response_tokens):
                    row[f'p_{idx}'] = dist.get(tok, 0.0)
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Order permutation test for cognitive foundation models"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model HF identifier (e.g. unsloth/Qwen3-8B)")
    parser.add_argument("--experiments", type=str, nargs="+", default=EXPERIMENTS,
                        choices=EXPERIMENTS,
                        help="Which experiments to test (default: both)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to local data (falls back to HuggingFace)")
    parser.add_argument("--n-permutations", type=int, default=50,
                        help="Number of context shuffles per target trial")
    parser.add_argument("--max-participants", type=int, default=None,
                        help="Cap on number of participants")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Max trials per participant (default: 60 for hebart, unlimited for ruggeri)")
    parser.add_argument("--min-context", type=int, default=5,
                        help="Minimum context trials required")
    parser.add_argument("--n-target-trials", type=int, default=3,
                        help="Number of target trials per participant")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for CSVs and config")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Enable 4-bit quantization")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset once
    if args.data_path:
        dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        dataset = load_dataset("marcelbinz/Psych-101-test", split="test")

    # Load model
    model_name = os.path.basename(args.model.rstrip('/'))
    print(f"Model:       {args.model}")
    print(f"Experiments: {args.experiments}")
    print(f"Permutations: {args.n_permutations}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    device = next(model.parameters()).device
    print(f"Loaded on {device} (dtype={args.dtype}, 4bit={args.load_in_4bit})")

    # Run test for each experiment
    for experiment in args.experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment}")
        print(f"{'='*60}")

        participants = load_participants(experiment, tokenizer, args, dataset)
        if not participants:
            print("No participants — skipping")
            continue

        t0 = time.time()
        df = run_order_permutation_test(
            model, tokenizer, participants, experiment, args, device, rng
        )
        path = os.path.join(args.output_dir, f"test1_order_{experiment}.csv")
        df.to_csv(path, index=False)
        print(f"{len(df)} rows -> {path} ({time.time() - t0:.1f}s)")

    # Save config
    config = {
        "model": args.model,
        "experiments": args.experiments,
        "test": "order_permutation",
        "n_permutations": args.n_permutations,
        "max_participants": args.max_participants,
        "max_trials_override": args.max_trials,
        "default_max_trials": {e: DEFAULT_MAX_TRIALS.get(e) for e in args.experiments},
        "min_context": args.min_context,
        "n_target_trials": args.n_target_trials,
        "seed": args.seed,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "max_seq_length": args.max_seq_length,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
