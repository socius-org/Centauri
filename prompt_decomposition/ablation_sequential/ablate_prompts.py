#!/usr/bin/env python3
"""
Ablate Psych-101 prompts under five conditions for shortcut-learning analysis.

Reads the Psych-101-test dataset and produces one JSONL file per condition:
  original.jsonl            — unmodified text
  instruction_ablated.jsonl — instruction header removed, trial content preserved
  content_masked.jsonl      — instruction replaced with I_min, stimulus/feedback masked
  history_only.jsonl        — I_min + "You {verb} <<response>>." per trial
  choice_only.jsonl         — bare "<<response>>" tokens only

Each JSONL has columns: {"text": ..., "experiment": ..., "participant": ...}
Compatible with eval_ablation.py.

Usage:
    python ablate_prompts.py
    python ablate_prompts.py --output-dir ablated_data
    python ablate_prompts.py --output-dir ablated_data --experiments peterson2021using bahrami2020four
"""

import argparse
import json
import os
import re

from tqdm import tqdm
from datasets import load_dataset
from ablation_utils import ablate_text, ablate_line_content


# ================================================================
# Constants
# ================================================================

CONDITIONS = [
    "original",
    "instruction_ablated",
    "content_masked",
    "history_only",
    "choice_only",
]

# All 32 experiments where ablation is meaningful
EXPERIMENTS = [
    # Original 23
    "badham2017deficits",
    "bahrami2020four",
    "collsiöö2023MCPL",
    "feng2021dynamics",
    "frey2017cct",
    "garcia2023experiential",
    "gershman2018deconstructing",
    "krueger2022identifying",
    "lefebvre2017behavioural",
    "peterson2021using",
    "plonsky2018when",
    "sadeghiyeh2020temporal",
    "schulz2020finding",
    "somerville2017charting",
    "speekenbrink2008learning",
    "steingroever2015data",
    "waltz2020differential",
    "wilson2014humans",
    "wise2019acomputational",
    "wu2018generalisation",
    "wulff2018description",
    "wulff2018sampling",
    "xiong2023neural",
    # 9 additional (previously Tier 2/4, included for evaluation ablation)
    "flesch2018comparing",
    "gershman2020reward",
    "hilbig2014generalized",
    "kool2016when",
    "kool2017cost",
    "levering2020revisiting",
    "tomov2020discovery",
    "tomov2021multitask",
    "zorowitz2023data",
]


# ================================================================
# Prompt parsing
# ================================================================

def extract_response_tokens(text):
    """Get unique response tokens from <<...>> markers and instruction text.

    Merges tokens found in <<...>> markers with options declared in the instruction
    (e.g., 'labeled A and B', 'pressing X or Y') to handle cases where a participant
    never chose one of the available options.
    """
    # Tokens from actual choices
    seen = set()
    unique = []
    for t in re.findall(r'<<([^>]+)>>', text):
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Also extract declared options from instruction and trial-header text
    instruction, trial_lines = parse_prompt(text)
    full_header = instruction + '\n' + '\n'.join(trial_lines)
    declared = set()
    # "labeled A, B, and C" or "labeled A and B"
    for m in re.findall(r'labeled[: ]+([A-Z](?:[, ]+(?:and )?[A-Z])*)', instruction):
        declared.update(re.findall(r'[A-Z]', m))
    # "pressing X or Y" (single letters only)
    for m in re.findall(r'pressing ([A-Z](?: or [A-Z]))', instruction):
        declared.update(re.findall(r'[A-Z]', m))
    # "Option X delivers..." or "option X and option Y" (peterson/plonsky/garcia)
    for m in re.findall(r'[Oo]ption ([A-Z]) delivers', full_header):
        declared.add(m)
    for m in re.findall(r'option ([A-Z]) and option ([A-Z])', full_header):
        declared.update(m)

    # Add any declared options not already seen (append at end)
    for t in sorted(declared):
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


def parse_prompt(text):
    """Split prompt text into instruction header and trial content lines.

    The boundary is the first line containing <<...>> markers, then we backtrack
    to include preceding trial-context lines (game headers, instructed trials, stimuli).

    Returns:
        (instruction_str, trial_lines_list)
    """
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if '<<' in line and '>>' in line:
            # Backtrack to include trial-context lines before the first choice
            split = i
            while split > 0:
                prev = lines[split - 1].strip()
                if (re.match(r'(?:Game|Block|Round|Trial|Environment number) \d', prev)
                        or re.match(r'There are \d+ (?:trials|loss cards)', prev)
                        or re.match(r'You are instructed to', prev)
                        or re.match(r'You (?:see|saw|go to|encounter|find|observe|are shown|are seeing|get a tree)', prev)
                        or re.match(r'You can choose between', prev)
                        or re.match(r'You will (?:be awarded|lose) \d', prev)
                        or re.match(r'You have \d+ choices', prev)
                        or re.match(r'(?:Stimulus|Lottery|Cue) \w', prev)
                        or re.match(r'(?:Progladine|Amalydine):', prev)
                        or re.match(r'(?:V|K|H): ', prev)
                        or re.match(r'(?:Round|Game) \d+:', prev)
                        or re.match(r'Option [A-Z] delivers', prev)
                        or re.match(r'The (?:hazard rate is|value of option|new starting|current market)', prev)
                        or re.match(r'Your station:', prev)
                        or re.match(r'You are in room', prev)
                        or re.match(r'Product \w ratings:', prev)
                        or re.match(r'There is (?:no|a) treasure multiplier', prev)
                        or prev == ''):
                    split -= 1
                else:
                    break
            return '\n'.join(lines[:split]).rstrip(), lines[split:]
    return text, []


def extract_verb(trial_lines):
    """Extract response verb (press/answer/say/etc.) from the first trial line that has one."""
    for line in trial_lines:
        if '<<' in line:
            m = re.search(r'You (press|answer|estimate|say|choose|select)', line, re.IGNORECASE)
            if m:
                return m.group(1).lower()
    return "press"


def build_minimal_instruction(response_tokens):
    """Construct I_min: defines valid response tokens without any task semantics."""
    n = len(response_tokens)
    if n == 0:
        return "You will make choices."
    if n == 1:
        return f"You will respond with {response_tokens[0]}."
    if n == 2:
        return f"You will respond with {response_tokens[0]} or {response_tokens[1]}."
    if n > 5:
        return f"You will respond with {', '.join(response_tokens[:4])}, etc."
    return f"You will respond with {', '.join(response_tokens[:-1])}, or {response_tokens[-1]}."


# ================================================================
# Condition builders
# ================================================================

# Per-trial/per-round instruction patterns to strip from instruction_ablated lines
_INLINE_INSTRUCTION_PATTERNS = [
    # Organizational / horizon info — strip in instruction_ablated
    # wu2018generalisation: "You have 5 choices to make in this environment."
    r'You have \d+ choices to make in this environment\.\s*',
    # xiong2023neural / feng2021dynamics: "There are 100 trials in this game."
    r'There are \d+ trials in this game\.\s*',
    # badham2017deficits: "You encounter a new problem with a new rule..."
    # wulff2018sampling: "You encounter a new choice problem."
    r'You encounter a new[^.]*[.:]\s*',
    # NOTE: Per-block STIMULI are NOT stripped here — they are kept in
    # instruction_ablated because they carry task-informative content:
    #   - Wu: "The value of option 21 is 70."  (revealed observation)
    #   - Xiong: "The hazard rate is 0.2."  (per-game parameter)
    #   - Frey: "You will be awarded 150 points..."  (per-round reward)
    #   - Frey: "You will lose 75 points..."  (per-round punishment)
    #   - Frey: "There are 20 loss cards..."  (per-round risk)
    # These are masked in content_masked via ablate_line_content().
]


def _strip_inline_instructions(line):
    """Remove per-round/per-environment instruction phrases from a line."""
    for pat in _INLINE_INSTRUCTION_PATTERNS:
        line = re.sub(pat, '', line)
    return line.strip()


def build_instruction_ablated(trial_lines):
    """Remove instruction header. Keep all trial content (stimuli, feedback, choices) unchanged.

    Merges any pre-choice lines (stimulus/context before the first <<>>) into the
    first choice line so the first trial has the same format as subsequent trials.
    Also strips per-round/per-environment instructions embedded in trial lines.
    """
    if not trial_lines:
        return ''
    # Strip inline instructions from each line
    cleaned = []
    for line in trial_lines:
        stripped = _strip_inline_instructions(line)
        if stripped:
            cleaned.append(stripped)
        elif cleaned and cleaned[-1].strip():
            cleaned.append('')
    trial_lines = cleaned
    # Remove trailing blank lines
    while trial_lines and not trial_lines[-1].strip():
        trial_lines.pop()
    return '\n'.join(trial_lines)


def build_content_masked(text, i_min_override=None):
    """I_min + trial content with values masked. All instruction replaced with I_min.

    Uses parse_prompt() to identify the instruction boundary, replaces the entire
    instruction block with I_min, then applies ablate_line_content() to mask
    task-informative values in trial lines while preserving trial structure.
    """
    _, trial_lines = parse_prompt(text)
    if i_min_override:
        i_min = i_min_override
    else:
        response_tokens = extract_response_tokens(text)
        i_min = build_minimal_instruction(response_tokens)

    # Per-block instruction lines to remove entirely from content_masked
    _CM_REMOVE = [
        r'You encounter a new problem with a new rule',
    ]

    ablated_lines = [i_min, ""]

    for line in trial_lines:
        # Remove per-block instruction lines
        if any(re.match(pat, line.strip()) for pat in _CM_REMOVE):
            continue
        if not line.strip():
            if ablated_lines and ablated_lines[-1].strip():
                ablated_lines.append("")
            continue

        masked = ablate_line_content(line)
        if masked.strip():
            ablated_lines.append(masked)
        elif ablated_lines and ablated_lines[-1].strip():
            ablated_lines.append("")

    # Clean up consecutive empty lines
    cleaned = []
    prev_empty = False
    for line in ablated_lines:
        if not line.strip():
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False

    return "\n".join(cleaned)


def build_history_only(i_min, trial_lines, verb):
    """I_min + choice history only. All <<response>> tokens per line preserved.

    Extracts the full verb phrase before each <<>> (e.g., 'say that the
    Caldionine concentration is <<30>>', 'press <<C>> and then type <<turquoise>>').
    Falls back to 'You {verb} <<response>>.' if no verb phrase is found.
    """
    out = [i_min, ""]
    for line in trial_lines:
        # Extract full "You verb ... <<response>>" phrases (may span multiple <<>>)
        full_matches = re.findall(
            r'You (?:press|type|choose|select|answer|say|estimate|predict)\b.*?<<[^>]+>>(?:[^.]*<<[^>]+>>)*',
            line, re.IGNORECASE
        )
        if full_matches:
            for m in full_matches:
                out.append(f"{m}.")
        else:
            # Fallback: bare <<response>> tokens
            matches = re.findall(r'<<([^>]+)>>', line)
            for m in matches:
                out.append(f"You {verb} <<{m}>>.")
    return '\n'.join(out)


def build_choice_only(trial_lines):
    """Bare choice tokens only. All <<response>> tokens per line preserved."""
    out = []
    for line in trial_lines:
        matches = re.findall(r'<<([^>]+)>>', line)
        for m in matches:
            out.append(f"<<{m}>>")
    return '\n'.join(out)


# ================================================================
# Main processing
# ================================================================

def ablate_sample(text):
    """Produce all ablation conditions for a single participant's prompt text.

    Returns:
        dict mapping condition name to ablated text string
    """
    _, trial_lines = parse_prompt(text)
    response_tokens = extract_response_tokens(text)

    # Special case: collsiöö MCPL uses fixed 10-90 scale
    if 'Caldionine' in text:
        i_min = "You will respond with one of nine values 10, 20, 30, 40, 50, 60, 70, 80, 90."
    else:
        i_min = build_minimal_instruction(response_tokens)

    verb = extract_verb(trial_lines)

    return {
        "original":             text,
        "instruction_ablated":  build_instruction_ablated(trial_lines),
        "content_masked":       build_content_masked(text, i_min_override=i_min if 'Caldionine' in text else None),
        "history_only":         build_history_only(i_min, trial_lines, verb),
        "choice_only":          build_choice_only(trial_lines),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablate Psych-101-test prompts for shortcut-learning analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ablate_prompts.py
  python ablate_prompts.py --output-dir ablated_data
  python ablate_prompts.py --experiments peterson2021using bahrami2020four
        """
    )
    parser.add_argument("--output-dir", default="./ablated_data",
                        help="Output directory for JSONL files (default: ./ablated_data)")
    parser.add_argument("--data-path", default=None,
                        help="Local dataset path (default: marcelbinz/Psych-101-test from HuggingFace)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Filter to specific experiments (default: all 32 ablation experiments)")
    parser.add_argument("--all-experiments", action="store_true",
                        help="Run on all 46 experiments instead of the default 32")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================
    # Load dataset
    # ============================================
    print("Loading Psych-101-test dataset...")
    if args.data_path:
        dataset = load_dataset("json", data_files=args.data_path)["train"]
    else:
        dataset = load_dataset("marcelbinz/Psych-101-test")["test"]
    print(f"  Total samples: {len(dataset):,}")

    # Filter to selected experiments (default: all 32 where ablation is meaningful)
    if args.experiments:
        exps = args.experiments
    elif not args.all_experiments:
        exps = EXPERIMENTS
    else:
        exps = None

    if exps:
        dataset = dataset.filter(
            lambda x: any(x['experiment'].startswith(e) for e in exps)
        )
        print(f"  Filtered to {len(dataset):,} samples ({len(exps)} experiments)")

    # ============================================
    # Open output files
    # ============================================
    writers = {}
    for cond in CONDITIONS:
        path = os.path.join(args.output_dir, f"{cond}.jsonl")
        writers[cond] = open(path, 'w', encoding='utf-8')

    # ============================================
    # Process each sample
    # ============================================
    print("\nAblating prompts...")
    n_choices_total = {c: 0 for c in CONDITIONS}

    for sample in tqdm(dataset, desc="Processing"):
        text = sample['text']
        ablated = ablate_sample(text)

        # Sanity check: all conditions must preserve the same choice count
        orig_count = len(re.findall(r'<<[^>]+>>', text))

        for cond in CONDITIONS:
            cond_count = len(re.findall(r'<<[^>]+>>', ablated[cond]))
            n_choices_total[cond] += cond_count

            row = {
                "text": ablated[cond],
                "experiment": sample["experiment"],
                "participant": sample.get("participant", ""),
            }
            writers[cond].write(json.dumps(row, ensure_ascii=False) + '\n')

    for w in writers.values():
        w.close()

    # ============================================
    # Summary
    # ============================================
    print(f"\nSaved ablated datasets to {args.output_dir}/")
    print(f"{'Condition':25s} {'Size (MB)':>10s} {'Total choices':>15s}")
    print("-" * 55)
    for cond in CONDITIONS:
        path = os.path.join(args.output_dir, f"{cond}.jsonl")
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {cond:23s} {size_mb:10.1f} {n_choices_total[cond]:15,}")

    # Verify choice preservation
    orig_choices = n_choices_total["original"]
    for cond in CONDITIONS:
        if n_choices_total[cond] != orig_choices:
            print(f"\n  WARNING: {cond} has {n_choices_total[cond]:,} choices "
                  f"vs {orig_choices:,} in original")


if __name__ == "__main__":
    main()
