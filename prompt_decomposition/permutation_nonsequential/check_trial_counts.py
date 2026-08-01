import re
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("marcelbinz/Psych-101-test", split="test")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

for exp in ["hebart2023things", "ruggeri2022globalizability"]:
    samples = [ex for ex in ds if ex["experiment"].startswith(exp)]
    trial_counts = []
    token_counts = []
    for ex in samples:
        n_trials = len(re.findall(r"<<[^>]+>>", ex["text"]))
        n_tokens = len(tokenizer.encode(ex["text"], add_special_tokens=False))
        trial_counts.append(n_trials)
        token_counts.append(n_tokens)

    print(f"\n{exp} ({len(samples)} participants)")
    print(f"  Min trials:    {min(trial_counts)}  ({min(token_counts)} tokens)")
    print(f"  Max trials:    {max(trial_counts)}  ({max(token_counts)} tokens)")
    print(f"  Mean trials:   {sum(trial_counts)/len(trial_counts):.1f}  ({sum(token_counts)/len(token_counts):.0f} tokens)")
    print(f"  Median trials: {sorted(trial_counts)[len(trial_counts)//2]}  ({sorted(token_counts)[len(token_counts)//2]} tokens)")

    # Distribution: group by trial count, show mean tokens per group
    from collections import defaultdict
    groups = defaultdict(list)
    for nt, ntok in zip(trial_counts, token_counts):
        groups[nt].append(ntok)

    print(f"  Distribution:")
    for n in sorted(groups.keys()):
        toks = groups[n]
        c = len(toks)
        mean_tok = sum(toks) / c
        bar = "#" * min(c, 50)
        print(f"    {n:>4} trials ({mean_tok:>6.0f} tok): {c:>4} participants  {bar}")
