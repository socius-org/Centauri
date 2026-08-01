from datasets import load_dataset
import random

random.seed(3407)
ds = load_dataset("marcelbinz/Psych-101-test", split="test")

for exp in ["hebart2023things", "ruggeri2022globalizability"]:
    samples = [ex for ex in ds if ex["experiment"].startswith(exp)]
    picked = random.sample(samples, min(3, len(samples)))

    with open(f"samples_{exp}.txt", "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp} ({len(samples)} total participants)\n")
        f.write("=" * 80 + "\n")
        for i, s in enumerate(picked):
            f.write(f"\n--- Sample {i+1} (full text) ---\n\n")
            f.write(s["text"])
            f.write("\n")
    print(f"Saved samples_{exp}.txt ({len(picked)} samples)")
