import csv, os, re, glob, json

# Flat per-task eval CSVs live in results/psych101 (this file is in
# results/psych101/figures). Llama-Centaur files are socius-Llama-Centaur-*.csv.
_HERE = os.path.dirname(os.path.abspath(__file__))
ABL_DIR = os.path.join(_HERE, "..")
AGG_CSV = os.path.join(_HERE, "..", "psych101_aggr.csv")

BASE_COL = {"1B": "Centaur-1B (bf16)", "3B": "Centaur-3B (bf16)", "8B": "Centaur-8B (bf16)"}

def read_loss_mean(path):
    losses = []
    tasks = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            tasks.append(row["task"])
            losses.append(float(row["loss"]))
    assert len(losses) == 46, f"{path} has {len(losses)} rows"
    return sum(losses) / len(losses), set(tasks)

# Step 1: parse all ablation files
pat = re.compile(r"socius-Llama-Centaur-(\dB|\d+B)-LoRA-r(\d+)(?:-f([\d.]+))?\.csv$")
files = sorted(glob.glob(os.path.join(ABL_DIR, "*.csv")))
records = []  # (size, rank, fraction, mean)
all_tasks = None
for p in files:
    m = pat.search(os.path.basename(p))
    assert m, f"no match: {p}"
    size = m.group(1)
    rank = int(m.group(2))
    frac = float(m.group(3)) if m.group(3) else 1.0
    mean, tasks = read_loss_mean(p)
    records.append((size, rank, frac, mean))
    if all_tasks is None:
        all_tasks = tasks
    else:
        assert all_tasks == tasks, f"task set mismatch in {p}"

# Step 2: baseline rank16/full-data from aggregate over the 46 ablation tasks
baselines = {}
with open(AGG_CSV, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
for size, col in BASE_COL.items():
    vals = []
    seen = set()
    for row in rows:
        exp = row["Experiment"]
        if exp in all_tasks:
            raw = row[col].replace("†", "").strip()  # strip dagger
            vals.append(float(raw))
            seen.add(exp)
    missing = all_tasks - seen
    assert not missing, f"missing tasks for {size}: {missing}"
    assert len(vals) == 46, f"{size}: {len(vals)} matched"
    baselines[size] = sum(vals) / len(vals)

# Step 3: assemble outputs
# rank_sweep: full-data files (frac==1.0) + baseline as r16
rank_sweep = []
for size, rank, frac, mean in records:
    if frac == 1.0:
        rank_sweep.append({"size": size, "rank": rank, "mean_nll": round(mean, 6)})
for size in BASE_COL:
    rank_sweep.append({"size": size, "rank": 16, "mean_nll": round(baselines[size], 6)})
rank_sweep.sort(key=lambda d: (d["size"], d["rank"]))

# datasize: rank16 -f files + baseline as f1.0
datasize = []
for size, rank, frac, mean in records:
    if rank == 16 and frac != 1.0:
        datasize.append({"size": size, "fraction": frac, "mean_nll": round(mean, 6)})
for size in BASE_COL:
    datasize.append({"size": size, "fraction": 1.0, "mean_nll": round(baselines[size], 6)})
datasize.sort(key=lambda d: (d["size"], d["fraction"]))

# derived
# rank_range: max-min over full-data rank points incl baseline
rank_range = {}
data_drop = {}
final_doubling = {}
for size in BASE_COL:
    full_pts = [r["mean_nll"] for r in rank_sweep if r["size"] == size]
    rank_range[size] = round(max(full_pts) - min(full_pts), 6)
    # data_drop = NLL(frac0.0625) - NLL(full=baseline)
    f0625 = next(d["mean_nll"] for d in datasize if d["size"] == size and abs(d["fraction"] - 0.0625) < 1e-9)
    f05 = next(d["mean_nll"] for d in datasize if d["size"] == size and abs(d["fraction"] - 0.5) < 1e-9)
    data_drop[size] = round(f0625 - baselines[size], 6)
    final_doubling[size] = round(f05 - baselines[size], 6)

out = {
    "baselines": {k: round(v, 6) for k, v in baselines.items()},
    "rank_sweep": rank_sweep,
    "datasize": datasize,
    "derived": {"rank_range": rank_range, "data_drop": data_drop, "final_doubling": final_doubling},
}
print(json.dumps(out, indent=2))
