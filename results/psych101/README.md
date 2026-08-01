# Psych101

Evaluation results and analysis for models on the [Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101) behavioral prediction benchmark (46 tasks).

## Data

- `socius-*.csv`, `unsloth-*.csv` — flat per-task NLL CSVs (bf16) for every fine-tuned
  adapter, base model, and control on Psych-101-test (46 tasks)
- `psych101_aggr.csv` — wide aggregate table across all models and tasks (read by the figures)
- 4-bit results live in `../4bit/`

## Scripts

| Script | Purpose |
|---|---|
| `eval_model.py` | Evaluate a model on Psych-101-test (supports LoRA/merged, Llama/Qwen, unsloth/transformers backends) |
| `figures/generate_scaling_plots.py` | bf16 scaling-law figure |
| `figures/generate_ablation_plots.py` | LoRA-rank + dataset-size ablation figures (per family) |
| `figures/viz_r8_and_ranks.py` | r=8 and rank-sweep scaling figures (all families) |
| `figures/cognitive_baseline_comparison.py` | Per-task comparison vs domain-specific cognitive models |
| `figures/generate_control_barplots.py` | Non-cognitive control-model bar plots |
| `tables/generate_heatmap_tables.py` | LaTeX heatmap tables of per-task NLL |
| `martingale_test.py` | Martingale property tests: order permutation and self-imputation drift |
| `plot_martingale.py` | Plot and summarize martingale test results across models |

## Cognitive-matched figures

The dotted "domain-specific cognitive models" line (NLL = 0.6851) is a mean over
only the **38 of 46** Psych-101 tasks for which Binz reports a cognitive model,
whereas the model points are means over all 46. The 8 tasks without a cognitive
model are disproportionately hard (pooled model NLL ≈ 1.36 vs 0.52 on the 38), so
the two are not measuring the same thing.

Every figure script that draws that line accepts `--cognitive-matched`, which
restricts **all** model/baseline means to the same 38 tasks and writes
`*_cognitive_matched` PNG/PDF (and, for the ablation script, matched summary
CSVs) beside the originals. Run order matters — the ablation summaries feed the
rank figures:

```bash
python figures/generate_scaling_plots.py --cognitive-matched
for f in llama qwen smollm olmo; do
  python figures/generate_ablation_plots.py --family $f --cognitive-matched
done
python figures/generate_ablation_plots.py --combined --cognitive-matched
python figures/viz_r8_and_ranks.py --cognitive-matched      # reads matched summaries
python figures/generate_control_barplots.py --cognitive-matched
# OOD companion (Psych-201), left in-distribution panel matched to the 38 tasks:
python ../psych201/figures/viz_ood_scaling.py --cognitive-matched
```

In the matched `scaling_r8_and_rank_sweep` the rank-sweep panel and the five
per-rank minis use a **broken y-axis** (double-wave `~` cut): every model point
now sits far below the cognitive line, so the line is parked in a thin top band
and the data region is expanded to keep the size/rank slopes legible. The
non-matched figure is unaffected (the cognitive line runs through the data there,
so no break is drawn).

On the matched subset every fine-tuned model beats the cognitive baseline with a
clear margin (see also `cognitive_baseline_comparison.py`, which does the paired
per-task test). Without the flag the scripts reproduce the original 46-task
figures unchanged.

### Companion data (matched CSVs)

`figures/build_cognitive_matched_data.py` writes the underlying numbers on the
38-task subset, in the same shapes as the committed 46-task exports, so slopes,
crossing thresholds, and size/rank substitution points can be computed rather
than read off the figures (the effects of interest are 0.005-0.02 nats):

- `figures/psych101_rank_datasize_figure_data_cognitive_matched.csv` — rank sweep
  + data-fraction + baseline + Centaur-70B/cognitive reference rows.
- `psych201/figures/r16_psych101_vs_psych201_figure_data_cognitive_matched.csv` —
  r=16 finetuned + base + Centaur-70B + cognitive rows. Psych-101 rows are
  recomputed over the 38 tasks; **Psych-201 (OOD) rows are copied unchanged**
  (no cognitive baseline exists there). means at 1e-6 nat precision.

The script recomputes each row's 46-task value and aborts unless it matches the
committed file, so a successful run proves the matched numbers share that exact
pipeline. It reads the `*_ablation_summary[_cognitive_matched].csv` files, so run
the ablation figures (with and without `--cognitive-matched`) for all four
families first.

## Outputs

- `figures/` — Generated plots (PNG + PDF); `*_cognitive_matched` variants use the 38-task subset
- `tables/` — Generated LaTeX tables
