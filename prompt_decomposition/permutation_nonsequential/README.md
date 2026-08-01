# Order Permutation Test (Non-Sequential)

Tests whether cognitive foundation models produce predictions invariant to the ordering of exchangeable context trials. Under exchangeability, shuffling context order should not change the predictive distribution for held-out target trials.

## Pipeline

```bash
# 1. Run permutation test (GPU required)
python order_permutation_test.py \
    --model socius/Qwentaur-8B-LoRA \
    --experiments hebart2023things ruggeri2022globalizability \
    --n-permutations 50 --output-dir results/Qwentaur-8B

# 2. Compute aggregate statistics from raw results
python analyse_permutation.py

# 3. Generate figures (reads CSVs produced by step 2)
python generate_permutation_plots.py
```

## Structure

| Path | Description |
|------|-------------|
| `order_permutation_test.py` | Main test script — generates per-model CSV results |
| `analyse_permutation.py` | Aggregation — computes per-participant variance and summary stats |
| `generate_permutation_plots.py` | Figure generator — combined bar, ECDF, violin plots |
| `results/` | Per-model subdirectories with raw `test1_order_*.csv` files |
| `figures/` | Output plots (PNG+PDF) and aggregate CSVs |

## Metric

For each (participant, target trial), the variance of each response-token probability across 50 context-order permutations is computed, averaged across tokens, then across targets per participant. Lower variance = stronger order invariance.

## Experiments

- **THINGS odd-one-out** (Hebart et al., 2023) — 3 response tokens, 768 participants
- **Intertemporal choice** (Ruggeri et al., 2022) — 2 response tokens, 1295 participants
