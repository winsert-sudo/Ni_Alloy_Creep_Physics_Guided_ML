# Frozen Paper 1 Results

## Headline benchmark

| Item | Frozen value |
|---|---|
| Target | `log10(rupture life / h)` |
| Cohort | Tier1 |
| Evaluation | Five-fold `source_id`-grouped pooled OOF |
| n | 220 |
| Model | P3-replace XGBoost |
| R² | 0.7497 |
| MAE | 0.3592 log10(h) |
| RMSE | 0.4931 log10(h) |

TASK 05.7-E reproduced the stored full-precision values exactly: R² 0.7497006307687231, MAE 0.3592029929766931, RMSE 0.4930923099654113. All 220 OOF predictions were identical.

## Interpretation boundary

These are frozen predictive results under the defined cohort and validation design. They do not establish complete material-state recovery, causal mechanisms, universal transferability, or direct Hastelloy-N validation. This package adds no scientific result or claim.

