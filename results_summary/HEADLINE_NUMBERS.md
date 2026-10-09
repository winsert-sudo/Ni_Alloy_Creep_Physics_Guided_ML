# Frozen Headline Numbers

All predictive headline metrics use the 220-observation Tier1 five-fold source-aware pooled OOF endpoint unless a different cohort/estimand is stated. MAE and RMSE are in log10(h).

## Best ML model

| Model | n | R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| P3-replace XGBoost | 220 | **0.7497** | **0.3592** | **0.4931** |

## Classical physics

| Model | n | R² | MAE | RMSE | Interpretation |
|---|---:|---:|---:|---:|---|
| A0 | 220 | 0.4280 | 0.5974 | 0.7454 | Global physics reference |
| A1 | 220 | 0.4802 | 0.5835 | 0.7106 | Strongest distinct global classical form |
| L1 | 220 | 0.4802 | 0.5835 | 0.7106 | Prediction-equivalent A1 reparameterization under the active constraint |

## State-specific physics

These entries are deliberately not a common ranking:

| Result | Cohort/estimand | Frozen value |
|---|---|---:|
| Global A1 | 220-row source-aware pooled OOF | R² = 0.4802 |
| State-specific A1 descriptive | Median across 10 eligible states; in-sample | R² = 0.9148 |
| State-specific A1 grouped OOF | Median across 9 calculable states; leave-one-condition-group-out | R² = 0.8428 |
| Oracle state-specific A1 | 186 observations in 10 eligible states; parameters fit to each state's outcomes | R² = 0.9301; MAE = 0.2009; RMSE = 0.2588 |

## Transfer

All rows below use XGBoost, the same 220 Tier1 validation observations, and pooled OOF metrics.

| Training/representation | R² | MAE | Interpretation |
|---|---:|---:|---|
| Tier1-only P0 | 0.7254 | 0.3830 | Reference |
| Naive Tier1+Tier2 P0 | 0.6893 | 0.4202 | Negative transfer |
| Mechanism-aware Tier1+Tier2 P0 | 0.7008 | 0.4165 | Partial recovery, still below reference |
| Tier1-only P3-replace | 0.7497 | 0.3592 | Physics-guided representation gain |

Supplementary fixed-block result: mechanism-aware Tier1+Tier2 with P3-replace gives pooled OOF R² 0.7279, MAE 0.3894, RMSE 0.5142. It is not one of the four primary transfer-table rows.

## Parameter learnability

The reconstruction cohort contains 186 observations from the 10 locally eligible states.

| Reconstruction | n | R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| Global A1 | 186 | 0.4341 | 0.6156 | 0.7367 |
| Predicted-state A1 | 186 | −447.2817 | 7.0520 | 20.7328 |
| Oracle state-specific A1 | 186 | 0.9301 | 0.2009 | 0.2588 |

The permitted conclusion is that available descriptors do not currently recover the state-associated effective parameters. The result is limited by identifiability, ten-state sample size, and source–state confounding.
