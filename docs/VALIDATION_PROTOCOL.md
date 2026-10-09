# Frozen validation protocol

The primary evaluation is five-fold source-aware grouped validation. Every record from a held-out `source_id` group is excluded from training. The frozen split builder uses `GroupKFold(n_splits=5)` with the entire eligible 914-row population and `groups=source_id`, without shuffle or tier stratification. The immutable assignments are then merged by unique record key and reused; they are not regenerated on Tier1 alone.

For the headline run, Tier1 records from the other four folds train the model, and the held-out Tier1 records receive predictions. Training–validation source intersections must be empty for every fold. Experiment-family identifiers are retained as provenance but are not the outer grouping unit. The source-level guarantee does not establish experiment-family independence, and the primary cohort's source/state confounding remains.

Imputation, encoding, model fitting and fold-dependent standardized interaction statistics use only outer-training data. Validation targets never enter preprocessing or physics features. For transfer comparisons, eligible Tier2 training records may be added only outside the held-out source groups; primary scores still use the same Tier1 endpoint.

Each Tier1 record receives one OOF prediction. Pooled metrics use all 220 held-out predictions together: MAE = mean(abs(y - prediction)); RMSE = sqrt(mean((y - prediction)^2)); R2 = 1 - sum((y - prediction)^2) / sum((y - mean(y))^2). Fold-mean and fold-SD metrics in released tables are separate summaries and are not the pooled headline estimand.

The builder's random reference split uses seed 42 but is not the primary evaluation. The unshuffled grouped split does not use that random-reference seed. XGBoost uses random_state=42. P3-replace was selected once from the frozen Tier1 XGBoost pooled OOF feature ablation and reused without retuning. This task neither runs nor revises that process.

Without the excluded rows, linked assignments and OOF outputs, readers can inspect this protocol and aggregate claims but cannot independently recompute per-record leakage checks or pooled scores from candidate files alone.
