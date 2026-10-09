# Method reproduction and verification guide

The frozen headline workflow is Tier1 + source-aware 5-fold validation + P3-replace XGBoost + pooled OOF evaluation. Target: y = log10(t_r / h).

| Frozen endpoint | Value |
|---|---|
| n | 220 |
| R2 | 0.7497006307687231 |
| MAE | 0.3592029929766931 log10(h) |
| RMSE | 0.4930923099654113 log10(h) |

These are validation targets from existing frozen summaries. They cannot be regenerated from this candidate alone. The prior internal reproduction matched 220/220 predictions with lawful local inputs; this task performs no model rerun.

The fixed XGBoost settings are n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, objective=reg:squarederror. See [feature definitions](FEATURE_DEFINITIONS.md) and [validation protocol](VALIDATION_PROTOCOL.md) for exact feature and fold-local processing contracts.

| Reproduction level | Current status | Practical scope |
|---|---|---|
| Level 1: method/workflow reproduction | CONDITIONAL | Method, schema and environment documented; 14 scientifically equivalent reviewed copies are staged with operational path/dependency cleanup; author license/release decision, complete source citations, lawful compatible inputs and clean numerical execution remain unresolved. |
| Level 2: verification against summaries | PARTIAL | Inspect frozen aggregate results and package checksums; cannot recompute OOF metrics or source assignments without withheld rows. |
| Level 3: exact numerical reproduction from public files alone | NO | The exact model-ready rows, immutable linked folds and two ancillary audit inputs are excluded. |

For environment preparation only, from the candidate root:

```bash
conda env create -f environment/environment.yml
conda activate paper1-creep-env
python --version
```

Python 3.12.14 and the exact pinned dependencies describe the reconstructed publication/reproduction environment, not a recovered historical training environment. Installing it supplies dependencies for the staged code but does not provide the omitted lawful inputs. This task has not installed or run it.

For Level 2, inspect `results_summary/METRIC_REFERENCE_TABLE.csv` for the full-precision headline values and `FINAL_MODEL_COMPARISON_TABLE.csv` for its documented decimal rounding. Verify candidate integrity with the relative-path checksum command documented in [manifest conventions](../manifests/MANIFEST_CONVENTIONS.md). Equality among summaries verifies packaging consistency rather than a new numerical reproduction.

For future Level 1 execution, first resolve the [code limitations](../REPRODUCIBILITY.md), lawful [reconstruction dependencies](../REPRODUCIBILITY.md) and exact-file release gates. Newly reconstructed inputs may yield different metrics. Reviewed code entries are documented in [code execution guide](CODE_EXECUTION_GUIDE.md). They require external private inputs and do not form a public-only exact-reproduction pipeline. No restricted-data workflow was run during remediation.
