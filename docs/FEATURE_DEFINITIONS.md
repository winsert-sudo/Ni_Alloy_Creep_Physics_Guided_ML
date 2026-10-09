# Frozen P3-replace feature definitions

The numeric feature order is:

`Ni_reconstructed_wt`, `Cr`, `Mo`, `Fe`, `Co`, `W`, `Nb`, `Ti`, `Al`, `C`, `B`, `phys_invT_Kinv`, `phys_stress_over_T_MPa_per_K`, `phys_logstress_over_T_Kinv`, `phys_zT_zStress`.

The categorical inputs are `alloy_family`, `base_alloy`, `heat_treatment_id`.

Let T be temperature in K and sigma stress in MPa. The existing `log_stress_MPa` field is log10(sigma).

| Feature | Exact definition | Unit |
|---|---|---|
| `phys_invT_Kinv` | 1 / T | K^-1 |
| `phys_stress_over_T_MPa_per_K` | sigma / T | MPa/K |
| `phys_logstress_over_T_Kinv` | log10(sigma) / T | K^-1 |
| `phys_zT_zStress` | ((T - mean_train(T)) / sd_train(T)) * ((sigma - mean_train(sigma)) / sd_train(sigma)) | Dimensionless |

The interaction uses training-fold arithmetic means and population standard deviations (`ddof=0`); a zero training standard deviation is replaced by 1. The same training statistics transform validation inputs. These operations were verified against the frozen implementation's `fold_coordinates` definition; no fitted statistics or feature rows are distributed.

P3-replace removes `temperature_K`, `stress_MPa`, `log_temperature_K` and `log_stress_MPa` from the predictor list and uses the four nonredundant physics coordinates instead. It retains the eleven chemistry fields and three categories. Neither target-containing Larson–Miller quantities nor fitted activation energies nor observed creep-rate/elongation responses enter the predictors. Mechanism and tier descriptors belong to the separate mechanism-aware transfer setting and are not additional headline P3-replace categories.

Numeric inputs use training-fold median imputation. Categories use training-fold most-frequent imputation and dense one-hot encoding with unknown categories ignored. XGBoost does not use the optional numeric StandardScaler used for Ridge. P3 standardization above is separate and remains fold-local.

Implementation caveat: the frozen chemistry builder populates `Ni_reconstructed_wt` only when original Ni is missing and its source qualifier explicitly says balance. Its value is 100 minus the sum of finite explicitly numeric source alloying additions. Reported Ni remains in `Ni`; the builder leaves `Ni_reconstructed_wt` missing for measured-Ni records. The headline predictor list uses `Ni_reconstructed_wt` alone, without a measured-Ni fallback. This documents the actual frozen behavior and corrects a loose description in the older dictionary; no algorithm or original data has been changed. Missing W/Nb and other elements are not filled with zero or nominal alloy values.

The definitions were checked by read-only inspection of frozen training and chemistry scripts. The original scripts remain CONDITIONAL; reviewed operationally sanitized copies are now staged under src/preprocessing and src/modeling, with whole-module scientific equivalence evidence. Released formulas are methodological documentation, not permission to release invertible feature values.
