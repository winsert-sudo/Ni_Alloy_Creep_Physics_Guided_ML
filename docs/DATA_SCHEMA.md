# Data schema

## Frozen modeling table

The referenced table `baseline_creep_dataset_v1p1.csv` has 914 rows and 36 columns. It is not included in this public source-code repository because numerical redistribution has not been cleared.

| Field | Type | Meaning / unit | Lineage |
|---|---|---|---|
| `creep_id` | string | Stable creep-record identifier | Observed-record key |
| `source_id` | string | Source grouping identifier | Provenance metadata |
| `experiment_family_id` | string | Experiment-family identifier | Provenance metadata |
| `material_state_id` | string | Recorded material-state grouping | Curated metadata; not complete physical state |
| `alloy_family` | string | Alloy-family label | Curated metadata |
| `base_alloy` | string | Base-alloy label | Curated metadata |
| `transfer_tier` | category | Tier 1, 2, or 3 analysis domain | Processed classification |
| `mechanism_class` | category | Approved coarse mechanism class | Processed classification; not causal identification |
| `heat_treatment_id` | string | Heat-treatment record identifier | Curated metadata |
| `temperature_K` | float | Test temperature, K | Observed/normalized descriptor |
| `stress_MPa` | float | Applied stress, MPa | Observed/normalized descriptor |
| `log_temperature_K` | float | log10 temperature in K | Generated descriptor |
| `log_stress_MPa` | float | log10 stress in MPa | Generated descriptor |
| `Ni`, `Cr`, `Mo`, `Fe`, `Co`, `W`, `Ti`, `Al`, `Nb`, `C`, `B` | float | Reported elemental composition, wt.% where recorded | Observed descriptors; missing remains missing |
| `target_log10_rupture_life_h` | float | log10 rupture life in hours | Target derived from reported rupture life |
| `Ni_reconstructed_wt` | float | Source-qualified Ni balance residual only; see implementation caveat below | Generated descriptor |
| `Ni_source` | category | Measured or balance-reconstructed provenance | Generated provenance flag |
| `Ni_missing`, `Cr_missing`, `Mo_missing`, `Fe_missing`, `W_missing`, `Nb_missing`, `Ti_missing`, `Al_missing` | integer | 1 if the original element value was missing, otherwise 0 | Generated missingness flags |
| `chemistry_completeness_score` | float | Fraction of expected chemistry fields considered reported | Generated descriptor |

## Headline P3-replace feature block

In addition to chemistry and categorical metadata, the headline model generates target-free stress–temperature coordinates inside the frozen workflow:

| Feature | Definition |
|---|---|
| `phys_invT_Kinv` | `1 / temperature_K` |
| `phys_stress_over_T_MPa_per_K` | `stress_MPa / temperature_K` |
| `phys_logstress_over_T_Kinv` | `log_stress_MPa / temperature_K` |
| `phys_zT_zStress` | Fold-training-standardized temperature × stress interaction |

These descriptors do not recover a hidden material state, do not provide complete microstructure information, and do not identify a causal mechanism.


## Input contract and implementation caveat

This describes fields, not experimental rows. Provenance fields and target are not predictor columns. `transfer_tier` and `mechanism_class` select domains or serve the separate transfer setting; the primary P3-replace categories are only those listed in [feature definitions](../docs/FEATURE_DEFINITIONS.md).

`Ni_reconstructed_wt` is populated only for source-qualified balance reconstruction when original Ni is missing. Measured Ni remains in `Ni` and is not copied into `Ni_reconstructed_wt`; original missingness flags remain unchanged. This differs from the older dictionary's general description and follows the frozen implementation. The completeness score counts the eleven expected chemistry elements, with measured Ni or a permitted balance residual counting as available for the Ni slot.

The fold assignment schema is `creep_id`, `source_id`, `material_state_id`, `fold_id`. Field names are documented; actual keys and assignments are excluded. The full headline script also depends on two excluded audit inputs for region diagnostics, including a record-linked condition-extrapolation flag. Their exact rows cannot be reconstructed from this schema alone.

The model builder converts temperature from Celsius to Kelvin by adding 273.15 and computes base-10 condition logs and log10 rupture life in hours. It rejects missing/nonpositive stress or source Celsius temperature and excludes missing/nonpositive rupture-life targets without target imputation. Composition values use the explicit numeric source mapping; qualifier and missingness distinctions must be preserved.
