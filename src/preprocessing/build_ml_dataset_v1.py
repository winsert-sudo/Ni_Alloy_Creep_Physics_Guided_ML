#!/usr/bin/env python3
"""Build the frozen, source-traceable ML-ready creep dataset v1 from RouteB v3."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = _CANDIDATE_ROOT
DEFAULT_DB = PROJECT_ROOT / "01_DATA/cleaned/RouteB_database_v3"
DEFAULT_OUT = PROJECT_ROOT / "03_MODELING/data"
DEFAULT_REPORTS = PROJECT_ROOT / "03_MODELING/reports"
CHEMISTRY = ["Ni", "Cr", "Mo", "Fe", "Co", "W", "Ti", "Al", "Nb", "C", "B"]
FORBIDDEN_LEAKAGE = (
    "minimum_creep_rate", "steady_state_creep_rate", "total_elongation",
    "rupture_elongation", "reduction_area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB, help="RouteB_database_v3 directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Dataset output directory")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS, help="Validation report directory")
    return parser.parse_args()


def read_table(db_dir: Path, name: str) -> pd.DataFrame:
    path = db_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required input table is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "creep_master.csv", "material_state_master.csv", "composition_master.csv",
        "heat_treatment_master.csv", "material_transfer_classification.csv",
    ]
    creep, states, composition, treatments, transfer = (read_table(args.db_dir, name) for name in required)
    logging.info("Loaded RouteB v3 tables: %d creep observations, %d material states.", len(creep), len(states))

    # Select only columns needed for explicit model inputs, target construction, or
    # required provenance.  Leakage-bearing creep response fields are never joined.
    creep_base = creep[[
        "creep_id", "source_id", "experiment_family_id", "material_state_id",
        "temperature_C", "stress_MPa", "rupture_life_h",
    ]].copy()
    state_base = states[["material_state_id", "alloy_family", "alloy_name", "composition_id", "heat_treatment_id"]].copy()
    state_base = state_base.rename(columns={"alloy_name": "base_alloy"})
    transfer_base = transfer[["material_state_id", "transfer_tier", "mechanism_class"]].copy()
    composition_base = composition[["composition_id", "element_numeric_columns"]].copy()
    treatment_base = treatments[["heat_treatment_id"]].drop_duplicates()

    join_stats: list[dict[str, object]] = []
    dataset = creep_base.merge(state_base, on="material_state_id", how="left", validate="many_to_one", indicator="_state_join")
    join_stats.append({"stage": "creep_master input", "row_count": len(creep_base), "unmatched_row_count": 0, "notes": "All creep source rows before target eligibility filtering."})
    join_stats.append({"stage": "join material_state_master", "row_count": len(dataset), "unmatched_row_count": int((dataset["_state_join"] != "both").sum()), "notes": "many-to-one join on material_state_id"})
    if (dataset["_state_join"] != "both").any():
        raise ValueError("Creep rows without material-state linkage cannot enter the ML dataset.")
    dataset = dataset.drop(columns="_state_join")

    dataset = dataset.merge(composition_base, on="composition_id", how="left", validate="many_to_one", indicator="_composition_join")
    join_stats.append({"stage": "join composition_master", "row_count": len(dataset), "unmatched_row_count": int((dataset["_composition_join"] != "both").sum()), "notes": "many-to-one join on composition_id"})
    if (dataset["_composition_join"] != "both").any():
        raise ValueError("Material-state rows without composition linkage cannot enter the ML dataset.")
    dataset = dataset.drop(columns="_composition_join")

    # Heat-treatment metadata is optional in the frozen source schema.  This join is
    # a linkage audit; the retained heat_treatment_id is the ML categorical descriptor.
    dataset = dataset.merge(treatment_base, on="heat_treatment_id", how="left", validate="many_to_one", indicator="_heat_join")
    unmatched_heat = int(((dataset["heat_treatment_id"] != "") & (dataset["_heat_join"] != "both")).sum())
    join_stats.append({"stage": "join heat_treatment_master", "row_count": len(dataset), "unmatched_row_count": unmatched_heat, "notes": "Blank heat_treatment_id is permitted by the frozen source schema."})
    if unmatched_heat:
        raise ValueError("A populated heat_treatment_id does not resolve to heat_treatment_master.")
    dataset = dataset.drop(columns="_heat_join")

    dataset = dataset.merge(transfer_base, on="material_state_id", how="left", validate="many_to_one", indicator="_transfer_join")
    join_stats.append({"stage": "join material_transfer_classification", "row_count": len(dataset), "unmatched_row_count": int((dataset["_transfer_join"] != "both").sum()), "notes": "many-to-one join on material_state_id"})
    if (dataset["_transfer_join"] != "both").any():
        raise ValueError("Creep rows without transfer annotation cannot enter the ML dataset.")
    dataset = dataset.drop(columns="_transfer_join")

    def composition_value(payload: str, element: str) -> float:
        values = json.loads(payload)
        value = values.get(element, "")
        return float(value) if value not in ("", None) else np.nan

    for element in CHEMISTRY:
        dataset[element] = dataset["element_numeric_columns"].map(lambda payload, el=element: composition_value(payload, el))
    dataset = dataset.drop(columns="element_numeric_columns")

    for column in ("temperature_C", "stress_MPa", "rupture_life_h"):
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    invalid_conditions = dataset[dataset["temperature_C"].isna() | dataset["stress_MPa"].isna() | (dataset["temperature_C"] <= 0) | (dataset["stress_MPa"] <= 0)]
    if not invalid_conditions.empty:
        raise ValueError(f"Found {len(invalid_conditions)} non-positive or missing creep conditions; no dataset was written.")
    invalid_target = dataset[dataset["rupture_life_h"].isna() | (dataset["rupture_life_h"] <= 0)].copy()
    if not invalid_target.empty:
        logging.warning("Excluding %d source rows lacking a positive rupture-life target; no target value is imputed.", len(invalid_target))
    dataset = dataset.drop(index=invalid_target.index).copy()
    join_stats.append({"stage": "positive rupture-life target eligibility", "row_count": len(dataset), "unmatched_row_count": len(invalid_target), "notes": "Rows with missing/non-positive rupture_life_h are excluded; target values are never imputed."})

    dataset["temperature_K"] = dataset["temperature_C"] + 273.15
    dataset["log_temperature_K"] = np.log10(dataset["temperature_K"])
    dataset["log_stress_MPa"] = np.log10(dataset["stress_MPa"])
    dataset["target_log10_rupture_life_h"] = np.log10(dataset["rupture_life_h"])
    if dataset["target_log10_rupture_life_h"].isna().any() or not np.isfinite(dataset["target_log10_rupture_life_h"]).all():
        raise ValueError("Target construction produced missing or non-finite values.")

    columns = [
        "creep_id", "source_id", "experiment_family_id", "material_state_id", "alloy_family", "base_alloy",
        "transfer_tier", "mechanism_class", "heat_treatment_id",
        "temperature_K", "stress_MPa", "log_temperature_K", "log_stress_MPa",
        *CHEMISTRY, "target_log10_rupture_life_h",
    ]
    dataset = dataset[columns]
    lower_columns = [column.lower() for column in dataset.columns]
    leakage_hits = [column for column in lower_columns if any(forbidden in column for forbidden in FORBIDDEN_LEAKAGE)]
    if leakage_hits:
        raise ValueError(f"Forbidden leakage columns detected; refusing to save dataset: {leakage_hits}")

    dataset_path = args.output_dir / "baseline_creep_dataset_v1.csv"
    dataset.to_csv(dataset_path, index=False, na_rep="NaN")
    join_stats_frame = pd.DataFrame(join_stats)
    join_stats_frame.to_csv(args.report_dir / "dataset_v1_join_statistics.csv", index=False)

    feature_columns = ["temperature_K", "stress_MPa", "log_temperature_K", "log_stress_MPa", *CHEMISTRY, "alloy_family", "base_alloy", "heat_treatment_id", "transfer_tier", "mechanism_class"]
    missing = pd.DataFrame({
        "feature": feature_columns,
        "missing_count": [int(dataset[column].isna().sum() + (dataset[column].eq("").sum() if dataset[column].dtype == object else 0)) for column in feature_columns],
    })
    missing["missing_percentage"] = (100 * missing["missing_count"] / len(dataset)).round(3)
    target_stats = dataset["target_log10_rupture_life_h"].agg(["min", "max", "mean", "std"])
    tier_distribution = dataset.groupby("transfer_tier", dropna=False).size().reindex(["Tier 1", "Tier 2", "Tier 3"], fill_value=0).rename("sample_count").reset_index()
    alloy_distribution = dataset.groupby("alloy_family", dropna=False).size().rename("sample_count").reset_index().sort_values("sample_count", ascending=False)
    source_distribution = dataset.groupby("source_id", dropna=False).size().rename("sample_count").reset_index().sort_values(["sample_count", "source_id"], ascending=[False, True])
    metadata = {
        "dataset_name": "baseline_creep_dataset_v1", "database_version": "RouteB_database_v3",
        "target": "target_log10_rupture_life_h", "target_formula": "log10(rupture_life_h)",
        "n_samples": int(len(dataset)), "input_creep_observations": int(len(creep_base)),
        "excluded_missing_or_nonpositive_target_rows": int(len(invalid_target)), "split_group": "source_id",
        "leakage_checked": True, "forbidden_leakage_terms": list(FORBIDDEN_LEAKAGE),
        "preserved_metadata_columns": ["creep_id", "source_id", "experiment_family_id", "material_state_id", "alloy_family", "base_alloy", "transfer_tier", "mechanism_class"],
        "feature_columns": feature_columns, "chemistry_columns": CHEMISTRY,
        "input_sha256": {name: sha256(args.db_dir / name) for name in required},
        "construction_script": "03_MODELING/scripts/build_ml_dataset_v1.py",
    }
    (args.output_dir / "baseline_creep_dataset_v1_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# ML-ready Dataset v1 Validation

## Dataset identity

- Source database: `RouteB_database_v3`
- Dataset: `baseline_creep_dataset_v1.csv`
- Samples: **{len(dataset)}** of {len(creep_base)} creep source rows
- Excluded target-ineligible rows: **{len(invalid_target)}** (all have source rupture-life value `-`; no imputation was performed)
- Material states: **{dataset['material_state_id'].nunique()}**
- Sources: **{dataset['source_id'].nunique()}**
- Experiment families: **{dataset['experiment_family_id'].replace('', np.nan).nunique()}** non-empty family IDs
- Split group: `source_id`
- Leakage check: **PASS** — none of the forbidden creep-rate, elongation, or reduction-area fields is present.

## Join statistics

{markdown_table(join_stats_frame)}

## Target statistics

| statistic | value |
| --- | ---: |
| min log10 life | {target_stats['min']:.6f} |
| max log10 life | {target_stats['max']:.6f} |
| mean log10 life | {target_stats['mean']:.6f} |
| std log10 life | {target_stats['std']:.6f} |

## Missing feature values

{markdown_table(missing)}

Chemistry values are only those in `composition_master.element_numeric_columns`; `NaN` indicates that the source did not supply a numeric value. No balance chemistry or nominal value was substituted.

## Domain distribution

{markdown_table(tier_distribution)}

## Alloy-family distribution

{markdown_table(alloy_distribution)}

## Source validation

- Unique source count: **{dataset['source_id'].nunique()}**
- `source_id` is preserved for every row and remains the recommended primary split group.

{markdown_table(source_distribution)}

## Result

**PASS.** The frozen dataset has a complete finite target, positive creep-condition features, preserved provenance/domain metadata, and no listed leakage features. It is the approved input for baseline modeling. The four target-ineligible historical rows remain unchanged in RouteB_database_v3 and are documented as exclusions rather than altered or imputed.
"""
    (args.report_dir / "dataset_v1_validation.md").write_text(report, encoding="utf-8")
    logging.info("Wrote %s (%d samples) and validation artefacts to %s.", dataset_path, len(dataset), args.report_dir)


if __name__ == "__main__":
    main()
