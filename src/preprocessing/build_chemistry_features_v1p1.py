#!/usr/bin/env python3
"""Build the provenance-preserving chemistry feature layer v1p1 from v1.

The approved v1 chemistry is never altered.  This script appends a nominal Ni
balance residual, its provenance, missingness indicators, and an effective
chemistry-completeness score.  RouteB_database_v3 is input-only.
"""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = _CANDIDATE_ROOT
BASE_PATH = ROOT / "03_MODELING/data/baseline_creep_dataset_v1.csv"
BASE_METADATA_PATH = ROOT / "03_MODELING/data/baseline_creep_dataset_v1_metadata.json"
COMPOSITION_PATH = ROOT / "01_DATA/cleaned/RouteB_database_v3/composition_master.csv"
OUT_PATH = ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv"
OUT_METADATA_PATH = ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1_metadata.json"
REPORT_DIR = ROOT / "03_MODELING/reports"
VALIDATION_PATH = REPORT_DIR / "chemistry_feature_v1p1_validation.md"
COMPARISON_PATH = REPORT_DIR / "chemistry_feature_v1_vs_v1p1.csv"

CHEMISTRY = ["Ni", "Cr", "Mo", "Fe", "Co", "W", "Ti", "Al", "Nb", "C", "B"]
INDICATOR_ELEMENTS = ["Ni", "Cr", "Mo", "Fe", "W", "Nb", "Ti", "Al"]
FORBIDDEN_LEAKAGE = (
    "minimum_creep_rate", "steady_state_creep_rate", "total_elongation",
    "rupture_elongation", "reduction_area",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.fillna("").to_markdown(index=False)


def numeric_sum_for_balance(payload: str) -> float:
    """Sum only finite, explicitly numeric source composition values.

    ``element_numeric_columns`` is the normalized source field: it excludes
    remarks and threshold/trace qualifiers.  The Ni balance token itself is
    absent from this mapping, so it cannot be counted.  A combined, explicitly
    measured field such as ``Nb+Ta`` is included once as an alloying addition;
    it is not mapped into the separate Nb model feature.
    """
    values = json.loads(payload)
    total = 0.0
    for value in values.values():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            total += numeric
    return total


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_PATH)
    # Keep an exact text copy for output so appending features cannot change the
    # decimal spelling (or empty-string convention) of any immutable v1 field.
    base_text = pd.read_csv(BASE_PATH, dtype=str, keep_default_na=False)
    compositions = pd.read_csv(COMPOSITION_PATH, dtype=str, keep_default_na=False)
    base_metadata = json.loads(BASE_METADATA_PATH.read_text(encoding="utf-8"))
    expected_composition_hash = base_metadata["input_sha256"]["composition_master.csv"]
    if sha256(COMPOSITION_PATH) != expected_composition_hash:
        raise ValueError("RouteB v3 composition_master differs from the immutable v1 build input.")

    composition_lookup = compositions.set_index("material_state_id")
    if not base["material_state_id"].isin(composition_lookup.index).all():
        raise ValueError("Every baseline record must resolve to composition_master.")

    derived = base.copy()
    reconstructed = pd.Series(np.nan, index=derived.index, dtype=float)
    ni_source = pd.Series("missing", index=derived.index, dtype="object")
    state_residuals: dict[str, float] = {}

    # Derive once per material state from the source-preserved composition data,
    # then apply the state value to all corresponding creep observations.
    for state_id in derived["material_state_id"].unique():
        source = composition_lookup.loc[state_id]
        qualifiers = json.loads(source["element_qualifier_columns"])
        state_mask = derived["material_state_id"].eq(state_id)
        ni_missing = derived.loc[state_mask, "Ni"].isna()
        source_ni_qualifier = qualifiers.get("Ni", "missing")

        if ni_missing.any() and source_ni_qualifier == "balance":
            value = 100.0 - numeric_sum_for_balance(source["element_numeric_columns"])
            if not 0 < value < 100:
                raise ValueError(f"Ni balance residual for {state_id} is outside (0, 100): {value}")
            reconstructed.loc[state_mask & derived["Ni"].isna()] = value
            ni_source.loc[state_mask & derived["Ni"].isna()] = "balance_reconstructed"
            state_residuals[state_id] = value
        elif ni_missing.any():
            # A missing source field remains missing; no nominal/alloy lookup is
            # substituted, and original chemistry remains untouched.
            ni_source.loc[state_mask & derived["Ni"].isna()] = "missing"

        if (~ni_missing).any():
            if source_ni_qualifier != "numeric":
                raise ValueError(f"Numeric Ni in v1 lacks a numeric source qualifier for {state_id}.")
            ni_source.loc[state_mask & derived["Ni"].notna()] = "measured"

    derived["Ni_reconstructed_wt"] = reconstructed
    derived["Ni_source"] = ni_source
    # Indicators intentionally reflect the *original* v1 fields.  Thus a
    # balance-derived Ni remains flagged as originally missing, while Ni_source
    # records that an effective value exists for the completeness score.
    for element in INDICATOR_ELEMENTS:
        derived[f"{element}_missing"] = derived[element].isna().astype("int8")

    effective_available = pd.DataFrame({element: derived[element].notna() for element in CHEMISTRY})
    effective_available["Ni"] = derived["Ni"].notna() | derived["Ni_reconstructed_wt"].notna()
    derived["chemistry_completeness_score"] = effective_available.sum(axis=1).div(len(CHEMISTRY))

    shared_columns = base.columns.tolist()
    if not derived[shared_columns].equals(base):
        raise ValueError("v1 fields changed while constructing v1p1.")
    for column in FORBIDDEN_LEAKAGE:
        if any(column in name.lower() for name in derived.columns):
            raise ValueError(f"Forbidden leakage field introduced: {column}")
    if len(derived) != len(base):
        raise ValueError("Sample count changed.")
    if not derived["creep_id"].equals(base["creep_id"]):
        raise ValueError("Creep record identities changed.")
    if not derived["source_id"].equals(base["source_id"]):
        raise ValueError("source_id changed.")
    if not derived["target_log10_rupture_life_h"].equals(base["target_log10_rupture_life_h"]):
        raise ValueError("Target changed.")

    output_text = base_text.copy()
    output_text["Ni_reconstructed_wt"] = derived["Ni_reconstructed_wt"].map(
        lambda value: "NaN" if pd.isna(value) else format(value, ".15g")
    )
    output_text["Ni_source"] = derived["Ni_source"]
    for element in INDICATOR_ELEMENTS:
        output_text[f"{element}_missing"] = derived[f"{element}_missing"].astype(str)
    output_text["chemistry_completeness_score"] = derived["chemistry_completeness_score"].map(
        lambda value: format(value, ".15g")
    )
    output_text.to_csv(OUT_PATH, index=False)
    written = pd.read_csv(OUT_PATH)
    if not written[shared_columns].equals(base):
        raise ValueError("Writing v1p1 changed a serialized v1 field.")

    comparison_rows = []
    for feature in CHEMISTRY:
        comparison_rows.append({
            "feature": feature,
            "missing_v1": int(base[feature].isna().sum()),
            "missing_v1p1": int(derived[feature].isna().sum()),
            "changed_values": int((base[feature].notna() & derived[feature].notna() & base[feature].ne(derived[feature])).sum()),
            "comparison_note": "preserved original chemistry column",
        })
    for feature in ["Ni_reconstructed_wt", "Ni_source", *[f"{item}_missing" for item in INDICATOR_ELEMENTS], "chemistry_completeness_score"]:
        comparison_rows.append({
            "feature": feature,
            "missing_v1": "not_applicable_new_feature",
            "missing_v1p1": int(derived[feature].isna().sum()),
            "changed_values": "new_feature",
            "comparison_note": "chemistry-derived v1p1 feature",
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)

    source_counts = derived["Ni_source"].value_counts().reindex(
        ["measured", "balance_reconstructed", "missing"], fill_value=0
    )
    reconstruction_values = derived.loc[derived["Ni_source"].eq("balance_reconstructed"), "Ni_reconstructed_wt"]
    if reconstruction_values.empty or not reconstruction_values.between(0, 100, inclusive="neither").all():
        raise ValueError("Ni reconstruction validation failed.")

    metadata = {
        "dataset_name": "baseline_creep_dataset_v1p1",
        "base_dataset": "baseline_creep_dataset_v1",
        "base_dataset_sha256": sha256(BASE_PATH),
        "database_version": "RouteB_database_v3",
        "composition_master_sha256": sha256(COMPOSITION_PATH),
        "Ni_balance_handling": "reconstructed",
        "Ni_reconstructed_wt_formula": "100 - sum(finite explicitly numeric source alloying elements)",
        "original_dataset_preserved": True,
        "n_samples": len(derived),
        "n_material_states": int(derived["material_state_id"].nunique()),
        "n_sources": int(derived["source_id"].nunique()),
        "original_chemistry_columns": CHEMISTRY,
        "new_feature_columns": ["Ni_reconstructed_wt", "Ni_source", *[f"{item}_missing" for item in INDICATOR_ELEMENTS], "chemistry_completeness_score"],
        "chemistry_completeness_expected_elements": CHEMISTRY,
        "chemistry_completeness_rule": "reported original element, except Ni may be source-qualified balance reconstruction",
        "Ni_source_counts": {key: int(value) for key, value in source_counts.items()},
        "leakage_checked": True,
        "forbidden_leakage_terms": list(FORBIDDEN_LEAKAGE),
        "preserved_from_v1": shared_columns,
        "construction_script": "03_MODELING/scripts/build_chemistry_features_v1p1.py",
        "base_dataset_metadata": {key: base_metadata[key] for key in ("target", "target_formula", "split_group")},
    }
    OUT_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    state_summary = pd.DataFrame([
        {"material_state_id": state, "Ni_reconstructed_wt": value}
        for state, value in sorted(state_residuals.items())
    ])
    validation = f"""# Chemistry Feature Layer v1p1 Validation

## Dataset size

- Samples: **{len(derived)}** (v1: {len(base)})
- Material states: **{derived['material_state_id'].nunique()}**
- Sources: **{derived['source_id'].nunique()}**

## Ni reconstruction

| Ni source | sample count |
| --- | ---: |
| measured | {source_counts['measured']} |
| balance_reconstructed | {source_counts['balance_reconstructed']} |
| missing | {source_counts['missing']} |

`Ni_reconstructed_wt` is populated only when the original Ni field is missing and the linked source qualifier is `balance`. The original `Ni` column remains unchanged; `Ni_missing=1` therefore preserves that original missingness.

## Reconstruction statistics

| statistic | Ni_reconstructed_wt |
| --- | ---: |
| minimum | {reconstruction_values.min():.6f} |
| maximum | {reconstruction_values.max():.6f} |
| mean | {reconstruction_values.mean():.6f} |
| all values satisfy 0 < Ni_reconstructed_wt < 100 | PASS |

### Reconstructed material states

{markdown_table(state_summary)}

## Chemistry completeness

`chemistry_completeness_score` is the fraction of the 11 expected chemistry elements that are source-reported, except that a source-qualified Ni balance residual counts as available. Its range is {derived['chemistry_completeness_score'].min():.6f} to {derived['chemistry_completeness_score'].max():.6f}.

## Integrity and scope checks

- Sample count unchanged: **PASS**
- `creep_id` unchanged and in the same order: **PASS**
- `target_log10_rupture_life_h` unchanged: **PASS**
- `source_id` unchanged: **PASS**
- Every original v1 column unchanged: **PASS**
- No forbidden leakage feature introduced: **PASS**
- RouteB v3 is input-only; its `composition_master.csv` SHA-256 is `{sha256(COMPOSITION_PATH)}`: **PASS**

## Result

**PASS — ready for TASK 04.3 baseline modeling.** Use v1 as the immutable comparator and v1p1 only with the documented `Ni_source` and missingness features. W/Nb are intentionally still missing where unreported; neither was zero-filled or inferred.
"""
    VALIDATION_PATH.write_text(validation, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_METADATA_PATH}")
    print(f"Wrote {VALIDATION_PATH}")
    print(f"Wrote {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
