#!/usr/bin/env python3
"""Train the frozen-split Tier 1 creep-life baseline models for """
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


ROOT = _CANDIDATE_ROOT
DATA_PATH = ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv"
SPLIT_PATH = ROOT / "03_MODELING/splits/source_groupkfold_5fold_assignment.csv"
OUT_DIR = ROOT / "03_MODELING/results/baseline_tier1_v1"
TARGET = "target_log10_rupture_life_h"
ID_COLUMNS = ["creep_id", "source_id", "fold_id"]
NUMERIC_A1 = [
    "temperature_K", "stress_MPa", "log_temperature_K", "log_stress_MPa",
    "Ni_reconstructed_wt", "Cr", "Mo", "Fe", "Co", "W", "Nb", "Ti", "Al", "C", "B",
]
CATEGORICAL_A1 = ["alloy_family", "base_alloy", "heat_treatment_id"]
NUMERIC_A2_EXTRA = [
    "Ni_missing", "Cr_missing", "Mo_missing", "Fe_missing", "W_missing", "Nb_missing",
    "Ti_missing", "Al_missing", "chemistry_completeness_score",
]
CATEGORICAL_A2_EXTRA = ["Ni_source"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
    }


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=True)


def model_pipeline(model_name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    if model_name == "Ridge":
        return Pipeline([("features", make_preprocessor(numeric, categorical, True)), ("model", Ridge(alpha=1.0))])
    if model_name == "RandomForest":
        return Pipeline([("features", make_preprocessor(numeric, categorical, False)), ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1,
        ))])
    if model_name == "XGBoost":
        return Pipeline([("features", make_preprocessor(numeric, categorical, False)), ("model", XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, objective="reg:squarederror",
        ))])
    raise ValueError(model_name)


def raw_feature(encoded_name: str, numeric: list[str], categorical: list[str]) -> str:
    if encoded_name.startswith("numeric__"):
        return encoded_name.removeprefix("numeric__")
    remainder = encoded_name.removeprefix("categorical__")
    for column in categorical:
        if remainder == column or remainder.startswith(column + "_"):
            return column
    return remainder


def importance_rows(pipe: Pipeline, model_name: str, experiment: str, fold: int,
                    numeric: list[str], categorical: list[str]) -> list[dict[str, object]]:
    estimator = pipe.named_steps["model"]
    if model_name == "Ridge":
        values = np.abs(estimator.coef_)
    else:
        values = estimator.feature_importances_
    names = pipe.named_steps["features"].get_feature_names_out()
    encoded = pd.DataFrame({"encoded_feature": names, "importance": values})
    encoded["feature"] = [raw_feature(x, numeric, categorical) for x in encoded["encoded_feature"]]
    grouped = encoded.groupby("feature", as_index=False)["importance"].sum()
    return [
        {"experiment": experiment, "model": model_name, "fold_id": fold,
         "feature": row.feature, "importance": float(row.importance)}
        for row in grouped.itertuples(index=False)
    ]


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    assignments = pd.read_csv(SPLIT_PATH)
    if assignments.creep_id.duplicated().any() or not assignments.creep_id.isin(data.creep_id).all():
        raise ValueError("Frozen split assignment has invalid creep_id values.")
    data = data.merge(assignments[["creep_id", "fold_id"]], on="creep_id", how="inner", validate="one_to_one")
    data = data.loc[data.transfer_tier.eq("Tier 1")].copy()
    required = set(NUMERIC_A1 + CATEGORICAL_A1 + NUMERIC_A2_EXTRA + CATEGORICAL_A2_EXTRA + ID_COLUMNS + [TARGET])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    data[CATEGORICAL_A1 + CATEGORICAL_A2_EXTRA] = data[CATEGORICAL_A1 + CATEGORICAL_A2_EXTRA].replace("", np.nan)
    if data.empty or data[TARGET].isna().any():
        raise ValueError("Tier 1 data are empty or target contains missing values.")

    experiments = {
        "A1_core_physical": (NUMERIC_A1, CATEGORICAL_A1),
        "A2_core_plus_chemistry_quality": (NUMERIC_A1 + NUMERIC_A2_EXTRA, CATEGORICAL_A1 + CATEGORICAL_A2_EXTRA),
    }
    models = ["Mean", "Ridge", "RandomForest", "XGBoost"]
    fold_metrics, prediction_rows, importance = [], [], []
    leakage_rows = []
    for fold in sorted(data.fold_id.unique()):
        validation = data.loc[data.fold_id.eq(fold)].copy()
        training = data.loc[~data.fold_id.eq(fold)].copy()
        overlap = sorted(set(training.source_id) & set(validation.source_id))
        leakage_rows.append({"fold_id": int(fold), "train_samples": len(training), "validation_samples": len(validation),
                             "train_sources": training.source_id.nunique(), "validation_sources": validation.source_id.nunique(),
                             "shared_sources": len(overlap)})
        if overlap:
            raise ValueError(f"Source leakage in frozen fold {fold}: {overlap}")
        for experiment, (numeric, categorical) in experiments.items():
            feature_columns = numeric + categorical
            for model_name in models:
                if model_name == "Mean":
                    prediction = np.repeat(training[TARGET].mean(), len(validation))
                else:
                    pipe = model_pipeline(model_name, numeric, categorical)
                    pipe.fit(training[feature_columns], training[TARGET])
                    prediction = pipe.predict(validation[feature_columns])
                    importance.extend(importance_rows(pipe, model_name, experiment, int(fold), numeric, categorical))
                record = {"experiment": experiment, "model": model_name, "fold_id": int(fold),
                          "n_train": len(training), "n_validation": len(validation)}
                record.update(metrics(validation[TARGET], prediction))
                fold_metrics.append(record)
                prediction_rows.extend({"creep_id": cid, "source_id": source, "fold_id": int(fold),
                                        "experiment": experiment, "model": model_name, "y_true": float(actual),
                                        "y_pred": float(pred)}
                                       for cid, source, actual, pred in zip(validation.creep_id, validation.source_id,
                                                                            validation[TARGET], prediction))

    leakage = pd.DataFrame(leakage_rows)
    if not leakage.shared_sources.eq(0).all():
        raise ValueError("Source leakage check failed.")
    metrics_frame = pd.DataFrame(fold_metrics).sort_values(["experiment", "model", "fold_id"])
    predictions = pd.DataFrame(prediction_rows).sort_values(["experiment", "model", "fold_id", "creep_id"])
    importance_frame = pd.DataFrame(importance).sort_values(["experiment", "model", "feature", "fold_id"])
    aggregate_rows = []
    for (experiment, model_name), group in predictions.groupby(["experiment", "model"], sort=True):
        subset_metrics = metrics_frame.loc[(metrics_frame.experiment == experiment) & (metrics_frame.model == model_name)]
        row = {"experiment": experiment, "model": model_name, "n_oof": len(group), "n_folds": group.fold_id.nunique()}
        row.update({"oof_" + key: value for key, value in metrics(group.y_true, group.y_pred).items()})
        for key in ["mae", "rmse", "r2"]:
            row["fold_mean_" + key] = float(subset_metrics[key].mean())
            row["fold_sd_" + key] = float(subset_metrics[key].std(ddof=1))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["experiment", "model"])
    importance_summary = (importance_frame.groupby(["experiment", "model", "feature"], as_index=False)
                           .agg(mean_importance=("importance", "mean"), sd_importance=("importance", "std"))
                           .sort_values(["experiment", "model", "mean_importance"], ascending=[True, True, False]))

    metrics_frame.to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    aggregate.to_csv(OUT_DIR / "cv_metrics_summary.csv", index=False)
    predictions.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    leakage.to_csv(OUT_DIR / "source_leakage_check.csv", index=False)
    importance_frame.to_csv(OUT_DIR / "feature_importance_by_fold.csv", index=False)
    importance_summary.to_csv(OUT_DIR / "feature_importance_summary.csv", index=False)
    report = f"""# TASK 04.3.2 — Tier 1 Baseline Training

## Scope and protocol

- Dataset: `baseline_creep_dataset_v1p1.csv` (SHA-256 `{sha256(DATA_PATH)}`)
- Domain: `transfer_tier == \"Tier 1\"` only; **{len(data)}** observations from **{data.source_id.nunique()}** sources.
- Target: `target_log10_rupture_life_h`.
- Evaluation: frozen `source_groupkfold_5fold_assignment.csv`; Tier-1 rows retain their original fold assignments.
- Source leakage: **PASS** — every train/validation source intersection is empty.
- No split was regenerated; no source dataset was modified.

## Fold leakage and size check

{markdown_table(leakage)}

## Cross-validation results

The `oof_*` values are computed over all out-of-fold predictions and are the primary source-aware result. Fold means and standard deviations describe variability across the uneven whole-source folds.

{markdown_table(aggregate)}

## Feature settings

- **A1:** physical condition, composition, and material-state descriptors only.
- **A2:** A1 plus `Ni_source`, chemistry missingness indicators, and `chemistry_completeness_score`. Any A2 change is information from reporting completeness/provenance, not automatically physical improvement.
- The source dataset names its original composition wt% fields `Cr`, `Mo`, `Fe`, `Co`, `W`, `Nb`, `Ti`, `Al`, `C`, and `B`; these are the task-specified `*_wt` quantities. They are used unchanged. `Ni_reconstructed_wt` is the separately documented balance-reconstructed field.

## Models

- Mean predictor.
- Ridge: median imputation, standardization, `alpha=1`.
- Random Forest: median imputation, 500 trees, `min_samples_leaf=3`, `random_state=42`.
- XGBoost: median-compatible preparation, 500 estimators, depth 4, learning rate 0.03, subsample/column sample 0.8, `random_state=42`.

## Outputs

- `fold_metrics.csv`, `cv_metrics_summary.csv`, and `oof_predictions.csv` record reproducible performance.
- `feature_importance_summary.csv` aggregates absolute Ridge coefficients or tree importances to the original feature level. Importances describe model association only and are not causal evidence.
"""
    (OUT_DIR / "baseline_training_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "task": "TASK 04.3.2 — Tier 1 Baseline Training", "dataset": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256(DATA_PATH), "split_assignment": str(SPLIT_PATH.relative_to(ROOT)),
        "domain_filter": "transfer_tier == 'Tier 1'", "target": TARGET,
        "n_tier1_samples": len(data), "n_tier1_sources": int(data.source_id.nunique()),
        "fold_ids": sorted(map(int, data.fold_id.unique())), "source_leakage_passed": True,
        "experiments": {key: {"numeric": numeric, "categorical": categorical}
                        for key, (numeric, categorical) in experiments.items()},
        "source_composition_column_note": "Original source wt% fields retain their dataset identifiers Cr, Mo, Fe, Co, W, Nb, Ti, Al, C, and B; only the derived Ni field is named Ni_reconstructed_wt.",
        "models": {"Mean": "training-fold target mean", "Ridge": {"alpha": 1.0},
                   "RandomForest": {"n_estimators": 500, "min_samples_leaf": 3, "random_state": 42},
                   "XGBoost": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
                               "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}},
        "script": "03_MODELING/scripts/train_tier1_baseline_v1.py",
    }
    (OUT_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Tier 1 baseline results to {OUT_DIR}")


if __name__ == "__main__":
    main()
