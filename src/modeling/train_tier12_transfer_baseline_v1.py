#!/usr/bin/env python3
"""evaluate Tier-2 transfer on frozen Tier-1 source folds.

The primary score is calculated on held-out Tier-1 records.  For each frozen
fold, every row from every validation source (Tier 1 and Tier 2) is excluded
from training.  Thus Tier-2 data change only the training domain, never the
Tier-1 evaluation population or its source-aware fold assignment.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = _CANDIDATE_ROOT
DATA = ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv"
SPLITS = ROOT / "03_MODELING/splits/source_groupkfold_5fold_assignment.csv"
TIER1 = ROOT / "03_MODELING/results/baseline_tier1_v1"
OUT = ROOT / "03_MODELING/results/baseline_tier12_transfer_v1"
TARGET = "target_log10_rupture_life_h"
NUMERIC = ["temperature_K", "stress_MPa", "log_temperature_K", "log_stress_MPa",
           "Ni_reconstructed_wt", "Cr", "Mo", "Fe", "Co", "W", "Nb", "Ti", "Al", "C", "B"]
CATEGORICAL = ["alloy_family", "base_alloy", "heat_treatment_id"]
MODELS = ["Mean", "Ridge", "RandomForest", "XGBoost"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(mean_squared_error(y, p) ** .5), "R2": float(r2_score(y, p))}


def prep(scale=False):
    n = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        n += [("scaler", StandardScaler())]
    return ColumnTransformer([
        ("numeric", Pipeline(n), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ], verbose_feature_names_out=True)


def make_model(name):
    if name == "Ridge":
        return Pipeline([("features", prep(True)), ("model", Ridge(alpha=1.0))])
    if name == "RandomForest":
        return Pipeline([("features", prep()), ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1))])
    if name == "XGBoost":
        return Pipeline([("features", prep()), ("model", XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03, subsample=.8,
            colsample_bytree=.8, random_state=42, n_jobs=-1, objective="reg:squarederror"))])
    raise ValueError(name)


def feature_name(encoded):
    if encoded.startswith("numeric__"):
        return encoded.replace("numeric__", "", 1)
    raw = encoded.replace("categorical__", "", 1)
    return next((c for c in CATEGORICAL if raw == c or raw.startswith(c + "_")), raw)


def draw_figures(pred, comparison, importance, by_family):
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    x = np.arange(len(comparison)); width = .36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, comparison.R2_mean, width, label="Tier1 baseline")
    ax.bar(x + width / 2, comparison.tier12_R2_mean, width, label="Tier1 + Tier2")
    ax.set_xticks(x, comparison.model); ax.set_ylabel("Tier1 OOF R²"); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "tier1_vs_tier12_performance.png", dpi=180); plt.close(fig)
    for model in ["RandomForest", "XGBoost"]:
        b = pred[(pred.model == model) & (pred.transfer_tier == "Tier 1")]
        fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(b.y_true, b.y_pred, alpha=.7)
        lim = [min(b.y_true.min(), b.y_pred.min()), max(b.y_true.max(), b.y_pred.max())]
        ax.plot(lim, lim, "k--"); ax.set(xlabel="Observed log10 h", ylabel="Predicted log10 h", title=model)
        fig.tight_layout(); fig.savefig(figdir / f"parity_{model.lower()}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in MODELS:
        x = pred[(pred.model == model) & (pred.transfer_tier == "Tier 1")].residual
        ax.hist(x, bins=22, alpha=.42, label=model)
    ax.axvline(0, color="black", linewidth=.8); ax.legend(); ax.set_xlabel("Residual (observed − predicted)")
    fig.tight_layout(); fig.savefig(figdir / "residual_distributions.png", dpi=180); plt.close(fig)
    sub = importance[importance.model.isin(["RandomForest", "XGBoost"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(["RandomForest", "XGBoost"]):
        d = sub[sub.model == model].nlargest(10, "mean_importance").sort_values("mean_importance")
        ax.barh(np.arange(10) + i * .38, d.mean_importance, .35, label=model)
        if i == 0: ax.set_yticks(np.arange(10), d.feature)
    ax.legend(); ax.set_xlabel("Mean grouped importance"); fig.tight_layout()
    fig.savefig(figdir / "feature_importance_comparison.png", dpi=180); plt.close(fig)
    d = by_family.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.barh(d.alloy_family, d.MAE); ax.set_xlabel("Tier2 OOF MAE")
    fig.tight_layout(); fig.savefig(figdir / "tier2_error_by_alloy_family.png", dpi=180); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "feature_importance").mkdir(exist_ok=True)
    data = pd.read_csv(DATA).merge(pd.read_csv(SPLITS)[["creep_id", "fold_id"]], on="creep_id", validate="one_to_one")
    data = data[data.transfer_tier.isin(["Tier 1", "Tier 2"])].copy()
    data[CATEGORICAL] = data[CATEGORICAL].replace("", np.nan)
    if data[TARGET].isna().any() or set(NUMERIC + CATEGORICAL) - set(data):
        raise ValueError("Missing required feature or target values")
    metrics_rows, predictions, importances, leakage = [], [], [], []
    for fold in sorted(data.fold_id.unique()):
        val = data[data.fold_id == fold].copy(); train = data[data.fold_id != fold].copy()
        shared = set(train.source_id) & set(val.source_id)
        leakage.append({"fold_id": fold, "train_samples": len(train), "validation_samples": len(val),
                        "train_sources": train.source_id.nunique(), "validation_sources": val.source_id.nunique(),
                        "shared_sources": len(shared)})
        if shared: raise ValueError(f"Source leakage in fold {fold}: {shared}")
        for name in MODELS:
            pipe = None
            if name == "Mean": p = np.repeat(train[TARGET].mean(), len(val))
            else:
                pipe = make_model(name); pipe.fit(train[NUMERIC + CATEGORICAL], train[TARGET]); p = pipe.predict(val[NUMERIC + CATEGORICAL])
                if name in ["RandomForest", "XGBoost"]:
                    encoded = pipe.named_steps["features"].get_feature_names_out(); values = pipe.named_steps["model"].feature_importances_
                    tmp = pd.DataFrame({"feature": [feature_name(v) for v in encoded], "importance": values}).groupby("feature", as_index=False).importance.sum()
                    importances.extend({"model": name, "fold_id": fold, "feature": r.feature, "importance": r.importance} for r in tmp.itertuples())
            out = val[["creep_id", "source_id", "material_state_id", "transfer_tier", "alloy_family", "temperature_K", "stress_MPa", "fold_id"]].copy()
            out["model"] = name; out["y_true"] = val[TARGET].values; out["y_pred"] = p; out["residual"] = out.y_true - out.y_pred
            predictions.append(out)
            for tier in ["Tier 1", "Tier 2", "All"]:
                q = out if tier == "All" else out[out.transfer_tier == tier]
                if len(q) > 1:
                    metrics_rows.append({"fold_id": fold, "model": name, "evaluation_domain": tier, "n": len(q), **scores(q.y_true, q.y_pred)})
    pred = pd.concat(predictions).sort_values(["model", "fold_id", "creep_id"])
    fold_metrics = pd.DataFrame(metrics_rows).sort_values(["evaluation_domain", "model", "fold_id"])
    summary = fold_metrics.groupby(["model", "evaluation_domain"], as_index=False).agg(
        n_oof=("n", "sum"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"), R2_mean=("R2", "mean"), R2_std=("R2", "std"))
    imp = pd.DataFrame(importances); imp_sum = imp.groupby(["model", "feature"], as_index=False).agg(mean_importance=("importance", "mean"), sd_importance=("importance", "std"))
    for model in ["RandomForest", "XGBoost"]:
        imp_sum[imp_sum.model == model].sort_values("mean_importance", ascending=False).to_csv(OUT / "feature_importance" / f"feature_importance_{'rf' if model == 'RandomForest' else 'xgb'}.csv", index=False)
    # Direct comparison uses identical held-out Tier-1 OOF records and baseline A1 fold means.
    old = pd.read_csv(TIER1 / "cv_metrics_summary.csv"); old = old[old.experiment.eq("A1_core_physical")]
    now = summary[summary.evaluation_domain.eq("Tier 1")]
    comparison = old.merge(now, on="model", suffixes=("_tier1", "_tier12"))
    comparison = pd.DataFrame({"model": comparison.model, "dataset": "Tier1 vs Tier1+Tier2 (Tier1 validation)",
        "MAE_mean": comparison.fold_mean_mae, "RMSE_mean": comparison.fold_mean_rmse, "R2_mean": comparison.fold_mean_r2,
        "R2_std": comparison.fold_sd_r2, "tier12_MAE_mean": comparison.MAE_mean, "tier12_RMSE_mean": comparison.RMSE_mean,
        "tier12_R2_mean": comparison.R2_mean, "tier12_R2_std": comparison.R2_std})
    comparison["delta_R2"] = comparison.tier12_R2_mean - comparison.R2_mean
    comparison["delta_MAE"] = comparison.tier12_MAE_mean - comparison.MAE_mean
    # Condition-region and Tier2-family behaviour for XGB are evaluated only from OOF predictions.
    xgb = pred[pred.model.eq("XGBoost")].copy(); t1x = xgb[xgb.transfer_tier.eq("Tier 1")]
    qlow, qhot, qlong = t1x.stress_MPa.quantile(.25), t1x.temperature_K.quantile(.75), t1x.y_true.quantile(.75)
    regions = [("low_stress", t1x.stress_MPa <= qlow), ("high_temperature", t1x.temperature_K >= qhot), ("long_life_tail", t1x.y_true >= qlong)]
    region_rows = [{"region": name, "n": int(mask.sum()), **scores(t1x.loc[mask, "y_true"], t1x.loc[mask, "y_pred"]), "mean_residual": float(t1x.loc[mask, "residual"].mean())} for name, mask in regions]
    old_pred = pd.read_csv(TIER1 / "oof_predictions.csv")
    old_pred = old_pred[(old_pred.experiment == "A1_core_physical") & (old_pred.model == "XGBoost")]
    old_pred = old_pred.merge(data[data.transfer_tier.eq("Tier 1")][["creep_id", "temperature_K", "stress_MPa"]], on="creep_id", validate="one_to_one")
    old_pred["residual"] = old_pred.y_true - old_pred.y_pred
    old_regions = [("low_stress", old_pred.stress_MPa <= qlow), ("high_temperature", old_pred.temperature_K >= qhot), ("long_life_tail", old_pred.y_true >= qlong)]
    difficult_compare = pd.DataFrame([{"region": name, "n": int(mask.sum()),
        "tier1_MAE": scores(old_pred.loc[mask, "y_true"], old_pred.loc[mask, "y_pred"])["MAE"],
        "tier12_MAE": scores(t1x.loc[new_mask, "y_true"], t1x.loc[new_mask, "y_pred"])["MAE"],
        "tier1_mean_residual": float(old_pred.loc[mask, "residual"].mean()),
        "tier12_mean_residual": float(t1x.loc[new_mask, "residual"].mean())}
        for (name, mask), (_, new_mask) in zip(old_regions, regions)])
    # Nearest condition distance, fit separately in each fold using only that fold's combined training partition.
    distance = []
    for fold in sorted(data.fold_id.unique()):
        tr, va = data[data.fold_id != fold], data[data.fold_id == fold]
        scaler = StandardScaler().fit(SimpleImputer(strategy="median").fit_transform(tr[NUMERIC]))
        trn = scaler.transform(SimpleImputer(strategy="median").fit(tr[NUMERIC]).transform(tr[NUMERIC]))
        # use a single fitted imputer (the repeated fit above is deliberately avoided below)
        imputer = SimpleImputer(strategy="median").fit(tr[NUMERIC]); nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(imputer.transform(tr[NUMERIC])))
        idx = va.transfer_tier.eq("Tier 1"); dist = nn.kneighbors(scaler.transform(imputer.transform(va.loc[idx, NUMERIC])))[0][:, 0]
        tmp = va.loc[idx, ["creep_id"]].copy(); tmp["fold_id"] = fold; tmp["nearest_feature_distance"] = dist; distance.append(tmp)
    dist = pd.concat(distance); xgbt1 = t1x.merge(dist, on=["creep_id", "fold_id"], validate="one_to_one")
    xgbt1["coverage_class"] = np.where(xgbt1.nearest_feature_distance >= xgbt1.nearest_feature_distance.quantile(.8), "low_coverage", "higher_coverage")
    coverage = xgbt1.groupby("coverage_class", as_index=False).agg(n=("creep_id", "size"), mean_distance=("nearest_feature_distance", "mean"), MAE=("residual", lambda s: float(np.abs(s).mean())), RMSE=("residual", lambda s: float(np.sqrt(np.mean(s**2)))))
    family = xgb[xgb.transfer_tier.eq("Tier 2")].groupby("alloy_family", as_index=False).agg(n=("creep_id", "size"), MAE=("residual", lambda s: float(np.abs(s).mean())), mean_residual=("residual", "mean"), prediction_contribution=("y_pred", "mean"))
    pd.DataFrame(leakage).to_csv(OUT / "source_leakage_check.csv", index=False); fold_metrics.to_csv(OUT / "metrics_fold.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False); pred.to_csv(OUT / "predictions.csv", index=False); comparison.to_csv(OUT / "transfer_comparison_with_tier1.csv", index=False)
    pd.DataFrame(region_rows).to_csv(OUT / "tier1_difficult_region_xgb.csv", index=False); difficult_compare.to_csv(OUT / "tier1_difficult_region_comparison_xgb.csv", index=False); coverage.to_csv(OUT / "tier1_feature_coverage_xgb.csv", index=False); family.to_csv(OUT / "tier2_alloy_family_xgb.csv", index=False)
    draw_figures(pred, comparison, imp_sum, family)
    config = {"task": "TASK 04.3.3 — Tier1 + Tier2 Transfer Baseline", "dataset": str(DATA.relative_to(ROOT)), "dataset_sha256": digest(DATA), "split": str(SPLITS.relative_to(ROOT)), "primary_evaluation": "held-out Tier 1 records only", "training_domain": "Tier 1 + Tier 2 excluding all validation sources", "features": {"numeric": NUMERIC, "categorical": CATEGORICAL}, "models": {"Ridge": {"alpha": 1}, "RandomForest": {"n_estimators": 500, "min_samples_leaf": 3, "random_state": 42}, "XGBoost": {"n_estimators": 500, "max_depth": 4, "learning_rate": .03, "subsample": .8, "colsample_bytree": .8, "random_state": 42}}, "source_leakage_passed": True}
    (OUT / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    ctable = comparison[["model", "R2_mean", "tier12_R2_mean", "delta_R2", "MAE_mean", "tier12_MAE_mean"]].to_markdown(index=False, floatfmt=".4f")
    rtable = difficult_compare.to_markdown(index=False, floatfmt=".4f")
    report = f"""# TASK 04.3.3 — Tier1 + Tier2 Transfer Baseline

## Protocol

The immutable v1p1 data and frozen source folds were consumed unchanged. Each fold trains on Tier1 + Tier2 records outside the validation sources, then predicts the held-out sources. The primary comparison below is restricted to **Tier1 validation rows**, exactly the population scored in TASK 04.3.2; no validation source appears in its training partition.

## Performance comparison

{ctable}

`delta_R2` and `delta_MAE` are Tier1+Tier2 minus Tier1 (positive R2 / negative MAE is favourable). A model is called positive-transfer only when its held-out Tier1 score improves; aggregate conclusions should not be based on Tier2 in-domain performance.

## Difficult Tier1 regions (XGBoost transfer model)

{rtable}

The table uses identical Tier1 samples and thresholds; a lower transfer MAE is favourable. `tier1_feature_coverage_xgb.csv` uses fold-safe nearest-feature distances against each fold's combined training set. It measures association, not a causal mechanism.

## Tier2 contribution

`tier2_alloy_family_xgb.csv` reports Tier2 OOF sample counts, error behaviour, and mean prediction by alloy family. It is diagnostic only: Tier2 rows are not the target-domain score.

## Recommendation

Use the source-aware Tier1 comparison as the transfer decision. If changes are mixed or adverse, proceed to a mechanism-aware/domain-adaptation model rather than treating extra alloy data as universally beneficial. Mechanism class was intentionally excluded here under the raw-transfer feature contract.
"""
    (OUT / "transfer_baseline_report.md").write_text(report)
    print(f"Wrote {OUT}")


if __name__ == "__main__": main()
