#!/usr/bin/env python3
"""Assemble figures from frozen outputs; never fits a model."""

from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = _CANDIDATE_ROOT
OUT = ROOT / "05_PAPER_CONVERGENCE"
MAIN = OUT / "figures/main"
SUPP = OUT / "figures/supplementary"

BLUE = "#2F6690"
TEAL = "#3A8D8D"
ORANGE = "#D9822B"
RED = "#B64B4B"
GREY = "#6B7280"
LIGHT = "#E8EEF3"
DARK = "#263238"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.06, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold")


def box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str = BLUE, width: float = 0.72, height: float = 0.16) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.025", linewidth=1.2, edgecolor=color, facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=DARK, fontsize=8.2)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GREY, style: str = "-|>") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=1.2, color=color))


dataset = pd.read_csv(ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv")
model_table = pd.read_csv(OUT / "FINAL_MODEL_COMPARISON_TABLE.csv")
transfer = pd.read_csv(OUT / "FINAL_TRANSFER_COMPARISON_TABLE.csv")
st_predictions = pd.read_csv(ROOT / "03_MODELING/results/stress_temperature_physics_v1/predictions.csv")
physics_oof = pd.read_csv(ROOT / "03_MODELING/results/classical_physics_baseline_v1/physics_baseline_oof_predictions.csv")
state_params = pd.read_csv(ROOT / "03_MODELING/results/within_state_physics_diagnostic_v1/state_physics_parameters.csv")
state_fit = pd.read_csv(ROOT / "03_MODELING/results/within_state_physics_diagnostic_v1/state_fit_metrics.csv")
state_cv = pd.read_csv(ROOT / "03_MODELING/results/within_state_physics_diagnostic_v1/state_cv_metrics.csv")
parameter_metrics = pd.read_csv(ROOT / "03_MODELING/results/state_parameter_learnability_v1/parameter_model_metrics.csv")
reconstruction = pd.read_csv(ROOT / "03_MODELING/results/state_parameter_learnability_v1/state_parameter_reconstruction_metrics.csv")
nn_seeds = pd.read_csv(ROOT / "03_MODELING/results/unified_ml_physics_benchmark_v1/nn_seed_metrics.csv")
importance = pd.read_csv(ROOT / "03_MODELING/results/baseline_tier1_v1/feature_importance_summary.csv")


# Figure 1: database and validation
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
counts = dataset.groupby(["transfer_tier", "alloy_family"]).size().unstack(fill_value=0).reindex(["Tier 1", "Tier 2", "Tier 3"])
bottom = np.zeros(len(counts))
colors = [BLUE, ORANGE]
for idx, col in enumerate(counts.columns):
    ax.bar(counts.index, counts[col], bottom=bottom, label=col.replace("-based superalloy", "-based"), color=colors[idx % len(colors)])
    bottom += counts[col].to_numpy()
for i, tier in enumerate(counts.index):
    part = dataset[dataset.transfer_tier == tier]
    ax.text(i, bottom[i] + 18, f"{part.material_state_id.nunique()} states\n{part.source_id.nunique()} sources", ha="center", fontsize=7.5)
ax.set_ylabel("Creep records")
ax.set_title("Database heterogeneity")
ax.legend(loc="upper right", fontsize=7)
ax.set_ylim(0, max(bottom) * 1.22)
panel(ax, "a")

ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, (0.5, 0.82), "Observed descriptors\nT, stress, composition,\nmechanism information", TEAL)
box(ax, (0.5, 0.50), "Incomplete material-state\nrepresentation", ORANGE)
box(ax, (0.5, 0.18), "Creep rupture behavior", BLUE)
arrow(ax, (0.5, 0.72), (0.5, 0.61)); arrow(ax, (0.5, 0.40), (0.5, 0.29))
ax.set_title("Representation gap")
panel(ax, "b")

ax = axes[2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ys = [0.88, 0.68, 0.48, 0.28, 0.08]
labels = ["Route B database", "Group by source_id", "5-fold OOF validation", "Predict log10 life", "MAE / RMSE / R²"]
for y, label in zip(ys, labels): box(ax, (0.5, y), label, BLUE, width=0.68, height=0.105)
for y1, y2 in zip(ys[:-1], ys[1:]): arrow(ax, (0.5, y1 - 0.06), (0.5, y2 + 0.06))
ax.text(0.5, -0.07, "Zero validation-source overlap", transform=ax.transAxes, ha="center", color=TEAL, fontweight="bold")
ax.set_title("Source-aware validation")
panel(ax, "c")
fig.suptitle("Database structure and validation design", y=1.02, fontsize=12, fontweight="bold")
save(fig, MAIN / "Fig1_database_validation")


# Figure 2: benchmark, parity, residual
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55))
names = ["Ridge", "Random Forest", "XGBoost baseline", "P3-replace XGBoost"]
vals = [float(model_table.loc[model_table.model_name == n, "R2"].iloc[0]) for n in names]
ax = axes[0]
ax.barh(["Ridge", "Random\nForest", "XGBoost\nbaseline", "P3 physics-guided\nXGBoost"], vals, color=[GREY, GREY, BLUE, TEAL])
ax.set_xlim(0, 0.82); ax.set_xlabel("Pooled OOF R²")
for i, value in enumerate(vals): ax.text(value + 0.012, i, f"{value:.4f}", va="center")
ax.set_title("Harmonized Tier1 benchmark")
panel(ax, "a")

p3 = st_predictions.query("analysis == 'tier1_physics' and `feature_set` == 'P3-replace' and model == 'XGBoost'").copy()
assert len(p3) == 220
ax = axes[1]
ax.scatter(p3.y_true, p3.y_pred, s=16, alpha=0.68, color=BLUE, edgecolors="none")
lims = [min(p3.y_true.min(), p3.y_pred.min()) - 0.2, max(p3.y_true.max(), p3.y_pred.max()) + 0.2]
ax.plot(lims, lims, "--", color=GREY, lw=1)
ax.set(xlim=lims, ylim=lims, xlabel="Experimental log10(rupture life / h)", ylabel="Predicted log10(rupture life / h)")
ax.text(0.04, 0.96, "R² = 0.7497\nMAE = 0.3592\nRMSE = 0.4931", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": .85, "edgecolor": LIGHT})
ax.set_title("P3-XGBoost parity (n=220)")
panel(ax, "b")

ax = axes[2]
ax.hist(p3.residual, bins=22, color=BLUE, alpha=.82, edgecolor="white")
ax.axvline(0, color=DARK, lw=1, ls="--")
ax.axvline(p3.residual.mean(), color=ORANGE, lw=1.5, label=f"Mean = {p3.residual.mean():.3f}")
ax.set(xlabel="Residual (log10 h)", ylabel="Count", title="Residual distribution")
ax.legend(fontsize=7.5)
ax.text(.04, .82, "Positive = life underprediction", transform=ax.transAxes, va="top", fontsize=7.5)
panel(ax, "c")
fig.suptitle("Source-aware machine-learning performance", y=1.02, fontsize=12, fontweight="bold")
save(fig, MAIN / "Fig2_ml_benchmark")


# Figure 3: transfer limitation
fig, axes = plt.subplots(1, 3, figsize=(11, 3.45))
labels = ["Tier1\nP0", "Tier1+Tier2\nnaive", "Tier1+Tier2\nmechanism", "Tier1\nP3"]
colors4 = [BLUE, RED, ORANGE, TEAL]
for ax, metric, ylabel, letter in zip(axes[:2], ["R2", "MAE"], ["Pooled OOF R²", "MAE (log10 h)"], ["a", "b"]):
    values = transfer[metric].to_numpy()
    ax.bar(np.arange(4), values, color=colors4)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel(ylabel)
    ax.set_title("Transfer comparison" if metric == "R2" else "Prediction error")
    for i, value in enumerate(values): ax.text(i, value + (0.012 if metric == "R2" else 0.008), f"{value:.4f}", ha="center", fontsize=7.5)
    if metric == "R2": ax.set_ylim(0.63, 0.78)
    else: ax.set_ylim(0, 0.47)
    panel(ax, letter)
ax = axes[2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, (.5, .82), "More cross-alloy samples", BLUE)
ax.text(.5, .61, "≠", ha="center", va="center", fontsize=27, color=RED)
box(ax, (.5, .43), "Automatically better transfer", RED)
arrow(ax, (.5, .33), (.5, .22), ORANGE)
box(ax, (.5, .10), "Consistent with unresolved\nsource-associated variability", ORANGE)
ax.set_title("Interpretation")
panel(ax, "c")
fig.suptitle("Cross-Alloy Transfer Under Evaluated Pooling Strategies", y=1.02, fontsize=12, fontweight="bold")
save(fig, MAIN / "Fig3_transfer_limitation")


# Figure 4: state-dependent physics
fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.65))
a1 = physics_oof.query("physics_model == 'A1'")
ax = axes[0]
ax.scatter(a1.y_true, a1.y_physics_pred, s=15, alpha=.65, color=ORANGE, edgecolors="none")
lims = [min(a1.y_true.min(), a1.y_physics_pred.min()) - .2, max(a1.y_true.max(), a1.y_physics_pred.max()) + .2]
ax.plot(lims, lims, "--", color=GREY, lw=1)
ax.set(xlim=lims, ylim=lims, xlabel="Experimental log10(rupture life / h)", ylabel="Predicted log10(rupture life / h)", title="Global A1 (source-aware OOF)")
ax.text(.04, .95, "R² = 0.4802\nn = 220", transform=ax.transAxes, va="top")
panel(ax, "a")

ax = axes[1]
eligible = state_fit.query("model == 'A1'").sort_values("n", ascending=False).head(3)
markers = ["o", "s", "^"]
for (_, metric), marker in zip(eligible.iterrows(), markers):
    state = metric.material_state_id
    obs = dataset[(dataset.transfer_tier == "Tier 1") & (dataset.material_state_id == state)]
    pars = state_params.query("material_state_id == @state and model == 'A1'").set_index("parameter").estimate
    x = obs.log_stress_MPa.to_numpy()
    temp = obs.temperature_K.to_numpy()
    pred = pars.beta_0 + pars.beta_sigma * x + pars.beta_T / temp + pars.beta_sigmaT * x / temp
    ax.scatter(obs.target_log10_rupture_life_h, pred, s=25, alpha=.72, marker=marker, label=f"{state} (R²={metric.R2:.2f})")
local_lims = [0, 6]
ax.plot(local_lims, local_lims, "--", color=GREY, lw=1)
ax.set(xlim=local_lims, ylim=local_lims, xlabel="Experimental log10(rupture life / h)", ylabel="Descriptive fitted log10(rupture life / h)", title="Representative local A1 fits")
ax.legend(fontsize=6.4, loc="lower right")
panel(ax, "b")

ax = axes[2]
names = ["Global\nA1", "State grouped\nOOF median", "Oracle state\nreconstruction"]
values = [0.4801863845, 0.8428, 0.9301416715]
ax.bar(np.arange(3), values, color=[ORANGE, BLUE, TEAL])
ax.set_xticks(np.arange(3), names)
ax.set_ylim(0, 1.02); ax.set_ylabel("R² (distinct estimands)")
for i, value in enumerate(values): ax.text(i, value + .025, f"{value:.4f}", ha="center")
ax.set_title("Global versus state-specific capacity")
panel(ax, "c")
fig.suptitle("Classical physics is locally effective but state dependent", y=1.03, fontsize=12, fontweight="bold")
save(fig, MAIN / "Fig4_state_dependent_physics")


# Figure 5: parameter recovery
fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.55))
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, (.28, .78), "Composition", BLUE, width=.45, height=.12)
box(ax, (.72, .78), "Mechanism", BLUE, width=.45, height=.12)
box(ax, (.5, .51), "Available state descriptors", TEAL, width=.70, height=.13)
arrow(ax, (.5, .43), (.5, .27), RED, style="-[")
ax.text(.57, .35, "unresolved mapping", color=RED, fontsize=7.5)
box(ax, (.5, .16), "θ_state\nstate-associated effective parameters", ORANGE, width=.78, height=.16)
ax.set_title("Descriptor-to-parameter question")
panel(ax, "a")

ax = axes[1]
params = ["beta_0", "beta_T", "beta_sigma", "beta_sigmaT"]
b0 = parameter_metrics[(parameter_metrics.method == "B0_global_mean")].set_index("parameter").loc[params]
d2 = parameter_metrics[(parameter_metrics.method == "D2")].set_index("parameter").loc[params]
x = np.arange(4); width = .36
ax.bar(x - width/2, b0.R2, width, color=GREY, label="Global-mean baseline")
ax.bar(x + width/2, d2.R2, width, color=RED, label="Composition + mechanism Ridge")
ax.axhline(0, color=DARK, lw=.8)
ax.set_xticks(x, ["β₀", "βT", "βσ", "βσT"])
ax.set_ylabel("LOSO parameter R²")
ax.set_title("Parameter recovery (10 states)")
ax.legend(fontsize=6.8)
ax.set_ylim(min(d2.R2.min() * 1.12, -27), 1)
panel(ax, "b")

ax = axes[2]
order = ["global_A1", "predicted_state_A1", "oracle_state_A1"]
rec = reconstruction.set_index("model").loc[order]
bars = ax.bar(["Global\nA1", "Predicted-state\nA1", "Oracle-state\nA1"], rec.R2, color=[ORANGE, RED, TEAL])
ax.axhline(0, color=DARK, lw=.8)
ax.set_yscale("symlog", linthresh=1)
ax.set_ylabel("Life-reconstruction R² (symlog)")
ax.set_title("Parameter-error propagation")
for bar, value in zip(bars, rec.R2):
    y = value + .12 if value >= 0 else -25
    ax.text(bar.get_x() + bar.get_width()/2, y, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5, color="white" if value < 0 else DARK)
ax.text(.03, .96, "Large negative R² is consistent with\nparameter-error propagation", transform=ax.transAxes, va="top", fontsize=7.2, color=RED)
panel(ax, "c")
fig.suptitle("Current descriptors do not recover state-associated effective parameters", y=1.02, fontsize=12, fontweight="bold")
save(fig, MAIN / "Fig5_state_parameter_recovery")


# Figure 6: scientific synthesis
fig, ax = plt.subplots(figsize=(8.6, 4.7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, (.17, .80), "Observed database variables\nT, stress, composition, mechanism", BLUE, width=.29, height=.18)
box(ax, (.50, .80), "Incomplete material-state\nrepresentation", ORANGE, width=.29, height=.18)
box(ax, (.83, .80), "State-associated effective\nphysics parameters", TEAL, width=.29, height=.18)
arrow(ax, (.32, .80), (.35, .80)); arrow(ax, (.65, .80), (.68, .80))
box(ax, (.50, .48), "Limited cross-alloy transfer\nnaive pooling degrades prediction", RED, width=.40, height=.17)
arrow(ax, (.83, .70), (.59, .57), ORANGE); arrow(ax, (.17, .70), (.41, .57), BLUE)
box(ax, (.50, .18), "Future evidence direction\nadditional measured state descriptors\n+ physics-guided ML", BLUE, width=.47, height=.19)
arrow(ax, (.50, .39), (.50, .29))
ax.text(.5, .965, "Evidence synthesis", ha="center", fontsize=13, fontweight="bold")
ax.text(.5, .02, "Unresolved material-state-related variability is an inference, not a direct microstructural observation.", ha="center", fontsize=8, color=GREY)
save(fig, MAIN / "Fig6_scientific_synthesis")


# Supplementary Figure S1: data distributions
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, col, label, letter in zip(
    axes,
    ["temperature_K", "stress_MPa", "target_log10_rupture_life_h"],
    ["Temperature (K)", "Stress (MPa)", "log10(rupture life / h)"],
    ["a", "b", "c"],
):
    for tier, color in zip(["Tier 1", "Tier 2", "Tier 3"], [BLUE, ORANGE, GREY]):
        ax.hist(dataset.loc[dataset.transfer_tier == tier, col], bins=20, histtype="step", lw=1.4, color=color, label=tier)
    ax.set_xlabel(label); ax.set_ylabel("Count"); panel(ax, letter)
axes[0].legend(fontsize=7)
fig.suptitle("Route B modeling-data distributions", fontsize=12, fontweight="bold")
save(fig, SUPP / "FigS1_dataset_distributions")


# Supplementary Figure S2: split topology, not performance
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
axes[0].bar(["Shared sources", "Non-shared sources"], [39, 5], color=[RED, GREY])
axes[0].set_title("Random 80/20 split"); axes[0].set_ylabel("Sources")
axes[0].text(.5, .90, "39/44 shared", transform=axes[0].transAxes, ha="center", color=RED, fontweight="bold")
axes[1].bar(["Shared sources", "Grouped sources"], [0, 44], color=[TEAL, BLUE])
axes[1].set_title("Source-aware 5-fold split")
axes[1].text(.5, .90, "0 shared in every fold", transform=axes[1].transAxes, ha="center", color=TEAL, fontweight="bold")
for ax, letter in zip(axes, ["a", "b"]): panel(ax, letter); ax.set_ylim(0, 48)
fig.suptitle("Why source-aware validation is required", fontsize=12, fontweight="bold")
save(fig, SUPP / "FigS2_split_comparison")


# Supplementary Figure S3: P0 versus P3 residuals
fig, ax = plt.subplots(figsize=(6.2, 3.8))
p0 = st_predictions.query("analysis == 'tier1_physics' and `feature_set` == 'P0' and model == 'XGBoost'")
ax.hist(p0.residual, bins=24, histtype="step", lw=1.6, color=GREY, label="P0 XGBoost")
ax.hist(p3.residual, bins=24, histtype="step", lw=1.6, color=TEAL, label="P3-replace XGBoost")
ax.axvline(0, color=DARK, lw=.8, ls="--")
ax.set(xlabel="Residual (log10 h)", ylabel="Count", title="Tier1 source-aware OOF residual comparison")
ax.legend()
save(fig, SUPP / "FigS3_residual_comparison")


# Supplementary Figure S4: strict Tier1 NN stability
fig, ax = plt.subplots(figsize=(6.6, 3.8))
nn_names = ["NN life-only", "NN life+mechanism", "NN multi-view"]
for i, name in enumerate(nn_names):
    values = nn_seeds.loc[nn_seeds.model == name, "MAE"].to_numpy()
    ax.scatter(np.full(len(values), i) + np.linspace(-.08, .08, len(values)), values, color=[BLUE, ORANGE, RED][i], s=28, alpha=.8)
    ax.plot([i-.16, i+.16], [np.median(values)]*2, color=DARK, lw=1.5)
ax.axhline(.3592029930, color=TEAL, ls="--", label="P3-XGBoost pooled OOF MAE")
ax.set_xticks(range(3), ["Life-only", "Life +\nmechanism", "Multi-view"])
ax.set_yscale("log"); ax.set_ylabel("Seed-level MAE (log10 h; log scale)")
ax.set_title("Strict Tier1 neural-network rerun stability")
ax.legend(fontsize=7.5)
save(fig, SUPP / "FigS4_nn_benchmark")


# Supplementary Figure S5: associative feature importance
fig, ax = plt.subplots(figsize=(6.6, 4.3))
imp = importance.query("experiment == 'A1_core_physical' and model == 'XGBoost'").nlargest(10, "mean_importance").sort_values("mean_importance")
ax.barh(imp.feature.str.replace("_", " "), imp.mean_importance, xerr=imp.sd_importance, color=BLUE, alpha=.85)
ax.set_xlabel("Relative feature importance (dimensionless)")
ax.set_title("Baseline XGBoost feature associations")
ax.text(.98, .03, "Association, not causal effect", transform=ax.transAxes, ha="right", color=RED, fontsize=8)
save(fig, SUPP / "FigS5_feature_interpretation")


manifest = {
    "task": "TASK 05.3",
    "generated_main_figures": 6,
    "generated_supplementary_figures": 5,
    "formats": ["pdf", "svg", "png"],
    "models_trained": 0,
    "frozen_outputs_modified": False,
}
(OUT / "figure_metadata/run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Generated 11 figures in PDF, SVG, and 600-dpi PNG formats; no model fitting performed.")
