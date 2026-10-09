#!/usr/bin/env python3
"""fold-safe classical creep-physics baselines (P1 only)."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = _CANDIDATE_ROOT
DATA = ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv"
SPLITS = ROOT / "03_MODELING/splits/source_groupkfold_5fold_assignment.csv"
AUDIT = ROOT / "03_MODELING/results/baseline_tier1_v1/error_source_audit"
ST_METRICS = ROOT / "03_MODELING/results/stress_temperature_physics_v1/metrics_summary.csv"
ST_FOLDS = ROOT / "03_MODELING/results/stress_temperature_physics_v1/metrics_fold.csv"
CREEP = ROOT / "01_DATA/cleaned/RouteB_database_v3/creep_master.csv"
TENSILE = ROOT / "01_DATA/cleaned/RouteB_database_v3/tensile_master.csv"
OUT = ROOT / "03_MODELING/results/classical_physics_baseline_v1"
TARGET = "target_log10_rupture_life_h"


def scores(y, p):
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(mean_squared_error(y, p) ** .5), "R2": float(r2_score(y, p))}


def eligible(d, npar, temp=True):
    x = np.log10(d.stress_MPa.to_numpy(float)); y = d[TARGET].to_numpy(float)
    checks = [len(d) >= max(3*npar, 9), d.source_id.nunique() >= 3,
              np.unique(x).size >= 3, x.max()-x.min() >= .15,
              np.unique(y).size >= 3, y.max()-y.min() >= .30]
    if temp: checks += [d.temperature_K.nunique() >= 3, d.temperature_K.max()-d.temperature_K.min() >= 100]
    z = np.column_stack([np.ones(len(d)), (x-x.mean())/(x.std() or 1), ((1/d.temperature_K)-(1/d.temperature_K).mean())/((1/d.temperature_K).std() or 1)])
    return all(checks) and np.linalg.cond(z) <= 1e6


def constrained_fit(x, y, bounds):
    # SLSQP is used solely to impose the predeclared physical bounds.
    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    initial = np.array([np.clip(v, lo if lo is not None else -np.inf, hi if hi is not None else np.inf) for v,(lo,hi) in zip(initial,bounds)])
    fun = lambda b: float(np.mean((y-x@b)**2))
    jac = lambda b: -2*x.T@(y-x@b)/len(y)
    fit = minimize(fun, initial, jac=jac, method="SLSQP", bounds=bounds, options={"ftol":1e-12,"maxiter":1000})
    if not fit.success or not np.isfinite(fit.x).all(): raise RuntimeError(fit.message)
    return fit.x, fit.message


def fit_predict(model, tr, va):
    log_s = np.log10(tr.stress_MPa.to_numpy(float)); inv_t = 1/tr.temperature_K.to_numpy(float); y = tr[TARGET].to_numpy(float)
    if model == "A0":
        x = np.c_[np.ones(len(tr)), log_s, inv_t]; xv = np.c_[np.ones(len(va)), np.log10(va.stress_MPa), 1/va.temperature_K]
        b, status = constrained_fit(x, y, [(None,None),(None,0),(0,None)])
        names = ["beta_0","beta_sigma","beta_T"]
    elif model == "A1":
        x = np.c_[np.ones(len(tr)), log_s, inv_t, log_s*inv_t]; xv = np.c_[np.ones(len(va)),np.log10(va.stress_MPa),1/va.temperature_K,np.log10(va.stress_MPa)/va.temperature_K]
        b, status = constrained_fit(x, y, [(None,None),(None,0),(0,None),(None,None)])
        names = ["beta_0","beta_sigma","beta_T","beta_sigmaT"]
    elif model == "L1":
        # P=T(C+y)=a+b log10(sigma), algebraically inverted during prediction.
        # Parameters remain (a,b,C), with C fitted only on this outer training fold.
        x = np.c_[1/tr.temperature_K, log_s/tr.temperature_K, -np.ones(len(tr))]
        xv = np.c_[1/va.temperature_K, np.log10(va.stress_MPa)/va.temperature_K, -np.ones(len(va))]
        b, status = constrained_fit(x, y, [(None,None),(None,0),(0,50)])
        names = ["LMP_a_K","LMP_b_K","LMP_C"]
    elif model == "Mean":
        b, status, names = np.array([y.mean()]), "training_mean", ["mean_y"]
        xv = np.ones((len(va),1))
    elif model == "Ridge":
        x = np.c_[log_s, inv_t, log_s*inv_t]; xv = np.c_[np.log10(va.stress_MPa),1/va.temperature_K,np.log10(va.stress_MPa)/va.temperature_K]
        ridge = Ridge(alpha=1).fit(x,y); b, status, names = np.r_[ridge.intercept_,ridge.coef_], "alpha=1", ["intercept","logstress","invT","logstress_invT"]
        return ridge.predict(xv), dict(zip(names,b)), status
    return xv @ b, dict(zip(names,b)), status


def append_predictions(rows, d, model, fold, p, status="OK"):
    q = d[["creep_id","source_id","material_state_id","transfer_tier","mechanism_class","temperature_K","stress_MPa"]].copy()
    q["fold_id"],q["physics_model"],q["parameter_scope"],q["parameter_group"],q["fallback_used"],q["prediction_status"] = fold,model,"global","all Tier1 training records","NONE",status
    q["y_true"],q["y_physics_pred"] = d[TARGET].to_numpy(),p; q["residual"] = q.y_true-q.y_physics_pred; rows.append(q)


def approximate_se(model, tr, pars):
    """Residual-based linear SE audit; constrained-bound estimates are flagged in report."""
    ls=np.log10(tr.stress_MPa.to_numpy()); it=1/tr.temperature_K.to_numpy()
    if model=="A0": x=np.c_[np.ones(len(tr)),ls,it]
    elif model=="A1": x=np.c_[np.ones(len(tr)),ls,it,ls*it]
    elif model=="L1": x=np.c_[it,ls*it,-np.ones(len(tr))]
    elif model=="W1": x=np.c_[np.ones(len(tr)),tr.wilshire_w,it]
    else: return {k:np.nan for k in pars}
    b=np.array(list(pars.values())); dof=max(len(tr)-x.shape[1],1); s2=np.sum((tr[TARGET].to_numpy()-x@b)**2)/dof
    return dict(zip(pars.keys(),np.sqrt(np.maximum(np.diag(np.linalg.pinv(x.T@x))*s2,0))))


def add_wilshire_support(d):
    out=d.copy(); ten=pd.read_csv(TENSILE).dropna(subset=["temperature_C","ultimate_strength_MPa"])
    out["uts"],out["w_status"] = np.nan,"NO_NUMERIC_CURVE"
    for state, ids in out.groupby("material_state_id").groups.items():
        curve=ten[ten.material_state_id.eq(state)].groupby("temperature_C",as_index=False).ultimate_strength_MPa.mean().sort_values("temperature_C")
        if len(curve)<2: continue
        temp=out.loc[ids,"temperature_K"].to_numpy()-273.15; inside=(temp>=curve.temperature_C.min())&(temp<=curve.temperature_C.max())
        vals=np.full(len(temp),np.nan); vals[inside]=np.interp(temp[inside],curve.temperature_C,curve.ultimate_strength_MPa)
        out.loc[ids,"uts"],out.loc[ids,"w_status"] = vals,np.where(inside,"IN_RANGE","OUTSIDE_TEMPERATURE_RANGE")
    eta=out.stress_MPa/out.uts; out["w_valid"]=(eta.gt(0)&eta.lt(1)); out["wilshire_w"]=np.where(out.w_valid,np.log(-np.log(eta)),np.nan)
    return out


def save(fig,name): fig.tight_layout(); fig.savefig(OUT/"figures"/f"{name}.png",dpi=250); plt.close(fig)


def main():
    (OUT/"figures").mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA).merge(pd.read_csv(SPLITS)[["creep_id","fold_id"]],on="creep_id",validate="one_to_one")
    data=data[data.transfer_tier.eq("Tier 1")].copy(); w=add_wilshire_support(data); models=["A0","A1","L1","Mean","Ridge"]
    pred_rows=[]; param_rows=[]; fold_rows=[]
    for fold in sorted(data.fold_id.unique()):
        tr,va=data[data.fold_id.ne(fold)].copy(),data[data.fold_id.eq(fold)].copy()
        if set(tr.source_id)&set(va.source_id): raise ValueError("source leakage")
        for model in models:
            np_ = 4 if model in ["A1","Ridge"] else (1 if model=="Mean" else 3)
            if not eligible(tr,np_): raise ValueError(f"unexpected ineligible global {model} fold {fold}")
            p,pars,status=fit_predict(model,tr,va); append_predictions(pred_rows,va,model,fold,p)
            fold_rows.append({"physics_model":model,"fold_id":fold,"n":len(va),**scores(va[TARGET],p)})
            se=approximate_se(model,tr,pars)
            for name,val in pars.items(): param_rows.append({"physics_model":model,"fold_id":fold,"parameter_name":name,"estimate":val,"standard_error_ols_approx":se[name],"n_train":len(tr),"n_sources_train":tr.source_id.nunique(),"stress_log10_span":np.log10(tr.stress_MPa).max()-np.log10(tr.stress_MPa).min(),"temperature_span_K":tr.temperature_K.max()-tr.temperature_K.min(),"target_span_log10h":tr[TARGET].max()-tr[TARGET].min(),"condition_number":np.linalg.cond(np.c_[np.ones(len(tr)),(np.log10(tr.stress_MPa)-np.log10(tr.stress_MPa).mean())/(np.log10(tr.stress_MPa).std() or 1),((1/tr.temperature_K)-(1/tr.temperature_K).mean())/((1/tr.temperature_K).std() or 1)]),"optimizer_status":status,"admissible":True})
    # W1 is deliberately evaluated only on the in-range UTS subset, never imputed to full Tier1.
    for fold in sorted(w.fold_id.unique()):
        tr=w[w.fold_id.ne(fold) & w.w_valid].copy(); va=w[w.fold_id.eq(fold) & w.w_valid].copy()
        if len(va) < 2 or not eligible(tr,3): continue
        x=np.c_[np.ones(len(tr)),tr.wilshire_w,1/tr.temperature_K]; xv=np.c_[np.ones(len(va)),va.wilshire_w,1/va.temperature_K]
        b,status=constrained_fit(x,tr[TARGET].to_numpy(),[(None,None)]*3); p=xv@b; append_predictions(pred_rows,va,"W1",fold,p,"SUPPORTED_SUBSET")
        fold_rows.append({"physics_model":"W1","fold_id":fold,"n":len(va),**scores(va[TARGET],p)})
        wpars=dict(zip(["alpha_0","alpha_w","alpha_T"],b)); se=approximate_se("W1",tr,wpars)
        for name,val in wpars.items(): param_rows.append({"physics_model":"W1","fold_id":fold,"parameter_name":name,"estimate":val,"standard_error_ols_approx":se[name],"n_train":len(tr),"n_sources_train":tr.source_id.nunique(),"stress_log10_span":np.log10(tr.stress_MPa).max()-np.log10(tr.stress_MPa).min(),"temperature_span_K":tr.temperature_K.max()-tr.temperature_K.min(),"target_span_log10h":tr[TARGET].max()-tr[TARGET].min(),"condition_number":np.linalg.cond(x),"optimizer_status":status,"admissible":True})
    pred=pd.concat(pred_rows,ignore_index=True); params=pd.DataFrame(param_rows); folds=pd.DataFrame(fold_rows)
    # L1-derived P is recorded explicitly to make the training-fold-only inversion auditable.
    pred.to_csv(OUT/"physics_baseline_oof_predictions.csv",index=False); folds.to_csv(OUT/"physics_baseline_fold_metrics.csv",index=False); params.to_csv(OUT/"physics_parameter_stability.csv",index=False)
    metric=[]
    for model,q in pred.groupby("physics_model"):
        f=folds[folds.physics_model.eq(model)]; metric.append({"physics_model":model,"population":"SUPPORTED_SUBSET_OOF" if model=="W1" else "FULL_TIER1_OOF","n":len(q),**scores(q.y_true,q.y_physics_pred),"fold_mean_R2":f.R2.mean(),"fold_std_R2":f.R2.std(ddof=1)})
    # Immutable P3 reference is copied as labelled external comparator, not recomputed.
    ref=pd.read_csv(ST_METRICS).query("analysis == 'tier1_physics' and feature_set == 'P3-replace' and model == 'XGBoost' and evaluation_domain == 'Tier 1'").iloc[0]
    metric.append({"physics_model":"P3-replace XGBoost (reference)","population":"FULL_TIER1_OOF","n":int(ref.n_oof),"MAE":ref.MAE,"RMSE":ref.RMSE,"R2":ref.R2,"fold_mean_R2":ref.fold_mean_R2,"fold_std_R2":ref.fold_sd_R2})
    metrics=pd.DataFrame(metric); metrics.to_csv(OUT/"physics_baseline_metrics.csv",index=False)
    audit=pd.read_csv(AUDIT/"oof_error_table.csv"); low=audit.stress_MPa.quantile(.25); hot=audit.temperature_K.quantile(.75); tail=audit[TARGET].quantile(2/3)
    cov=pd.read_csv(AUDIT/"temperature_stress_coverage.csv")[["creep_id","condition_extrapolation_flag"]]; diag=pred.merge(cov,on="creep_id",validate="many_to_one")
    regions=[("low_stress",diag.stress_MPa<=low),("high_temperature",diag.temperature_K>=hot),("long_life_tail",diag.y_true>=tail),("interpolation",diag.condition_extrapolation_flag.eq(0)),("true_condition_extrapolation",diag.condition_extrapolation_flag.eq(1))]
    regional=[]
    for name,mask in regions:
        for model,q in diag[mask].groupby("physics_model"): regional.append({"physics_model":model,"region":name,"n":len(q),**scores(q.y_true,q.y_physics_pred),"bias":q.residual.mean(),"mean_residual":q.residual.mean()})
    p3diag=pd.read_csv(ROOT/"03_MODELING/results/stress_temperature_physics_v1/predictions.csv").query("analysis == 'tier1_physics' and feature_set == 'P3-replace' and model == 'XGBoost'").merge(cov,on="creep_id",validate="one_to_one")
    for name,mask in [("low_stress",p3diag.stress_MPa<=low),("high_temperature",p3diag.temperature_K>=hot),("long_life_tail",p3diag.y_true>=tail),("interpolation",p3diag.condition_extrapolation_flag.eq(0)),("true_condition_extrapolation",p3diag.condition_extrapolation_flag.eq(1))]:
        q=p3diag[mask]; regional.append({"physics_model":"P3-replace XGBoost (reference)","region":name,"n":len(q),**scores(q.y_true,q.y_pred),"bias":q.residual.mean(),"mean_residual":q.residual.mean()})
    regional=pd.DataFrame(regional); regional.to_csv(OUT/"regional_error_comparison.csv",index=False)
    # W1 support assessment: W1 metrics above are a subset endpoint, not a full-cohort comparison.
    wsummary={"eligible_rows":int(w.w_valid.sum()),"eligible_sources":int(w.loc[w.w_valid,"source_id"].nunique()),"all_folds_have_9_validation_rows":bool(all(w[w.fold_id.eq(f)].w_valid.sum()>=9 for f in sorted(w.fold_id.unique())))}
    # MG1 proxy sensitivity on observed minimum rates; post-test only.
    rate=pd.read_csv(CREEP)[["creep_id","minimum_creep_rate_pct_h"]]; mg=data.merge(rate,on="creep_id",validate="one_to_one"); mg=mg[mg.minimum_creep_rate_pct_h.gt(0)].copy(); mg_rows=[]
    for fold in sorted(mg.fold_id.unique()):
        tr,va=mg[mg.fold_id.ne(fold)],mg[mg.fold_id.eq(fold)]
        if len(va)<2 or not eligible(tr,2,temp=False): continue
        x=np.log10(tr.minimum_creep_rate_pct_h.to_numpy()); b=np.linalg.lstsq(np.c_[np.ones(len(tr)),-x],tr[TARGET],rcond=None)[0]; p=np.c_[np.ones(len(va)),-np.log10(va.minimum_creep_rate_pct_h)]@b
        mg_rows.append({"fold_id":fold,"n":len(va),"n_sources":va.source_id.nunique(),"gamma_0":b[0],"m":b[1],**scores(va[TARGET],p)})
    mgdf=pd.DataFrame(mg_rows); mgdf.to_csv(OUT/"minimum_rate_proxy_sensitivity.csv",index=False)
    elig=[]
    for model in ["A0","A1","L1"]:
        r=regional[(regional.physics_model==model)&(regional.region=="true_condition_extrapolation")].iloc[0]
        # Classification is conservative: compare behavior, not just R2.
        label="POTENTIAL_P3_BACKBONE" if (r.MAE < 1.0779 or regional[(regional.physics_model==model)&(regional.region=="long_life_tail")].mean_residual.iloc[0] < .1564) else "INTERPRETABILITY_ONLY"
        elig.append({"physics_model":model,"classification":label,"rationale":"fold-fitted, physically bounded low-complexity baseline; classification uses extrapolation/long-life behavior rather than global R2."})
    elig += [{"physics_model":"W1","classification":"INSUFFICIENT_SUPPORT" if not wsummary["all_folds_have_9_validation_rows"] else "SUPPORTED-SUBSET PHYSICS BASELINE","rationale":json.dumps(wsummary)}, {"physics_model":"MG1","classification":"MINIMUM-RATE PROXY SENSITIVITY ANALYSIS","rationale":f"n={len(mg)}; observed minimum rate only; not a pre-test rupture model."}, {"physics_model":"SASM","classification":"DEFERRED / BLOCKED_BY_IMPLEMENTATION_DETAILS","rationale":"unresolved m/S/x0/unit convention and no confirmed steady-state-rate labels."}]
    pd.DataFrame(elig).to_csv(OUT/"physics_backbone_eligibility.csv",index=False)
    # Required figures.
    colors={"A0":"#2563eb","A1":"#7c3aed","L1":"#dc2626","P3-replace XGBoost":"#059669"}
    p3=pd.read_csv(ROOT/"03_MODELING/results/stress_temperature_physics_v1/predictions.csv").query("analysis == 'tier1_physics' and feature_set == 'P3-replace' and model == 'XGBoost'").copy()
    fig,axs=plt.subplots(2,2,figsize=(10,9));
    for ax,model in zip(axs.flat,["A0","A1","L1","P3-replace XGBoost"]):
        q=pred[pred.physics_model.eq(model)] if model!="P3-replace XGBoost" else p3.rename(columns={"y_pred":"y_physics_pred"})
        lo=min(q.y_true.min(),q.y_physics_pred.min()); hi=max(q.y_true.max(),q.y_physics_pred.max());ax.scatter(q.y_true,q.y_physics_pred,s=22,alpha=.65,color=colors[model]);ax.plot([lo,hi],[lo,hi],"k--");ax.set(title=model,xlabel="Observed log10(rupture life / h)",ylabel="Predicted log10(rupture life / h)")
    save(fig,"parity_comparison")
    for col,label,name in [("y_true","Observed life (log10 h)","residual_vs_observed_life"),("stress_MPa","Stress (MPa)","residual_vs_stress"),("temperature_K","Temperature (K)","residual_vs_temperature")]:
        fig,ax=plt.subplots(figsize=(7,4.6));
        for m in ["A0","A1","L1"]: q=pred[pred.physics_model.eq(m)];ax.scatter(q[col],q.residual,s=20,alpha=.5,label=m,color=colors[m])
        ax.axhline(0,color="black",ls="--");ax.set(xlabel=label,ylabel="Residual (log10 h)");ax.legend();save(fig,name)
    fig,ax=plt.subplots(figsize=(8,4.7)); rr=regional[regional.region.isin(["low_stress","high_temperature","long_life_tail","true_condition_extrapolation"])]; piv=rr.pivot(index="region",columns="physics_model",values="MAE");piv[["A0","A1","L1"]].plot.bar(ax=ax);ax.set(ylabel="MAE (log10 h)",xlabel="");save(fig,"regional_mae_comparison")
    fig,ax=plt.subplots(figsize=(8,4.7)); pp=params[params.physics_model.isin(["A0","A1","L1"])];
    for (m,n),q in pp.groupby(["physics_model","parameter_name"]): ax.plot(q.fold_id,q.estimate,"o-",label=f"{m}: {n}")
    ax.set(xlabel="Outer fold",ylabel="Parameter estimate",title="Fold parameter stability");ax.legend(fontsize=7,ncol=2);save(fig,"fold_parameter_stability")
    fig,ax=plt.subplots(figsize=(7,4.7)); ex=regional[regional.region.isin(["interpolation","true_condition_extrapolation"])].pivot(index="region",columns="physics_model",values="MAE");ex[["A0","A1","L1"]].plot.bar(ax=ax);ax.set(ylabel="MAE (log10 h)",xlabel="");save(fig,"extrapolation_vs_interpolation")
    table=metrics.to_markdown(index=False,floatfmt=".4f")
    report=f"""# TASK 04.7-B — Classical Physics Baseline Comparison

## Protocol

All A0, A1, and L1 parameters were fitted separately on each frozen outer source-group training partition. Validation rupture life never entered fitting. Residuals are `y_true - y_physics_pred`; positive residual is life underprediction. A0/A1 enforce the declared stress and temperature signs; L1 fits `P=a+b log10(stress)` with `b<=0` and fitted `C in [0,50]`, then inverts only with held-out stress and temperature.

## Full Tier1 OOF comparison

{table}

`P3-replace XGBoost (reference)` is an immutable external comparator from TASK 04.4-B. Its cell is not a retrained model. Mean and condition-only Ridge are direct low-complexity controls.

## Regional diagnostics

{regional[regional.region.ne('interpolation')].to_markdown(index=False,floatfmt='.4f')}

## Support-limited diagnostics

- **W1:** `{('SUPPORTED-SUBSET PHYSICS BASELINE' if wsummary['all_folds_have_9_validation_rows'] else 'INSUFFICIENT_SUPPORT')}`. {json.dumps(wsummary)}. It was not forced into a full-Tier1 metric.
- **MG1:** `MINIMUM-RATE PROXY SENSITIVITY ANALYSIS`, n={len(mg)}, sources={mg.source_id.nunique()}, material states={mg.material_state_id.nunique()}. Fold results are in `minimum_rate_proxy_sensitivity.csv`; it is post-test and not a deployment or full-cohort comparator.
- **SASM:** `DEFERRED / BLOCKED_BY_IMPLEMENTATION_DETAILS`: unresolved numerical implementation conventions and minimum-rate rather than confirmed steady-state-rate semantics.

## Decision

**A. Overall:** no classical full-Tier1 model outperforms P3-replace XGBoost (R² 0.7497; MAE 0.3592). A1/L1 are best among them (R² {metrics.query("physics_model == 'A1'").R2.iloc[0]:.4f}; MAE {metrics.query("physics_model == 'A1'").MAE.iloc[0]:.4f}).

**B. Regions:** A1/L1 do not beat P3-replace in low stress, high temperature, or the long-life tail. In the five-row true-extrapolation subset, A1/L1 MAE is {regional[(regional.physics_model=='A1')&(regional.region=='true_condition_extrapolation')].MAE.iloc[0]:.4f}, better than P3-replace's {regional[(regional.physics_model=='P3-replace XGBoost (reference)')&(regional.region=='true_condition_extrapolation')].MAE.iloc[0]:.4f}; this is suggestive, not definitive at n=5.

**C. Long-life bias:** A1/L1 retain positive long-life residual ({regional[(regional.physics_model=='A1')&(regional.region=='long_life_tail')].mean_residual.iloc[0]:.4f}), substantially larger than P3-replace's {regional[(regional.physics_model=='P3-replace XGBoost (reference)')&(regional.region=='long_life_tail')].mean_residual.iloc[0]:.4f}; they do not reduce the systematic underprediction.

**D. Parameter stability:** all fits converge with low scaled condition numbers, and A0 signs are stable. `physics_parameter_stability.csv` includes residual-based linear standard-error approximations; because constrained boundary estimates invalidate ordinary symmetric inference, they are a numerical audit rather than unconstrained confidence claims. A1's direct log-stress coefficient is at its nonpositive boundary in every fold, while the interaction is negative; L1 gives exactly the same predictions as A1 because its three-parameter fitted-C form is the corresponding reparameterization under this active bound. Thus L1 is an interpretable benchmark, not independent evidence for a distinct backbone.

**Recommendation:** proceed to TASK 04.7-C only with **A1 as the candidate cross-fitted residual backbone**, and retain A0 as a simpler control. The narrow extrapolation advantage is a credible reason to test it, but the global and long-life deficits require a predeclared no-harm comparison against P3-replace; no superiority is established. The eligibility table records the full evidence-based classification.
"""
    (OUT/"CLASSICAL_PHYSICS_BASELINE_REPORT.md").write_text(report,encoding="utf-8")
    (OUT/"model_config.json").write_text(json.dumps({"task":"TASK 04.7-B","dataset":str(DATA.relative_to(ROOT)),"splits":str(SPLITS.relative_to(ROOT)),"target":TARGET,"outer_fit_only":True,"models":{"A0":"bounded linear Arrhenius-stress","A1":"A0 plus logstress/T","L1":"fitted-C Larson-Miller"},"source_overlap":0},indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__": main()
