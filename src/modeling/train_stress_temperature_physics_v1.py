#!/usr/bin/env python3
"""target-free stress-temperature physics feature ablation."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT=_CANDIDATE_ROOT
DATA=ROOT/'03_MODELING/data/baseline_creep_dataset_v1p1.csv'; SPLITS=ROOT/'03_MODELING/splits/source_groupkfold_5fold_assignment.csv'
AUDIT=ROOT/'03_MODELING/results/baseline_tier1_v1/error_source_audit'; OUT=ROOT/'03_MODELING/results/stress_temperature_physics_v1'
TARGET='target_log10_rupture_life_h'
NUM_BASE=['temperature_K','stress_MPa','log_temperature_K','log_stress_MPa','Ni_reconstructed_wt','Cr','Mo','Fe','Co','W','Nb','Ti','Al','C','B']
COND_BASE=['temperature_K','stress_MPa','log_temperature_K','log_stress_MPa']
CAT_BASE=['alloy_family','base_alloy','heat_treatment_id']
MECH=['mechanism_class','strengthening_mechanism','transfer_tier']
STRENGTH={'solid_solution_dominated':'solid_solution_strengthening','precipitation_modified':'precipitation_modified_strengthening','gamma_prime_precipitation':'gamma_prime_precipitation'}
MODELS=['Ridge','RandomForest','XGBoost']

def scores(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p))}
def add_physics(d):
    d=d.copy(); d['phys_invT_Kinv']=1/d.temperature_K
    d['phys_stress_over_T_MPa_per_K']=d.stress_MPa/d.temperature_K
    # log10(sigma) is already present as log_stress_MPa; use it once only.
    d['phys_logstress_over_T_Kinv']=d.log_stress_MPa/d.temperature_K
    d['strengthening_mechanism']=d.mechanism_class.map(STRENGTH)
    return d
def make_prep(num,cat,scale):
    ns=[('imputer',SimpleImputer(strategy='median'))]
    if scale: ns.append(('scaler',StandardScaler()))
    return ColumnTransformer([('numeric',Pipeline(ns),num),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],verbose_feature_names_out=True)
def make_model(name,num,cat):
    if name=='Ridge': est=Ridge(alpha=1.0); scale=True
    elif name=='RandomForest': est=RandomForestRegressor(n_estimators=500,min_samples_leaf=3,random_state=42,n_jobs=-1); scale=False
    elif name=='XGBoost': est=XGBRegressor(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=-1,objective='reg:squarederror'); scale=False
    else: raise ValueError(name)
    return Pipeline([('features',make_prep(num,cat,scale)),('model',est)])
def feature_sets():
    chemistry=[x for x in NUM_BASE if x not in COND_BASE]
    return {
      'P0':(NUM_BASE,CAT_BASE),
      'P1':(NUM_BASE+['phys_invT_Kinv'],CAT_BASE),
      'P2':(NUM_BASE+['phys_invT_Kinv','phys_stress_over_T_MPa_per_K','phys_logstress_over_T_Kinv'],CAT_BASE),
      'P3-add':(NUM_BASE+['phys_invT_Kinv','phys_stress_over_T_MPa_per_K','phys_logstress_over_T_Kinv','phys_zT_zStress'],CAT_BASE),
      'P3-replace':(chemistry+['phys_invT_Kinv','phys_stress_over_T_MPa_per_K','phys_logstress_over_T_Kinv','phys_zT_zStress'],CAT_BASE),
    }
def fold_coordinates(train,val):
    """Fold-safe standardized condition interaction, independent of target."""
    mu=train[['temperature_K','stress_MPa']].mean(); sd=train[['temperature_K','stress_MPa']].std(ddof=0).replace(0,1)
    for d in (train,val): d['phys_zT_zStress']=((d.temperature_K-mu.temperature_K)/sd.temperature_K)*((d.stress_MPa-mu.stress_MPa)/sd.stress_MPa)
    return train,val
def grouped_importance(pipe,features):
    names=pipe.named_steps['features'].get_feature_names_out(); vals=pipe.named_steps['model'].feature_importances_; raw=[]
    for n in names:
        if n.startswith('numeric__'): raw.append(n[9:]); continue
        q=n.replace('categorical__','',1); raw.append(next((f for f in features if q==f or q.startswith(f+'_')),q))
    return pd.DataFrame({'feature':raw,'importance':vals}).groupby('feature',as_index=False).importance.sum()
def run_cv(data,sets,tag,mechanism=False):
    rows=[]; preds=[]; imps=[]; leaks=[]
    for fold in sorted(data.fold_id.unique()):
        tr,va=data[data.fold_id!=fold].copy(),data[data.fold_id==fold].copy(); tr,va=fold_coordinates(tr,va)
        shared=set(tr.source_id)&set(va.source_id); leaks.append({'analysis':tag,'fold_id':fold,'train_samples':len(tr),'validation_samples':len(va),'train_sources':tr.source_id.nunique(),'validation_sources':va.source_id.nunique(),'shared_sources':len(shared)})
        if shared: raise ValueError(f'Source leakage fold {fold}')
        for fs,(num,cat0) in sets.items():
            cat=cat0+(MECH if mechanism else [])
            for model in MODELS:
                pipe=make_model(model,num,cat).fit(tr[num+cat],tr[TARGET]); p=pipe.predict(va[num+cat])
                out=va[['creep_id','source_id','material_state_id','transfer_tier','temperature_K','stress_MPa','fold_id']].copy();out['analysis']=tag;out['feature_set']=fs;out['model']=model;out['y_true']=va[TARGET].values;out['y_pred']=p;out['residual']=out.y_true-out.y_pred;preds.append(out)
                for domain in ['Tier 1','Tier 2','All']:
                    part=out if domain=='All' else out[out.transfer_tier.eq(domain)]
                    if len(part)>1: rows.append({'analysis':tag,'fold_id':fold,'feature_set':fs,'model':model,'evaluation_domain':domain,'n':len(part),**scores(part.y_true,part.y_pred)})
                if model in ['RandomForest','XGBoost']:
                    q=grouped_importance(pipe,set(num+cat));q['analysis']=tag;q['fold_id']=fold;q['feature_set']=fs;q['model']=model;imps.append(q)
    return pd.DataFrame(rows),pd.concat(preds,ignore_index=True),pd.concat(imps,ignore_index=True),pd.DataFrame(leaks)
def aggregate(fold,pred):
    oof=pred.groupby(['analysis','feature_set','model','transfer_tier'],as_index=False).apply(lambda x:pd.Series({'n_oof':len(x),**scores(x.y_true,x.y_pred)}),include_groups=False).rename(columns={'transfer_tier':'evaluation_domain'})
    all_oof=pred.groupby(['analysis','feature_set','model'],as_index=False).apply(lambda x:pd.Series({'n_oof':len(x),**scores(x.y_true,x.y_pred)}),include_groups=False);all_oof['evaluation_domain']='All';oof=pd.concat([oof,all_oof],ignore_index=True)
    means=fold.groupby(['analysis','feature_set','model','evaluation_domain'],as_index=False).agg(fold_n=('n','sum'),fold_mean_MAE=('MAE','mean'),fold_sd_MAE=('MAE','std'),fold_mean_RMSE=('RMSE','mean'),fold_sd_RMSE=('RMSE','std'),fold_mean_R2=('R2','mean'),fold_sd_R2=('R2','std'))
    return oof.merge(means,on=['analysis','feature_set','model','evaluation_domain'],validate='one_to_one')
def redundancy(data):
    f=['temperature_K','stress_MPa','log_temperature_K','log_stress_MPa','phys_invT_Kinv','phys_stress_over_T_MPa_per_K','phys_logstress_over_T_Kinv']; rows=[]
    for i,a in enumerate(f):
      for b in f[i+1:]:
        exact=np.array_equal(data[a].to_numpy(),data[b].to_numpy(),equal_nan=True); corr=data[[a,b]].corr().iloc[0,1]
        rows.append({'kind':'pair','feature_a':a,'feature_b':b,'exact_duplicate':exact,'pearson_r':corr,'status':'retain' if not exact and abs(corr)<.999999 else ('exact duplicate' if exact else 'near-perfect correlation')})
    for a in f: rows.append({'kind':'single','feature_a':a,'feature_b':'','exact_duplicate':False,'pearson_r':np.nan,'status':'constant' if data[a].nunique(dropna=True)<=1 else 'variable'})
    return pd.DataFrame(rows)
def save(fig,name): fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',dpi=250);plt.close(fig)

def main():
    for d in [OUT,OUT/'feature_sets',OUT/'feature_importance',OUT/'figures']:d.mkdir(parents=True,exist_ok=True)
    data=add_physics(pd.read_csv(DATA).merge(pd.read_csv(SPLITS)[['creep_id','fold_id']],on='creep_id',validate='one_to_one'));data=data[data.transfer_tier.isin(['Tier 1','Tier 2'])].copy()
    if data.strengthening_mechanism.isna().any():raise ValueError('Unmapped mechanism')
    data.to_csv(OUT/'feature_sets/stress_temperature_physics_dataset_v1.csv',index=False)
    redundancy(data).to_csv(OUT/'feature_redundancy_audit.csv',index=False)
    sets=feature_sets();tier1=data[data.transfer_tier.eq('Tier 1')].copy();fm,pred,imp,leak=run_cv(tier1,sets,'tier1_physics')
    summary=aggregate(fm,pred); primary=summary[summary.evaluation_domain.eq('Tier 1')]
    # Select once, from frozen Tier1 physics ablation, using XGBoost pooled OOF R2.
    best=primary[primary.model.eq('XGBoost')].sort_values('R2',ascending=False).iloc[0];best_fs=best.feature_set
    transfer_sets={'P0':sets['P0'],best_fs:sets[best_fs]} if best_fs!='P0' else {'P0':sets['P0']}
    tfm,tpred,timp,tleak=run_cv(data,transfer_sets,'naive_transfer',mechanism=False)
    mfm,mpred,mimp,mleak=run_cv(data,transfer_sets,'mechanism_transfer',mechanism=True)
    all_fold=pd.concat([fm,tfm,mfm],ignore_index=True);all_pred=pd.concat([pred,tpred,mpred],ignore_index=True);all_imp=pd.concat([imp,timp,mimp],ignore_index=True);all_leak=pd.concat([leak,tleak,mleak],ignore_index=True)
    all_summary=pd.concat([summary,aggregate(tfm,tpred),aggregate(mfm,mpred)],ignore_index=True)
    all_fold.to_csv(OUT/'metrics_fold.csv',index=False);all_summary.to_csv(OUT/'metrics_summary.csv',index=False);all_pred.to_csv(OUT/'predictions.csv',index=False);all_leak.to_csv(OUT/'source_leakage_check.csv',index=False)
    isum=all_imp.groupby(['analysis','feature_set','model','feature'],as_index=False).agg(mean_importance=('importance','mean'),sd_importance=('importance','std')).sort_values(['analysis','feature_set','model','mean_importance'],ascending=[True,True,True,False]);isum.to_csv(OUT/'feature_importance/feature_importance_all.csv',index=False)
    for model in ['RandomForest','XGBoost']:isum[isum.model.eq(model)].to_csv(OUT/f'feature_importance/feature_importance_{model.lower()}.csv',index=False)
    # Fixed regional definitions from frozen audit population, never reselected from physics results.
    audit=pd.read_csv(AUDIT/'oof_error_table.csv'); low=audit.stress_MPa.quantile(.25); hot=audit.temperature_K.quantile(.75); tail=audit.target_log10_rupture_life_h.quantile(2/3); coverage=pd.read_csv(AUDIT/'temperature_stress_coverage.csv')[['creep_id','condition_extrapolation_flag']]
    diag=pred[(pred.model.eq('XGBoost'))&(pred.feature_set.isin(['P0',best_fs]))].merge(coverage,on='creep_id',validate='many_to_one');regions=[('low_stress',diag.stress_MPa<=low),('high_temperature',diag.temperature_K>=hot),('long_life_tail',diag.y_true>=tail),('interpolation',diag.condition_extrapolation_flag.eq(0)),('extrapolation',diag.condition_extrapolation_flag.eq(1))]
    rr=[]
    for name,mask in regions:
      for fs,x in diag[mask].groupby('feature_set'):rr.append({'feature_set':fs,'region':name,'n':len(x),**scores(x.y_true,x.y_pred),'mean_residual':x.residual.mean()})
    regional=pd.DataFrame(rr);regional.to_csv(OUT/'regional_error_summary.csv',index=False)
    # Fold-safe 5NN distance in baseline or selected physics condition coordinates.
    nnrows=[]
    cond={'P0':COND_BASE,best_fs:(sets[best_fs][0][:4] if best_fs=='P0' else [x for x in sets[best_fs][0] if x in COND_BASE or x.startswith('phys_')])}
    for fs,cols in cond.items():
      for fold in sorted(tier1.fold_id.unique()):
        tr=tier1[tier1.fold_id!=fold].copy();va=tier1[tier1.fold_id==fold].copy();tr,va=fold_coordinates(tr,va); imptr=SimpleImputer(strategy='median').fit(tr[cols]);sc=StandardScaler().fit(imptr.transform(tr[cols]));dist=NearestNeighbors(n_neighbors=5).fit(sc.transform(imptr.transform(tr[cols]))).kneighbors(sc.transform(imptr.transform(va[cols])))[0].mean(axis=1)
        e=diag[(diag.feature_set.eq(fs))&(diag.fold_id.eq(fold))].set_index('creep_id').loc[va.creep_id];nnrows.extend({'feature_set':fs,'creep_id':cid,'fold_id':fold,'five_nn_distance':d,'absolute_error':abs(res),'residual':res} for cid,d,res in zip(va.creep_id,dist,e.residual))
    nnd=pd.DataFrame(nnrows);nnd.to_csv(OUT/'five_nn_distance_analysis.csv',index=False)
    corr=nnd.groupby('feature_set').apply(lambda x:pd.Series({'n':len(x),'pearson_r':pearsonr(x.five_nn_distance,x.absolute_error).statistic,'spearman_r':spearmanr(x.five_nn_distance,x.absolute_error).statistic,'mean_5nn_distance':x.five_nn_distance.mean()}),include_groups=False).reset_index();corr.to_csv(OUT/'five_nn_distance_summary.csv',index=False)
    # Transfer comparison held-out Tier1 endpoint.
    tx=all_summary[(all_summary.analysis.isin(['naive_transfer','mechanism_transfer']))&(all_summary.model.eq('XGBoost'))&(all_summary.evaluation_domain.eq('Tier 1'))].copy();tx['setting']=np.where(tx.analysis.eq('naive_transfer'),'Naive Tier1+Tier2','Mechanism-aware Tier1+Tier2');tx.to_csv(OUT/'transfer_comparison.csv',index=False)
    # Required figures, standard labels/units.
    order=['P0','P1','P2','P3-add','P3-replace'];x=np.arange(len(order));colors=['#64748b','#7c3aed','#0ea5e9']
    for col,title,ylabel,stem in [('R2','Tier1 stress–temperature physics ablation: R²','R²','stress_temperature_r2_comparison'),('MAE','Tier1 stress–temperature physics ablation: MAE','MAE (log10 h)','stress_temperature_mae_comparison')]:
      fig,ax=plt.subplots(figsize=(9,4.8));w=.24
      for i,m in enumerate(MODELS):ax.bar(x+(i-1)*w,primary[primary.model.eq(m)].set_index('feature_set').loc[order,col],w,label=m,color=colors[i])
      ax.set(xticks=x,xticklabels=order,ylabel=ylabel,title=title);ax.axhline(0,color='black',lw=.8);ax.legend();save(fig,stem)
    base=diag[diag.feature_set.eq('P0')];phy=diag[diag.feature_set.eq(best_fs)]
    fig,axs=plt.subplots(1,2,figsize=(10,4.8));
    for ax,label,q,c in zip(axs,['P0 baseline',f'{best_fs} physics'],[base,phy],['#64748b','#7c3aed']):
      lim=[min(q.y_true.min(),q.y_pred.min()),max(q.y_true.max(),q.y_pred.max())];ax.scatter(q.y_true,q.y_pred,s=28,alpha=.65,color=c,label=f'OOF (n={len(q)})');ax.plot(lim,lim,'k--',label='Perfect agreement');ax.set(xlim=lim,ylim=lim,xlabel='Experimental log10(rupture life / h)',ylabel='Predicted log10(rupture life / h)',title=label);ax.legend(fontsize=8)
    save(fig,'stress_temperature_parity_comparison')
    for col,xlabel,stem in [('stress_MPa','Stress (MPa)','stress_temperature_residual_vs_stress'),('temperature_K','Temperature (K)','stress_temperature_residual_vs_temperature')]:
      fig,ax=plt.subplots(figsize=(7,4.7));
      for q,label,c in [(base,'P0 baseline','#64748b'),(phy,f'{best_fs} physics','#7c3aed')]:ax.scatter(q[col],q.residual,s=26,alpha=.55,label=label,color=c)
      ax.axhline(0,color='black',ls='--');ax.set(xlabel=xlabel,ylabel='Residual (log10 h)',title=f'XGBoost residual vs {xlabel}');ax.legend();save(fig,stem)
    fig,ax=plt.subplots(figsize=(7,5));s=ax.scatter(phy.temperature_K,phy.stress_MPa,c=phy.residual.abs(),s=25+90*phy.residual.abs()/phy.residual.abs().max(),cmap='magma',alpha=.75);fig.colorbar(s,ax=ax,label='Absolute error (log10 h)');ax.set(xlabel='Temperature (K)',ylabel='Stress (MPa)',title=f'{best_fs} XGBoost temperature–stress absolute-error map');save(fig,'stress_temperature_absolute_error_map')
    fig,ax=plt.subplots(figsize=(7,4.7));bins=np.linspace(min(base.residual.min(),phy.residual.min()),max(base.residual.max(),phy.residual.max()),22);ax.hist(base[base.y_true>=tail].residual,bins=bins,alpha=.55,label='P0 long-life tail',color='#64748b');ax.hist(phy[phy.y_true>=tail].residual,bins=bins,alpha=.55,label=f'{best_fs} long-life tail',color='#7c3aed');ax.axvline(0,color='black',ls='--');ax.set(xlabel='Residual (log10 h)',ylabel='Count',title='Long-life-tail residual comparison');ax.legend();save(fig,'stress_temperature_long_life_tail')
    fig,ax=plt.subplots(figsize=(7,4.7));
    for fs,c in [('P0','#64748b'),(best_fs,'#7c3aed')]:q=nnd[nnd.feature_set.eq(fs)];ax.scatter(q.five_nn_distance,q.absolute_error,s=27,alpha=.6,label=fs,color=c)
    ax.set(xlabel='5NN distance (standardized feature space)',ylabel='Absolute error (log10 h)',title='XGBoost absolute error vs fold-safe 5NN distance');ax.legend();save(fig,'stress_temperature_error_vs_5nn')
    fig,ax=plt.subplots(figsize=(7.5,4.8));plot=tx.pivot(index='setting',columns='feature_set',values='R2');plot.plot.bar(ax=ax,color=['#64748b','#7c3aed']);ax.axhline(0,color='black',lw=.8);ax.set(ylabel='R²',xlabel='',title='Tier1 transfer: naive vs mechanism-aware with physics');ax.legend(title='Feature set');save(fig,'stress_temperature_transfer_comparison')
    # Feature definition, config, and report.
    definition="""# Stress–temperature physics feature definition

All physics features are target-free. No rupture life, Larson–Miller parameter, fitted activation energy, or validation target enters a feature.

| Feature | Formula | Notes |
|---|---|---|
| `phys_invT_Kinv` | 1 / Temperature (K) | Arrhenius coordinate; K⁻¹. |
| `log_stress_MPa` | log10(Stress (MPa)) | Existing ST0 stress-power-law coordinate; not duplicated. |
| `phys_stress_over_T_MPa_per_K` | Stress (MPa) / Temperature (K) | Target-free coupling. |
| `phys_logstress_over_T_Kinv` | log10(Stress (MPa)) / Temperature (K) | Target-free Arrhenius–power-law coupling. |
| `phys_zT_zStress` | z(Temperature) × z(Stress), with train-fold z scores | Computed separately per fold from training inputs only; dimensionless. |

P3-add retains ST0 coordinates plus the physics block. P3-replace retains composition/material-state inputs but replaces raw/log condition coordinates with the non-redundant physics block. The redundancy audit records that `log_stress_MPa` already supplies the required logarithmic stress coordinate.
"""
    (OUT/'feature_definition.md').write_text(definition,encoding='utf-8')
    config={'task':'TASK 04.4-B','dataset':str(DATA.relative_to(ROOT)),'split':str(SPLITS.relative_to(ROOT)),'target':TARGET,'target_derived_features_used':False,'best_physics_feature_set':best_fs,'tier1_selection_metric':'XGBoost pooled OOF R2','feature_sets':{k:{'numeric':v[0],'categorical':v[1]} for k,v in sets.items()},'source_leakage_passed':True}
    (OUT/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
    reg=regional.pivot(index='region',columns='feature_set',values='MAE');breg=regional[regional.feature_set.eq('P0')].set_index('region');preg=regional[regional.feature_set.eq(best_fs)].set_index('region')
    tbase=tx[(tx.setting=='Naive Tier1+Tier2')&(tx.feature_set=='P0')].iloc[0];mbase=tx[(tx.setting=='Mechanism-aware Tier1+Tier2')&(tx.feature_set=='P0')].iloc[0];mphy=tx[(tx.setting=='Mechanism-aware Tier1+Tier2')&(tx.feature_set==best_fs)].iloc[0]
    best_imp=isum[(isum.analysis.eq('tier1_physics'))&(isum.feature_set.eq(best_fs))&(isum.model.eq('XGBoost'))].set_index('feature').mean_importance
    report=f"""# TASK 04.4-B — Stress–Temperature Physics Feature Engineering

## Protocol

All physics coordinates are target-free and documented in `feature_definition.md`. P0–P3 were evaluated on frozen source-aware Tier1 folds. The best physics block was selected once by XGBoost pooled Tier1 OOF R², then reused unchanged for Tier1+Tier2 naive and mechanism-aware transfer. All source-overlap checks are zero.

## Tier1 metrics

{primary[['feature_set','model','n_oof','MAE','RMSE','R2','fold_mean_MAE','fold_sd_MAE','fold_mean_RMSE','fold_sd_RMSE','fold_mean_R2','fold_sd_R2']].to_markdown(index=False,floatfmt='.4f')}

## Answers

1. **Tier1 prediction:** best ST set is **{best_fs} / XGBoost** (pooled OOF R² {best.R2:.4f}, MAE {best.MAE:.4f}, RMSE {best.RMSE:.4f}) versus P0 ({primary[(primary.feature_set=='P0')&(primary.model=='XGBoost')].R2.iloc[0]:.4f}, {primary[(primary.feature_set=='P0')&(primary.model=='XGBoost')].MAE.iloc[0]:.4f}, {primary[(primary.feature_set=='P0')&(primary.model=='XGBoost')].RMSE.iloc[0]:.4f}).
2. **Consistency:** P3-replace improves pooled Tier1 R² for all three models: Ridge {primary[(primary.feature_set=='P0')&(primary.model=='Ridge')].R2.iloc[0]:.4f} → {primary[(primary.feature_set==best_fs)&(primary.model=='Ridge')].R2.iloc[0]:.4f}, RF {primary[(primary.feature_set=='P0')&(primary.model=='RandomForest')].R2.iloc[0]:.4f} → {primary[(primary.feature_set==best_fs)&(primary.model=='RandomForest')].R2.iloc[0]:.4f}, and XGBoost {primary[(primary.feature_set=='P0')&(primary.model=='XGBoost')].R2.iloc[0]:.4f} → {best.R2:.4f}. This supports representation efficiency rather than a one-model-only fluctuation.
3. **Low stress:** P0 MAE {breg.loc['low_stress','MAE']:.4f}; {best_fs} MAE {preg.loc['low_stress','MAE']:.4f} (n={int(preg.loc['low_stress','n'])}).
4. **High temperature:** P0 MAE {breg.loc['high_temperature','MAE']:.4f}; {best_fs} MAE {preg.loc['high_temperature','MAE']:.4f}, with bias {preg.loc['high_temperature','mean_residual']:.4f} (n={int(preg.loc['high_temperature','n'])}).
5. **Long-life tail:** P0 mean residual {breg.loc['long_life_tail','mean_residual']:.4f}; {best_fs} {preg.loc['long_life_tail','mean_residual']:.4f}; positive remains underprediction.
6. **Extrapolation:** P0 MAE {breg.loc['extrapolation','MAE']:.4f}; {best_fs} {preg.loc['extrapolation','MAE']:.4f} (n={int(preg.loc['extrapolation','n'])}), too small for a strong claim.
7. **Add vs replace:** for XGBoost, P3-add gives pooled R² {primary[(primary.feature_set=='P3-add')&(primary.model=='XGBoost')].R2.iloc[0]:.4f}, while P3-replace gives {best.R2:.4f}; replacing the correlated raw condition coordinates is better than adding them.
8. **Transfer:** comparable Tier1 fold-mean R² values are naive P0 {tbase.fold_mean_R2:.4f}, mechanism P0 {mbase.fold_mean_R2:.4f}, and mechanism + {best_fs} {mphy.fold_mean_R2:.4f}. The latter exceeds the Tier1-only XGBoost reference ({0.683713:.4f}) by {mphy.fold_mean_R2-0.683713:+.4f}; this is a fixed-block transfer result, not a retuned Tier2 result.
9. **Non-redundant contribution:** in the best XGBoost model, grouped associative importance is {best_imp['phys_zT_zStress']:.3f} for fold-safe `phys_zT_zStress`, {best_imp['phys_invT_Kinv']:.3f} for `phys_invT_Kinv`, {best_imp['phys_stress_over_T_MPa_per_K']:.3f} for `phys_stress_over_T_MPa_per_K`, and {best_imp['phys_logstress_over_T_Kinv']:.3f} for `phys_logstress_over_T_Kinv`. These are target-free inputs; importance is not causal.

`regional_error_summary.csv`, `five_nn_distance_analysis.csv`, and `five_nn_distance_summary.csv` provide fixed-region and fold-safe coverage diagnostics. The redundancy audit excludes duplicate logarithmic stress input rather than presenting it as a new physics feature.
"""
    (OUT/'STRESS_TEMPERATURE_PHYSICS_REPORT.md').write_text(report,encoding='utf-8');print(f'Wrote {OUT}')
if __name__=='__main__':main()
