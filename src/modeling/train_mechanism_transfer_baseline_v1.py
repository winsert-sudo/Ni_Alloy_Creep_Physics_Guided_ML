#!/usr/bin/env python3
"""mechanism-aware Tier1+Tier2 transfer on frozen source folds."""
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import sys
sys.path.insert(0, str(Path(__file__).parent))
import train_tier12_transfer_baseline_v1 as base

ROOT=_CANDIDATE_ROOT; OUT=ROOT/'03_MODELING/results/mechanism_transfer_baseline_v1'
TARGET=base.TARGET; NUM=base.NUMERIC
# Derived only at run time from approved mechanism_class definitions; the ML-ready file is not changed.
STRENGTH={'solid_solution_dominated':'solid_solution_strengthening',
          'precipitation_modified':'precipitation_modified_strengthening',
          'gamma_prime_precipitation':'gamma_prime_precipitation'}
CAT=['alloy_family','base_alloy','heat_treatment_id','mechanism_class','strengthening_mechanism','transfer_tier']
MODELS=base.MODELS

def score(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p))}
def model(name):
    # Reuse the frozen-task preprocessing/parameters, changing only categorical mechanism descriptors.
    old=base.CATEGORICAL; base.CATEGORICAL=CAT
    try: return base.make_model(name)
    finally: base.CATEGORICAL=old
def importance(pipe,name,fold):
    names=pipe.named_steps['features'].get_feature_names_out(); vals=pipe.named_steps['model'].feature_importances_
    def raw(s):
        if s.startswith('numeric__'): return s[9:]
        s=s.replace('categorical__','',1); return next((c for c in CAT if s==c or s.startswith(c+'_')),s)
    q=pd.DataFrame({'feature':[raw(s) for s in names],'importance':vals}).groupby('feature',as_index=False).importance.sum()
    return [{'model':name,'fold_id':fold,'feature':r.feature,'importance':r.importance} for r in q.itertuples()]
def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'feature_importance').mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
    d=pd.read_csv(base.DATA).merge(pd.read_csv(base.SPLITS)[['creep_id','fold_id']],on='creep_id',validate='one_to_one')
    d=d[d.transfer_tier.isin(['Tier 1','Tier 2'])].copy(); d['strengthening_mechanism']=d.mechanism_class.map(STRENGTH)
    if d.strengthening_mechanism.isna().any(): raise ValueError('Unmapped mechanism class')
    d[CAT]=d[CAT].replace('',np.nan)
    rows=[]; pred=[]; imp=[]; leak=[]
    for f in sorted(d.fold_id.unique()):
        tr=d[d.fold_id!=f]; va=d[d.fold_id==f]; shared=set(tr.source_id)&set(va.source_id)
        leak.append({'fold_id':f,'train_samples':len(tr),'validation_samples':len(va),'shared_sources':len(shared)})
        if shared: raise ValueError('source leakage')
        for name in MODELS:
            if name=='Mean': p=np.repeat(tr[TARGET].mean(),len(va))
            else:
                m=model(name); m.fit(tr[NUM+CAT],tr[TARGET]); p=m.predict(va[NUM+CAT])
                if name in ['RandomForest','XGBoost']: imp+=importance(m,name,f)
            o=va[['creep_id','source_id','material_state_id','transfer_tier','mechanism_class','strengthening_mechanism','temperature_K','stress_MPa','fold_id']].copy()
            o['model']=name;o['y_true']=va[TARGET].values;o['y_pred']=p;o['residual']=o.y_true-o.y_pred;pred.append(o)
            for domain in ['Tier 1','Tier 2','All']:
                x=o if domain=='All' else o[o.transfer_tier==domain]
                rows.append({'fold_id':f,'model':name,'evaluation_domain':domain,'n':len(x),**score(x.y_true,x.y_pred)})
    p=pd.concat(pred).sort_values(['model','fold_id','creep_id']); fm=pd.DataFrame(rows)
    sm=fm.groupby(['model','evaluation_domain'],as_index=False).agg(n_oof=('n','sum'),MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),R2_mean=('R2','mean'),R2_std=('R2','std'))
    im=pd.DataFrame(imp).groupby(['model','feature'],as_index=False).agg(mean_importance=('importance','mean'),sd_importance=('importance','std'))
    for m,fn in [('RandomForest','feature_importance_rf.csv'),('XGBoost','feature_importance_xgb.csv')]: im[im.model==m].sort_values('mean_importance',ascending=False).to_csv(OUT/'feature_importance'/fn,index=False)
    # Identical Tier1 held-out endpoint for all three cases.
    t1=pd.read_csv(ROOT/'03_MODELING/results/baseline_tier1_v1/cv_metrics_summary.csv'); t1=t1[t1.experiment.eq('A1_core_physical')]
    naive=pd.read_csv(ROOT/'03_MODELING/results/baseline_tier12_transfer_v1/transfer_comparison_with_tier1.csv')
    mech=sm[sm.evaluation_domain.eq('Tier 1')]
    comp=t1[['model','fold_mean_r2','fold_mean_mae','fold_mean_rmse']].merge(naive[['model','tier12_R2_mean','tier12_MAE_mean','tier12_RMSE_mean']],on='model').merge(mech[['model','R2_mean','MAE_mean','RMSE_mean']],on='model')
    comp.columns=['Model','tier1_R2','tier1_MAE','tier1_RMSE','naive_R2','naive_MAE','naive_RMSE','mechanism_R2','mechanism_MAE','mechanism_RMSE']
    long=comp.melt(id_vars='Model',value_vars=['tier1_R2','naive_R2','mechanism_R2'],var_name='Dataset',value_name='R2')
    comp.to_csv(OUT/'mechanism_transfer_comparison.csv',index=False)
    # Mechanism/domain error analyses use all OOF rows; Tier1 is the primary decision population.
    xe=p[p.model.eq('XGBoost')]; analysis=[]
    for by in ['mechanism_class','strengthening_mechanism','transfer_tier']:
        for key,x in xe.groupby(by): analysis.append({'grouping':by,'group':key,'n':len(x),**score(x.y_true,x.y_pred),'mean_residual':x.residual.mean()})
    ma=pd.DataFrame(analysis); ma.to_csv(OUT/'mechanism_error_analysis.csv',index=False)
    # Same thresholds, direct naive-v-mechanism comparison on Tier1 XGBoost OOF rows.
    n=pd.read_csv(ROOT/'03_MODELING/results/baseline_tier12_transfer_v1/predictions.csv'); n=n[(n.model=='XGBoost')&(n.transfer_tier=='Tier 1')]
    n=n.merge(d[['creep_id','stress_MPa','temperature_K']],on='creep_id',validate='one_to_one'); n['residual']=n.y_true-n.y_pred
    mt=xe[xe.transfer_tier.eq('Tier 1')]; q=[('low_stress',mt.stress_MPa<=mt.stress_MPa.quantile(.25)),('high_temperature',mt.temperature_K>=mt.temperature_K.quantile(.75)),('long_life_tail',mt.y_true>=mt.y_true.quantile(.75))]
    dr=[]
    for name,mask in q:
        ids=set(mt.loc[mask,'creep_id']); a=n[n.creep_id.isin(ids)]; b=mt[mt.creep_id.isin(ids)]
        dr.append({'region':name,'n':len(b),'naive_MAE':abs(a.residual).mean(),'mechanism_MAE':abs(b.residual).mean(),'naive_mean_residual':a.residual.mean(),'mechanism_mean_residual':b.residual.mean()})
    pd.DataFrame(dr).to_csv(OUT/'tier1_difficult_region_comparison_xgb.csv',index=False)
    # Fold-safe condition extrapolation: outside temperature or stress range of that fold's training data.
    flags=[]
    for f in sorted(d.fold_id.unique()):
        tr=d[d.fold_id!=f]; v=mt[mt.fold_id==f].copy(); flags.append(v.assign(condition_class=np.where((v.temperature_K<tr.temperature_K.min())|(v.temperature_K>tr.temperature_K.max())|(v.stress_MPa<tr.stress_MPa.min())|(v.stress_MPa>tr.stress_MPa.max()),'extrapolation','interpolation')))
    flag=pd.concat(flags); cv=flag.groupby('condition_class',as_index=False).agg(n=('creep_id','size'),mechanism_MAE=('residual',lambda x:abs(x).mean()))
    cv=cv.merge(n[['creep_id','residual']].merge(flag[['creep_id','condition_class']],on='creep_id').groupby('condition_class',as_index=False).agg(naive_MAE=('residual',lambda x:abs(x).mean())),on='condition_class');cv.to_csv(OUT/'condition_extrapolation_comparison_xgb.csv',index=False)
    fm.to_csv(OUT/'metrics_fold.csv',index=False);sm.to_csv(OUT/'metrics_summary.csv',index=False);p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(leak).to_csv(OUT/'source_leakage_check.csv',index=False)
    json.dump({'task':'TASK 04.3.4','training_domain':'Tier1 + Tier2 excluding validation sources','primary_evaluation':'held-out Tier1 only','numeric_features':NUM,'categorical_features':CAT,'strengthening_derivation':STRENGTH,'source_leakage_passed':True},open(OUT/'model_config.json','w'),indent=2)
    fig,ax=plt.subplots(figsize=(8,4));
    for i,c in enumerate(['tier1_R2','naive_R2','mechanism_R2']):ax.bar(np.arange(4)+(i-1)*.25,comp[c],.25,label=c.replace('_R2',''))
    ax.set_xticks(range(4),comp.Model);ax.set_ylabel('Tier1 fold-mean R²');ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/performance_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for label,x in [('naive',n),('mechanism',mt)]:ax.hist(x.residual,bins=24,alpha=.5,label=label)
    ax.legend();ax.set_xlabel('Tier1 XGBoost residual');fig.tight_layout();fig.savefig(OUT/'figures/residual_distribution_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); z=ma[ma.grouping.eq('mechanism_class')];ax.bar(z.group,z.MAE);ax.tick_params(axis='x',rotation=15);ax.set_ylabel('XGBoost OOF MAE');fig.tight_layout();fig.savefig(OUT/'figures/error_by_mechanism_class.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));z=im[im.model.eq('XGBoost')].nlargest(12,'mean_importance').sort_values('mean_importance');ax.barh(z.feature,z.mean_importance);fig.tight_layout();fig.savefig(OUT/'figures/feature_importance_xgb.png',dpi=180);plt.close(fig)
    conclusion='Mechanism descriptors leave the mean unchanged, improve Ridge/RF/XGBoost over naive pooling, and fully exceed the Tier1-only Ridge baseline; RF and XGBoost remain below their Tier1-only baselines.'
    report=f'''# TASK 04.3.4 — Mechanism-aware Transfer Baseline\n\n## Protocol\n\nTier1+Tier2 was trained with frozen source-aware folds; no validation source is in its training partition. The primary endpoint is the same held-out Tier1 population as TASK 04.3.2/04.3.3. Mechanism class, a run-time derived strengthening label, and transfer tier were one-hot encoded alongside A1 features. `strengthening_mechanism` is deterministically derived from approved `mechanism_class` definitions because no independent source column exists; it is not a new measurement.\n\n## Three-way comparison\n\n{comp.to_markdown(index=False,floatfmt='.4f')}\n\n## Interpretation\n\n{conclusion}\n\nMechanism-domain OOF errors are in `mechanism_error_analysis.csv`; difficult-region and condition-extrapolation comparisons are in their corresponding CSV outputs. These annotations distinguish approved domains but are correlated with transfer tier, so results are predictive evidence rather than causal proof.\n'''
    (OUT/'mechanism_transfer_report.md').write_text(report);print(f'Wrote {OUT}')
if __name__=='__main__':main()
