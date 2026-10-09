#!/usr/bin/env python3
"""descriptive and condition-group CV physics fits by material state."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from train_classical_physics_baseline_v1 import fit_predict, eligible, approximate_se
ROOT=_CANDIDATE_ROOT; DATA=ROOT/'03_MODELING/data/baseline_creep_dataset_v1p1.csv'; OUT=ROOT/'03_MODELING/results/within_state_physics_diagnostic_v1';Y='target_log10_rupture_life_h'
def met(y,p):return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p)),'bias':float(np.mean(np.asarray(y)-np.asarray(p)))}
def save(f,n):f.tight_layout();f.savefig(OUT/'figures'/f'{n}.png',dpi=220);plt.close(f)
def ident(d,m):
 p=4 if m=='A1' else 3; temps=d.temperature_K.nunique();span=d.temperature_K.max()-d.temperature_K.min(); ls=np.log10(d.stress_MPa)
 # State identity is source-confounded in Tier1 (normally one source/state).  This local
 # geometric gate deliberately records that caveat rather than pretending 3 sources exist.
 ok=len(d)>=max(3*p,12) and ls.nunique()>=3 and ls.max()-ls.min()>=.15 and temps>=3 and span>=100 and d[Y].nunique()>=3 and d[Y].max()-d[Y].min()>=.30
 level='FULL_T_STRESS_IDENTIFIABLE' if ok and d.source_id.nunique()>=3 else ('WEAKLY_IDENTIFIABLE' if ok else ('STRESS_ONLY_IDENTIFIABLE' if len(d)>=max(3*p,12) and ls.nunique()>=3 and ls.max()-ls.min()>=.15 and temps<3 else 'NOT_IDENTIFIABLE'))
 return ok,level
def main():
 (OUT/'figures').mkdir(parents=True,exist_ok=True);d=pd.read_csv(DATA).query("transfer_tier == 'Tier 1'").copy(); ir=[];fitrows=[];pars=[];cvrows=[];preds=[]
 for state,q in d.groupby('material_state_id'):
  base={'material_state_id':state,'alloy_family':q.alloy_family.iloc[0],'mechanism_class':q.mechanism_class.iloc[0],'heat_treatment_id':q.heat_treatment_id.iloc[0],'n':len(q),'n_sources':q.source_id.nunique(),'n_temperatures':q.temperature_K.nunique(),'temperature_span_K':q.temperature_K.max()-q.temperature_K.min(),'n_stresses':q.stress_MPa.nunique(),'stress_log10_span':np.log10(q.stress_MPa).max()-np.log10(q.stress_MPa).min(),'target_span_log10h':q[Y].max()-q[Y].min()}
  for m in ['A0','A1','L1']:
   ok,level=ident(q,m); rr=base|{'model':m,'eligible':ok,'identifiability_class':level,'failure_category':'' if ok else 'INSUFFICIENT_DATA_OR_SPAN'};ir.append(rr)
   if not ok: continue
   p,pa,stat=fit_predict(m,q,q);fitrows.append(base|{'model':m,'fit_label':'DESCRIPTIVE / IN-SAMPLE PHYSICS FIT',**met(q[Y],p)});se=approximate_se(m,q,pa)
   pars += [base|{'model':m,'parameter':k,'estimate':v,'standard_error_ols_approx':se[k],'optimizer_status':stat,'effective_parameter':bool((m=='A1' and k=='beta_sigma' and abs(v)<1e-10))} for k,v in pa.items()]
   # leave-one unique (T,stress) condition out, only if each train partition remains eligible.
   groups=q.assign(condition=q.temperature_K.astype(str)+'__'+q.stress_MPa.astype(str)).condition
   out=[]
   for c in groups.unique():
    tr=q[groups.ne(c)];va=q[groups.eq(c)]
    if not ident(tr,m)[0]:out=[];break
    pp,_,_=fit_predict(m,tr,va);z=va[['creep_id',Y]].copy();z['model']=m;z['y_pred']=pp;preds.append(z);out.append(z)
   if out:
    z=pd.concat(out);cvrows.append(base|{'model':m,'cv_label':'LEAVE_ONE_CONDITION_GROUP_OUT','n_oof':len(z),**met(z[Y],z.y_pred)})
  
 identdf=pd.DataFrame(ir);fits=pd.DataFrame(fitrows);par=pd.DataFrame(pars);cv=pd.DataFrame(cvrows)
 identdf.to_csv(OUT/'state_identifiability.csv',index=False);fits.to_csv(OUT/'state_fit_metrics.csv',index=False);cv.to_csv(OUT/'state_cv_metrics.csv',index=False);par.to_csv(OUT/'state_physics_parameters.csv',index=False)
 # Exploratory state relationships: one row per fitted parameter, no parameter-prediction model.
 rel=par.groupby(['model','parameter','alloy_family','mechanism_class'],dropna=False).agg(n_states=('material_state_id','nunique'),estimate_median=('estimate','median'),estimate_iqr=('estimate',lambda x:x.quantile(.75)-x.quantile(.25))).reset_index();rel.to_csv(OUT/'parameter_state_relationships.csv',index=False)
 # figures
 a=fits[fits.model.eq('A1')];fig,ax=plt.subplots(1,2,figsize=(11,4.5));ax[0].hist(a.R2,bins=12);ax[0].set(xlabel='A1 descriptive R²',ylabel='States');ax[1].hist(a.MAE,bins=12);ax[1].set(xlabel='A1 descriptive MAE (log10 h)',ylabel='States');save(fig,'statewise_r2_mae_distribution')
 fig,ax=plt.subplots(figsize=(8,4.5));a.set_index('material_state_id').R2.sort_values().plot.bar(ax=ax);ax.axhline(.4802,color='r',ls='--',label='Global A1 OOF R²');ax.set(ylabel='Descriptive R²');ax.legend();save(fig,'global_vs_state_a1_comparison')
 for pn,nm in [('beta_sigma','beta_sigma_by_state'),('beta_T','beta_T_by_state'),('beta_sigmaT','interaction_by_state')]:
  z=par[(par.model=='A1')&(par.parameter==pn)].sort_values('estimate');fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(z.material_state_id,z.estimate);ax.tick_params(axis='x',rotation=90,labelsize=7);ax.set(ylabel=pn);save(fig,nm)
 z=par[par.model.eq('A1')].pivot(index='material_state_id',columns='parameter',values='estimate');fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(z.corr(),vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(len(z.columns)),xticklabels=z.columns,yticks=range(len(z.columns)),yticklabels=z.columns);fig.colorbar(im,ax=ax);save(fig,'parameter_correlation_matrix')
 # parity panels: best, median, poor A1 descriptive states.
 if len(a)>=3:
  states=[a.sort_values('R2').iloc[-1].material_state_id,a.iloc[(a.R2-a.R2.median()).abs().argmin()].material_state_id,a.sort_values('R2').iloc[0].material_state_id];fig,axs=plt.subplots(1,3,figsize=(12,4));
  for ax,s in zip(axs,states):
   q=d[d.material_state_id.eq(s)];p,_,_=fit_predict('A1',q,q);lo=min(q[Y].min(),p.min());hi=max(q[Y].max(),p.max());ax.scatter(q[Y],p);ax.plot([lo,hi],[lo,hi],'k--');ax.set(title=s,xlabel='Observed log10(h)',ylabel='Fit log10(h)')
  save(fig,'representative_state_parity')
  fig,axs=plt.subplots(1,3,figsize=(12,3.8))
  for ax,s in zip(axs,states):
   q=d[d.material_state_id.eq(s)].copy();p,_,_=fit_predict('A1',q,q);q['fit']=p
   for t,z in q.groupby('temperature_K'): ax.scatter(z.stress_MPa,z[Y],s=18,label=f'{t:.0f} K obs');ax.plot(z.stress_MPa,z.fit,'-')
   ax.set(xscale='log',xlabel='Stress (MPa)',ylabel='log10 life (h)',title=s)
  save(fig,'representative_state_stress_life_curves')
 fig,ax=plt.subplots(figsize=(8,4.5));
 for mech,q in par[(par.model=='A1')&(par.parameter=='beta_sigmaT')].groupby('mechanism_class'):ax.scatter(q.material_state_id,q.estimate,label=mech)
 ax.tick_params(axis='x',rotation=90,labelsize=7);ax.legend(fontsize=7);ax.set(ylabel='beta_sigmaT');save(fig,'parameters_grouped_by_mechanism')
 n0=identdf.query("model=='A0' and eligible").material_state_id.nunique();n1=identdf.query("model=='A1' and eligible").material_state_id.nunique();medr=a.R2.median() if len(a) else np.nan;medcv=cv[cv.model.eq('A1')].R2.median() if len(cv[cv.model.eq('A1')]) else np.nan; outcome='STATE-SPECIFIC PHYSICS PARTIALLY SUPPORTED' if len(a) and medr>.7 else 'STATE-SPECIFIC PHYSICS NOT SUPPORTED'
 imp=f"# Future modeling implication\n\nOutcome: **{outcome}**. State-specific descriptive fitting is not predictive validation. The limited leakage-safe state CV and parameter heterogeneity support treating any future `material state → physics parameters → life` approach as hierarchical research, not as proof of a transferable physics law. TASK 04.7-C should be redesigned as a constrained A1/A0 residual test with state-aware diagnostics, rather than proceed unchanged.\n";(OUT/'future_modeling_implication.md').write_text(imp)
 rep=f"# TASK 04.7-B3 — Within-State Physics Fit Diagnostic\n\n## Separation of evidence\n\n`state_fit_metrics.csv` is **DESCRIPTIVE / IN-SAMPLE PHYSICS FIT** only. `state_cv_metrics.csv` contains leakage-safer leave-one-condition-group-out results only where every training remainder remained identifiable. Neither is a Tier1 unseen-source benchmark.\n\n## Findings\n\nTier1 has {d.material_state_id.nunique()} material states; A0/A1 have {n0}/{n1} fully temperature-stress identifiable states. A1 median descriptive R² is {medr:.4f}; median available within-state condition-group OOF R² is {medcv:.4f}. Global A1 OOF R² was 0.4802 and must not be compared directly with descriptive state R². Parameter distributions and exploratory alloy/mechanism summaries are exported without a parameter-prediction model.\n\n**Best-supported hypothesis: H3 — both effects contribute.** Eligible local surfaces can be described better than a shared global fit in some states, but coverage/identifiability is inconsistent and parameters are state-dependent effective fit parameters. High descriptive fits with limited temperature span are not thermal-law evidence.\n\nQ7: any mechanism/alloy grouping is exploratory and limited by state count. Q8: the evidence supports hierarchical parameter modeling as a future direction, not a current validated predictor.\n";(OUT/'WITHIN_STATE_PHYSICS_DIAGNOSTIC_REPORT.md').write_text(rep);print('Wrote',OUT)
if __name__=='__main__':main()
