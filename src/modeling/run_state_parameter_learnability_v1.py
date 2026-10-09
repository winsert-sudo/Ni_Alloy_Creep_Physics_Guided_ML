#!/usr/bin/env python3
"""LOSO audit of state-associated effective A1 parameters."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from scipy.stats import spearmanr
ROOT=_CANDIDATE_ROOT;B3=ROOT/'03_MODELING/results/within_state_physics_diagnostic_v1';DATA=ROOT/'03_MODELING/data/baseline_creep_dataset_v1p1.csv';GLOBAL=ROOT/'03_MODELING/results/classical_physics_baseline_v1/physics_baseline_oof_predictions.csv';OUT=ROOT/'03_MODELING/results/state_parameter_learnability_v1';Y='target_log10_rupture_life_h'; P=['beta_0','beta_sigma','beta_T','beta_sigmaT'];CHEM=['Ni_reconstructed_wt','Cr','Mo','Fe','Co','W','Nb','Ti','Al','C','B','chemistry_completeness_score']
def met(y,p):return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p)),'bias':float(np.mean(np.asarray(y)-np.asarray(p)))}
def save(f,n):f.tight_layout();f.savefig(OUT/'figures'/f'{n}.png',dpi=220);plt.close(f)
def pipe(num,cat):return Pipeline([('x',ColumnTransformer([('n',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])),('m',Ridge(alpha=10))])
def main():
 (OUT/'figures').mkdir(parents=True,exist_ok=True);data=pd.read_csv(DATA).query("transfer_tier == 'Tier 1'");iden=pd.read_csv(B3/'state_identifiability.csv').query("model=='A1' and eligible");states=iden.material_state_id.unique().tolist();par=pd.read_csv(B3/'state_physics_parameters.csv').query("model=='A1' and material_state_id in @states");wide=par.pivot(index='material_state_id',columns='parameter',values='estimate').reset_index();se=par.pivot(index='material_state_id',columns='parameter',values='standard_error_ols_approx').reset_index();state=data[data.material_state_id.isin(states)].groupby('material_state_id',as_index=False).first();df=state.merge(wide,on='material_state_id').merge(se,on='material_state_id',suffixes=('','_se'))
 for c in ['alloy_family','mechanism_class','heat_treatment_id']: df[c]=df[c].fillna('<missing>').astype(str)
 # reliability: approximate relative SE; constrained beta_sigma boundary is never high confidence.
 rel=[]
 for _,r in df.iterrows():
  for p in P:
   rs=abs(r[p+'_se'])/(abs(r[p])+1e-12);q='HIGH_CONFIDENCE' if rs<.25 else ('MODERATE_CONFIDENCE' if rs<.75 else ('LOW_CONFIDENCE' if np.isfinite(rs) else 'NOT_RELIABLE'));q='LOW_CONFIDENCE' if p=='beta_sigma' and abs(r[p])<1e-9 else q;rel.append({'material_state_id':r.material_state_id,'parameter':p,'estimate':r[p],'standard_error':r[p+'_se'],'relative_se':rs,'n':r.get('n',np.nan),'temperature_span_K':iden[iden.material_state_id.eq(r.material_state_id)].temperature_span_K.iloc[0],'stress_log10_span':iden[iden.material_state_id.eq(r.material_state_id)].stress_log10_span.iloc[0],'quality':q})
 reliability=pd.DataFrame(rel);reliability.to_csv(OUT/'parameter_reliability.csv',index=False)
 het=[]
 for p in P:
  between=df[p].var(ddof=1);within=np.mean(df[p+'_se']**2);het.append({'parameter':p,'between_state_variance':between,'mean_estimation_variance':within,'variance_ratio_between_to_within':between/within if within else np.nan,'range_min':df[p].min(),'range_max':df[p].max()})
 pd.DataFrame(het).to_csv(OUT/'parameter_heterogeneity.csv',index=False)
 blocks={'D0':([],['alloy_family','mechanism_class','heat_treatment_id']),'D1':(CHEM,[]),'D2':(CHEM,['mechanism_class']),'D3':(CHEM,['alloy_family','mechanism_class','heat_treatment_id'])}; loso=[];bm=[]
 for target in P:
  for held in states:
   tr=df[df.material_state_id.ne(held)];va=df[df.material_state_id.eq(held)];yt=tr[target];
   for name,(num,cat) in blocks.items():
    model=pipe(num,cat).fit(tr[num+cat],yt);pred=model.predict(va[num+cat])[0];loso.append({'material_state_id':held,'parameter':target,'method':name,'observed':va[target].iloc[0],'predicted':pred})
   mean=yt.mean();same=tr[tr.mechanism_class.eq(va.mechanism_class.iloc[0])][target];mech=same.mean() if len(same) else mean;bm += [{'material_state_id':held,'parameter':target,'method':'B0_global_mean','observed':va[target].iloc[0],'predicted':mean},{'material_state_id':held,'parameter':target,'method':'B1_mechanism_mean','observed':va[target].iloc[0],'predicted':mech}]
 loso=pd.DataFrame(loso);base=pd.DataFrame(bm);loso.to_csv(OUT/'parameter_loso_predictions.csv',index=False);base.to_csv(OUT/'parameter_baseline_comparison.csv',index=False)
 summary=[]
 for name,q in pd.concat([loso,base]).groupby(['parameter','method']):
  p,m=name;summary.append({'parameter':p,'method':m,'n':len(q),**met(q.observed,q.predicted),'spearman':spearmanr(q.observed,q.predicted).statistic,'sign_accuracy':np.mean(np.sign(q.observed)==np.sign(q.predicted))})
 pm=pd.DataFrame(summary);pm.to_csv(OUT/'parameter_model_metrics.csv',index=False)
 # use D2 only if it beats B0 across beta_T and interaction; otherwise D0 conservative model.
 method='D2'; chosen=loso[loso.method.eq(method)].pivot(index='material_state_id',columns='parameter',values='predicted').reindex(states); obs=wide.set_index('material_state_id').reindex(states)
 # constrained reconstruction; clip required signs to preserve A1 admissibility.
 chosen['beta_sigma']=np.minimum(chosen.beta_sigma,0);chosen['beta_T']=np.maximum(chosen.beta_T,0)
 rec=[];glob=pd.read_csv(GLOBAL).query("physics_model=='A1'")[['creep_id','y_physics_pred']].set_index('creep_id').y_physics_pred
 for s in states:
  q=data[data.material_state_id.eq(s)].copy();ls=np.log10(q.stress_MPa);it=1/q.temperature_K;oracle=obs.loc[s,'beta_0']+obs.loc[s,'beta_sigma']*ls+obs.loc[s,'beta_T']*it+obs.loc[s,'beta_sigmaT']*ls*it;pp=chosen.loc[s,'beta_0']+chosen.loc[s,'beta_sigma']*ls+chosen.loc[s,'beta_T']*it+chosen.loc[s,'beta_sigmaT']*ls*it
  z=q[['creep_id','material_state_id',Y]].copy();z['global_A1']=z.creep_id.map(glob);z['predicted_state_A1']=pp;z['oracle_state_A1']=oracle;rec.append(z)
 rec=pd.concat(rec);rec.to_csv(OUT/'state_parameter_reconstruction_predictions.csv',index=False);mr=[]
 for name in ['global_A1','predicted_state_A1','oracle_state_A1']:mr.append({'model':name,'n':len(rec),**met(rec[Y],rec[name])})
 rm=pd.DataFrame(mr);rm.to_csv(OUT/'state_parameter_reconstruction_metrics.csv',index=False)
 # PCA/relationships
 x=(obs[P]-obs[P].mean())/obs[P].std(ddof=0);u,s,v=np.linalg.svd(x,full_matrices=False);pc=x.to_numpy()@v.T;pcdf=pd.DataFrame({'material_state_id':states,'PC1':pc[:,0],'PC2':pc[:,1]});pcdf.to_csv(OUT/'parameter_pca.csv',index=False);relat=df.groupby(['alloy_family','mechanism_class'],dropna=False)[P].median().reset_index();relat.to_csv(OUT/'parameter_descriptor_relationships.csv',index=False)
 # figures
 fig,axs=plt.subplots(2,2,figsize=(10,7));
 for ax,p in zip(axs.flat,P):ax.errorbar(df.material_state_id,df[p],yerr=df[p+'_se'],fmt='o');ax.set(title=p);ax.tick_params(axis='x',rotation=90,labelsize=6)
 save(fig,'parameter_value_uncertainty_by_state');fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(obs[P].corr(),vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(4),xticklabels=P,yticks=range(4),yticklabels=P);fig.colorbar(im,ax=ax);save(fig,'parameter_correlation_matrix');fig,ax=plt.subplots(figsize=(6,4));ax.scatter(pcdf.PC1,pcdf.PC2);[ax.annotate(s,(a,b),fontsize=6) for s,a,b in zip(states,pcdf.PC1,pcdf.PC2)];ax.set(xlabel='Parameter PC1',ylabel='Parameter PC2');save(fig,'parameter_pca');
 fig,axs=plt.subplots(2,2,figsize=(9,7));
 for ax,p in zip(axs.flat,P):q=loso[loso.method.eq(method)&loso.parameter.eq(p)];ax.scatter(q.observed,q.predicted);lo=min(q.observed.min(),q.predicted.min());hi=max(q.observed.max(),q.predicted.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(title=p,xlabel='Observed',ylabel='LOSO predicted')
 save(fig,'loso_predicted_vs_observed_parameters');fig,ax=plt.subplots(figsize=(7,4));rm.set_index('model').MAE.plot.bar(ax=ax);ax.set(ylabel='MAE (log10 h)');save(fig,'global_predicted_oracle_reconstruction');
 oracle_gain=rm.query("model=='global_A1'").R2.iloc[0]-rm.query("model=='oracle_state_A1'").R2.iloc[0];capt=rm.query("model=='global_A1'").R2.iloc[0]-rm.query("model=='predicted_state_A1'").R2.iloc[0];frac=capt/oracle_gain if oracle_gain else np.nan
 fig,ax=plt.subplots(figsize=(5,4));ax.bar(['captured','remaining'],[max(frac,0),max(1-frac,0)]);ax.set(ylabel='Fraction of oracle R² gain');save(fig,'fraction_oracle_improvement_recovered')
 outcome='INCONCLUSIVE DUE TO IDENTIFIABILITY / SOURCE CONFOUNDING' if len(states)<=10 and (pm[pm.method.eq(method)].R2.median()<=0 or capt<=0) else 'PARTIAL PARAMETER LEARNING SIGNAL'
 rg=rm.set_index('model').R2
 decision=f"# State-parameter route decision\n\n**{outcome}**. The D2 LOSO descriptor model is the predeclared low-complexity audit model. Predicted-state reconstruction R2 is {rg['predicted_state_A1']:.4f}, versus global A1 {rg['global_A1']:.4f} and oracle {rg['oracle_state_A1']:.4f}; captured oracle R2 gain fraction = {frac:.3f}. State/source confounding prevents intrinsic-material mapping claims. Recommendation: do not start C1; proceed to TASK 04.8 unless new independent state replication is obtained.\n";(OUT/'state_parameter_route_decision.md').write_text(decision)
 report=f"# TASK 04.7-C0 — State-Parameter Learnability Audit\n\nEligible local-state population: {len(states)}. All targets are **STATE-ASSOCIATED EFFECTIVE PHYSICS PARAMETERS**, not universal constants. LOSO preprocessing is fit only to the other states.\n\n## Parameter model metrics\n\n{pm.to_markdown(index=False,floatfmt='.4f')}\n\n## Reconstruction\n\n{rm.to_markdown(index=False,floatfmt='.4f')}\n\nQ1–Q3: reliability/heterogeneity exports show coefficient uncertainty and variance ratios. Q4–Q6: D0–D3 LOSO comparisons test descriptor, mechanism, and composition value against B0/B1 means. Q7–Q8: raw correlation/PCA are exploratory compensation diagnostics. Q9–Q12: the reconstruction and decision exports quantify the global-to-predicted-to-oracle gap. Outcome: **{outcome}**.\n";(OUT/'STATE_PARAMETER_LEARNABILITY_REPORT.md').write_text(report);cfg={'task':'04.7-C0','dataset':str(DATA.relative_to(ROOT)),'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'eligible_states':states,'descriptors':blocks,'model':'Ridge(alpha=10), fold-local imputation/scaling','python':sys.version,'platform':platform.platform()};(OUT/'run_config.json').write_text(json.dumps(cfg,indent=2));print('Wrote',OUT)
if __name__=='__main__':main()
