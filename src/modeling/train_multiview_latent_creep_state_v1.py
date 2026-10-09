#!/usr/bin/env python3
"""source-aware multi-view latent creep-state feasibility study."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))
import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT=_CANDIDATE_ROOT
DATA=ROOT/'03_MODELING/data/baseline_creep_dataset_v1p1.csv'; CREEP=ROOT/'01_DATA/cleaned/RouteB_database_v3/creep_master.csv'; TENSILE=ROOT/'01_DATA/cleaned/RouteB_database_v3/tensile_master.csv'; SPLITS=ROOT/'03_MODELING/splits/source_groupkfold_5fold_assignment.csv'; OUT=ROOT/'03_MODELING/results/multiview_latent_creep_state_v1'
LIFE='target_log10_rupture_life_h'; RATE='minimum_creep_rate_pct_h'; SEEDS=[7,19,31,43,59]
NUM=['Ni_reconstructed_wt','Cr','Mo','Fe','Co','W','Nb','Ti','Al','C','B','phys_invT_Kinv','phys_stress_over_T_MPa_per_K','phys_logstress_over_T_Kinv','phys_zT_zStress']; CAT=['alloy_family','base_alloy','heat_treatment_id']

def scores(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p))}
def save(fig,name): fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',dpi=250);plt.close(fig)
def physics(d,mean=None,sd=None):
 d=d.copy();d['phys_invT_Kinv']=1/d.temperature_K;d['phys_stress_over_T_MPa_per_K']=d.stress_MPa/d.temperature_K;d['phys_logstress_over_T_Kinv']=d.log_stress_MPa/d.temperature_K
 if mean is None: mean=d[['temperature_K','stress_MPa']].mean();sd=d[['temperature_K','stress_MPa']].std(ddof=0).replace(0,1)
 d['phys_zT_zStress']=((d.temperature_K-mean.temperature_K)/sd.temperature_K)*((d.stress_MPa-mean.stress_MPa)/sd.stress_MPa);return d,mean,sd
def add_ys(d,tensile):
 d=d.copy();d['yield_strength_at_creep_MPa']=np.nan;d['ys_view_status']='no_numeric_tensile_curve';use=tensile.dropna(subset=['temperature_C','yield_strength_MPa'])
 for state,idx in d.groupby('material_state_id').groups.items():
  curve=use[use.material_state_id.eq(state)].groupby('temperature_C',as_index=False).yield_strength_MPa.mean().sort_values('temperature_C')
  if len(curve)<2: continue
  t=d.loc[idx,'temperature_K'].to_numpy()-273.15;x=curve.temperature_C.to_numpy();y=curve.yield_strength_MPa.to_numpy();ok=(t>=x.min())&(t<=x.max());values=np.full(len(t),np.nan);values[ok]=np.interp(t[ok],x,y);d.loc[idx,'yield_strength_at_creep_MPa']=values;d.loc[idx,'ys_view_status']=np.where(ok,'interpolated_in_range','outside_tensile_temperature_range')
 return d
def encode_fit(tr,va):
 med=tr[NUM].median().fillna(0);sd=tr[NUM].fillna(med).std(ddof=0).replace(0,1);cats=pd.get_dummies(tr[CAT].fillna('<missing>').astype(str),prefix=CAT,dtype=float)
 def transform(d):
  n=((d[NUM].fillna(med)-med)/sd).to_numpy(float);c=pd.get_dummies(d[CAT].fillna('<missing>').astype(str),prefix=CAT,dtype=float).reindex(columns=cats.columns,fill_value=0).to_numpy(float);return np.hstack([n,c])
 return transform(tr),transform(va)

class Net:
 def __init__(self,n_in,n_class,seed):
  rng=np.random.default_rng(seed);sizes=[n_in,64,32,8];self.p={f'W{i}':rng.normal(0,np.sqrt(2/sizes[i]),(sizes[i],sizes[i+1])) for i in range(3)};self.p|={f'b{i}':np.zeros(sizes[i+1]) for i in range(3)};self.p|={'W_life':rng.normal(0,.1,(8,1)),'b_life':np.zeros(1),'W_rate':rng.normal(0,.1,(8,1)),'b_rate':np.zeros(1),'W_ys':rng.normal(0,.1,(8,1)),'b_ys':np.zeros(1),'W_mech':rng.normal(0,.1,(8,n_class)),'b_mech':np.zeros(n_class)};self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()};self.step=0
 def forward(self,x,training,rng):
  cache={'a0':x};a=x
  for i in range(3):
   z=a@self.p[f'W{i}']+self.p[f'b{i}'];a=np.maximum(z,0);cache[f'z{i}']=z
   if training and i<2: mask=(rng.random(a.shape)>=.1)/.9;a*=mask;cache[f'm{i}']=mask
   cache[f'a{i+1}']=a
  return (a@self.p['W_life']+self.p['b_life']).ravel(),(a@self.p['W_rate']+self.p['b_rate']).ravel(),(a@self.p['W_ys']+self.p['b_ys']).ravel(),a@self.p['W_mech']+self.p['b_mech'],a,cache
 def update(self,g):
  self.step+=1
  for k,v in g.items():
   self.m[k]=.9*self.m[k]+.1*v;self.v[k]=.999*self.v[k]+.001*v*v;mh=self.m[k]/(1-.9**self.step);vh=self.v[k]/(1-.999**self.step);self.p[k]-=.004*mh/(np.sqrt(vh)+1e-8)
 def fit(self,x,life,rate,ys,mech,rate_mask,ys_mask,mode,seed):
  rng=np.random.default_rng(seed+1000);idx=rng.permutation(len(x));hold,fit=idx[:max(2,int(.15*len(x)))],idx[max(2,int(.15*len(x))):];best=copy.deepcopy(self.p);bestloss=np.inf;wait=0
  for _ in range(500):
   pl,pr,py,logits,z,c=self.forward(x[fit],True,rng);g={k:np.zeros_like(v) for k,v in self.p.items()};dz=np.zeros_like(z)
   e=2*(pl-life[fit])/len(fit);g['W_life']=z.T@e[:,None];g['b_life']=np.array([e.sum()]);dz+=e[:,None]*self.p['W_life'].T
   if mode in ('mechanism','multiview'):
    ex=np.exp(logits-logits.max(axis=1,keepdims=True));prob=ex/ex.sum(axis=1,keepdims=True);prob[np.arange(len(fit)),mech[fit]]-=1;prob*=.15/len(fit);g['W_mech']=z.T@prob;g['b_mech']=prob.sum(axis=0);dz+=prob@self.p['W_mech'].T
   if mode=='multiview':
    for name,pred,target,mask,weight in [('rate',pr,rate,rate_mask,.30),('ys',py,ys,ys_mask,.15)]:
     m=mask[fit]
     if m.any():
      er=np.zeros(len(fit));er[m]=2*weight*(pred[m]-target[fit][m])/m.sum();g[f'W_{name}']=z.T@er[:,None];g[f'b_{name}']=np.array([er.sum()]);dz+=er[:,None]*self.p[f'W_{name}'].T
   for i in reversed(range(3)):
    if i<2:dz*=c[f'm{i}']
    dz*=c[f'z{i}']>0;g[f'W{i}']=c[f'a{i}'].T@dz;g[f'b{i}']=dz.sum(axis=0);dz=dz@self.p[f'W{i}'].T
   self.update(g);val=self.forward(x[hold],False,rng)[0];loss=np.mean((val-life[hold])**2)
   if loss<bestloss-1e-5:best=copy.deepcopy(self.p);bestloss=loss;wait=0
   else:
    wait+=1
    if wait>=45:break
  self.p=best

def main():
 for d in [OUT,OUT/'figures']:d.mkdir(parents=True,exist_ok=True)
 data=pd.read_csv(DATA).merge(pd.read_csv(CREEP,usecols=['creep_id',RATE]),on='creep_id',how='left',validate='one_to_one').merge(pd.read_csv(SPLITS)[['creep_id','fold_id']],on='creep_id',validate='one_to_one');data[RATE]=pd.to_numeric(data[RATE],errors='coerce');data['log_rate']=np.nan;m=data[RATE]>0;data.loc[m,'log_rate']=np.log10(data.loc[m,RATE]);data=add_ys(data,pd.read_csv(TENSILE));classes=sorted(data.mechanism_class.unique());class_to_i={x:i for i,x in enumerate(classes)};data['mechanism_index']=data.mechanism_class.map(class_to_i)
 pred=[];latent=[];leaks=[];xgb=[]
 for fold in sorted(data.fold_id.unique()):
  tr0,va0=data[data.fold_id.ne(fold)].copy(),data[data.fold_id.eq(fold)].copy();shared=set(tr0.source_id)&set(va0.source_id);leaks.append({'fold_id':fold,'train_samples':len(tr0),'validation_samples':len(va0),'train_sources':tr0.source_id.nunique(),'validation_sources':va0.source_id.nunique(),'shared_sources':len(shared),'train_rate_labels':int(tr0.log_rate.notna().sum()),'train_ys_labels':int(tr0.yield_strength_at_creep_MPa.notna().sum())});
  if shared:raise ValueError('source leakage')
  tr,mean,sd=physics(tr0);va,_,_=physics(va0,mean,sd);xt,xv=encode_fit(tr,va);lm,ls=tr[LIFE].mean(),tr[LIFE].std(ddof=0) or 1;rm,rs=tr.log_rate.mean(),tr.log_rate.std(ddof=0) or 1;ym,ysd=tr.yield_strength_at_creep_MPa.mean(),tr.yield_strength_at_creep_MPa.std(ddof=0) or 1
  life=((tr[LIFE]-lm)/ls).to_numpy();rate=((tr.log_rate.fillna(rm)-rm)/rs).to_numpy();ystr=((tr.yield_strength_at_creep_MPa.fillna(ym)-ym)/ysd).to_numpy();mech=tr.mechanism_index.to_numpy();rmask=tr.log_rate.notna().to_numpy();ymask=tr.yield_strength_at_creep_MPa.notna().to_numpy()
  ref=XGBRegressor(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=-1,objective='reg:squarederror').fit(xt,tr[LIFE]);xgb.append(va.assign(model='XGB physics baseline',seed=-1,life_prediction=ref.predict(xv),rate_prediction=np.nan,ys_prediction=np.nan,mechanism_prediction=''))
  for seed in SEEDS:
   for label,mode in [('NN life-only','life'),('NN life+mechanism','mechanism'),('NN multi-view','multiview')]:
    net=Net(xt.shape[1],len(classes),seed);net.fit(xt,life,rate,ystr,mech,rmask,ymask,mode,seed);pl,pr,py,logits,z,_=net.forward(xv,False,np.random.default_rng(seed));rec=va[['creep_id','source_id','material_state_id','alloy_family','mechanism_class','transfer_tier','temperature_K','stress_MPa','fold_id',LIFE,'log_rate','yield_strength_at_creep_MPa']].copy();rec['model']=label;rec['seed']=seed;rec['life_prediction']=pl*ls+lm;rec['rate_prediction']=pr*rs+rm;rec['ys_prediction']=py*ysd+ym;rec['mechanism_prediction']=[classes[i] for i in logits.argmax(axis=1)];rec['life_residual']=rec[LIFE]-rec.life_prediction;pred.append(rec)
    if label=='NN multi-view' and seed==SEEDS[0]:
     q=va[['creep_id','source_id','material_state_id','alloy_family','mechanism_class','transfer_tier','fold_id',LIFE]].copy()
     for i in range(8):q[f'z{i+1}']=z[:,i]
     latent.append(q)
 allpred=pd.concat(pred+xgb,ignore_index=True);allpred.to_csv(OUT/'predictions.csv',index=False);pd.concat(latent,ignore_index=True).to_csv(OUT/'latent_vectors.csv',index=False);pd.DataFrame(leaks).to_csv(OUT/'source_leakage_check.csv',index=False)
 if not pd.DataFrame(leaks).shared_sources.eq(0).all():raise ValueError('source leakage')
 rows=[]
 for (model,seed),q in allpred.groupby(['model','seed']):
  row={'model':model,'seed':seed,'n_life':len(q),**{f'life_{k}':v for k,v in scores(q[LIFE],q.life_prediction).items()}}
  if model=='NN multi-view':
   for view,true,prediction in [('rate','log_rate','rate_prediction'),('ys','yield_strength_at_creep_MPa','ys_prediction')]:
    v=q[q[true].notna() & q[prediction].notna()]
    if len(v):row[f'n_{view}']=len(v);row|={f'{view}_{k}':val for k,val in scores(v[true],v[prediction]).items()}
  if model!='XGB physics baseline':row['mechanism_accuracy']=float((q.mechanism_prediction==q.mechanism_class).mean()) if model!='NN life-only' else np.nan
  rows.append(row)
 seed=pd.DataFrame(rows);seed.to_csv(OUT/'seed_results.csv',index=False);summary=seed.groupby('model',as_index=False).agg(n_seeds=('seed','nunique'),life_R2_mean=('life_R2','mean'),life_R2_std=('life_R2','std'),life_MAE_mean=('life_MAE','mean'),life_MAE_std=('life_MAE','std'),life_RMSE_mean=('life_RMSE','mean'),life_RMSE_std=('life_RMSE','std'),rate_R2_mean=('rate_R2','mean'),ys_R2_mean=('ys_R2','mean'),mechanism_accuracy_mean=('mechanism_accuracy','mean'));summary.to_csv(OUT/'metrics_summary.csv',index=False)
 # Mean seed predictions give one OOF value per neural model for diagnostics.
 nn=allpred[allpred.seed.ge(0)].groupby(['model','creep_id'],as_index=False).agg({LIFE:'first','life_prediction':'mean','material_state_id':'first','alloy_family':'first','mechanism_class':'first','transfer_tier':'first','temperature_K':'first','stress_MPa':'first'});ref=allpred[allpred.model.eq('XGB physics baseline')][nn.columns];diag=pd.concat([nn,ref],ignore_index=True);diag['residual']=diag[LIFE]-diag.life_prediction
 regional=[]
 for region,mask in [('long_life_tail',diag[LIFE]>=diag[LIFE].quantile(2/3)),('low_stress',diag.stress_MPa<=diag.stress_MPa.quantile(.25)),('high_temperature',diag.temperature_K>=diag.temperature_K.quantile(.75))]:
  for name,q in diag[mask].groupby('model'):regional.append({'region':region,'model':name,'n':len(q),**scores(q[LIFE],q.life_prediction),'mean_residual':q.residual.mean()})
 pd.DataFrame(regional).to_csv(OUT/'regional_error_analysis.csv',index=False);state=diag.groupby(['model','material_state_id'],as_index=False).apply(lambda q:pd.Series({'n':len(q),**scores(q[LIFE],q.life_prediction),'mean_residual':q.residual.mean()}),include_groups=False);state.to_csv(OUT/'material_state_error_analysis.csv',index=False)
 # Figures.
 a,b=diag[diag.model.eq('NN life-only')],diag[diag.model.eq('NN multi-view')];fig,ax=plt.subplots(1,2,figsize=(10,4.5))
 for x,q,title,c in zip(ax,[a,b],['Life-only NN','Multi-view NN'],['#0ea5e9','#d9485f']):lim=[min(q[LIFE].min(),q.life_prediction.min()),max(q[LIFE].max(),q.life_prediction.max())];x.scatter(q[LIFE],q.life_prediction,s=20,alpha=.6,color=c);x.plot(lim,lim,'k--');x.set(xlabel='Observed log10(rupture life / h)',ylabel='Predicted log10(rupture life / h)',title=title,xlim=lim,ylim=lim)
 save(fig,'life_prediction_parity');fig,ax=plt.subplots(figsize=(7,4.5));bins=np.linspace(min(a.residual.min(),b.residual.min()),max(a.residual.max(),b.residual.max()),24);ax.hist(a.residual,bins=bins,alpha=.55,label='Life-only',color='#0ea5e9');ax.hist(b.residual,bins=bins,alpha=.55,label='Multi-view',color='#d9485f');ax.axvline(0,color='black',ls='--');ax.set(xlabel='Residual (log10 h)',ylabel='Count',title='Life residual comparison');ax.legend();save(fig,'residual_distribution_comparison')
 z=pd.read_csv(OUT/'latent_vectors.csv');zc=[f'z{i}' for i in range(1,9)];pc=PCA(n_components=2,random_state=42).fit_transform(z[zc]);z['PC1'],z['PC2']=pc[:,0],pc[:,1];z['life_bin']=pd.qcut(z[LIFE],4,duplicates='drop').astype(str);fig,axs=plt.subplots(2,2,figsize=(11,8));
 for ax,col in zip(axs.ravel(),['mechanism_class','transfer_tier','alloy_family','life_bin']):
  for name,q in z.groupby(col):ax.scatter(q.PC1,q.PC2,s=15,alpha=.6,label=str(name))
  ax.set(xlabel='PC1',ylabel='PC2',title=f'Latent PCA by {col}');ax.legend(fontsize=6)
 save(fig,'latent_pca_representation');tail=diag[LIFE]>=diag[LIFE].quantile(2/3);fig,ax=plt.subplots(figsize=(7,4.5));ax.hist(a[tail.loc[a.index]].residual,bins=18,alpha=.55,label='Life-only',color='#0ea5e9');ax.hist(b[tail.loc[b.index]].residual,bins=18,alpha=.55,label='Multi-view',color='#d9485f');ax.axvline(0,color='black',ls='--');ax.set(xlabel='Residual (log10 h)',ylabel='Count',title='Long-life-tail residual comparison');ax.legend();save(fig,'long_life_tail_residual_comparison')
 st=state[state.model.isin(['NN life-only','NN multi-view'])].pivot(index='material_state_id',columns='model',values='MAE').sort_values('NN life-only');fig,ax=plt.subplots(figsize=(8,max(5,.27*len(st))));st.plot.barh(ax=ax,color=['#0ea5e9','#d9485f']);ax.set(xlabel='MAE (log10 h)',ylabel='Material state',title='Error by material state');save(fig,'error_by_material_state')
 config={'task':'TASK 04.5-B','dataset':str(DATA.relative_to(ROOT)),'split':str(SPLITS.relative_to(ROOT)),'seeds':SEEDS,'architecture':'input-64-32-8 latent; life/rate/YS regression heads and mechanism classification head','dropout':.1,'early_stopping_patience':45,'loss_weights':{'mechanism':.15,'rate':.30,'yield_strength':.15},'rate_labels':int(data.log_rate.notna().sum()),'yield_strength_labels':int(data.yield_strength_at_creep_MPa.notna().sum()),'mechanism_labels':len(data),'source_leakage_passed':True};(OUT/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
 sm=summary.set_index('model');life,mech,mv=(sm.loc[x] for x in ['NN life-only','NN life+mechanism','NN multi-view']);reg=pd.DataFrame(regional);tail0=reg[(reg.region=='long_life_tail')&(reg.model=='NN life-only')].iloc[0];tailmv=reg[(reg.region=='long_life_tail')&(reg.model=='NN multi-view')].iloc[0]
 report=f'''# TASK 04.5-B — Multi-view Latent Creep State Learning

## Protocol

All models use frozen source-aware OOF folds and five fixed seeds. The encoder input is the P3-replace physics/composition/material representation. Mechanism, log minimum creep rate, and yield strength at creep temperature are **auxiliary heads only**, never life-prediction inputs. UTS is not used. Tensile yield strength is linearly interpolated only inside a material state's observed numeric tensile-temperature range; missing rate/YS labels are masked rather than dropping records. All five source overlaps are zero.

## Repeated OOF metrics

{summary.to_markdown(index=False,floatfmt='.4f')}

## Answers

1. **Does multi-view learning outperform physics XGB?** XGB life R2 is {sm.loc['XGB physics baseline','life_R2_mean']:.4f}; multi-view NN is {mv.life_R2_mean:.4f} +/- {mv.life_R2_std:.4f}. {'It exceeds the reference in this feasibility protocol.' if mv.life_R2_mean>sm.loc['XGB physics baseline','life_R2_mean'] else 'It does not exceed the reference.'}
2. **Which auxiliary information contributes?** Life-only/mechanism/multi-view R2 values are {life.life_R2_mean:.4f}/{mech.life_R2_mean:.4f}/{mv.life_R2_mean:.4f}. Mechanism-head accuracy is {mech.mechanism_accuracy_mean:.4f} for Model B and {mv.mechanism_accuracy_mean:.4f} for Model C; these labels encode approved domain classes, not new measurements.
3. **Does sparse rate help?** The multi-view rate head has R2 {mv.rate_R2_mean:.4f} on its masked labels, but its life comparison versus Model B determines transfer value rather than rate-head fit alone.
4. **Does tensile help as an auxiliary view?** The multi-view YS head has R2 {mv.ys_R2_mean:.4f} on in-range labels. Compare Model C with Model B; this is a feasibility signal only because tensile availability is state-conditioned.
5. **Does latent space reflect mechanism?** PCA plots are descriptive, colored by mechanism, tier, alloy, and life bins. They can reveal organization but do not establish a physical state variable without independent microstructure measurements.

Long-life-tail life-only/multi-view residual is {tail0.mean_residual:.4f}/{tailmv.mean_residual:.4f}; positive means underprediction. Regional and state/tier/mechanism diagnostics are exported for the full source-aware OOF population. {'The results support further controlled latent-state work.' if mv.life_R2_mean>life.life_R2_mean else 'The results do not yet support a general hidden-state improvement claim; explicit microstructure or broader independent auxiliary views remain needed.'}
''';(OUT/'MULTIVIEW_LATENT_STATE_REPORT.md').write_text(report,encoding='utf-8');print(f'Wrote {OUT}')
if __name__=='__main__':main()
