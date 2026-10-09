#!/usr/bin/env python3
"""harmonized Tier1 source-aware ML--physics benchmark."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))
import hashlib, json, platform, sys, types
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# The historical module imports XGBoost for its own standalone main routine.
# This benchmark reuses only its NumPy NN components and reconstructs frozen XGB OOF.
if 'xgboost' not in sys.modules:
 stub=types.ModuleType('xgboost'); stub.XGBRegressor=object; sys.modules['xgboost']=stub
from train_multiview_latent_creep_state_v1 import Net, physics, add_ys, encode_fit, SEEDS
from train_classical_physics_baseline_v1 import fit_predict

ROOT=_CANDIDATE_ROOT; DATA=ROOT/'03_MODELING/data/baseline_creep_dataset_v1p1.csv'; SPLIT=ROOT/'03_MODELING/splits/source_groupkfold_5fold_assignment.csv'; CREEP=ROOT/'01_DATA/cleaned/RouteB_database_v3/creep_master.csv'; TENSILE=ROOT/'01_DATA/cleaned/RouteB_database_v3/tensile_master.csv'; AUDIT=ROOT/'03_MODELING/results/baseline_tier1_v1/error_source_audit'; OUT=ROOT/'03_MODELING/results/unified_ml_physics_benchmark_v1'; Y='target_log10_rupture_life_h'
def met(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(mean_squared_error(y,p)**.5),'R2':float(r2_score(y,p)),'bias':float(np.mean(np.asarray(y)-np.asarray(p)))}
def save(fig,n): fig.tight_layout();fig.savefig(OUT/'figures'/f'{n}.png',dpi=220);plt.close(fig)
def main():
 (OUT/'figures').mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(DATA).merge(pd.read_csv(SPLIT)[['creep_id','fold_id']],on='creep_id',validate='one_to_one');d=d[d.transfer_tier.eq('Tier 1')].copy();d=d.merge(pd.read_csv(CREEP,usecols=['creep_id','minimum_creep_rate_pct_h']),on='creep_id',how='left',validate='one_to_one');d['log_rate']=np.where(d.minimum_creep_rate_pct_h.gt(0),np.log10(d.minimum_creep_rate_pct_h),np.nan);d=add_ys(d,pd.read_csv(TENSILE));classes=sorted(d.mechanism_class.unique());cmap={x:i for i,x in enumerate(classes)};d['mechanism_index']=d.mechanism_class.map(cmap)
 p3prior=pd.read_csv(ROOT/'03_MODELING/results/stress_temperature_physics_v1/predictions.csv').query("analysis == 'tier1_physics' and feature_set == 'P3-replace' and model == 'XGBoost'")[['creep_id','y_pred']].set_index('creep_id').y_pred
 rows=[]; params=[]; leak=[]
 for fold in sorted(d.fold_id.unique()):
  tr0,va0=d[d.fold_id.ne(fold)].copy(),d[d.fold_id.eq(fold)].copy(); shared=set(tr0.source_id)&set(va0.source_id); leak.append({'fold_id':fold,'train_n':len(tr0),'validation_n':len(va0),'shared_sources':len(shared)}); assert not shared
  tr,mu,sd=physics(tr0);va,_,_=physics(va0,mu,sd);xt,xv=encode_fit(tr,va)
  # Exact reconstruction of the frozen P3-replace OOF vector; no retraining/tuning.
  p=va.creep_id.map(p3prior).to_numpy()
  for model in ['P3-replace XGBoost']:
   q=va[['creep_id','source_id','temperature_K','stress_MPa','fold_id',Y]].copy();q['model']=model;q['seed']=-1;q['y_pred']=p;rows.append(q)
  for pm in ['A0','A1','L1']:
   p,par,status=fit_predict(pm,tr,va);q=va[['creep_id','source_id','temperature_K','stress_MPa','fold_id',Y]].copy();q['model']=pm;q['seed']=-1;q['y_pred']=p;rows.append(q)
   params += [{'model':pm,'fold_id':fold,'parameter':k,'estimate':v,'status':status} for k,v in par.items()]
  lm,ls=tr[Y].mean(),tr[Y].std(ddof=0) or 1;rm,rs=tr.log_rate.mean(),tr.log_rate.std(ddof=0) or 1;ym,ys=tr.yield_strength_at_creep_MPa.mean(),tr.yield_strength_at_creep_MPa.std(ddof=0) or 1
  life=((tr[Y]-lm)/ls).to_numpy();rate=((tr.log_rate.fillna(rm)-rm)/rs).to_numpy();ystr=((tr.yield_strength_at_creep_MPa.fillna(ym)-ym)/ys).to_numpy();mi=tr.mechanism_index.to_numpy();rmask=tr.log_rate.notna().to_numpy();ymask=tr.yield_strength_at_creep_MPa.notna().to_numpy()
  for seed in SEEDS:
   for label,mode in [('NN life-only','life'),('NN life+mechanism','mechanism'),('NN multi-view','multiview')]:
    net=Net(xt.shape[1],len(classes),seed);net.fit(xt,life,rate,ystr,mi,rmask,ymask,mode,seed);pl,pr,py,logits,_,_=net.forward(xv,False,np.random.default_rng(seed));q=va[['creep_id','source_id','temperature_K','stress_MPa','fold_id',Y]].copy();q['model']=label;q['seed']=seed;q['y_pred']=pl*ls+lm;q['rate_pred']=pr*rs+rm;q['ys_pred']=py*ys+ym;q['mechanism_pred']=[classes[i] for i in logits.argmax(axis=1)];rows.append(q)
 pred=pd.concat(rows,ignore_index=True);pred['residual']=pred[Y]-pred.y_pred;pred.to_csv(OUT/'unified_oof_predictions.csv',index=False);pd.DataFrame(leak).to_csv(OUT/'source_leakage_check.csv',index=False);pd.DataFrame(params).to_csv(OUT/'physics_parameter_stability.csv',index=False)
 # deterministic and seed-pooled metrics; ensemble is direct NN point estimate.
 seedrows=[]
 for (m,s),q in pred[pred.seed.ge(0)].groupby(['model','seed']): seedrows.append({'model':m,'seed':s,'n':len(q),**met(q[Y],q.y_pred)})
 seeds=pd.DataFrame(seedrows);seeds.to_csv(OUT/'nn_seed_metrics.csv',index=False)
 ens = pred[pred.seed.ge(0)].groupby(['model','creep_id'],as_index=False).agg(
  {Y:'first', 'y_pred':'mean', 'source_id':'first', 'temperature_K':'first',
   'stress_MPa':'first', 'fold_id':'first'})
 ens['seed'] = -2
 ens['residual'] = ens[Y] - ens.y_pred
 det=pred[pred.seed.eq(-1)].copy(); allp=pd.concat([det,ens],ignore_index=True)
 metrics=[]
 for m,q in allp.groupby('model'):
  x={'Model':m,'Population_n':len(q),**{f'OOF_{k}':v for k,v in met(q[Y],q.y_pred).items()},'model_class':'Neural network' if m.startswith('NN') else ('Classical physics' if m in ['A0','A1','L1'] else 'Tree ML')}
  if m.startswith('NN'):
   z=seeds[seeds.model.eq(m)];x.update({'seed_mean_R2':z.R2.mean(),'seed_std_R2':z.R2.std(),'seed_mean_MAE':z.MAE.mean(),'seed_std_MAE':z.MAE.std()})
  metrics.append(x)
 metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'unified_model_metrics.csv',index=False)
 audit=pd.read_csv(AUDIT/'oof_error_table.csv'); low=audit.stress_MPa.quantile(.25); hot=audit.temperature_K.quantile(.75); tail=audit[Y].quantile(2/3);cov=pd.read_csv(AUDIT/'temperature_stress_coverage.csv')[['creep_id','condition_extrapolation_flag']];allp=allp.merge(cov,on='creep_id',validate='many_to_one')
 regs=[]
 for n,mask in [('low_stress',allp.stress_MPa<=low),('high_temperature',allp.temperature_K>=hot),('long_life_tail',allp[Y]>=tail),('condition_extrapolation',allp.condition_extrapolation_flag.eq(1)),('condition_interpolation',allp.condition_extrapolation_flag.eq(0))]:
  for m,q in allp[mask].groupby('model'): regs.append({'model':m,'region':n,'n':len(q),**met(q[Y],q.y_pred),'mean_residual':q.residual.mean()})
 regional=pd.DataFrame(regs);regional.to_csv(OUT/'regional_metrics.csv',index=False)
 ext=allp[allp.condition_extrapolation_flag.eq(1)].pivot(index=['creep_id','source_id','temperature_K','stress_MPa',Y],columns='model',values='y_pred').reset_index(); ext.to_csv(OUT/'condition_extrapolation_predictions.csv',index=False)
 exseed=pred.merge(cov,on='creep_id',validate='many_to_one').query('condition_extrapolation_flag == 1 and seed >= 0').groupby(['model','creep_id']).y_pred.agg(['mean','std']).reset_index();exseed.to_csv(OUT/'condition_extrapolation_seed_variation.csv',index=False)
 pairs=[]
 for a,b in [('NN life+mechanism','P3-replace XGBoost'),('NN life-only','P3-replace XGBoost'),('A1','P3-replace XGBoost'),('A1','NN life+mechanism'),('NN life+mechanism','NN multi-view')]:
  qa=allp[allp.model.eq(a)].set_index('creep_id');qb=allp[allp.model.eq(b)].set_index('creep_id');diff=(qa.loc[qb.index,Y]-qa.loc[qb.index,'y_pred']).abs()-(qb[Y]-qb.y_pred).abs();pairs.append({'model_A':a,'model_B':b,'n':len(diff),'mean_abs_error_difference_A_minus_B':diff.mean(),'median':diff.median(),'A_better_fraction':(diff<0).mean()})
 pd.DataFrame(pairs).to_csv(OUT/'paired_model_error_comparison.csv',index=False)
 roles=pd.DataFrame([['P3-replace XGBoost','STRONG_INTERPOLATION_MODEL'],['NN life-only','AUXILIARY_STATE_MODEL'],['NN life+mechanism','CURRENT_BEST_DATA_DRIVEN_MODEL'],['NN multi-view','AUXILIARY_STATE_MODEL'],['A0','PHYSICS_REFERENCE_ONLY'],['A1','POTENTIAL_EXTRAPOLATION_BACKBONE'],['L1','EQUIVALENT_REPARAMETERIZATION_IF_CONFIRMED'] ],columns=['model','role']);roles.to_csv(OUT/'model_role_classification.csv',index=False)
 # visual suite
 order=metrics.sort_values('OOF_R2',ascending=False).Model.tolist();fig,ax=plt.subplots(1,2,figsize=(12,4.5));metrics.set_index('Model').loc[order,'OOF_R2'].plot.bar(ax=ax[0]);metrics.set_index('Model').loc[order,'OOF_MAE'].plot.bar(ax=ax[1]);ax[0].set(ylabel='OOF R²');ax[1].set(ylabel='OOF MAE (log10 h)');save(fig,'overall_mae_r2_comparison')
 fig,axs=plt.subplots(2,2,figsize=(10,9))
 for ax,m in zip(axs.flat,['P3-replace XGBoost','NN life+mechanism','NN multi-view','A1']):
  q=allp[allp.model.eq(m)];lo=min(q[Y].min(),q.y_pred.min());hi=max(q[Y].max(),q.y_pred.max());ax.scatter(q[Y],q.y_pred,s=18,alpha=.55);ax.plot([lo,hi],[lo,hi],'k--');ax.set(title=m,xlabel='Observed life (log10 h)',ylabel='Predicted life (log10 h)',xlim=(lo,hi),ylim=(lo,hi))
 save(fig,'unified_parity_comparison')
 fig,ax=plt.subplots(figsize=(7,5));
 for m,c in [('P3-replace XGBoost','#059669'),('NN life+mechanism','#7c3aed'),('A1','#dc2626')]:q=allp[allp.model.eq(m)];ax.scatter(q[Y],q.residual,s=20,alpha=.55,label=m,color=c)
 ax.axhline(0,color='k',ls='--');ax.set(xlabel='Observed life (log10 h)',ylabel='Residual (log10 h)');ax.legend();save(fig,'residual_vs_life_comparison')
 for region,name in [('long_life_tail','long_life_tail_error_comparison'),('low_stress','low_stress_error_comparison'),('high_temperature','high_temperature_error_comparison'),('condition_extrapolation','condition_interpolation_vs_extrapolation')]:
  fig,ax=plt.subplots(figsize=(8,4.5)); q=regional[regional.region.eq(region)];q.set_index('model').MAE.reindex(order).plot.bar(ax=ax);ax.set(ylabel='MAE (log10 h)',title=region);save(fig,name)
 fig,ax=plt.subplots(figsize=(8,4.5));regional[regional.region.isin(['condition_interpolation','condition_extrapolation'])].pivot(index='model',columns='region',values='MAE').reindex(order).plot.bar(ax=ax);ax.set(ylabel='MAE (log10 h)',title='Condition interpolation versus extrapolation');save(fig,'condition_interpolation_vs_extrapolation')
 fig,ax=plt.subplots(figsize=(9,5));ext.set_index('creep_id')[['A1','P3-replace XGBoost','NN life-only','NN life+mechanism','NN multi-view']].plot.bar(ax=ax);ax.set(ylabel='Predicted life (log10 h)',title='Five condition-extrapolation predictions');save(fig,'five_extrapolation_point_comparison')
 fig,ax=plt.subplots(figsize=(8,4.5));
 for m,q in seeds.groupby('model'):ax.plot(q.seed,q.R2,'o-',label=m)
 ax.set(xlabel='Seed',ylabel='Pooled OOF R²');ax.legend();save(fig,'nn_seed_stability')
 q=allp[allp.model.eq('NN life+mechanism')].set_index('creep_id');x=allp[allp.model.eq('P3-replace XGBoost')].set_index('creep_id');fig,ax=plt.subplots(figsize=(8,4.5));ax.scatter(q[Y],(q[Y]-q.y_pred).abs()-(x.loc[q.index,Y]-x.loc[q.index,'y_pred']).abs(),s=20,alpha=.6);ax.axhline(0,color='k',ls='--');ax.set(xlabel='Observed life (log10 h)',ylabel='|error| NN-mech − XGB (log10 h)');save(fig,'nn_mechanism_vs_xgb_absolute_error_difference')
 # concise gate and report
 a1ex=regional.query("model=='A1' and region=='condition_extrapolation'").iloc[0]; best=metrics.sort_values('OOF_R2',ascending=False).iloc[0];gate='PASS_GATE_C_CONTROLLED_TEST_ONLY' if a1ex.MAE < regional.query("model=='NN life+mechanism' and region=='condition_extrapolation'").MAE.iloc[0] else 'DO_NOT_PROCEED'
 (OUT/'task_04_7_c_gate_decision.md').write_text(f"# TASK 04.7-C gate\n\nDecision: **{gate}**. A1 extrapolation MAE is {a1ex.MAE:.4f} on n=5; this is exploratory and cannot establish superiority. A controlled A1 residual test is justified only as Gate C, with P3-replace and A0 no-harm controls.\n",encoding='utf-8')
 report=f"# TASK 04.7-B2 — Unified ML–Physics Benchmark\n\nAll models use the identical 220-row Tier1 frozen source-aware OOF population. NN rows are five-seed OOF ensembles; seed summaries are separate.\n\n## Unified comparison\n\n{metrics.to_markdown(index=False,floatfmt='.4f')}\n\n## Decisions\n\n- Q1/Q2: compare the ensemble rows above; this is the harmonized answer, not a comparison of historical means.\n- Q3: NN multi-view versus mechanism is reported by paired errors and seed metrics; auxiliary-head accuracy is not a latent-physics claim.\n- Q4: A1 extrapolation MAE is {a1ex.MAE:.4f} (n=5); all model predictions and NN seed variation are exported.\n- Q5: long-life metrics are in `regional_metrics.csv`; positive residual means underprediction.\n- Q6: `condition_extrapolation_predictions.csv` gives the five individual points, so aggregate error is not interpreted without inspecting them.\n- Q7: {gate}; see `task_04_7_c_gate_decision.md`.\n\nThe n=5 extrapolation comparison is descriptive only.\n"; (OUT/'UNIFIED_ML_PHYSICS_BENCHMARK_REPORT.md').write_text(report,encoding='utf-8')
 cfg={'task':'TASK 04.7-B2','dataset':str(DATA.relative_to(ROOT)),'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'split':str(SPLIT.relative_to(ROOT)),'seeds':SEEDS,'nn_architecture':'reused Net: input-64-32-8, dropout=0.1, Adam, early stopping=45','xgb':'500 trees, depth=4, lr=.03, subsample=.8, colsample=.8','python':sys.version,'platform':platform.platform(),'rows':len(d)};(OUT/'run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 print('Wrote',OUT)
if __name__=='__main__':main()
