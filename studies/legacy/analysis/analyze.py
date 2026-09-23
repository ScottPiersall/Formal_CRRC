"""Post-hoc analysis using compact exports only. No model libraries or inference."""
import argparse, hashlib, itertools, json, math, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
P=argparse.ArgumentParser();P.add_argument('--handoff',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);P.add_argument('--out',type=pathlib.Path);a=P.parse_args()
H=a.handoff.resolve(); O=(a.out or H/'analysis/results').resolve();O.mkdir(parents=True,exist_ok=True);(O/'cases').mkdir(exist_ok=True)
sys.path.insert(0,str(H.parents[1]/'code/src'))
from formalcrrc.day3 import reachable_crossings, reachable_crossings_by_enumeration
from formalcrrc.sequence_recovery import curve_diagnostics
SEED=20260911;B=5000
def jfail(p):return next((i for i,x in enumerate(p) if not x),len(p))
def independent(m,y):
    # Enumerate all attainable decision states directly. Equal scores enter together.
    # This avoids intercept arithmetic, epsilon, and floating-point midpoint issues.
    cuts=sorted(set(m));states=[[v>=c for v in m] for c in cuts]+[[False]*len(m)]
    j=jfail(y);pred=[v>=0 for v in m]; reach=sorted({jfail(p) for p in states})
    trr=j in reach;fsrr=any(p==y for p in states)
    nontrivial=0<j<len(m)
    gap=min(m[:j])-m[j] if nontrivial else None
    fgap=min(m[:j])-max(m[j:]) if nontrivial else None
    if nontrivial:assert trr==(gap>0) and fsrr==(fgap>0)
    return dict(true_first_fail=j,predicted_first_fail=jfail(pred),errors=sum(x!=z for x,z in zip(pred,y)),accuracy=sum(x==z for x,z in zip(pred,y))/len(m),TCE=abs(jfail(pred)-j),TRR=int(trr),FSRR=int(fsrr),TRR_gap=gap,FSRR_gap=fgap,nontrivial=nontrivial,oracle_min_errors=min(sum(x!=z for x,z in zip(p,y)) for p in states),trr_success_fsrr_failure=int(trr and not fsrr),strict_inversion_blocks_trr=int(nontrivial and gap<0),tie_only_blocks_trr=int(nontrivial and gap==0),tie_only_blocks_fsrr=int(nontrivial and fgap==0),reachable=reach,prediction=pred)

boundary_checks=0
for m in itertools.product([-1.,0.,1.],repeat=5):
    mm=list(m)+[m[-1]]*4
    for j in range(10):
        r=independent(mm,[i<j for i in range(9)])
        assert r['reachable']==list(reachable_crossings(mm))
        if j not in [0,9]:
            repo=curve_diagnostics(mm,j);assert repo['trr']==bool(r['TRR']) and repo['fsrr']==bool(r['FSRR'])
        boundary_checks+=1
for mm in [[0.]*9,[1,1,0,0,-1,-1,-2,-2,-3],[1,0,np.nextafter(0.,-1),-1,2,2,2,2,2]]:
    for j in range(10):independent(mm,[i<j for i in range(9)]);boundary_checks+=1

d=pd.read_csv(H/'data/threshold_scores.csv',float_precision='round_trip',keep_default_na=True)
keys=['study','version','model','revision','protocol','template','score_representation','split','artifact_id']
curve=[]; disagreements=[]; raw_validation=[]
for key,g in d.groupby(keys,dropna=False,sort=True):
    g=g.sort_values('threshold'); base=dict(zip(keys,key));base['family']=g.family.iloc[0];base['latent_z']=int(g.latent_z.iloc[0]);base['n_rows']=len(g)
    base['n_valid_rows']=int(((g.completion_status=='complete')&g.semantic_margin.notna()).sum())
    complete=len(g)==9 and list(g.threshold)==list(range(9)) and base['n_valid_rows']==9
    base['completion_status']='complete' if complete else 'incomplete';base['missing_thresholds']=g.loc[(g.completion_status!='complete')|g.semantic_margin.isna(),'threshold'].tolist()
    if complete:
        m=g.semantic_margin.tolist();y=[bool(v) for v in g.truth_label];r=independent(m,y)
        assert list(y)==[i<int(g.true_first_fail.iloc[0]) for i in range(9)]
        base.update(r);base['margins']=m;base['truth']=y;base['printed_thresholds']=g.printed_threshold.astype(int).tolist()
        theorem=list(reachable_crossings(m));enum=list(reachable_crossings_by_enumeration(m))
        assert theorem==r['reachable']
        if enum!=r['reachable']:disagreements.append(dict(**dict(zip(keys,key)),kind='repo_intercept_enumeration_float_arithmetic',ours=r['reachable'],repo=enum))
        if 0<r['true_first_fail']<9:
            rr=curve_diagnostics(m,r['true_first_fail']);assert rr['trr']==bool(r['TRR']) and rr['fsrr']==bool(r['FSRR'])
        base['strict_accuracy']=sum(x==z for x,z in zip(r['prediction'][1:],y[1:]))/8
        strict=independent(m[1:],y[1:]);base['FSRR_excluding_k0']=strict['FSRR'];base['TRR_excluding_k0']=strict['TRR']
    curve.append(base)
c=pd.DataFrame(curve)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x.item() if hasattr(x,'item') else x
with (O/'curves.jsonl').open('w',encoding='utf-8') as h:
    for r in curve:h.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
scalar=c.drop(columns=['margins','truth','printed_thresholds','reachable','prediction','missing_thresholds'],errors='ignore');scalar.to_csv(O/'curve_metrics.csv',index=False,float_format='%.17g')
valid=c[c.completion_status=='complete'].copy();metadata=['study','version','model','revision','protocol','template','score_representation','split']
def interval(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return (None,None)
    rng=np.random.default_rng(SEED);boot=values[rng.integers(0,len(values),size=(B,len(values)))].mean(axis=1)
    return tuple(np.quantile(boot,[.025,.975]))
def summarize(g,base,expected,subset,scope,family='ALL'):
    ng=g[g.nontrivial==True];r=dict(**base,scope=scope,family=family,subset=subset,n_expected_tasks=expected,n_complete_tasks=len(g),n_nontrivial=len(ng),n_all_pass=int((g.true_first_fail==9).sum()),n_all_fail=int((g.true_first_fail==0).sum()))
    for m in ['accuracy','TCE','TRR','FSRR','trr_success_fsrr_failure']:
        use=ng if m in ['TRR','FSRR','trr_success_fsrr_failure'] else g
        r[m]=float(use[m].mean()) if len(use) else None;r[m+'_n']=len(use)
        lo,hi=interval(use[m]);r[m+'_lo95']=lo;r[m+'_hi95']=hi
        if m in ['TRR','FSRR','trr_success_fsrr_failure']:r[m+'_count']=int(use[m].sum())
    for m in ['TRR','FSRR']:r[m+'_including_trivial']=float(g[m].mean()) if len(g) else None
    r['TRR_success_FSRR_failure_share_of_TRR_success']=float(ng.trr_success_fsrr_failure.sum()/ng.TRR.sum()) if ng.TRR.sum() else None
    r['post_hoc']=True;r['bootstrap_seed']=SEED;r['bootstrap_replicates']=B;r['interval_method']='artifact/task-cluster percentile 95%; degenerate intervals are not population certainty'
    return r
summ=[];common_rows=[]
for key,g in c.groupby(metadata,sort=True):
    base=dict(zip(metadata,key)); vg=g[g.completion_status=='complete']
    for family in ['ALL']+sorted(g.family.unique().tolist()):
        full=g if family=='ALL' else g[g.family==family];v=vg if family=='ALL' else vg[vg.family==family]
        for subset in ['all','nontrivial','all_pass','all_fail']+(['partial'] if base['study']=='CODE' else []):
            if subset=='all': filt=lambda x:x
            elif subset=='nontrivial':filt=lambda x:x[(x.latent_z<8)] if base['study']=='CODE' else x[(x.true_first_fail>0)&(x.true_first_fail<9)]
            elif subset=='partial':filt=lambda x:x[(x.latent_z>=1)&(x.latent_z<=7)]
            elif subset=='all_pass':filt=lambda x:x[x.true_first_fail==9]
            else:filt=lambda x:x[x.true_first_fail==0]
            # Incomplete code rows need a known truth boundary for subset denominators.
            if 'true_first_fail' in full:full=full.copy();full['true_first_fail']=full['true_first_fail'].fillna(full.latent_z+1) if base['study']=='CODE' else full.true_first_fail
            chosen=filt(v);expected=len(filt(full));summ.append(summarize(chosen,base,expected,subset,'all_valid',family))
for version,cc in c[(c.study=='CODE')&(c.split=='main')].groupby('version'):
    cells=[set(g.loc[g.completion_status=='complete','artifact_id']) for _,g in cc.groupby(['model','template','score_representation'])]
    common=set.intersection(*cells)
    for task in sorted(common):common_rows.append(dict(version=version,artifact_id=task))
    for key,g in cc[cc.artifact_id.isin(common)].groupby(metadata):
        for subset,v in [('all',g),('nontrivial',g[g.latent_z<8]),('partial',g[(g.latent_z>=1)&(g.latent_z<=7)])]:summ.append(summarize(v,dict(zip(metadata,key)),len(v),subset,'five_models_two_templates_common'))
s=pd.DataFrame(summ);s.to_csv(O/'condition_summary.csv',index=False,float_format='%.17g');pd.DataFrame(common_rows).to_csv(O/'code_common_ids.csv',index=False)
s[(s.study=='CODE')&(s.split=='main')&(s.family=='ALL')].to_csv(O/'code_subsets.csv',index=False,float_format='%.17g')
s[(s.study!='CODE')&(s.study!='S2_CAL')&(s.family=='ALL')&(s.subset=='nontrivial')].to_csv(O/'synthetic_nontrivial.csv',index=False,float_format='%.17g')

# Paired contrasts: keep whole curves and intersect the actual IDs in each condition.
paired=[];pair_details=[]
def pair(left,right,label):
    x=left.set_index('artifact_id');y=right.set_index('artifact_id');ids=sorted(set(x.index)&set(y.index))
    if not ids:return
    for subset,chosen in [('all',ids),('nontrivial',[i for i in ids if x.loc[i,'nontrivial']]),('partial',[i for i in ids if x.loc[i,'study']=='CODE' and 1<=x.loc[i,'latent_z']<=7])]:
        if not chosen:continue
        for metric in ['accuracy','TCE','TRR','FSRR']:
            vals=y.loc[chosen,metric].astype(float).values-x.loc[chosen,metric].astype(float).values;lo,hi=interval(vals)
            paired.append(dict(contrast=label,subset=subset,metric=metric,n_paired=len(chosen),delta_right_minus_left=float(vals.mean()),lo95=lo,hi95=hi,seed=SEED,replicates=B,method='post-hoc paired task-cluster percentile'))
    pair_details.append(dict(contrast=label,task_ids=ids))
for model in sorted(valid.model.unique()):
    orig=valid[(valid.study=='S3')&(valid.model==model)];swap=valid[(valid.study=='LS')&(valid.model==model)]
    if len(orig) and len(swap):pair(orig,swap,'LS minus S3 / '+model)
for key,g in valid[(valid.study=='CODE')&(valid.split=='main')].groupby(['version','model','score_representation']):
    pair(g[g.template=='original'],g[g.template=='explicit'],'CODE explicit minus original / '+' / '.join(key))
for key,g in valid[(valid.study=='CODE')&(valid.split=='main')&(valid.model.str.contains('Qwen3'))].groupby(['template','score_representation']):
    for left,right in [('original_8k_old_scorer','original_8k_shared_prefix_scorer'),('original_8k_shared_prefix_scorer','two_stage_32k_shared_prefix_scorer')]:pair(g[g.version==left],g[g.version==right],f'Q3 code {right} minus {left} / '+str(key))
pd.DataFrame(paired).to_csv(O/'paired_contrasts.csv',index=False,float_format='%.17g');(O/'paired_ids.json').write_text(json.dumps(pair_details,indent=2),encoding='utf-8')

# Representative cases, deterministic lexicographic selection; all counts reported by condition.
cats={'trr_success_fsrr_failure':lambda x:(x.TRR==1)&(x.FSRR==0),'raw_error_fsrr_success':lambda x:(x.errors>0)&(x.FSRR==1),'strict_inversion_trr_failure':lambda x:x.strict_inversion_blocks_trr==1,'tie_only_trr_failure':lambda x:x.tie_only_blocks_trr==1,'tie_only_fsrr_failure':lambda x:x.tie_only_blocks_fsrr==1}
counts=[];chosen=[]
for category,predicate in cats.items():
    for key,g in valid.groupby(metadata):
        ng=g[g.nontrivial==True];selected=ng[predicate(ng)];counts.append(dict(**dict(zip(metadata,key)),category=category,n_nontrivial=len(ng),count=len(selected)))
    primary=valid[(valid.nontrivial==True)&(valid.score_representation=='bare')&((valid.study!='CODE')|((valid.version=='two_stage_32k_shared_prefix_scorer')&(valid.split=='main')))]
    candidates=primary[predicate(primary)].sort_values(['study','version','model','template','artifact_id'])
    for domain,gg in [('synthetic',candidates[candidates.study!='CODE']),('code',candidates[candidates.study=='CODE'])]:
        for _,r in gg.head(2).iterrows():chosen.append(dict(category=category,domain=domain,**r.to_dict()))
pd.DataFrame(counts).to_csv(O/'cases/category_counts.csv',index=False)
with (O/'cases/representative_cases.jsonl').open('w',encoding='utf-8') as f:
    for r in chosen:f.write(json.dumps(r,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else x)+'\n')
case_ids={(r['study'],r['version'],r['model'],r['template'],r['artifact_id']) for r in chosen}
selected_rows=d[[tuple(r) in case_ids for r in d[['study','version','model','template','artifact_id']].itertuples(index=False,name=None)]]
selected_rows[selected_rows.score_representation=='bare'].to_csv(O/'cases/threshold_details.csv',index=False,float_format='%.17g')
lines=['# Representative cases (post-hoc)','Selection: first two lexicographic IDs per category and domain; bare only; complete nontrivial curves; latest code version. Category totals are in category_counts.csv. No examples substitute for estimates.','Immediate versus RTS within Qwen2.5: NOT VERIFIABLE; RTS raw rows absent. Qwen3 versus Qwen2.5 is not a within-model protocol pair.']
for r in chosen:
    lines.extend([f"\n## {r['category']} / {r['study']} / {r['model']} / {r['template']} / {r['artifact_id']}",f"TRR={r['TRR']}; FSRR={r['FSRR']}; accuracy={r['accuracy']:.4f}; TCE={r['TCE']}; j*={r['true_first_fail']}",'|s|printed threshold|truth|margin|raw verdict|','|---|---|---|---|---|'])
    for k in range(9):lines.append(f"|{k}|{r['printed_thresholds'][k]}|{int(r['truth'][k])}|{r['margins'][k]:.17g}|{int(r['prediction'][k])}|")
(O/'cases/README.md').write_text('\n'.join(lines),encoding='utf-8')

# Exploratory screen rule is fixed in this script: accuracy <= 1 percentage point apart,
# TRR or FSRR >= 10 percentage points apart, same study/version/split/subset/family.
comp=s[(s.scope=='all_valid')&(s.subset=='nontrivial')&(s.score_representation=='bare')&(s.split!='smoke')&(s.n_complete_tasks>=20)]
close=[]
for key,g in comp.groupby(['study','version','split','family']):
    for (_,x),(_,y) in itertools.combinations(g.iterrows(),2):
        if abs(x.accuracy-y.accuracy)<=.01 and max(abs(x.TRR-y.TRR),abs(x.FSRR-y.FSRR))>=.1:
            close.append(dict(study=key[0],version=key[1],family=key[3],condition_left=x.model+'/'+x.template,condition_right=y.model+'/'+y.template,n_left=x.n_complete_tasks,n_right=y.n_complete_tasks,accuracy_left=x.accuracy,accuracy_right=y.accuracy,TRR_left=x.TRR,TRR_right=y.TRR,FSRR_left=x.FSRR,FSRR_right=y.FSRR,exploratory=True))
pd.DataFrame(close).to_csv(O/'exploratory_similar_accuracy.csv',index=False)
comp.to_csv(O/'accuracy_recovery_all_conditions.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for study,g in comp[comp.family=='ALL'].groupby('study'):
    for i,m in enumerate(['TRR','FSRR']):ax[i].scatter(g.accuracy,g[m],label=study,s=30,alpha=.75)
for i,m in enumerate(['TRR','FSRR']):ax[i].set(xlabel='Raw accuracy (complete nontrivial curves)',ylabel=m,xlim=(0,1.02),ylim=(-.02,1.02));ax[i].grid(alpha=.2)
ax[1].legend(fontsize=8,ncol=2);fig.suptitle('Post-hoc: all available model/protocol/template conditions, bare labels')
fig.savefig(O/'accuracy_vs_recovery.png',dpi=180);fig.savefig(O/'accuracy_vs_recovery.svg');plt.close(fig)

# Compare with pre-existing code metrics without treating them as inputs.
expected=json.loads((H/'evidence/artifacts/code_extension_v4_q3_continuation/analysis/curve_metrics_by_version.json').read_text())
lookup={(r.version,r.model.split('/')[-1],r.template,r.score_representation,r.artifact_id):r for _,r in valid[(valid.study=='CODE')&(valid.split=='main')].iterrows()}
modelmap={'qwen':'Qwen2.5-14B-Instruct','mistral':'Mistral-7B-Instruct-v0.3','llama':'Llama-3.1-8B-Instruct','gemma':'gemma-2-9b-it','qwen3':'Qwen3-30B-A3B-Thinking-2507'}
checked=0
for version,rr in expected.items():
    for r in rr:
        key=(version,modelmap[r['model_key']],r['variant'],r['representation'],r['task_id']);ours=lookup.get(key)
        if ours is None:disagreements.append(dict(kind='saved_curve_not_reproduced',key=str(key)));continue
        for old,new in [('accuracy','accuracy'),('tce','TCE'),('trr','TRR'),('fsrr','FSRR'),('oracle_min_errors','oracle_min_errors')]:
            if r[old] is None:continue
            # Metric serialization only: accuracy is rational errors/9; compare exact count.
            match=(round(r[old]*9)==round(ours[new]*9)) if old=='accuracy' else r[old]==ours[new]
            if not match:disagreements.append(dict(kind='saved_metric_mismatch',key=str(key),metric=old,saved=r[old],posthoc=ours[new]))
        checked+=1
(O/'validation.json').write_text(json.dumps(dict(post_hoc=True,seed=SEED,bootstrap_replicates=B,boundary_checks=boundary_checks,complete_curves_checked=len(valid),saved_code_curves_compared=checked,disagreements=disagreements,total_export_rows=len(d),complete_rows=int((d.completion_status=='complete').sum()),missing_rows=int((d.completion_status!='complete').sum()),similar_accuracy_pairs=len(close)),indent=2),encoding='utf-8')
print(json.dumps({'complete_curves':len(valid),'condition_rows':len(s),'saved_code_curves_compared':checked,'disagreements':len(disagreements),'similar_accuracy_pairs':len(close)}))
