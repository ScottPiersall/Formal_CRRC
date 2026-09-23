"""CPU-only analysis of frozen existing outputs. No imports from model code.

python -X utf8 analyze.py [--output /new/empty/directory]
Existing output files are never overwritten; --output permits clean reproduction.
"""
import argparse, collections, csv, hashlib, json, math, pathlib, sys
from datetime import datetime, timezone
import numpy as np

BASE=pathlib.Path(__file__).resolve().parent
PROTOCOLS=['immediate','rts']; MODES=['raw','learned','oracle']
def load(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def check_freeze():
    # Original freeze is retained locally; verify the anonymous export manifest.
    import subprocess
    subprocess.run([sys.executable, str(BASE.parents[1]/'verify.py')], check=True)
    return load(BASE/'config.json')

def read_scores():
    gg=collections.defaultdict(dict)
    with (BASE/'normalized_scores.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            k=int(r['k']);key=(r['condition'],r['protocol'],r['task_id'])
            assert k not in gg[key]
            gg[key][k]=dict(r,k=k,z=int(r['z']),truth=int(r['truth']),margin=float(r['margin']) if r['status']=='ok' else None)
    return gg
def fit(m,y):
    m=np.asarray(m,dtype=np.float64).reshape(-1);y=np.asarray(y,dtype=bool).reshape(-1)
    assert len(m) and np.isfinite(m).all()
    cs=np.unique(np.r_[m,0.,np.nextafter(m.max(),np.inf)])
    error=np.count_nonzero((m[None,:]>=cs[:,None])!=y[None,:],axis=1)
    best=int(error.min());ties=cs[error==best].tolist();cut=min(ties,key=lambda c:(abs(c),c))
    return dict(cut=float(cut),train_errors=best,train_points=len(m),train_objective=best/len(m),candidate_count=len(cs),optimal_cut_count=len(ties),optimal_cuts=sorted(ties),tie_after_abs_count=sum(abs(c)==abs(cut) for c in ties))
def first_false(p):return next((i for i,v in enumerate(p) if not v),len(p))
def metric(m,y,c):
    pred=np.asarray(m)>=c;truth=np.asarray(y,dtype=bool);e=int(np.count_nonzero(pred!=truth))
    return dict(errors=e,correct=len(y)-e,points=len(y),accuracy=(len(y)-e)/len(y),sequence_correct=int(e==0),tce=abs(first_false(pred)-first_false(truth)))
def ranking(m,y):
    good=[v for v,t in zip(m,y) if t];bad=[v for v,t in zip(m,y) if not t]
    nc=bool(good and bad);fsr=not nc or min(good)>max(bad)
    tr=not nc or min(good)>m[first_false(y)]
    minimum=fit(m,y)['train_errors']
    return dict(nonconstant=int(nc),fsr=int(fsr),tr=int(tr),oracle_minimum_errors=minimum,strict_inversion=int(nc and min(good)<max(bad)),tie_only_failure=int(nc and min(good)==max(bad)))

def main(out):
    cfg=check_freeze();out.mkdir(parents=True,exist_ok=True)
    started=now();gg=read_scores();fold=cfg['fold_assignments'];cuts=[];predictions=[];metrics=[];oracle_cuts=[]
    def curve(c,p,t):
        rows=gg[c,p,t];assert set(rows)==set(range(9)) and all(r['status']=='ok' for r in rows.values())
        return [rows[k]['margin'] for k in range(9)],[rows[k]['truth'] for k in range(9)]
    # Fitter receives ONLY train arrays. Held-out arrays are accessed afterwards.
    for cond,pop in cfg['condition_populations'].items():
        if pop['status']!='run':continue
        ids=pop['task_ids']
        for p in PROTOCOLS:
            for f in range(cfg['n_folds']):
                train=sorted(t for t in ids if fold[t]!=f);test=sorted(t for t in ids if fold[t]==f)
                assert set(train).isdisjoint(test) and set(train)|set(test)==set(ids)
                train_curves=[curve(cond,p,t) for t in train]
                fitted=fit([v for m,y in train_curves for v in m],[v for m,y in train_curves for v in y])
                cuts.append(dict(condition=cond,protocol=p,fold_id=f,train_n=len(train),test_n=len(test),train_task_ids=json.dumps(train),test_task_ids=json.dumps(test),**{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in fitted.items()}))
                for t in test:
                    m,y=curve(cond,p,t);oc=fit(m,y);z=gg[cond,p,t][0]['z']
                    oracle_cuts.append(dict(condition=cond,protocol=p,task_id=t,fold_id=f,z=z,oracle_cut=oc['cut'],oracle_minimum_errors=oc['train_errors'],optimal_cuts=json.dumps(oc['optimal_cuts'])))
                    for k in range(9):
                        predictions.append(dict(condition=cond,protocol=p,task_id=t,z=z,fold_id=f,k=k,truth=y[k],margin=m[k],raw_cut=0.,learned_cut=fitted['cut'],oracle_cut=oc['cut'],raw_pred=int(m[k]>=0),learned_pred=int(m[k]>=fitted['cut']),oracle_pred=int(m[k]>=oc['cut'])))
                    for start in ([0,1] if cond=='original_bare' else [0]):
                        mm,yy=m[start:],y[start:];r=dict(condition=cond,eval_domain='k0_8' if start==0 else 'k1_8',protocol=p,task_id=t,z=z,fold_id=f,k_start=start,**ranking(mm,yy))
                        for mode,c in [('raw',0.),('learned',fitted['cut']),('oracle',oc['cut'])]:
                            r.update({mode+'_'+key:v for key,v in metric(mm,yy,c).items()})
                        r['raw_minus_learned_errors']=r['raw_errors']-r['learned_errors']
                        r['learned_minus_oracle_errors']=r['learned_errors']-r['oracle_errors']
                        metrics.append(r)
    predictions.sort(key=lambda r:(r['condition'],r['task_id'],r['protocol'],r['k']))
    metrics.sort(key=lambda r:(r['condition'],r['eval_domain'],r['task_id'],r['protocol']))
    csvout(out/'fitted_cuts.csv',cuts);csvout(out/'oracle_cuts.csv',oracle_cuts);csvout(out/'oof_predictions.csv',predictions)
    csvout(out/'task_metrics_long.csv',metrics)
    paired={}
    for r in metrics:
        key=(r['condition'],r['eval_domain'],r['task_id'])
        if key not in paired:paired[key]={k:r[k] for k in ['condition','eval_domain','task_id','z','fold_id','k_start']}
        paired[key].update({r['protocol']+'_'+k:v for k,v in r.items() if k not in ['condition','eval_domain','task_id','z','fold_id','k_start','protocol']})
    csvout(out/'paired_task_metrics.csv',list(paired.values()))
    summary=[];rank_summary=[];dist=[];identity=[]
    specs=sorted({(r['condition'],r['eval_domain']) for r in metrics})
    for cond,domain in specs:
        nbase=len(cfg['primary_task_ids']); planned_partial=len(cfg['partial_task_ids'])
        for subset in ['all','partial']:
            for p in PROTOCOLS:
                rr=[r for r in metrics if (r['condition'],r['eval_domain'],r['protocol'])==(cond,domain,p) and (subset=='all' or 1<=r['z']<=7)]
                n=len(rr);planned=nbase if subset=='all' else planned_partial
                for mode in MODES:
                    sums={k:sum(r[mode+'_'+k] for r in rr) for k in ['correct','points','errors','sequence_correct','tce']}
                    summary.append(dict(condition=cond,eval_domain=domain,subset=subset,protocol=p,regime=mode,n_tasks=n,primary_population_n=planned,extra_missing_tasks=planned-n,missing_points_in_evaluated_population=0,correct=sums['correct'],points=sums['points'],accuracy=sums['correct']/sums['points'],sequence_successes=sums['sequence_correct'],sequence_denominator=n,sequence_label='all-nine' if domain=='k0_8' else 'all-eight',sequence_correctness=sums['sequence_correct']/n,tce_total=sums['tce'],mean_tce=sums['tce']/n,total_errors=sums['errors'],mean_errors=sums['errors']/n))
                nc=[r for r in rr if r['nonconstant']];fs=sum(r['fsr'] for r in nc);tr=sum(r['tr'] for r in nc)
                rank_summary.append(dict(condition=cond,eval_domain=domain,subset=subset,protocol=p,n_tasks=n,nonconstant_n=len(nc),fsr_successes=fs,fsrr=fs/len(nc) if nc else None,tr_successes=tr,trr=tr/len(nc) if nc else None,constant_n=n-len(nc),oracle_recoverable_all=sum(r['fsr'] for r in rr),oracle_recoverability_all=sum(r['fsr'] for r in rr)/n,minimum_errors_total=sum(r['oracle_minimum_errors'] for r in rr),minimum_errors_mean=sum(r['oracle_minimum_errors'] for r in rr)/n,strict_inversion_tasks=sum(r['strict_inversion'] for r in rr),tie_only_failure_tasks=sum(r['tie_only_failure'] for r in rr),tr_success_fsr_failure_tasks=sum(r['tr'] and not r['fsr'] for r in nc)))
                count=collections.Counter(r['oracle_minimum_errors'] for r in rr)
                for e in range((9 if domain=='k0_8' else 8)+1):dist.append(dict(condition=cond,eval_domain=domain,subset=subset,protocol=p,oracle_minimum_errors=e,n_tasks=count[e],denominator=n))
    for p in PROTOCOLS:
        rr=[r for r in metrics if r['condition']=='original_bare' and r['eval_domain']=='k0_8' and r['protocol']==p and 1<=r['z']<=7]
        actual=dict(n=len(rr),correct=sum(r['raw_correct'] for r in rr),points=sum(r['raw_points'] for r in rr),all_nine=sum(r['raw_sequence_correct'] for r in rr),tce_total=sum(r['raw_tce'] for r in rr),fsr=sum(r['fsr'] for r in rr),tr=sum(r['tr'] for r in rr))
        assert actual==cfg['table8_identity_expected'][p],(p,actual)
        identity.append(dict(protocol=p,**actual,expected=cfg['table8_identity_expected'][p],matches=True))
    csvout(out/'summary_metrics.csv',summary);csvout(out/'ranking_summary.csv',rank_summary);csvout(out/'oracle_error_distribution.csv',dist)
    write(out/'table8_identity_check.json',dict(status='passed',source='original code_protocol_followup_v2/report_zh.md and manuscript/results_table.tex; raw 23-task Table 8 estimates',rows=identity))
    # Paired bootstrap: one resampled task-index matrix per distinct population.
    indices={};populations={};replicates={};bootcols={};boot=[];sensitivity=[]
    def draw(ids):
        key='p_'+hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:12]
        if key not in indices:
            indices[key]=np.random.Generator(np.random.PCG64(cfg['seed'])).integers(0,len(ids),(cfg['n_bootstrap'],len(ids)),dtype=np.int64)
            populations[key]=ids
        return key,indices[key]
    def boot_block(label,ids,vectors,meta):
        key,idx=draw(ids);cols=list(vectors)
        arr=np.asarray([vectors[c] for c in cols],float).T
        bs=arr[idx].mean(axis=1);replicates[label]=bs;bootcols[label]=dict(columns=cols,population_key=key,**meta)
        for j,c in enumerate(cols):
            lo,hi=np.quantile(bs[:,j],[.025,.975],method='linear')
            boot.append(dict(**meta,contrast=c,n_tasks=len(ids),estimate=float(arr[:,j].mean()),ci_low=float(lo),ci_high=float(hi),replicates=cfg['n_bootstrap'],population_key=key,replicate_array=label,units='errors_per_task' if 'errors' in c else 'proportion'))
    for cond,domain in specs:
        for subset in ['all','partial']:
            ids=sorted(t for c,d,t in paired if c==cond and d==domain and (subset=='all' or 1<=paired[c,d,t]['z']<=7))
            rr=[paired[cond,domain,t] for t in ids];vv={}
            for p in PROTOCOLS:
                for met in ['accuracy','sequence_correct']:
                    vv[f'{p}_learned_minus_raw_{met}']=[r[p+'_learned_'+met]-r[p+'_raw_'+met] for r in rr]
                vv[p+'_raw_minus_learned_errors']=[r[p+'_raw_minus_learned_errors'] for r in rr]
                vv[p+'_learned_minus_oracle_errors']=[r[p+'_learned_minus_oracle_errors'] for r in rr]
            vv['rts_minus_immediate_learned_accuracy']=[r['rts_learned_accuracy']-r['immediate_learned_accuracy'] for r in rr]
            vv['rts_minus_immediate_oracle_minimum_errors']=[r['rts_oracle_minimum_errors']-r['immediate_oracle_minimum_errors'] for r in rr]
            boot_block('_'.join([cond,domain,subset]),ids,vv,dict(condition=cond,eval_domain=domain,subset=subset))
            if cond!='original_bare':
                vectors={}
                for p in PROTOCOLS:
                    for mode in MODES:
                        vectors[p+'_'+mode+'_sensitivity_minus_primary_accuracy']=[paired[cond,domain,t][p+'_'+mode+'_accuracy']-paired['original_bare','k0_8',t][p+'_'+mode+'_accuracy'] for t in ids]
                boot_block(cond+'_versus_primary_'+subset,ids,vectors,dict(condition=cond+'_versus_primary_same_tasks',eval_domain=domain,subset=subset))
    csvout(out/'bootstrap_summary.csv',boot)
    np.savez_compressed(out/'bootstrap_task_indices.npz',**indices)
    np.savez_compressed(out/'bootstrap_replicates.npz',**replicates)
    write(out/'bootstrap_metadata.json',dict(seed=cfg['seed'],n_replicates=cfg['n_bootstrap'],population_task_order=populations,replicate_arrays=bootcols,method='paired task bootstrap, fixed fitted cuts/folds, linear percentile 95% CI; descriptive post-hoc'))
    write(out/'run_receipt.json',dict(started_utc=started,completed_utc=now(),config_sha256=sha(BASE/'config.json'),analysis_script_sha256=sha(pathlib.Path(__file__)),python=sys.version,numpy=np.__version__,fitted_cuts=len(cuts),oof_prediction_rows=len(predictions),task_metric_rows=len(metrics),paired_task_metric_rows=len(paired),bootstrap_contrasts=len(boot)))
    for r in summary:
        if r['condition']=='original_bare' and r['eval_domain']=='k0_8':print(json.dumps(r))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=BASE);main(p.parse_args().output)
