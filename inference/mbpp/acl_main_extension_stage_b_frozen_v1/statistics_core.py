"""Unchanged validated task-level statistical core from Stage B preparation."""
import math
import numpy as np
def curve_metrics(margins,z):
    if type(z) is not int or not 0<=z<=8 or len(margins)!=9:raise ValueError('Nine thresholds and integer z required')
    truth=[k<=z for k in range(9)];known=[m is not None and math.isfinite(m) for m in margins]
    correct=sum((m>=0)==t for m,t,ok in zip(margins,truth,known) if ok)
    output={'point_coverage':sum(known)/9,'complete':all(known),'accuracy_bounds':[correct/9,(correct+9-sum(known))/9],
      'accuracy':None,'FSRR':None,'oracle_errors':None,'direct_all_nine':None}
    if not all(known):return output
    cut_predictions=[[False]*9]+[[m>=cut for m in margins] for cut in sorted(set(margins))]
    output.update(accuracy=correct/9,FSRR=None if z==8 else int(min(margins[:z+1])>max(margins[z+1:])),
      oracle_errors=min(sum(a!=b for a,b in zip(prediction,truth)) for prediction in cut_predictions),direct_all_nine=int(correct==9))
    return output

def mcnemar(gain,loss):
    n=gain+loss
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.0

def holm(pvalues):
    # NA hypotheses remain in the prespecified family with p=1 for adjustment.
    m=len(pvalues);order=sorted(range(m),key=lambda i:1.0 if pvalues[i] is None else pvalues[i]);result=[None]*m;running=0
    for rank,i in enumerate(order):
        p=1.0 if pvalues[i] is None else pvalues[i];running=max(running,min(1.0,(m-rank)*p))
        result[i]=None if pvalues[i] is None else running
    return result

def paired_bootstrap(values,seed=20260914,replicates=9999):
    x=np.asarray(values,dtype=float);n=len(x)
    if n<2 or not np.isfinite(x).all():return {'n':n,'estimate':None,'p':None,'CI95':None,'status':'insufficient_or_nonfinite'}
    mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(n))
    if se==0:return {'n':n,'estimate':mean,'p':None,'CI95':None,'status':'degenerate_observed_variance'}
    rng=np.random.Generator(np.random.PCG64(seed));sample=x[rng.integers(0,n,size=(replicates,n))]
    bs_se=sample.std(axis=1,ddof=1)/math.sqrt(n)
    if np.any(bs_se==0):return {'n':n,'estimate':mean,'p':None,'CI95':None,'status':'degenerate_bootstrap_variance','degenerate_draws':int((bs_se==0).sum())}
    t=(sample.mean(axis=1)-mean)/bs_se;quantiles=np.quantile(t,[.025,.975],method='linear')
    return {'n':n,'estimate':mean,'SE':se,'p':float((1+np.sum(np.abs(t)>=abs(mean/se)))/(replicates+1)),
       'CI95':[float(mean-quantiles[1]*se),float(mean-quantiles[0]*se)],'status':'ok','replicates':replicates}

def count_metrics(predicted,z):
    if predicted is None:return {'parsed':False,'count_exact_match':None,'accuracy':None,'direct_all_nine':None}
    if type(predicted) is not int or not 0<=predicted<=8:raise ValueError('Count must be an integer 0..8')
    return {'parsed':True,'count_exact_match':int(predicted==z),'accuracy':1-abs(predicted-z)/9,'direct_all_nine':int(predicted==z)}

def metric_bound(metric,z,value):
    if value is not None:return [value,value]
    return [0,min(z+1,8-z)] if metric=='oracle_errors' else [0,1]

def contrasts(records,truth,readout='bare',template='original'):
    results=[]
    for judge in ['qwen','mistral']:
        for metric in ['accuracy','FSRR','oracle_errors']:
            values=[];gains=losses=0;lower=upper=0;planned=0
            for task,z in truth.items():
                if not 1<=z<=7:continue
                planned+=1;pair=[]
                for protocol in ['same_engine_immediate','corrected_RTS_v2']:
                    margins=[records.get((judge,task,template,protocol,k),{}).get('scores',{}).get(readout+'_margin') for k in range(9)]
                    pair.append(curve_metrics(margins,z))
                immediate,rts=[m[metric] for m in pair]
                a=pair[0]['accuracy_bounds'] if metric=='accuracy' else metric_bound(metric,z,immediate)
                b=pair[1]['accuracy_bounds'] if metric=='accuracy' else metric_bound(metric,z,rts)
                lower+=b[0]-a[1];upper+=b[1]-a[0]
                if immediate is not None and rts is not None:
                    delta=rts-immediate;values.append(delta)
                    if metric=='FSRR':gains+=int(delta==1);losses+=int(delta==-1)
            result=paired_bootstrap(values)
            if metric=='FSRR':result.update(p=mcnemar(gains,losses) if values else None,gain=gains,loss=losses,discordant=gains+losses)
            result.update(judge=judge,metric=metric,template=template,readout=readout,planned_partial_tasks=planned,
              complete_paired_tasks=len(values),full_population_identification_bounds=[lower/planned,upper/planned] if planned else None)
            results.append(result)
    adjusted=holm([r['p'] for r in results])
    for r,p in zip(results,adjusted):r['Holm_p']=p
    return results

def interactions(records,truth):
    # Six secondary contrasts: three metrics x two interaction types.
    # Template interaction averages both judges equally; judge interaction
    # averages both templates equally. Cells are descriptive alongside these.
    results=[]
    for kind in ['template','judge']:
        for metric in ['accuracy','FSRR','oracle_errors']:
            values=[]
            for task,z in truth.items():
                if not 1<=z<=7:continue
                cell={}
                for judge in ['qwen','mistral']:
                    for template in ['original','explicit']:
                        pair=[]
                        for protocol in ['same_engine_immediate','corrected_RTS_v2']:
                            m=[records.get((judge,task,template,protocol,k),{}).get('scores',{}).get('bare_margin') for k in range(9)]
                            pair.append(curve_metrics(m,z)[metric])
                        cell[judge,template]=None if None in pair else pair[1]-pair[0]
                if None in cell.values():continue
                value=sum(cell[j,'explicit']-cell[j,'original'] for j in ['qwen','mistral'])/2 if kind=='template' else sum(cell['mistral',t]-cell['qwen',t] for t in ['original','explicit'])/2
                values.append(value)
            results.append(dict(paired_bootstrap(values),interaction=kind,metric=metric,readout='bare',population='partial_all_eight_cells_complete'))
    for row,p in zip(results,holm([r['p'] for r in results])):row['Holm_p']=p
    return results
