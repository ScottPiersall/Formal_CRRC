"""Full task-level descriptive tables and frozen confirmatory comparisons."""
import collections
import math
import numpy as np
from statistics_core import curve_metrics,count_metrics,contrasts,interactions

STRATA=['all','partial','z0','z8','nonconstant']
def included(z,s):return s=='all' or s=='partial' and 1<=z<=7 or s=='z0' and z==0 or s=='z8' and z==8 or s=='nonconstant' and z<8
def interval(values):
    if not values:return {'n':0,'estimate':None,'CI95':None}
    x=np.array(values,float);rng=np.random.Generator(np.random.PCG64(20260914))
    boot=x[rng.integers(0,len(x),size=(9999,len(x)))].mean(axis=1)
    return {'n':len(x),'estimate':float(x.mean()),'CI95':[float(v) for v in np.quantile(boot,[.025,.975],method='linear')],'CI_type':'marginal task-percentile bootstrap, descriptive'}
def bounds_mean(pairs):return [sum(p[i] for p in pairs)/len(pairs) for i in [0,1]] if pairs else None

def aggregate(records,truth,counts,trace_observed):
    """Input records contain only validated metrics/diagnostics, never fake samples."""
    curves=[];lookup={}
    for task,z in truth.items():
        for judge in ['qwen','mistral']:
            for template in ['original','explicit']:
                for protocol in ['same_engine_immediate','corrected_RTS_v2']:
                    for readout in ['bare','union']:
                        margins=[records.get((judge,task,template,protocol,k),{}).get('scores',{}).get(readout+'_margin') for k in range(9)]
                        row=dict(curve_metrics(margins,z),judge=judge,task_id=task,z=z,template=template,protocol=protocol,readout=readout,margins=margins)
                        if z==8:fs_bounds=None
                        else:fs_bounds=[row['FSRR']]*2 if row['FSRR'] is not None else [0,1]
                        row['FSRR_bounds']=fs_bounds;row['oracle_errors_bounds']=[row['oracle_errors']]*2 if row['oracle_errors'] is not None else [0,min(z+1,8-z)]
                        known_wrong=any(m is not None and (m>=0)!=(k<=z) for k,m in enumerate(margins))
                        row['direct_all_nine_bounds']=[row['direct_all_nine']]*2 if row['complete'] else [0,0] if known_wrong else [0,1]
                        curves.append(row);lookup[judge,task,template,protocol,readout]=row
    summaries=[];paired=[]
    for stratum in STRATA:
        tasks=[t for t,z in truth.items() if included(z,stratum)]
        for judge in ['qwen','mistral']:
            for template in ['original','explicit']:
                for readout in ['bare','union']:
                    for protocol in ['same_engine_immediate','corrected_RTS_v2']:
                        rows=[lookup[judge,t,template,protocol,readout] for t in tasks]
                        for metric in ['accuracy','FSRR','oracle_errors','direct_all_nine']:
                            eligible=[r for r in rows if metric!='FSRR' or r['z']<8];values=[r[metric] for r in eligible if r[metric] is not None]
                            summaries.append(dict(interval(values),stratum=stratum,judge=judge,template=template,readout=readout,protocol=protocol,metric=metric,
                              planned_tasks=len(tasks),eligible_tasks=len(eligible),complete_metric_tasks=len(values),
                              point_coverage=sum(r['point_coverage'] for r in rows)/len(rows) if rows else None,
                              full_population_identification_bounds=bounds_mean([r[metric+'_bounds'] for r in eligible]),
                              note='FSRR undefined for z=8; oracle errors for z=8 are structurally zero regardless of ranking.' if metric in ['FSRR','oracle_errors'] and stratum=='z8' else None))
                    for metric in ['accuracy','FSRR','oracle_errors','direct_all_nine']:
                        values=[];bounds=[];gain=loss=0;eligible=0
                        for task in tasks:
                            if metric=='FSRR' and truth[task]==8:continue
                            eligible+=1;a=lookup[judge,task,template,'same_engine_immediate',readout];b=lookup[judge,task,template,'corrected_RTS_v2',readout]
                            lo,hi=a[metric+'_bounds'];blo,bhi=b[metric+'_bounds'];bounds.append([blo-hi,bhi-lo])
                            if a[metric] is not None and b[metric] is not None:
                                d=b[metric]-a[metric];values.append(d)
                                if metric=='FSRR':gain+=d==1;loss+=d==-1
                        paired.append(dict(interval(values),stratum=stratum,judge=judge,template=template,readout=readout,metric=metric,direction='RTS minus immediate',
                           planned_tasks=len(tasks),eligible_tasks=eligible,common_complete_tasks=len(values),FSRR_gain=gain if metric=='FSRR' else None,
                           FSRR_loss=loss if metric=='FSRR' else None,full_population_identification_bounds=bounds_mean(bounds),significance_test=None))
    count_task_rows=[];count_summaries=[]
    for task,z in truth.items():
        for judge in ['qwen','mistral']:
            for template in ['original','explicit']:
                c=counts.get((judge,task,template),{'observed':False,'parsed':False,'failure_reason':'no_complete_generation_receipt'})
                row=dict(c,task_id=task,z=z,judge=judge,template=template)
                row.update(count_metrics(c.get('predicted_count') if c.get('parsed') else None,z));count_task_rows.append(row)
    for stratum in STRATA:
        for judge in ['qwen','mistral']:
            for template in ['original','explicit']:
                rows=[r for r in count_task_rows if r['judge']==judge and r['template']==template and included(r['z'],stratum)];n=len(rows)
                observed=sum(r['observed'] for r in rows);parsed=sum(r['parsed'] for r in rows);failed=sum(r['observed'] and not r['parsed'] for r in rows)
                base={'stratum':stratum,'judge':judge,'template':template,'planned_tasks':n,'observed_generation_receipts':observed,'parsed':parsed,
                  'parse_failures':failed,'unobserved':n-observed,'parse_failure_rate_observed':failed/observed if observed else None,
                  'parse_failure_rate_planned':failed/n if n else None,'parsed_yield_planned':parsed/n if n else None,
                  'parse_failure_task_CI95':interval([int(r['observed'] and not r['parsed']) for r in rows if r['observed']])}
                for metric in ['count_exact_match','accuracy','direct_all_nine']:
                    values=[r[metric] for r in rows if r['parsed']]
                    count_summaries.append(dict(base,metric=metric,parsed_only=interval(values),
                      observed_success_yield_planned=sum(values)/n if n else None,
                      full_population_identification_bounds=[sum(values)/n,(sum(values)+n-parsed)/n] if n else None,
                      note='direct all-nine equals count exact match by deterministic comparison; not independent evidence. No recoverability-score superiority claim.'))
    visible_tasks=[];visible_groups=[]
    for task in truth:
        for judge in ['qwen','mistral']:
            for template in ['original','explicit']:
                keys=[(judge,task,template,'corrected_RTS_v2',k) for k in range(9)]
                rows=[records.get(key,{}) for key in keys];valid=[r for r in rows if r.get('status')=='ok' and r.get('rts',{}).get('valid_boundary')]
                marker=sum(r['rts']['marker_only'] for r in valid);visible=sum(r['rts']['visible_prefix'] for r in valid)
                visible_tasks.append({'task_id':task,'judge':judge,'template':template,'planned':9,'observed_requests':sum(key in trace_observed for key in keys),
                  'valid_scored_contexts':len(valid),'valid_boundaries_in_condition_receipts':sum(r.get('rts',{}).get('valid_boundary',False) for r in rows),
                  'marker_only':marker,'visible_prefix':visible,'missing_scored_contexts':9-len(valid),
                  'missing_reasons':dict(collections.Counter(r.get('reason','no_complete_score_receipt') for r in rows if r not in valid))})
    for judge in ['qwen','mistral']:
        for template in ['original','explicit']:
            rows=[r for r in visible_tasks if r['judge']==judge and r['template']==template];n=len(rows);planned=9*n
            valid=sum(r['valid_scored_contexts'] for r in rows);marker=sum(r['marker_only'] for r in rows)
            # Ratio interval resamples whole tasks, retaining all nine thresholds.
            ratio_ci=None
            if rows and valid:
                rng=np.random.Generator(np.random.PCG64(20260914));indices=rng.integers(0,n,size=(9999,n))
                numer=np.array([r['marker_only'] for r in rows])[indices].sum(1);denom=np.array([r['valid_scored_contexts'] for r in rows])[indices].sum(1)
                if (denom>0).all():ratio_ci=[float(x) for x in np.quantile(numer/denom,[.025,.975])]
            visible_groups.append({'judge':judge,'template':template,'planned':planned,'observed_requests':sum(r['observed_requests'] for r in rows),
              'valid_scored_contexts':valid,'marker_only':marker,'visible_prefix':sum(r['visible_prefix'] for r in rows),
              'valid_over_planned':valid/planned if planned else None,'marker_only_over_valid':marker/valid if valid else None,
              'marker_only_over_planned':marker/planned if planned else None,'marker_only_over_valid_task_bootstrap_CI95':ratio_ci,
              'marker_only_over_planned_task_CI95':interval([r['marker_only']/9 for r in rows]),'extra_significance_tests':0})
    primary_records={k:v for k,v in records.items() if v.get('status')=='ok'}
    primary=contrasts(primary_records,truth);secondary=interactions(primary_records,truth);joint=[]
    for judge in ['qwen','mistral']:
        a=next(r for r in primary if r['judge']==judge and r['metric']=='accuracy');f=next(r for r in primary if r['judge']==judge and r['metric']=='FSRR')
        joint.append({'judge':judge,'accuracy_estimate':a['estimate'],'FSRR_estimate':f['estimate'],
          'both_Holm_significant':all(r['Holm_p'] is not None and r['Holm_p']<=.05 for r in [a,f]),
          'accuracy_up_FSRR_down':a['estimate'] is not None and f['estimate'] is not None and a['estimate']>0 and f['estimate']<0,
          'required_direction':False,'joint_power_guarantee':False})
    return {'curve_task_rows':curves,'stratified_metrics':summaries,'paired_descriptive':paired,'primary_six_tests':primary,
       'secondary_six_interactions':secondary,'joint_direction_summary':joint,'count_task_rows':count_task_rows,'count_summaries':count_summaries,
       'visible_prefix_task_rows':visible_tasks,'visible_prefix_groups':visible_groups}
