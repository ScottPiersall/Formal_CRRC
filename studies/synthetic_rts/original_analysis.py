"""Reproducible CPU-only paired post-hoc analysis of original saved Qwen2.5 S3 outputs.
No torch/transformers, model execution, sampling of traces, or score modification.
"""
from pathlib import Path
import argparse,hashlib,json,platform,re,sys,zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def jsonout(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,f):f.to_csv(p,index=False,float_format='%.17g',lineterminator='\n',na_rep='')

def cut_states(m):
    # m >= each unique stored value, plus the upper-tail all-fail state.
    # Equal margins are indivisible; no midpoint or float addition is required.
    return [m>=v for v in np.unique(m)]+[np.zeros(len(m),dtype=bool)]

def curve_metrics(m,y):
    m=np.asarray(m,dtype=np.float64);y=np.asarray(y,dtype=bool)
    assert len(m)==len(y)==9 and np.isfinite(m).all()
    j=next((s for s,b in enumerate(y) if not b),9)
    assert np.array_equal(y,np.arange(9)<j)
    prediction=m>=0;jhat=next((s for s,b in enumerate(prediction) if not b),9)
    low=float(m[y].min()) if y.any() else None
    high=float(m[~y].max()) if (~y).any() else None
    nontrivial=bool(y.any() and (~y).any())
    gap=low-high if nontrivial else None
    fsr=int(gap>0) if nontrivial else 1
    tr=int(m[j]<m[:j].min()) if nontrivial else 1
    states=cut_states(m)
    matching=[i for i,p in enumerate(states) if np.array_equal(p,y)]
    reached={next((i for i,b in enumerate(p) if not b),9) for p in states}
    assert bool(matching)==bool(fsr) and (j in reached)==bool(tr)
    if nontrivial:assert fsr<=tr
    return {'fsr':fsr,'tr':tr,'separation_gap':gap,
      'gap_class':'constant_oracle_recoverable' if not nontrivial else 'recoverable' if gap>0 else 'exact_tie' if gap==0 else 'strict_inversion',
      'min_passing_margin':low,'max_failing_margin':high,
      'argmin_passing_strictness':json.dumps(np.flatnonzero(y&(m==low)).tolist()) if low is not None else '[]',
      'argmax_failing_strictness':json.dumps(np.flatnonzero((~y)&(m==high)).tolist()) if high is not None else '[]',
      'correct_points':int((prediction==y).sum()),'accuracy':float((prediction==y).mean()),
      'direct_all_nine_correct':int(np.array_equal(prediction,y)),
      'predicted_first_fail':jhat,'tce':abs(jhat-j),'cut_enumeration_fsr':int(bool(matching)),
      'cut_enumeration_tr':int(j in reached),'cut_states_enumerated':len(states),
      'cut_witness_state_index':matching[0] if matching else None}

def bootstrap(n,cfg,seed):
    rng=np.random.default_rng(seed);B=cfg['bootstrap']['replicates'];draws=[];family_results={};fingerprints={}
    for fam in cfg['families']:
        f=n[n.family==fam].sort_values('artifact_id');a=f.immediate_fsr.to_numpy(dtype=float);b=f.rts_fsr.to_numpy(dtype=float)
        ix=rng.integers(0,len(f),size=(B,len(f)))
        # One index matrix draws both protocols, retaining each nine-score curve.
        x=a[ix];y=b[ix];draws.append((x,y))
        family_results[fam]=(x.mean(axis=1),y.mean(axis=1),(y-x).mean(axis=1)*100)
        fingerprints[fam]={'n_artifacts':len(f),'artifact_id_order_sha256':digest('\n'.join(f.artifact_id).encode()),
          'index_matrix_int64_le_sha256':digest(ix.astype('<i8').tobytes())}
    a=np.concatenate([x for x,y in draws],axis=1);b=np.concatenate([y for x,y in draws],axis=1)
    results={'all':(a.mean(axis=1),b.mean(axis=1),(b-a).mean(axis=1)*100),**family_results}
    return results,fingerprints

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE);ap.add_argument('--pyarrow-path',type=Path)
    ap.add_argument('--verify-original-archive',action='store_true');args=ap.parse_args()
    if args.pyarrow_path:sys.path.insert(0,str(args.pyarrow_path.resolve()))
    import pyarrow
    import pandas as pd
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'));root=HERE/'inputs';s3=root/'artifacts/day3';rts=root/'artifacts/reason_then_score_ablation'
    source=json.loads((HERE/'archive_source.json').read_text())
    needed=[s3/n for n in ['test_dataset.parquet','test_manifest.json','prompt_manifest.parquet','preregistration.json',
      'preregistration.sha256','model_provenance.json','metrics_by_artifact.parquet','summary.json',
      'raw_scores/qwen2_5_14b_instruct.parquet','raw_scores/qwen2_5_14b_instruct_run.json']]
    needed+=sorted(p for p in rts.rglob('*') if p.is_file())
    before={str(p):sha(p) for p in needed};by_member={x['archive_member']:x for x in source['selected_members']}
    assert all(sha(p)==by_member[p.relative_to(root).as_posix()]['sha256'] for p in needed)
    ro=json.loads((s3/'raw_scores/qwen2_5_14b_instruct_run.json').read_text())
    rr=json.loads((rts/'run.json').read_text());po=json.loads((s3/'preregistration.json').read_text())
    pr=json.loads((rts/'preregistration.json').read_text());tm=json.loads((s3/'test_manifest.json').read_text())
    prompt=json.loads((rts/'prompt_diff.json').read_text());generation=json.loads((rts/'generation_config.json').read_text())
    integrity=json.loads((rts/'integrity_manifest.json').read_text());fingerprint=json.loads((rts/'frozen_record_fingerprint.json').read_text())
    assert ro['model_id']==rr['model_id']==cfg['model_id']
    assert ro['revision']==rr['revision']==rr['tokenizer_revision']==cfg['revision']
    assert ro['partition']==cfg['partition'] and rr['condition']=='reason_then_score' and rr['protocol']=='elicited_reason_then_score'
    assert rr['marker']=='FINAL:' and rr['rows_scored']==rr['rows_requested'] and rr['retries']==[]
    assert sha(s3/'raw_scores/qwen2_5_14b_instruct.parquet')==ro['raw_scores_sha256']
    assert sha(rts/'raw_rows.parquet')==rr['raw_rows_sha256']==integrity['raw_rows']['raw_rows_sha256']
    assert sha(s3/'test_dataset.parquet')==ro['dataset_file_sha256']==pr['dataset']['validation']['dataset_file_sha256']
    assert sha(s3/'preregistration.json')==ro['preregistration_sha256']
    assert sha(rts/'preregistration.json')==rr['preregistration_sha256']
    assert (rts/'preregistration.sha256').read_text().split()[0]==rr['preregistration_sha256']
    assert pr['frozen_at']<rr['finished_at'] and pr['amendments'] in ([],0)
    assert integrity['status']=='PASS' and integrity['frozen_record']['combined_now']==pr['frozen_record']['combined_sha256']
    assert generation['stage_d']['candidate_met']==' A' and generation['stage_d']['candidate_not_met']==' B'
    assert generation['stage_d']['scoring_method']=='sequence_loglikelihood'
    assert ro['label_tokenization']['label_met']=='A' and ro['label_tokenization']['label_not_met']=='B'
    assert prompt['rts_template']==prompt['original_template']+prompt['added_suffix']
    assert len(prompt['added_suffix'].encode())==prompt['added_bytes']==229
    for k in ['original_template','rts_template','added_suffix']:assert digest(prompt[k].encode())==prompt[k+'_sha256']
    assert prompt['original_template_sha256']==ro['prompt_template_sha256']
    d=pd.read_parquet(s3/'test_dataset.parquet');o=pd.read_parquet(s3/'raw_scores/qwen2_5_14b_instruct.parquet')
    r=pd.read_parquet(rts/'raw_rows.parquet');pm=pd.read_parquet(s3/'prompt_manifest.parquet')
    keys=['artifact_id','strictness_index'];audit={}
    for name,f in [('dataset',d),('immediate',o),('rts',r),('prompt_manifest',pm)]:
        audit[name]={'rows':len(f),'duplicate_key_rows':int(f.duplicated(keys,keep=False).sum()),
          'duplicate_prompt_id_rows':int(f.prompt_id.duplicated(keep=False).sum()),'unique_artifacts':int(f.artifact_id.nunique())}
        assert audit[name]['duplicate_key_rows']==audit[name]['duplicate_prompt_id_rows']==0
        assert set(zip(f.artifact_id,f.strictness_index))==set(zip(d.artifact_id,d.strictness_index))
    assert set(o.model_id)==set(r.model_id)=={cfg['model_id']}
    assert set(o.revision)==set(r.model_revision)==set(r.tokenizer_revision)=={cfg['revision']}
    assert set(o.partition)=={cfg['partition_tag']} and set(r.condition)=={'reason_then_score'}
    assert set(d.seed)=={cfg['dataset_seed']} and d.artifact_id.str.startswith('D3T-').all()
    assert np.isfinite(o[['raw_score_met','raw_score_not_met','margin']]).all().all()
    assert np.isfinite(r[['logprob_a','logprob_b','margin']]).all().all()
    assert np.array_equal(o.raw_score_met-o.raw_score_not_met,o.margin)
    assert np.array_equal(r.logprob_a-r.logprob_b,r.margin)
    assert np.array_equal(r.margin>=0,r.semantic_decision.astype(bool))
    assert set(r.status)=={'ok'} and r.error.isna().all()
    assert r.marker_reached.all() and not r.reasoning_truncated.any() and not r.premature_label_observed.any()
    assert set(r.stop_reason)=={'marker'} and r.text_through_marker.str.endswith('FINAL:').all()
    audit['immediate'].update(nonfinite_score_rows=0,run_failures=ro['failures'])
    audit['rts'].update(nonfinite_score_rows=0,error_rows=0,truncated_rows=0,premature_labels=0,marker_reached_rows=int(r.marker_reached.sum()),retries=rr['retries'])
    content=hashlib.sha256()
    for col in d.columns:content.update(f'#{col}\n'.encode())
    for row in d.sort_values(keys).itertuples(index=False,name=None):
        content.update('\x1f'.join('' if v is None or v is pd.NA else str(v) for v in row).encode());content.update(b'\x1e')
    assert content.hexdigest()==tm['hashes']['dataset_content_sha256']==pr['dataset']['validation']['dataset_content_sha256']
    dg=d.set_index(keys);om=o.set_index(keys);rm=r.set_index(keys);pmap=pm.set_index('prompt_id')
    for row in d.to_dict('records'):
        key=(row['artifact_id'],row['strictness_index']);a=om.loc[key];b=rm.loc[key];s=int(row['strictness_index'])
        for field in ['family','latent_level','prompt_id']:assert row[field]==a[field]==b[field]
        if row['family'] in ['coverage','max_violation']:
            literals={x[2:] for x in row['candidate_response'].splitlines() if x.startswith('- ')}
            listed={x[2:] for x in row['rubric_text'].splitlines() if x.startswith('- ')}
            assert literals==set(json.loads(row['response_markers_json'])) and listed==set(json.loads(row['list_items_json']))
            latent=len(literals&listed)
        else:
            observed=int(re.search(r'value reported by this submission is (-?\d+)\.',row['candidate_response']).group(1))
            target=int(re.search(r'Target value: (-?\d+)',row['rubric_text']).group(1))
            assert observed==row['reported_value'] and target==row['target_value'];latent=abs(observed-target)
        assert latent==row['latent_level'];z=latent if row['family']=='coverage' else 8-latent
        assert int(z>=s)==row['formal_truth']==b.formal_truth and z+1==row['true_first_fail_index']==b.true_first_fail_index
        for text,hashkey in [('candidate_response','candidate_sha256'),('rubric_text','rubric_sha256')]:assert digest(row[text].encode())==row[hashkey]
        msg=prompt['original_template'].format(rubric_text=row['rubric_text'],candidate_response=row['candidate_response'])
        msg_r=prompt['rts_template'].format(rubric_text=row['rubric_text'],candidate_response=row['candidate_response'])
        assert msg==pmap.loc[row['prompt_id']].user_message
        assert digest(msg.encode())==pmap.loc[row['prompt_id']].user_message_sha256==b.day3_user_message_sha256
        assert digest(msg_r.encode())==b.rts_message_sha256
        expected_seed=int.from_bytes(hashlib.sha256(f"42|{row['artifact_id']}|{s}".encode()).digest()[:8],'big')%2147483647
        assert expected_seed==b.generation_seed
        assert len(b.generated_token_ids)==b.generated_token_count
        assert list(b.candidate_a_token_ids)==generation['stage_d']['candidates']['met_token_ids']
        assert list(b.candidate_b_token_ids)==generation['stage_d']['candidates']['not_met_token_ids']
    curves=[]
    for aid,f in d.groupby('artifact_id',sort=True):
        f=f.sort_values('strictness_index');assert f.strictness_index.tolist()==cfg['strictness']
        assert f.family.nunique()==f.latent_level.nunique()==f.candidate_sha256.nunique()==1
        y=f.formal_truth.to_numpy(dtype=bool);z=int(f.latent_level.iloc[0]) if f.family.iloc[0]=='coverage' else 8-int(f.latent_level.iloc[0])
        rec={'artifact_id':aid,'family':f.family.iloc[0],'latent_level_original':int(f.latent_level.iloc[0]),
          'canonical_z':z,'j_star':z+1,'nontrivial':int(y.any() and (~y).any()),'truth_curve':''.join(str(int(v)) for v in y),
          'pair_complete':True,'immediate_rows':9,'rts_rows':9}
        for protocol,frame in [('immediate',om),('rts',rm)]:
            m=frame.loc[aid].sort_index().margin.to_numpy(dtype=np.float64)
            rec.update({protocol+'_'+k:v for k,v in curve_metrics(m,y).items()})
        rec['fsr_change']=rec['rts_fsr']-rec['immediate_fsr']
        rec['transition']={(1,1):'both_recoverable',(1,0):'lost_recoverability',(0,1):'gained_recoverability',(0,0):'neither_recoverable'}[(rec['immediate_fsr'],rec['rts_fsr'])]
        curves.append(rec)
    c=pd.DataFrame(curves);n=c[c.nontrivial==1].copy();csvout(out/'paired_curve_metrics.csv',c)
    csvout(out/'exclusions.csv',pd.DataFrame(columns=['artifact_id','protocol','reason']))
    thresholds=d[['artifact_id','family','latent_level','strictness_index','formal_truth']].sort_values(keys).copy()
    for protocol,f in [('immediate',om),('rts',rm)]:
        scores=f.loc[pd.MultiIndex.from_frame(thresholds[keys])].margin.to_numpy(dtype=np.float64)
        thresholds[protocol+'_margin']=scores;thresholds[protocol+'_margin_hex']=[float(x).hex() for x in scores]
    csvout(out/'paired_threshold_scores.csv',thresholds)
    bootstrap_by_seed={};plans={}
    for label,seed in [('primary',cfg['bootstrap']['primary_seed']),('secondary_seed42',cfg['bootstrap']['secondary_seed_sensitivity'])]:
        samples,fp=bootstrap(n,cfg,seed);bootstrap_by_seed[label]=samples;plans[label]=fp
        rows=[]
        for fam,(a,b,delta) in samples.items():
            rows.extend({'replicate':i+1,'family':fam,'immediate_fsrr':float(a[i]),'rts_fsrr':float(b[i]),'delta_pp':float(delta[i])} for i in range(len(a)))
        csvout(out/('bootstrap_replicates.csv' if label=='primary' else 'bootstrap_replicates_seed42_secondary.csv'),pd.DataFrame(rows))
    fsrsummary=[];transitions=[];secondary=[];summary=[]
    for fam in ['all',*cfg['families']]:
        q=n if fam=='all' else n[n.family==fam]
        base={'family':fam,'n_artifacts':len(q),'immediate_successes':int(q.immediate_fsr.sum()),'rts_successes':int(q.rts_fsr.sum()),
          'immediate_fsrr_percent':float(100*q.immediate_fsr.mean()),'rts_fsrr_percent':float(100*q.rts_fsr.mean()),
          'delta_pp':float(100*q.fsr_change.mean()),'analysis_role':'primary_overall' if fam=='all' else 'secondary_descriptive_family'}
        for label,samples in bootstrap_by_seed.items():
            record=base.copy();record['bootstrap_seed']=cfg['bootstrap']['primary_seed'] if label=='primary' else 42
            for metric,values,factor in zip(['immediate','rts','delta_pp'],samples[fam],[100,100,1]):
                lo,hi=np.quantile(values,[.025,.975],method=cfg['bootstrap']['quantile_method'])*factor
                record[metric+'_ci_low']=float(lo);record[metric+'_ci_high']=float(hi)
            (fsrsummary if label=='primary' else secondary).append(record)
        counts=q.transition.value_counts().to_dict()
        transitions.append({'family':fam,'nontrivial_pairs':len(q),'coverage':1.0,**{k:int(counts.get(k,0)) for k in ['both_recoverable','lost_recoverability','gained_recoverability','neither_recoverable']}})
        fq=c if fam=='all' else c[c.family==fam]
        for subset in ['all','nontrivial','constant']:
            sq=fq if subset=='all' else fq[fq.nontrivial==(1 if subset=='nontrivial' else 0)]
            for protocol in ['immediate','rts']:
                record={'family':fam,'subset':subset,'protocol':protocol,'population':'common_complete_pairs',
                  'n_curves':len(sq),'n_scoring_rows':9*len(sq),'correct_points':int(sq[protocol+'_correct_points'].sum()),
                  'accuracy_percent':float(100*sq[protocol+'_correct_points'].sum()/(9*len(sq))),
                  'tce_sum':int(sq[protocol+'_tce'].sum()),'mean_tce':float(sq[protocol+'_tce'].mean()),
                  'all_nine_correct_count':int(sq[protocol+'_direct_all_nine_correct'].sum()),
                  'all_nine_correct_percent':float(100*sq[protocol+'_direct_all_nine_correct'].mean()),
                  'tr_successes':int(sq[protocol+'_tr'].sum()) if subset=='nontrivial' else None,
                  'trr_percent':float(100*sq[protocol+'_tr'].mean()) if subset=='nontrivial' else None,
                  'fsr_successes':int(sq[protocol+'_fsr'].sum()) if subset=='nontrivial' else None,
                  'fsrr_percent':float(100*sq[protocol+'_fsr'].mean()) if subset=='nontrivial' else None,
                  'constant_oracle_fsr_successes':int(sq[protocol+'_fsr'].sum()) if subset=='constant' else None}
                summary.append(record)
    for record in summary:
        if record['subset']=='nontrivial':
            fs=next(x for x in fsrsummary if x['family']==record['family'])
            prefix=record['protocol']
            record.update(fsrr_ci_low_percent=fs[prefix+'_ci_low'],fsrr_ci_high_percent=fs[prefix+'_ci_high'],
              paired_fsrr_delta_pp=fs['delta_pp'],paired_fsrr_delta_ci_low_pp=fs['delta_pp_ci_low'],
              paired_fsrr_delta_ci_high_pp=fs['delta_pp_ci_high'])
    for record in fsrsummary:
        record['rts_bootstrap_degenerate']=record['rts_ci_low']==record['rts_ci_high']
        record['rts_exact_binomial_ci_low_percent']=float(100*.025**(1/record['n_artifacts'])) if record['rts_successes']==record['n_artifacts'] else None
        record['rts_exact_binomial_ci_high_percent']=100.0 if record['rts_successes']==record['n_artifacts'] else None
    csvout(out/'fsrr_summary.csv',pd.DataFrame(fsrsummary));csvout(out/'transitions.csv',pd.DataFrame(transitions))
    csvout(out/'summary.csv',pd.DataFrame(summary));csvout(out/'seed42_sensitivity.csv',pd.DataFrame(secondary))
    # Compare saved per-artifact results, not hardcoded paper targets.
    old=pd.read_parquet(rts/'metrics_by_artifact.parquet').set_index('artifact_id');checks=0;difference=[]
    for row in curves:
        for prefix,suffix in [('immediate','o'),('rts','r')]:
            for new,previous in [('tce','tce'),('tr','reachable'),('predicted_first_fail','j_hat'),('accuracy','accuracy')]:
                checks+=1
                if row[prefix+'_'+new]!=old.loc[row['artifact_id'],previous+'_'+suffix]:difference.append({'artifact_id':row['artifact_id'],'metric':prefix+'_'+new})
    assert not difference
    paper=[]
    for metric,value,ref in [('immediate_nontrivial_tr_count',int(n.immediate_tr.sum()),138),('rts_nontrivial_tr_count',int(n.rts_tr.sum()),258),
      ('immediate_nontrivial_fsr_count',int(n.immediate_fsr.sum()),136),
      ('immediate_all_accuracy_percent_1dp',round(100*c.immediate_correct_points.sum()/(9*len(c)),1),76.8),
      ('rts_all_accuracy_percent_1dp',round(100*c.rts_correct_points.sum()/(9*len(c)),1),96.5),
      ('immediate_all_tce_3dp',round(c.immediate_tce.mean(),3),3.373),('rts_all_tce_3dp',round(c.rts_tce.mean(),3),1.522)]:
        paper.append({'metric':metric,'recomputed':value,'paper_reference':ref,'match':bool(value==ref)})
    assert all(x['match'] for x in paper)
    constant=c[c.nontrivial==0]
    validation={'status':'PASS','counts':{'curves':len(c),'nontrivial':len(n),'constant':len(constant),'paired_rows':len(d)},
      'family_counts':c.groupby('family').size().to_dict(),'family_nontrivial_counts':n.groupby('family').size().to_dict(),
      'input_audit':audit,'input_pairing_complete':True,'excluded_curves':0,
      'raw_margin_difference_exact':True,'truth_prompt_and_seed_checks':len(d),
      'dataset_content_sha256':content.hexdigest(),'cut_enumeration_comparisons':2*len(c),'cut_disagreements':0,
      'fsr_le_tr_comparisons':2*len(n),'fsr_le_tr_violations':0,
      'saved_per_artifact_comparisons':checks,'saved_per_artifact_discrepancies':difference,'paper_crosschecks':paper,
      'gap_classes':{p:n[p+'_gap_class'].value_counts().to_dict() for p in ['immediate','rts']},
      'tr_success_fsr_failure_ids':{p:n[(n[p+'_tr']==1)&(n[p+'_fsr']==0)].artifact_id.tolist() for p in ['immediate','rts']},
      'original_RTS_statistics':pr['statistics'],'original_summary_bootstrap':json.loads((rts/'summary.json').read_text())['bootstrap']['bootstrap'],
      'bootstrap_draw_fingerprints':plans,'original_full_bootstrap_source_missing':True,
      'original_source_missing_does_not_prevent_independent_score_analysis':True,
      'all_input_hashes_unchanged':all(sha(Path(p))==h for p,h in before.items()),
      'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pyarrow.__version__}}
    assert validation['all_input_hashes_unchanged']
    if args.verify_original_archive:
        archive=Path(source['archive']);assert sha(archive)==source['archive_sha256']
        coverage=[]
        with zipfile.ZipFile(archive) as z:
            for name,expected in fingerprint['files'].items():
                present=name in z.namelist();actual=digest(z.read(name)) if present else None
                coverage.append({'original_fingerprint_path':name,'present_in_archive':present,'expected_sha256':expected,'actual_sha256':actual,'matches':actual==expected if present else None})
        csvout(out/'original_fingerprint_coverage.csv',pd.DataFrame(coverage))
        validation['original_fingerprint_present_files']=sum(x['present_in_archive'] for x in coverage)
        validation['original_fingerprint_total_files']=len(coverage)
        validation['original_fingerprint_present_mismatches']=[x for x in coverage if x['present_in_archive'] and not x['matches']]
        assert not validation['original_fingerprint_present_mismatches']
        validation['original_archive_sha256_unchanged']=sha(archive)==source['archive_sha256']
    jsonout(out/'validation.json',validation)
    manifest={'analysis_type':cfg['analysis_type'],'config':cfg,'config_sha256':sha(HERE/'config.json'),
      'analysis_script_sha256':sha(Path(__file__)),'source_archive':{k:v for k,v in source.items() if k!='selected_members'},
      'model_id':cfg['model_id'],'revision':cfg['revision'],'dataset_content_sha256':content.hexdigest(),
      'protocols':{'immediate':{'margin':'raw_score_met - raw_score_not_met','job_id':ro['slurm']['job_id'],'finished_at':ro['finished_at']},
        'rts':{'margin':'logprob_a - logprob_b','job_id':rr['slurm']['job_id'],'finished_at':rr['finished_at'],'preregistration_sha256':rr['preregistration_sha256']}},
      'inputs':[{'source_path':source['archive']+'::'+p.relative_to(root).as_posix(),'local_path':str(p),
        'package_path':p.relative_to(HERE).as_posix(),'sha256_before':before[str(p)],'sha256_after':sha(p),'bytes':p.stat().st_size} for p in needed],
      'all_input_bytes_unchanged':validation['all_input_hashes_unchanged'],'runtime':validation['runtime']}
    jsonout(out/'input_manifest.json',manifest)
    print(pd.DataFrame(fsrsummary).to_string(index=False));print(pd.DataFrame(transitions).to_string(index=False))
    print(pd.DataFrame(summary).query("family=='all'").to_string(index=False));print('Validation PASS')

if __name__=='__main__':main()
