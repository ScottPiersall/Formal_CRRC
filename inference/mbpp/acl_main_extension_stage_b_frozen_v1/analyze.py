"""Complete frozen analysis from terminal, hash-verified raw request snapshots."""
import argparse
import collections
import math
import sys
import platform
import numpy as np
from common import *
from gates import verify_bundle,TERMINAL
from pipeline import load_rows,validate_scored_result
from ledger import validate_response
sys.path.insert(0,str(ROOT/'runtime'))
from scoring_core import parse_count,ProtocolFailure
from reporting import aggregate

def safe_ref(work,ref):
    p=(work/ref['path']).resolve()
    if not p.is_relative_to(work.resolve()) or sha(p)!=ref['sha256']:raise RuntimeError('Raw evidence reference mismatch')
    return read(p)
def scheduler_cost(records,work):
    submissions=[read(p) for p in (work/'slots').glob('*/submission.json')]
    if {r[0] for r in records}!={r['job_id'] for r in submissions} or len(records)!=len(submissions) or len(records)>24:raise RuntimeError('Scheduler job set differs')
    jobs=[]
    for r in records:
        gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',r[4]);seconds=int(r[3])
        if r[1].split()[0] not in TERMINAL or not gpu or int(gpu[1])!=1 or not 0<=seconds<=28800:raise RuntimeError('Unexpected scheduler accounting')
        jobs.append({'job_id':r[0],'state':r[1],'exit_code':r[2],'physical_GPU_seconds':seconds,'AllocTRES':r[4]})
    total=sum(r['physical_GPU_seconds'] for r in jobs)
    if total>691200:raise RuntimeError('Actual GPU cap exceeded')
    return {'jobs':jobs,'physical_GPU_seconds':total,'physical_GPU_hours':total/3600,'hard_cap_GPU_hours':192,'billing_weight_is_not_GPU_count':True}

def audit_raw(root,work,stage_hash):
    """Validate source membership and recompute scores/parses; retain missing cells."""
    expected={row['result_id']:row for judge in ['qwen','mistral'] for row in load_rows(root,judge)}
    config=read(root/'configs/experiment.json')
    records={};counts={};trace_observed=set();all_requests={};invalid=[];costs=collections.defaultdict(lambda:dict(received_requests=0,generated_tokens_known=0,input_tokens_received=0,elapsed_seconds_known=0.0,elapsed_seconds_unknown_receipts=0,cap_or_length_generation=0))
    missing=[];received_files=set();generation_by_result={}
    for attempt_path in sorted((work/'attempts').glob('*.json')):
        attempt=read(attempt_path);rid=attempt['request_id'];mode=attempt['mode']
        match=re.fullmatch(r'(.+)_(generation|prefix_[0-9]+|teacher_[0-9]+)',rid)
        if not match or match[1] not in expected:raise RuntimeError('Unknown raw request identity')
        row=expected[match[1]];category=row['kind'] if mode=='generation' else 'conditional_scoring_probe' if mode=='score' else 'teacher_forced_parity_probe'
        key=(row['binding']['judge'],row['kind'],category)
        costs[key]
        required_mode='generation' if match[2]=='generation' else 'score' if match[2].startswith('prefix_') else 'parity'
        if mode!=required_mode or mode=='generation' and row['kind']=='same_engine_immediate':raise RuntimeError('Unexpected request mode')
        if attempt['stage_B_sha256']!=stage_hash or attempt_path.name!=rid+'_a'+str(attempt['attempt_index'])+'.json' or attempt['attempt_index'] not in range(3):raise RuntimeError('Attempt header differs')
        if mode=='generation' and attempt['attempt_index']!=0:raise RuntimeError('Generation was retried')
        p=work/'requests'/attempt_path.name
        if not p.exists():missing.append({'request_id':rid,'mode':mode,'judge':key[0],'protocol':row['kind'],'category':category,'attempt_index':attempt['attempt_index']});continue
        raw=read(p);spec=raw['spec'];response=raw['response'];received_files.add(p.name)
        if raw['stage_B_sha256']!=stage_hash or raw['attempt_sha256']!=sha(attempt_path) or digest(spec)!=attempt['spec_sha256'] or raw['spec_sha256']!=digest(spec) or spec['binding']!=row['binding'] or spec['request_id']!=rid or spec['result_id']!=row['result_id'] or spec['mode']!=mode or spec['kind']!=row['kind']:raise RuntimeError('Raw request provenance differs')
        params={'n':1,'max_tokens':1,'temperature':0,'seed':row['seed']}
        if mode=='score':params['logprob_token_ids']=spec['needed_token_ids']
        elif mode=='parity':params['prompt_logprobs']=1
        else:params.update(max_tokens=8192,top_p=1,top_k=-1,min_p=0,repetition_penalty=1,stop=['FINAL:'] if row['kind']=='corrected_RTS_v2' else None,include_stop_str_in_output=True)
        if spec['seed']!=row['seed'] or spec['params']!=params:raise RuntimeError('Sampling parameters differ')
        stat=costs[key];stat['received_requests']+=1;stat['input_tokens_received']+=len(spec['context_token_ids'])
        stat['generated_tokens_known']+=len(response.get('generated_token_ids',[]))
        if isinstance(response.get('elapsed_seconds'),(int,float)) and math.isfinite(response['elapsed_seconds']) and response['elapsed_seconds']>=0:stat['elapsed_seconds_known']+=response['elapsed_seconds']
        else:stat['elapsed_seconds_unknown_receipts']+=1
        if mode=='generation':stat['cap_or_length_generation']+=len(response.get('generated_token_ids',[]))>=8192 or response.get('finish_reason')=='length'
        try:validate_response(response,spec)
        except (RuntimeError,KeyError,TypeError) as exc:invalid.append({'request_id':rid,'reason':str(exc)});continue
        if mode=='generation':
            if spec['context_token_ids']!=row['prompt_token_ids'] or spec['params']['max_tokens']!=8192 or spec['params'].get('stop')!=(['FINAL:'] if row['kind']=='corrected_RTS_v2' else None):raise RuntimeError('Generation context/stop differs')
            if spec['result_id'] in generation_by_result:raise RuntimeError('Duplicate received generation')
            generation_by_result[spec['result_id']]=(p,raw)
            if row['kind']=='corrected_RTS_v2':trace_observed.add((row['binding']['judge'],row['task_id'],row['template'],row['kind'],row['k']))
            elif row['kind']=='count':
                c={'observed':True,'parsed':False,'failure_reason':None,'generated_tokens':response['generated_tokens'],'elapsed_seconds':response['elapsed_seconds']}
                try:c.update(parse_count(response['decoded_without_terminal_eos'],response['generated_tokens'],response['finish_reason'],response['stop_reason'],eos_token_ids=response['eos_token_ids']),parsed=True)
                except ProtocolFailure as exc:c['failure_reason']=str(exc)
                counts[row['binding']['judge'],row['task_id'],row['template']]=c
        elif spec['params']['max_tokens']!=1:raise RuntimeError('Probe generated more than one token')
        # Keep only the path/index, not all large context arrays, across conditions.
        all_requests[p.relative_to(work).as_posix()]=p
    if {p.name for p in (work/'requests').glob('*.json')}!=received_files:raise RuntimeError('Response without a tracked attempt')
    result_count=0
    for p in sorted((work/'results').glob('*.json')):
        r=read(p);rid=r['result_id']
        if rid not in expected or p.name!=rid+'.json':raise RuntimeError('Unknown or duplicated condition file')
        native=expected[rid]
        if r['binding']!=native['binding'] or r['judge']!=native['binding']['judge'] or r['stage_B_sha256']!=stage_hash or r['status'] not in ['ok','terminal_protocol_failure']:raise RuntimeError('Condition provenance differs')
        if any(r[k]!=native[k] for k in ['task_id','condition_id','kind','template','k']):raise RuntimeError('Condition metadata differs')
        raw={ref['path']:safe_ref(work,ref) for ref in r['request_refs']}
        if any(ref not in all_requests for ref in raw):raise RuntimeError('Result references invalid/missing raw evidence')
        for rec in raw.values():
            if rec['spec']['result_id']!=rid:raise RuntimeError('Condition references another task-condition')
        if r['kind']=='count':
            c=counts.get((r['judge'],r['task_id'],r['template']))
            if c is None or (r['status']=='ok')!=c['parsed'] or c['parsed'] and (r['predicted_count']!=c['predicted_count'] or r['verdicts']!=c['verdicts']):raise RuntimeError('Count parse differs from saved generation')
        else:
            key=(r['judge'],r['task_id'],r['template'],r['kind'],r['k'])
            if r['kind']=='same_engine_immediate' and r['status']=='ok' and (r['context_token_ids']!=native['prompt_token_ids'] or r['answer_paths']!=native['answer_paths']):raise RuntimeError('Immediate score context/paths changed')
            if 'rts' in r:
                trace=r['rts'];gen=raw[r['generation_ref']['path']]['response'];n=trace['accepted_generated_prefix_length'];text=trace['accepted_trace_text']
                if not 0<n<=len(gen['generated_token_ids']) or not text.endswith('FINAL:') or text.count('FINAL:')!=1 or not gen['decoded_generated_context'].startswith(text):raise RuntimeError('RTS boundary evidence differs')
                if gen['finish_reason']!='stop' or gen['stop_reason']!='FINAL:' or gen['generated_tokens']>=8192:raise RuntimeError('RTS stop invalid')
                prefix=text[:-6]
                if trace['visible_prefix']!=bool(prefix.strip()) or trace['marker_only']!=not_bool(prefix.strip()) or trace['extra_returned_token_ids']!=gen['generated_token_ids'][n:]:raise RuntimeError('Visible protocol diagnostic differs')
                if r['status']=='ok' and r['context_token_ids']!=native['prompt_token_ids']+gen['generated_token_ids'][:n]:raise RuntimeError('RTS token prefix altered')
            validate_scored_result(r,raw)
            if r['status']=='ok':
                required_parity=r['task_id']==config['parity']['task_id'] and r['k']==0
                if required_parity and [p['form'] for p in r.get('parity',[])]!=list(r['answer_paths']):raise RuntimeError('Required six-form parity missing')
                for parity in r.get('parity',[]):
                    record=raw[parity['request_ref']['path']];context_len=len(r['context_token_ids']);values=record['response']['teacher_forced_actual_token_logprobs'][context_len:]
                    if record['spec']['mode']!='parity' or record['spec']['context_token_ids']!=r['context_token_ids']+r['answer_paths'][parity['form']] or len(values)!=len(r['answer_paths'][parity['form']]) or parity['tolerance']!=config['parity']['max_abs_sequence_logprob_difference']:raise RuntimeError('Parity context/configuration differs')
                    teacher=sum(values);delta=abs(teacher-r['scores']['sequence_logprobs'][parity['form']])
                    if teacher!=parity['teacher_logprob'] or delta!=parity['absolute_difference'] or not parity['passed'] or delta>parity['tolerance']:raise RuntimeError('Parity receipt failed')
            records[key]={k:v for k,v in r.items() if k in ['result_id','status','scores','rts','reason']}
        result_count+=1
    unknown=collections.Counter((m['judge'],m['protocol'],m['category']) for m in missing)
    cost_rows=[dict(judge=j,protocol=protocol,category=category,**values,attempts_without_receipts=unknown[j,protocol,category],
       output_token_total_is_lower_bound=bool(unknown[j,protocol,category]),throughput_output_tokens_per_request_second=values['generated_tokens_known']/values['elapsed_seconds_known'] if values['elapsed_seconds_known'] and not values['elapsed_seconds_unknown_receipts'] else None)
       for (j,protocol,category),values in sorted(costs.items())]
    return records,counts,trace_observed,{'planned_condition_count':len(expected),'completed_condition_receipts':result_count,'raw_request_receipts':len(received_files),
       'attempts_without_receipts':missing,'invalid_engine_responses':invalid,'cost_by_judge_and_category':cost_rows,
       'retained_uncommitted_write_files':[p.relative_to(work).as_posix() for p in work.rglob('*.writing-*')],
       'parsed_counts_recomputed_from_raw_generation':sum(c['parsed'] for c in counts.values()),'raw_score_recomputation':True}
def not_bool(value):return not bool(value)

def markdown_report(tables,audit,cost,truth):
    lines=['# Formal judge analysis\n',f"Independent tasks: {len(truth)}; partial: {sum(1<=z<=7 for z in truth.values())}. Planned conditions: {audit['planned_condition_count']}; complete condition receipts: {audit['completed_condition_receipts']}.\n",
      'All signs and missingness are reported under the frozen rules. Thirty-three partial tasks do not provide adequate power for the previously assumed small effects. Non-significance is not evidence of equivalence or stability.\n',
      '| Judge | Primary metric | Paired n | Estimate (RTS − immediate) | Holm p |\n|---|---|---:|---:|---:|\n']
    def fmt(v):return 'NA' if v is None else f'{v:.6g}'
    for r in tables['primary_six_tests']:lines.append(f"| {r['judge']} | {r['metric']} | {r['n']} | {fmt(r['estimate'])} | {fmt(r['Holm_p'])} |\n")
    lines.append('\nFull stratified tables, task rows, identification bounds, count-control metrics, visible-prefix denominators and all six secondary interactions are in the adjacent JSON files. Descriptive CIs are marginal and do not introduce extra significance tests.\n')
    lines.append(f"\nActual physical GPU hours: {cost['physical_GPU_hours']:.6f}. Input/output token counts and request wall time are separated into RTS, count, scoring and parity probes. Attempted requests without responses make reported output-token totals lower bounds; no missing attempt is assigned zero cost.\n")
    lines.append('\nCount direct-all-nine equals count exact match by deterministic construction; no recoverability score is used as evidence of superiority. Marker-only valid RTS contexts remain in the primary analysis.\n')
    return ''.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    config=verify_bundle()
    expected_numpy=config.get('analysis_runtime',{}).get('numpy')
    if expected_numpy is not None and np.__version__!=expected_numpy:raise RuntimeError('Frozen analysis NumPy version differs; no automatic installation')
    if a.snapshot is None:print(json.dumps({'mode':'schema_only','model_imports':[],'GPU_submissions':0,'report_components':['primary','interactions','all_strata','count','visible_prefix','coverage','cost','raw_provenance']}));return
    if a.output is None or a.output.exists():raise ValueError('A new output directory is required')
    snapshot=a.snapshot.resolve();transfer=read(snapshot/'transfer_receipt.json');work=snapshot/'work';stage=sha(ROOT/'stage_B_manifest.json')
    if transfer['stage_B_sha256']!=stage:raise RuntimeError('Snapshot belongs to another Stage B')
    actual={p.relative_to(snapshot).as_posix() for p in work.rglob('*') if p.is_file()}
    if actual!=set(transfer['files']):raise RuntimeError('Snapshot evidence file set changed')
    for rel,expected in transfer['files'].items():
        target=(snapshot/rel).resolve()
        if not target.is_relative_to(snapshot) or sha(target)!=expected:raise RuntimeError('Snapshot file hash mismatch')
    cost=scheduler_cost(transfer['scheduler_records'],work);records,counts,traces,audit=audit_raw(ROOT,work,stage)
    truth={r['task_id']:r['z'] for r in read(ROOT/'data/candidate_truth_bindings.json')['bindings']}
    tables=aggregate(records,truth,counts,traces);a.output.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items():save(a.output/(name+'.json'),rows)
    save(a.output/'raw_evidence_audit.json',audit);save(a.output/'scheduler_cost.json',cost)
    text_file(a.output/'REPORT.md',markdown_report(tables,audit,cost,truth))
    save(a.output/'analysis_manifest.json',{'created_utc':now(),'stage_B_sha256':stage,'snapshot_transfer_sha256':sha(snapshot/'transfer_receipt.json'),
      'analysis_runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
      'independent_tasks':len(truth),'partial_tasks':sum(1<=z<=7 for z in truth.values()),'new_inference_requests':0,'files':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}})
    print(json.dumps({'output':str(a.output),'completed_condition_receipts':audit['completed_condition_receipts'],'planned_condition_count':audit['planned_condition_count'],'new_inference_requests':0}))
if __name__=='__main__':main()
