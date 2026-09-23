"""Audit fetched, terminal candidate receipts without executing candidate code."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from common import read,sha,save,inspect_state,name

def quantile(values,q):
    values=sorted(values)
    if not values:return None
    position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(position-lo)*(values[hi]-values[lo])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);a=parser.parse_args()
    snapshot=a.snapshot.resolve()
    if not snapshot.is_relative_to((ROOT/'snapshots').resolve()):raise ValueError('Use an immutable fetched snapshot in this package')
    transfer=read(snapshot/'transfer_receipt.json');work=snapshot/'work'
    for rel,expected in transfer['files'].items():
        p=(snapshot/rel).resolve()
        if not p.is_relative_to(snapshot) or sha(p)!=expected:raise RuntimeError('Transferred work checksum mismatch')
    requests=read(ROOT/'data/requests.json');tasks={t['task_id']:t for t in read(ROOT/'stage_a/main_tasks.json')}
    execution_hash=sha(ROOT/'execution_manifest.json');stage_hash=sha(ROOT/'stage_a_manifest.json')
    state=inspect_state(work,requests,execution_hash,stage_hash)
    out=ROOT/'results'/snapshot.name
    if out.exists():raise FileExistsError('Never overwrite an analysis version')
    rows=[]
    for req in requests:
        p=work/'generation'/name(req['task_id'])
        if not p.exists():continue
        result=read(p);syntax_error=None;entry=None
        try:
            tree=ast.parse(result['candidate_code'])
            entry=any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==tasks[req['task_id']]['entry_point'] for n in tree.body)
        except (SyntaxError,ValueError,RecursionError) as exc:syntax_error=type(exc).__name__+': '+str(exc)
        rows.append({'task_id':req['task_id'],'ordinal':req['ordinal'],'generation_receipt_sha256':sha(p),'candidate_sha256':result['candidate_sha256'],'prompt_tokens':len(result['prompt_token_ids']),'generated_tokens':result['generated_tokens'],'elapsed_seconds':result['elapsed_seconds'],'truncated':result['truncated'],'raw_response_empty':not result['raw_response'].strip(),'candidate_code_empty':not result['candidate_code'].strip(),'syntax_error':syntax_error,'declares_top_level_entry_point':entry,'static_only_no_candidate_execution':True})
    accounting=[];total_seconds=0
    submissions=[read(p) for p in sorted(work.glob('submission_[12].json'))]
    if {r[0] for r in transfer['scheduler_records']}!={s['job_id'] for s in submissions}:raise RuntimeError('Scheduler receipts do not match submitted slots')
    for rec in transfer['scheduler_records']:
        job,status,exit_code,elapsed,tres=rec[:5];gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        if not gpu or int(gpu.group(1))!=1:raise RuntimeError('Exactly one physical GPU required in actual accounting')
        seconds=int(elapsed)
        if seconds>10800:raise RuntimeError('Slot exceeded its approved cap')
        total_seconds+=seconds;accounting.append({'job_id':job,'state':status,'exit_code':exit_code,'physical_GPU_seconds':seconds,'alloc_tres':tres})
    if total_seconds>21600 or len(accounting)>2:raise RuntimeError('Combined approved budget exceeded')
    final_ok=not state['pending'] and len(rows)==276 and all(r['state']=='COMPLETED' and r['exit_code']=='0:0' for r in accounting)
    tokens=[r['generated_tokens'] for r in rows];seconds=sum(r['elapsed_seconds'] for r in rows)
    duplicate_counts=collections.Counter(r['candidate_sha256'] for r in rows)
    summary={'status':'GENERATION_COMPLETE' if final_ok else 'INCOMPLETE_OR_REVIEW_REQUIRED','snapshot':str(snapshot),'execution_manifest_sha256':execution_hash,'stage_a_manifest_sha256':stage_hash,'planned_independent_tasks':276,'completed_candidate_receipts':len(rows),'pending_tasks':state['pending'],'fixed_order_and_all_receipt_bindings_verified':True,'candidates_per_task':1,'scheduler':accounting,'physical_GPU_seconds':total_seconds,'physical_GPU_hours':total_seconds/3600,'authorized_physical_GPU_hours_cap':6,'remaining_slot_needed':False if final_ok else 'requires_clean_checkpoint_review','output_tokens':sum(tokens),'output_length':{'min':min(tokens) if tokens else None,'p50':quantile(tokens,.5),'p95':quantile(tokens,.95),'max':max(tokens) if tokens else None,'mean':statistics.mean(tokens) if tokens else None},'input_tokens':sum(r['prompt_tokens'] for r in rows),'request_seconds':seconds,'output_tokens_per_request_second':sum(tokens)/seconds if seconds else None,'non_request_allocation_seconds':total_seconds-seconds,'truncated_candidates':sum(r['truncated'] for r in rows),'empty_raw_responses':sum(r['raw_response_empty'] for r in rows),'empty_candidate_code':sum(r['candidate_code_empty'] for r in rows),'syntax_parse_failures':sum(r['syntax_error'] is not None for r in rows),'top_level_entry_point_missing_among_parseable':sum(r['syntax_error'] is None and not r['declares_top_level_entry_point'] for r in rows),'duplicate_candidate_code_groups':sum(n>1 for n in duplicate_counts.values()),'duplicate_codes_retained_as_original_distinct_tasks':True,'candidate_execution_performed':False,'z_and_partial_counts':'pending independent CPU truth execution; syntax/entry checks are not correctness','judge_inference_requests':0,'formal_Stage_A_frozen':True,'formal_Stage_B_frozen':False,'raw_receipts_rewritten':False,'transferred_files_verified':len(transfer['files'])}
    save(out/'candidate_static_audit.json',rows);save(out/'generation_summary.json',summary)
    if final_ok:
        save(out/'truth_handoff_manifest.json',{'status':'READY_FOR_CPU_TRUTH_PREPARATION_NOT_STAGE_B','task_order':[r['task_id'] for r in requests],'stage_a_manifest_sha256':stage_hash,'tasks_file':'stage_a/main_tasks.json','tasks_file_sha256':sha(ROOT/'stage_a/main_tasks.json'),'generation_snapshot_relative':snapshot.relative_to(ROOT).as_posix(),'candidate_receipt_bindings':[{k:r[k] for k in ['task_id','ordinal','generation_receipt_sha256','candidate_sha256']} for r in rows],'truth_execution_and_judge_requests_executed':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
