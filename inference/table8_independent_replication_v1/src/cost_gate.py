"""Compute timing gate and reconcile physical GPU use without summing steps."""
import argparse,re
from protocol import *
def physical_jobs(lines):
 jobs={}
 for line in lines.splitlines():
  p=line.strip().split('|')
  if len(p)!=6 or not re.fullmatch(r'\d+(?:\.[a-zA-Z0-9_-]+)?',p[0]):continue
  job=p[0].split('.')[0];state,exit_code,elapsed,tres,node=p[1:]
  match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
  g=int(match[1]) if match else 0
  d=jobs.setdefault(job,dict(max_elapsed_seconds=0,physical_gpu_count=0,states=[],raw_rows=[]))
  d['max_elapsed_seconds']=max(d['max_elapsed_seconds'],int(elapsed));d['physical_gpu_count']=max(d['physical_gpu_count'],g)
  d['states'].append(state);d['raw_rows'].append(line)
 for d in jobs.values():d['physical_gpu_seconds']=d['max_elapsed_seconds']*d['physical_gpu_count']
 return jobs
def smoke_gate():
 inputs=rows(OUT/'preregistration/inference_inputs.jsonl');smoke=[r for r in inputs if r['split']=='smoke'];observed=[]
 missing=[];bad_scoring=[]
 for r in smoke:
  p=OUT/'traces'/(r['condition_id']+'.json')
  if not p.exists():missing.append(r['condition_id']);continue
  t=load(p);t.update(experiment=r['experiment'],variant=r['variant']);observed.append(t)
  if r['experiment']=='A' and t['status']=='ok':
   sp=OUT/'data/scored'/(r['condition_id']+'.json')
   if not sp.exists() or load(sp)['status']!='ok':bad_scoring.append(r['condition_id'])
 calls=[r for p in (OUT/'costs/runtime').glob('*.jsonl') for r in rows(p) if r.get('event')=='completed' and r.get('stage')=='smoke']
 phase={}
 for name in ('A','B','score'):
  rr=[r for r in calls if r['phase']==name];seconds=sum(r['elapsed_seconds'] for r in rr)
  phase[name]=dict(requests=len(rr),seconds=seconds,generated_tokens=sum(r['generated_tokens'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr),
   max_request_seconds=max((r['elapsed_seconds'] for r in rr),default=None),mean_request_seconds=seconds/len(rr) if rr else None,
   generated_tokens_per_second=sum(r['generated_tokens'] for r in rr)/seconds if seconds else None,input_tokens_per_second=sum(r['input_tokens'] for r in rr)/seconds if seconds else None)
 valid_each=all(any(t['experiment']==ex and t['variant']==v and t['status']=='ok' for t in observed) for ex in ('A','B') for v in ('original','explicit'))
 ready=not missing and not bad_scoring and valid_each and all(phase[x]['requests'] for x in phase)
 estimate=None
 if ready:
  # Sequential requests; no unmeasured batched speedup. A has max two trie nodes
  # in the frozen Qwen tokenizer probe; verify actual paths and reserve worst.
  a_valid=sum(t['experiment']=='A' and t['status']=='ok' for t in observed)
  node_multiplier=max(2,phase['score']['requests']/a_valid)
  max_load=max((load(p)['load_seconds'] for p in (OUT/'costs/runtime').glob('*.json') if 'load_seconds' in load(p)),default=0)
  estimate=1.5*(1440*phase['A']['max_request_seconds']+160*phase['B']['max_request_seconds']+1440*node_multiplier*phase['score']['max_request_seconds'])+max_load+600
 result=dict(created_at=now(),technical_gate_passed=ready,planned_A_smoke=54,planned_B_smoke=6,observed=len(observed),missing=missing,scoring_errors=bad_scoring,
  phase=phase,full_main_estimated_physical_gpu_seconds=estimate,account_quota_gate_separate=True,budget_limit_seconds=360000)
 write(OUT/'costs/smoke_gate.json',result);print(canonical(result))
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','accounting']);p.add_argument('--sacct');a=p.parse_args()
 if a.action=='smoke':smoke_gate()
 else:
  jobs=physical_jobs(pathlib.Path(a.sacct).read_text());write(OUT/'costs/physical_accounting.json',dict(jobs=jobs,total_gpu_seconds=sum(x['physical_gpu_seconds'] for x in jobs.values()),billing_conversion_confirmed=False))
if __name__=='__main__':main()
