"""Audit untouched project evidence and copy a self-contained frozen source closure."""
import collections,csv,hashlib,io,json,os,pathlib,re,shutil,subprocess,sys,unicodedata
from datetime import datetime,timezone
O=pathlib.Path(__file__).resolve().parents[1];ROOT=O.parents[1];V=ROOT/'artifacts/code_protocol_followup_v2'
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,str(p)
 else:p.write_bytes(b)
def js(p,x):put(p,(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
def csvput(p,rows,fields):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(rows);put(p,s.getvalue().encode())
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def main():
 assert not (O/'preregistration/stage1_design_freeze.json').exists()
 sources={}
 def cp(p,q):
  b=p.read_bytes();put(q,b);sources[p.relative_to(ROOT).as_posix()]=dict(sha256=sha(b),copy=q.relative_to(O).as_posix())
 for folder in ('reused_source','tokenizer','verified_model_files'):
  for p in sorted((V/'preregistration'/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:cp(p,O/'preregistration'/folder/p.relative_to(V/'preregistration'/folder))
 for name in ('model.json','generation_config.json','chat_rendering.json','chat_template.jinja','scaffold_v2.txt'):
  cp(V/'preregistration'/name,O/'preregistration'/name)
 for name in ('protocol.py','run_inference.py','cost_gate.py','test_protocol.py'):
  cp(V/'src'/name,O/'reference/table8_v2/src'/name)
 for name in ('analysis_plan.md','freeze.json','main_freeze.json','remote_preflight.json','input_audit.json'):
  cp(V/'preregistration'/name,O/'reference/table8_v2/preregistration'/name)
 for p in [V/'report_zh.md',V/'costs/physical_accounting.json',ROOT/'artifacts/code_extension_v2/pilot_protocol.json',ROOT/'artifacts/code_extension_v2/model_inventory.json',ROOT/'artifacts/code_extension_v2/task_selection.json',ROOT/'artifacts/code_extension_v2/runtime/generation_mistral_803174.json',ROOT/'artifacts/code_extension_v2/instances.json',ROOT/'artifacts/code_pilot_v1/instances.json',ROOT/'artifacts/code_pilot_v1/task_pool.json']:
  cp(p,O/'reference'/p.relative_to(ROOT/'artifacts'))
 for name in ('README.md','docs/CODE_EXTENSION_V2.md','docs/CODE_PILOT_V1.md','docs/CLUSTER_CLUSTER'):
  cp(ROOT/name,O/'reference/project_docs'/name)
 for p in (ROOT/'artifacts/code_pilot_v1/upstream').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:cp(p,O/'reference/upstream'/p.relative_to(ROOT/'artifacts/code_pilot_v1/upstream'))
 # Freeze protection is additive and checks existing data, scores, code and manuscript.
 baseline={}
 for parent in [ROOT/'src',ROOT/'scripts',ROOT/'docs',ROOT/'manuscript']+list((ROOT/'artifacts').glob('code_*')):
  for p in parent.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.png','.pdf') and not any(x in p.parts for x in ('portable_repro_a84fn3lq',)):
    baseline[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
 js(O/'preregistration/protected_sources.json',baseline)
 put(O/'preregistration/git_status_before.txt',subprocess.check_output(['git','status','--short'],cwd=ROOT))
 put(O/'preregistration/git_head.txt',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
 # The applicable directory/ancestor scan found no AGENTS.md.
 candidates=[ROOT, *ROOT.parents, O, O.parent]
 agents=[str(p/'AGENTS.md') for p in candidates if (p/'AGENTS.md').exists()]
 assert not agents,agents
 if not (O/'preregistration/instruction_audit.json').exists():
  js(O/'preregistration/instruction_audit.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),applicable_AGENTS=agents,read_documents=['README.md','docs/CODE_EXTENSION_V2.md','docs/CODE_PILOT_V1.md','docs/CLUSTER_CLUSTER','artifacts/code_protocol_followup_v2/README.md','artifacts/code_protocol_followup_v2/preregistration/analysis_plan.md'],all_writes_restricted_to=O.name))
 a=load(ROOT/'artifacts/code_pilot_v1/instances.json');b=load(ROOT/'artifacts/code_extension_v2/instances.json')
 used=collections.defaultdict(set)
 for study,items in [('pilot_v1',a),('extension_v2',b)]:
  for r in items:used[r['task_id']].add(study+'_'+('development30' if study=='pilot_v1' and r['split']=='main' else r['split']))
 remote_files=sorted((O/'operations').glob('audit_remote_tasks_*.out'));assert remote_files
 remote=load(remote_files[-1])
 for t in remote['task_ids']:used[t].add('remote_actual_candidate_or_judge_evidence')
 # Search exact new-ID content throughout local artifacts and code; preserve all hits.
 pool=load(ROOT/'artifacts/code_pilot_v1/task_pool.json');byid={r['task_id']:r for r in pool['tasks']}
 dataset=list((O/'reference/upstream').rglob('HumanEvalPlus-v0.1.10.jsonl'))[0]
 raw=dataset.read_bytes();assert sha(raw)=='42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f'
 tasks=[json.loads(l) for l in raw.splitlines()];full={r['task_id']:r for r in tasks}
 norms={t:sha(norm(r['prompt']).encode()) for t,r in full.items()}
 eligible=[t for t,r in byid.items() if r['eligible'] and t not in used]
 pattern=r'HumanEval[/_]('+ '|'.join(t.split('/')[1] for t in eligible)+r')([^0-9]|$)'
 cmd=['rg','-l',pattern,'artifacts','src','scripts','tests','docs','-g','*.json','-g','*.jsonl','-g','*.csv','-g','*.py','-g','*.md','-g','!table8_independent_replication_v1/**']
 proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True);assert proc.returncode in (0,1)
 hits=[]
 for name in proc.stdout.splitlines():
  p=ROOT/name
  if O in p.parents:continue
  rel=p.relative_to(ROOT).as_posix()
  category='reference_only_or_reserve' if ('/screening/' in rel or '/upstream/' in rel or rel.endswith(('task_pool.json','task_selection.json'))) else ('integrity_hash_index_only' if rel in ('artifacts/code_pilot_v1/handoff_manifest.json','artifacts/code_extension_v2/prior_v1_baseline.json') else 'review_required')
  hits.append(dict(path=rel,sha256=sha(p.read_bytes()),classification=category))
 assert all(x['classification']!='review_required' for x in hits),[x['path'] for x in hits if x['classification']=='review_required']
 js(O/'preregistration/local_reserve_usage_audit.json',dict(command=cmd,hits=hits,interpretation='Reference-only eligibility screening predates this replication, explicitly inherited per original input rules; no candidate generation, tuning or judge scoring found for remaining IDs.'))
 usednorm={norms[t] for t in used if t in norms}
 excluded=[];selected=[];seen=set()
 for t in sorted(full):
  why=sorted(used[t]) if t in used else []
  if not byid[t]['eligible']:why.append('original_reference_ineligible:'+str(byid[t].get('reason')))
  if t not in used and norms[t] in usednorm:why.append('normalized_problem_duplicate_of_used_task')
  if not why and norms[t] in seen:why.append('normalized_problem_duplicate_within_new_pool')
  if not why:selected.append(t);seen.add(norms[t])
  else:excluded.append(dict(task_id=t,reason=';'.join(why),normalized_problem_sha256=norms[t]))
 if len(selected)>100:
  import numpy as np
  selected=sorted(np.random.default_rng(20260912).choice(selected,100,replace=False).tolist())
 instances=[]
 for t in selected:
  r=byid[t]
  assert r['problem']==full[t]['prompt']
  instances.append(dict(task_id=t,split='main',problem=r['problem'],entry_point=r['entry_point'],reference_code=r['reference_code'],tests=[dict(index=j,args=x,expected_repr=r['expected_repr'][j],comparison='Python ==',assertion=f"assert {r['entry_point']}(*{repr(x)}) == {r['expected_repr'][j]}") for j,x in enumerate(r['inputs'])]))
  cp(ROOT/'artifacts/code_pilot_v1/screening'/(t.replace('/','_')+'.json'),O/'reference/screening'/(t.replace('/','_')+'.json'))
 csvput(O/'old_task_exclusions.csv',excluded,['task_id','reason','normalized_problem_sha256'])
 csvput(O/'task_manifest.csv',[dict(task_id=t,status='selected_before_generation',problem_sha256=sha(full[t]['prompt'].encode()),normalized_problem_sha256=norms[t],candidate_sha256='',z='',primary_population='pending_execution_truth') for t in selected],['task_id','status','problem_sha256','normalized_problem_sha256','candidate_sha256','z','primary_population'])
 js(O/'instances.json',instances)
 js(O/'preregistration/task_selection.json',dict(benchmark='HumanEval+ v0.1.10',benchmark_tasks=len(full),reference_eligible=sum(r['eligible'] for r in byid.values()),used_count=len(used),pilot_main=30,extension_main=80,old_smoke=6,remaining_eligible=len(eligible),selected_task_ids=selected,fixed_n=len(selected),sampling='All remaining eligible when <=100; otherwise sorted-ID NumPy PCG64 seed 20260912 choice without replacement 100',normalization='Unicode NFKC, casefold, all whitespace collapsed; SHA256 of complete problem prompt',normalized_duplicates_removed=sum('normalized_problem_duplicate' in r['reason'] for r in excluded),reference_screening_is_not_candidate_or_judge_use=True))
 js(O/'preregistration/source_manifest.json',sources)
 print('PREPARED',len(selected),'new tasks; excluded',len(excluded),'of',len(full),'protected files',len(baseline))
if __name__=='__main__':main()
