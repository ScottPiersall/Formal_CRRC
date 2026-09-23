"""Actual timestamped pre-generation design freeze, never backdated."""
import importlib.metadata,pathlib,sys
O=pathlib.Path(__file__).resolve().parents[1];ROOT=O.parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(O/'preregistration/reused_source'))
from protocol import load,write,sha,now
from formalcrrc import code_pilot as pilot,code_extension as cp,code_extension_validation as direct
def main():
 assert not list((O/'generation').glob('*.json')) and not list((O/'traces').rglob('*.json'))
 f=O/'preregistration/stage1_design_freeze.json';assert not f.exists()
 pilot.outdir=cp.outdir=lambda _:O
 pilot.sandbox_smoke(O);direct.smoke(O)
 assert load(O/'sandbox_smoke.json')['passed'] and load(O/'direct_equality_smoke.json')['passed']
 tests=(O/'preregistration/cpu_tests.xml').read_text();assert 'failures="0"' in tests and 'errors="0"' in tests
 write(O/'preregistration/cpu_engineering_gate.json',dict(passed=True,at=now(),scope='Exact archived Table 8 token/scorer/math tests and original isolated executor/direct-equality fixtures; no judge calls on old or new tasks.'),True)
 p=load(O/'reference/code_extension_v2/pilot_protocol.json')
 write(O/'preregistration/templates.json',dict(original=pilot.PROMPT_TEMPLATE,explicit=cp.EXPLICIT_TEMPLATE,RTS_v2=(O/'preregistration/scaffold_v2.txt').read_text(),generation=p['generation']['message']),True)
 write(O/'preregistration/local_environment.json',dict(at=now(),python=sys.version,packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','pytest','tokenizers')}),True)
 n=len(load(O/'instances.json'));assert n==load(O/'preregistration/task_selection.json')['fixed_n'] and n>0
 old=load(ROOT/'artifacts/code_protocol_followup_v2/costs/summary.json')
 gen=load(ROOT/'artifacts/code_extension_v2/runtime/generation_mistral_803174.json')
 write(O/'reference/table8_v2/costs/summary.json',old,True)
 write(O/'preregistration/cost_estimate.json',dict(at=now(),new_task_n=n,judge_contexts_per_protocol=n*18,total_judge_contexts=n*36,candidate_generation_cap=n*1024,RTS_generated_token_cap=n*18*8192,RTS_expected_generated_tokens=n/80*884086,scaled_historical_main_GPU_seconds=n/80*17148,initial_walltime_seconds=7200,total_physical_GPU_budget_seconds=14400,GPUs=1,cpu_per_job=8,source='Archived actual Table 8 805025 main physical allocation (17148 seconds) and cost summary; scale by task count before new generation.'),True)
 write(O/'preregistration/task_manifest_stage1.csv',(O/'task_manifest.csv').read_text(),True)
 files={}
 for d in ('src','preregistration','reference','operations'):
  for q in (O/d).rglob('*'):
   if q.is_file() and '__pycache__' not in q.parts and q.suffix not in ('.out','.err') and q!=f:
    files[q.relative_to(O).as_posix()]=sha(q.read_bytes())
 for name in ('instances.json','old_task_exclusions.csv','sandbox_smoke.json','direct_equality_smoke.json'):
  files[name]=sha((O/name).read_bytes())
 # Read-only resource and remote usage receipts are bound explicitly.
 for pattern in ('inspect_*.out','audit_remote_tasks_*.out','cache_audit_*.out'):
  for q in (O/'operations').glob(pattern):files[q.relative_to(O).as_posix()]=sha(q.read_bytes())
 inherited={k:p[k] for k in ('generator','generation','sandbox','models')}
 record=dict(study='table8_independent_replication_v1',stage=1,frozen_at=now(),identity='Prospective independent-task replication, separate from original preregistration; no new candidate or judge inference exists',task_n=n,statistical_seed=20260912,protocol=inherited,judge_protocol='Exact Table8 corrected RTS v2 and same-engine immediate; see analysis_plan and generation_config, not the old candidate-generation protocol judge fields',analysis_plan='preregistration/analysis_plan.md',files=files)
 write(f,record,True);write(O/'freeze_A.json',f.read_text(),True)
 write(f.with_suffix('.json.sha256'),sha(f.read_bytes())+'  stage1_design_freeze.json\n',True)
 print('STAGE1_FROZEN',record['frozen_at'],n,sha(f.read_bytes()),len(files))
if __name__=='__main__':main()
