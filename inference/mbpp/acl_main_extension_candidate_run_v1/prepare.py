"""CPU-only assembly of a new formal-candidate review package; exclusive outputs."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEF=ROOT.parent/'acl_main_extension_protocol_definition_v1'
PLAN=ROOT.parent/'acl_main_extension_planning_v1'
REVIEW=ROOT.parent/'acl_main_extension_stage_a_review_v1'
SMOKE=ROOT.parent/'acl_main_extension_smoke_preparation_v1'
RETRY=SMOKE/'retry_review_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(rel,v):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def copy_file(src,rel):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(src.read_bytes())

def main():
    c=read(DEF/'configs/experiment.reviewed_draft.json');tasks=read(DEF/'data/main_tasks.reviewed_draft.json')
    before={t['task_id']:t for t in read(PLAN/'reference_screening_v2/reference_eligible_manifest.json')}
    early=read(REVIEW/'review/semantic_exclusions.json');late=read(SMOKE/'review/source_decisions.json')
    excluded={t['task_id'] for t in early}|{t['task_id'] for t in late['additional_exclusions']}
    dev=set(c['selection']['development_task_ids']);ids=[t['task_id'] for t in tasks]
    assert len(excluded)==25 and set(ids)==set(before)-dev-excluded and len(ids)==276
    assert ids==c['selection']['main_task_ids'] and len(ids)==len(set(ids))
    clarifications={r['task_id']:r for r in late['clarifications']}
    for t in tasks:
        old=before[t['task_id']]
        for key in ['reference_code','inputs_repr','expected_repr','selected_input_hashes','fixed_order']:
            assert t[key]==old[key],(t['task_id'],key)
        assert t['problem']==clarifications.get(t['task_id'],{}).get('adapted',old['problem'])
        assert len(t['inputs_repr'])==len(t['expected_repr'])==8
        for value in t['inputs_repr']+t['expected_repr']:ast.literal_eval(value)
    files=[(DEF/'data/main_tasks.reviewed_draft.json','stage_a/main_tasks.json'),(DEF/'data/generation_prompts.reviewed_draft.json','data/generation_prompts.json'),(DEF/'data/generation_native.reviewed_draft.json','data/generation_native.json'),(DEF/'configs/templates.json','stage_a/judge_templates.json'),(DEF/'configs/protocol_definition.json','stage_a/protocol_definition.json'),(DEF/'protocol_definition_zh.md','stage_a/protocol_definition_zh.md'),(DEF/'reporting_plan_zh.md','stage_a/reporting_plan_zh.md'),(DEF/'data/development_exclusions.json','stage_a/development_exclusions.json'),(REVIEW/'review/semantic_exclusions.json','stage_a/prior_semantic_exclusions.json'),(SMOKE/'review/source_decisions.json','stage_a/source_decisions.json'),(REVIEW/'review/reference_review_summary.json','stage_a/reference_review_summary.json'),(PLAN/'task_screening_v2/historical_task_exclusions.csv','stage_a/historical_task_exclusions.csv'),(DEF/'audit/native_refresh_comparison.json','stage_a/native_refresh_comparison.json')]
    for src,dest in files:copy_file(src,dest)
    # Preserve provenance of the source statuses; this package's Stage A manifest
    # will govern the new experiment, without rewriting earlier draft records.
    c=copy.deepcopy(c);c.update(study_id='acl_main_extension_formal_v1',status='STAGE_A_CONTENT_READY_EXECUTION_UNAUTHORIZED')
    c['source'].update(task_manifest='stage_a/main_tasks.json',task_manifest_sha256=sha(ROOT/'stage_a/main_tasks.json'),history_path='stage_a/historical_task_exclusions.csv',semantic_review='Consolidated exact set review passed; 25 exclusions, four clarifications; limitations recorded in attribution_and_review_zh.md')
    c['generator'].update(prompt_manifest='data/generation_prompts.json',native_token_manifest='data/generation_native.json',native_token_manifest_sha256=sha(ROOT/'data/generation_native.json'))
    c['judging']['template_file']='stage_a/judge_templates.json';c['judging']['protocol_definition']='stage_a/protocol_definition.json'
    c['freeze']={'stage_A':False,'stage_B':False,'stage_A_pending':['Final CPU validation and content checksum record in this new package'],'stage_B_pending':c['freeze']['stage_B_pending']}
    save('stage_a/scientific_config.json',c)
    info=read(RETRY/'configs/model_inventory.json')['mistral']
    save('configs/model_inventory.json',info)
    save('configs/candidate_run.json',{'scope':'formal_276_candidates_only_no_judge','study_id':'acl_main_extension_formal_v1','task_ids':ids,'candidate_per_task':1,'generator':c['generator'],
      'versions':{'torch':'2.13.0+cu130','transformers':'5.16.1','tokenizers':'0.23.2'},
      'resources':{'partition':'normal','account':'<SLURM_ACCOUNT>','gres':'gpu:nvidia_h100_pcie:1','exclude':'COMPUTE_NODE','cpus':8,'memory':'128G','slots':[{'slot':1,'minutes':180},{'slot':2,'minutes':180}],'total_GPU_seconds_cap':21600,'checkpoint_reserve_seconds':300,'automatic_submissions':False,'slurm_requeue':False},
      'runtime_python':'/REDACTED_LOCAL_PATH','CUDA_HOME':'/apps/cuda/cuda-13.1.0',
      'model_downloads':False,'trust_remote_code':False,'judge_inference_allowed':False,
      'continuation':'Slot 2 only after slot 1 is confirmed COMPLETED exit 0, its immutable checkpoint is clean, and unfinished tasks have no begun request; no automatic slot submission',
      'ambiguous_attempt':'Fail closed: preserve the attempt and report unknown completion; do not regenerate or skip forward',
      'infrastructure_retries_this_package':0,'prior_infrastructure_retry_ceiling':2,'retry_note':'Conservative realization: this package does not exercise the prior allowance for retrying a begun request; any such recovery needs a separately reviewed provenance decision',
      'termination':'Fixed entire 276-task order; no partial-count/effect stopping. Allocation cap may leave incomplete work, never lowers scientific target'})
    # Only immutable public prompts and pinned generator settings enter requests.
    native=read(ROOT/'data/generation_native.json');prompts=read(ROOT/'data/generation_prompts.json')
    requests=[]
    for ordinal,(p,n) in enumerate(zip(prompts,native['rows']),1):
        assert p['task_id']==n['task_id']==ids[ordinal-1]
        requests.append({'task_id':p['task_id'],'ordinal':ordinal,'user_message':p['user_message'],'user_sha256':p['user_sha256'],'rendered_prompt':n['rendered_prompt'],'rendered_prompt_sha256':n['prompt_sha256'],'prompt_token_ids':n['prompt_token_ids'],'model_id':info['model_id'],'revision':info['revision'],'seed':20260914,'do_sample':False,'max_new_tokens':1024,'n':1})
    save('data/requests.json',requests)
    save('audit/consolidated_review.json',{'status':'PASS','formal_tasks':276,'development_excluded':sorted(dev),'excluded_total':25,'clarified_tasks':sorted(clarifications),'task_set_exact_original_minus_exclusions':True,'fixed_order_preserved':True,'all_eight_tests_reference_code_and_expected_values_unchanged':True,'selected_judge_outputs_consulted_for_task_selection':False,'reviewer':'Current assistant; existing independent execution harness, not a new independent human semantic review','new_candidates':0})
    save('audit/source_bindings.json',{'files':{str(src.relative_to(ROOT.parent)).replace('\\','/'):sha(src) for src,_ in files},'previous_definition_manifest_sha256':sha(DEF/'completed_package_manifest.sha256.json')})
    print(json.dumps({'tasks':276,'slots':2,'max_GPU_hours_proposed':6,'new_requests_executed':0}))

if __name__=='__main__':main()
