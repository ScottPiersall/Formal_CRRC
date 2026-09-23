"""Read-only local validation. Execution actions are refused unconditionally."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runtime'))
from common import read,sha,digest
from gates import verify_bundle,state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['validate','dry-run','inspect','generate','run','judge','submit']);a=p.parse_args()
    if a.action in ['generate','run','judge','submit']:
        print(json.dumps({'status':'BLOCKED','reason':'This entry is read-only. Formal candidate execution uses separately authorized submit.py; judge execution is not implemented.'}));return 3
    c=verify_bundle();science=read(ROOT/'stage_a/scientific_config.json');requests=read(ROOT/'data/requests.json')
    tasks=read(ROOT/'stage_a/main_tasks.json');prompts=read(ROOT/'data/generation_prompts.json');native=read(ROOT/'data/generation_native.json')
    assert c['task_ids']==[t['task_id'] for t in tasks]==[r['task_id'] for r in requests]==[r['task_id'] for r in prompts]==[r['task_id'] for r in native['rows']]
    assert len(set(c['task_ids']))==276 and not set(c['task_ids'])&set(science['selection']['development_task_ids'])
    assert science['selection']['total_candidate_generation_cap']==276 and science['selection']['partial_planning_target']==80
    assert science['freeze']['stage_A'] and not science['freeze']['stage_B']
    for t,prompt,n,req in zip(tasks,prompts,native['rows'],requests):
        assert len(t['inputs_repr'])==len(t['expected_repr'])==8
        for value in t['inputs_repr']+t['expected_repr']:ast.literal_eval(value)
        assert req['user_message']==prompt['user_message'] and req['rendered_prompt']==n['rendered_prompt'] and req['prompt_token_ids']==n['prompt_token_ids']
        assert hashlib.sha256(req['user_message'].encode()).hexdigest()==req['user_sha256']
        assert hashlib.sha256(req['rendered_prompt'].encode()).hexdigest()==req['rendered_prompt_sha256']
        assert len(req['prompt_token_ids'])+1024<=32768
        assert req['do_sample'] is False and req['max_new_tokens']==1024 and req['n']==1 and req['seed']==20260914
        assert not set(req)&{'z','pass_bits','expected_repr','reference_code','inputs_repr'}
        assert not prompt['contains_selected_eight_tests'] and not prompt['contains_reference_solution_body']
    assert science['judges']['qwen']['revision']=='cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'
    assert science['judges']['mistral']['revision']==c['generator']['revision']=='c170c708c41dac9275d15a8fff4eca08d52bab71'
    current=state();forbidden=[k for k in ['torch','transformers','vllm','tokenizers','jinja2'] if k in sys.modules]
    assert not forbidden
    print(json.dumps({'status':'PASS_STAGE_A_CONTENT_AND_CANDIDATE_PACKAGE','formal_Stage_A_frozen':True,'formal_Stage_B_frozen':False,'candidate_requests_planned':276,'candidate_requests_executed_in_preparation':0,'work_state':current if a.action=='inspect' else {k:v for k,v in current.items() if k not in ['completed','pending']},'judge_contexts_executed':0,'proposed_max_physical_GPU_hours':6,'authorization_file_present':(ROOT/'authorization.json').exists(),'execution_authorization_verified':False,'model_or_tokenizer_modules_imported':forbidden,'scheduler_contacted':False},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
