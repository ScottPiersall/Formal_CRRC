"""Local read-only standard-library validation; no execution implementation exists."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
EXPECTED={'qwen':('Qwen/Qwen2.5-14B-Instruct','cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'),
          'mistral':('mistralai/Mistral-7B-Instruct-v0.3','c170c708c41dac9275d15a8fff4eca08d52bab71')}


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validate','dry-run','run','generate','submit','freeze'])
    parser.add_argument('--require-stage-a',action='store_true')
    args=parser.parse_args()
    if args.action in {'run','generate','submit','freeze'}:
        print(json.dumps({'status':'BLOCKED','reason':'Definition/review package has no model, scheduler or formal-freeze execution path','requested_action':args.action}))
        return 3
    errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    c=read(ROOT/'configs/experiment.reviewed_draft.json')
    d=read(ROOT/'configs/protocol_definition.json')
    check(c['planning_only'] and not any(c[k] for k in ['allow_model_load','allow_generation','allow_job_submission']),'Execution flags must be disabled')
    check(not c['freeze']['stage_A'] and not c['freeze']['stage_B'],'This package is not formal Stage A/B')
    check(c['resources']['authorized_total_GPU_hour_cap'] is None,'No GPU authorization in this package')
    for judge,pin in EXPECTED.items():check((c['judges'][judge]['model_id'],c['judges'][judge]['revision'])==pin,'Pinned judge mismatch: '+judge)
    check((c['generator']['model_id'],c['generator']['revision'])==EXPECTED['mistral'],'Pinned generator mismatch')
    check(c['selection']['main_tasks']==276 and c['selection']['candidate_per_task']==1 and c['selection']['total_candidate_generation_cap']==276,'Full 276-task one-candidate design')
    check(c['truth']['tests_per_task']==8 and c['truth']['thresholds']==list(range(9)),'Eight tests/nine thresholds')
    check(c['analysis']['primary_template']=='original' and c['analysis']['primary_readout']=='bare','Primary cell changed')
    check(c['count_control']['requests_per_task']==4 and c['count_control']['recoverability_score'] is None,'Count structure changed')
    check(d['retention']['valid_marker_only']=='keep in primary analysis' and not d['retention']['protocol_failure_regeneration'],'Marker-only retention / no regeneration')
    check(sha(ROOT/'configs/templates.json')==d['templates_sha256'],'Prompt templates changed')
    check(sha(ROOT/c['source']['task_manifest'])==c['source']['task_manifest_sha256'],'Task bytes changed')
    tasks=read(ROOT/c['source']['task_manifest']);prompts=read(ROOT/c['generator']['prompt_manifest']);native=read(ROOT/c['generator']['native_token_manifest'])
    check(sha(ROOT/c['generator']['native_token_manifest'])==c['generator']['native_token_manifest_sha256'],'Native manifest not bound')
    ids=[t['task_id'] for t in tasks]
    check(ids==c['selection']['main_task_ids']==[r['task_id'] for r in prompts]==[r['task_id'] for r in native['rows']],'Task/prompt/native ordering mismatch')
    check(len(set(ids))==276 and not set(ids)&set(c['selection']['development_task_ids']),'Main/development overlap or duplicate')
    check(native['revision']==EXPECTED['mistral'][1] and not native['model_loaded'] and not native['new_generation'],'Native CPU provenance')
    for task,prompt,row in zip(tasks,prompts,native['rows']):
        check(len(task['inputs_repr'])==len(task['expected_repr'])==8,'Eight values required: '+task['task_id'])
        for value in task['inputs_repr']+task['expected_repr']:ast.literal_eval(value)
        check(hashlib.sha256(prompt['user_message'].encode()).hexdigest()==prompt['user_sha256'],'Prompt hash: '+task['task_id'])
        check(hashlib.sha256(row['rendered_prompt'].encode()).hexdigest()==row['prompt_sha256'],'Native rendered hash')
        check(len(row['prompt_token_ids'])==row['n_prompt_tokens'] and row['n_prompt_tokens']+1024<=32768,'Generator length receipt')
        check(not prompt['contains_selected_eight_tests'] and not prompt['contains_reference_solution_body'],'Generator payload flags')
    for rel,expected in read(ROOT/'audit/input_bindings.json')['files'].items():check(sha(ROOT.parent/rel)==expected,'Old bound input changed: '+rel)
    for rel,expected in read(ROOT/'definition_manifest.sha256.json')['files'].items():check(sha(ROOT/rel)==expected,'Definition package binding changed: '+rel)
    forbidden=[m for m in ['torch','transformers','vllm','tokenizers','jinja2'] if m in sys.modules]
    check(not forbidden,'Validation imported a model/tokenizer package')
    check(not args.require_stage_a,'Formal Stage A is not frozen' if args.require_stage_a else '')
    result={'status':'PASS_REVIEW_ONLY' if not errors else 'BLOCKED_OR_INVALID','errors':errors,
            'formal_tasks_planned':276,'partial_planning_reference':80,'partial_is_not_a_stop_trigger':True,
            'core_contexts_planned':276*72,'RTS_requests_planned':276*36,'count_requests_planned':276*4,
            'formal_Stage_A_frozen':False,'formal_Stage_B_frozen':False,'GPU_authorized':False,
            'new_model_requests':0,'model_or_tokenizer_modules_imported':forbidden,'pending':c['freeze']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
