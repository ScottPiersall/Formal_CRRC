"""Read-only review entrypoint. Execution actions are unconditionally unavailable."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'acl_main_extension_planning_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate():
    errors=[]
    approval=read(ROOT/'scope_approval.json')
    overlay=read(ROOT/'configs/approval_overlay.json')
    smoke=read(ROOT/'configs/development_smoke.draft.json')
    manifest=read(ROOT/'configs/task_pool.reviewed_draft.json')
    exclusions=read(ROOT/'review/semantic_exclusions.json')
    original=read(BASE/'reference_screening_v2/reference_eligible_manifest.json')
    base_by_id={t['task_id']:t for t in original}
    excluded={t['task_id'] for t in exclusions}
    if digest(BASE/'configs/experiment.draft.json')!=approval['base_config_sha256']:errors.append('base config hash mismatch')
    if not approval['scope_approved'] or not overlay['research_scope_approved']:errors.append('research scope missing')
    for key in ['candidate_generation_authorized','model_inference_authorized','GPU_submission_authorized']:
        if approval[key] is not False:errors.append('scope approval must not authorize execution: '+key)
    for key in ['allow_model_load','allow_generation','allow_submit']:
        if smoke[key] is not False:errors.append('smoke execution gate must remain closed: '+key)
    if smoke['authorized_GPU_hour_cap'] is not None:errors.append('GPU cap not authorized in this scope approval')
    if digest(ROOT/'configs/task_pool.reviewed_draft.json')!=smoke['task_manifest_sha256']:errors.append('task manifest hash mismatch')
    if digest(ROOT/'protocol/scoring_core.py')!=smoke['shared_CPU_protocol_sha256']:errors.append('protocol core hash mismatch')
    if set(t['task_id'] for t in manifest)!=set(base_by_id)-excluded:errors.append('pool not exact original-minus-reviewed-exclusions')
    dev=[t for t in manifest if t['proposed_split']=='development_smoke']
    main=[t for t in manifest if t['proposed_split']=='main_pool']
    if smoke['task_ids']!=[t['task_id'] for t in dev] or len(dev)!=6:errors.append('development IDs changed')
    if any(t in {m['task_id'] for m in main} for t in smoke['task_ids']):errors.append('smoke contains a formal task')
    for task in manifest:
        old=base_by_id[task['task_id']]
        for key in ['fixed_order','proposed_split','reference_code','inputs_repr','expected_repr','selected_input_hashes']:
            if task[key]!=old[key]:errors.append(task['task_id']+': changed '+key)
        if len(task['inputs_repr'])!=8 or len(task['expected_repr'])!=8:errors.append('not eight tests')
        for text in task['inputs_repr']+task['expected_repr']:ast.literal_eval(text)
    result={'status':'PASS_REVIEW_DRAFT' if not errors else 'FAIL','errors':errors,
            'research_scope_approved':True,'starting_main_tasks':301,'current_main_draft':len(main),
            'review_exclusions':len(excluded),'development_tasks':len(dev),
            'stage_A_frozen':False,'stage_B_frozen':False,
            'inference_authorized':False,'inference_executed':False,'model_modules_imported':[m for m in ['torch','transformers','vllm','tokenizers'] if m in sys.modules]}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if errors else 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['status','validate','dry-run','run','generate','submit'])
    args=parser.parse_args()
    if args.action in ['run','generate','submit']:
        print('BLOCKED: research-scope approval does not authorize model or GPU execution; no execution backend in this entrypoint.')
        raise SystemExit(3)
    raise SystemExit(validate())


if __name__=='__main__':main()
