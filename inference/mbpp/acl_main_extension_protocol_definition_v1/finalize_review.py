"""Bind the completed CPU refresh to this new, not-yet-frozen review draft."""
import json
from build_review import ROOT, BASE, RETRY, read, save, sha


def main():
    c=read(ROOT/'configs/experiment.reviewed_draft.json')
    native=read(ROOT/'data/generation_native.reviewed_draft.json')
    old={r['task_id']:r for r in read(BASE/'configs/generation_native_token_manifest.draft.json')['rows']}
    changed=[]
    for row in native['rows']:
        previous=old[row['task_id']]
        if row['rendered_prompt']!=previous['rendered_prompt']:
            changed.append(row['task_id'])
        else:
            assert row['prompt_token_ids']==previous['prompt_token_ids']
    assert set(changed)=={'Mbpp/69','Mbpp/237','Mbpp/310','Mbpp/576'}
    c['generator'].update(native_token_manifest_sha256=sha(ROOT/'data/generation_native.reviewed_draft.json'),native_token_manifest_status='CPU_verified_276_after_four_clarifications_not_formally_frozen',adapter_status='Six-task actual generation verified; all 276 final draft prompts CPU-rendered and bound')
    c['freeze']['stage_A_pending'].remove('CPU native prompt refresh and integrity review')
    # This file is a new mutable draft until the binding below, not an old freeze.
    (ROOT/'configs/experiment.reviewed_draft.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('audit/native_refresh_comparison.json',{'formal_tasks':276,'unchanged_rendered_prompts_and_token_ids':272,'changed_due_to_previously_reviewed_clarification':changed,'min_prompt_tokens':min(r['n_prompt_tokens'] for r in native['rows']),'max_prompt_tokens':max(r['n_prompt_tokens'] for r in native['rows']),'model_loaded':False,'new_candidates':0})
    cpu=read(ROOT/'audit/CLUSTER_cpu_refresh.json')
    save('environment_status.json',{'current_round':{'checked_at_utc':cpu['checked_at_utc'],'authorized_CLUSTER_connection':'directly_verified_read_only_CPU','tokenizer_revisions':'directly_verified_pinned_metadata_hashes_for_both_models','native_generation_prompts':276,'existing_RTS_actual_prefix_checks':216,'remote_writes':False,'models_loaded_this_round':False,'new_inference_requests':0},'existing_development_evidence':{'status':'actual_completed_inference_receipts_read_and_bound_this_round','judge_job':814629,'both_pinned_models_loaded_and_scored':True,'conditional_probability_checks_passed':48,'evidence_path':'../acl_main_extension_smoke_preparation_v1/retry_review_v1/REDACTED_LOCAL_PATH','not_a_current_GPU_availability_probe':True},'unverified_current_state':{'scheduler_queue':'not_queried_this_round','remaining_quota':'unknown_current_value','future_allocation_availability':'pending_at_execution','formal_runtime_and_stage_A_B':'pending'},'model_revisions':{k:{'model_id':v['model_id'],'revision':v['revision']} for k,v in c['judges'].items()}})
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['definition_manifest.sha256.json','completed_package_manifest.sha256.json']}
    save('definition_manifest.sha256.json',{'scope':'Versioned review definition and CPU artifacts; not formal Stage A/B freeze, not GPU authorization','files':files})
    print(json.dumps({'bound_review_files':len(files),'native_prompts':276,'unchanged':272,'changed':changed,'formal_Stage_A_frozen':False,'new_inference_requests':0}))


if __name__=='__main__':main()
