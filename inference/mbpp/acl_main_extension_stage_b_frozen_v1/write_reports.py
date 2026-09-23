# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""One-time CPU report builder in this new package; never overwrites reports."""
from common import *

def main():
    b = read(ROOT / 'results/judge_budget.json')
    old = read(ROOT / 'audit/current_resources.json')
    current = read(ROOT / 'audit/CLUSTER_cpu_check.json')
    lines = ['Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.']
    for r in b['scenarios']:
        calculated = sum((x['seconds_before_retry_reserve'] for x in r['rows'])) * (1 + r['technical_retry_and_variability_reserve_fraction']) / 3600
        if abs(calculated - r['total_GPU_hours']) > 1e-09:
            raise RuntimeError('Budget row sum differs')
        lines.append(f"| {r['scenario']} | {r['RTS_tokens']:,.0f} | {r['count_tokens']:,.0f} | {r['GPU_hours_by_judge']['qwen']:.2f} | {r['GPU_hours_by_judge']['mistral']:.2f} | {r['total_GPU_hours']:.2f} |\n")
    lines.append('Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    text_file(ROOT / 'budget.md', ''.join(lines))
    text_file(ROOT / 'inventory.md', 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.')
    native = {j: read(ROOT / 'data/native' / (j + '.json')) for j in ['qwen', 'mistral']}
    save(ROOT / 'environment_status.json', {'created_utc': now(), 'directly_verified_this_turn': {'local_CPU_tests': 'audit/CPU_acceptance.json', 'local_native_manifest_static_validation': 'audit/CPU_acceptance.json', 'current_CLUSTER_connection': current, 'local_WSL_Linux_source_syntax': 'audit/local_linux_syntax.json', 'formal_judge_requests': 0, 'GPU_jobs_submitted': 0, 'models_loaded': False}, 'prior_verified_artifacts_rechecked_for_integrity': {'fixed_judge_native_preflight': {j: {k: native[j][k] for k in ['checked_utc', 'revision', 'max_prompt_tokens', 'old_successful_native_rows_exact_match', 'model_cache_structure_checked', 'model_loaded']} for j in native}, 'CLUSTER_resources': old, 'actual_development_model_load_job': '814629', 'actual_formal_candidate_generation_job': '815821', 'CPU_truth_test_calls_already_completed': 4416}, 'not_currently_verified': {'cache_revision_and_shard_structure': 'pending_restored_connection', 'engine_versions_on_CLUSTER': 'pending_restored_connection', 'current_GPU_quota': 'unknown', 'GPU_price': 'unknown', 'fresh_formal_model_load': 'pending_authorized_execution', 'formal_FINAL_boundary_and_conditional_parity': 'pending_actual_saved_trace', 'formal_throughput_failures': 'pending_execution', 'remote_disk_quota': 'unknown'}, 'cache_warning': 'Prior tokenizer/config hashes and shard stat are not a fresh weight load; full weight files were not rehashed this turn.', 'historical_count_decoder_pending': read(ROOT / 'audit/legacy_regression.json')['token_decode_pending'], 'code_implementation_pending': []})
    save(ROOT / 'decision_record.json', {'recorded_utc': now(), 'user_instruction': 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'authorized_scope': 'Complete CPU engineering, validation, full analysis export, Stage B content freeze and reviewed deployment artifact.', 'formal_GPU_execution_authorized': False, 'new_candidates_authorized': False, 'freeze_is_internal_not_public_preregistration': True, 'scope_preserved': {'independent_tasks': 276, 'partial_tasks': 33, 'same_candidates_and_tests': True, 'no_effect_driven_stopping': True}, 'current_environment_block_does_not_change_frozen_scientific_design': True, 'dynamic_checks_remain_enforced_by_runtime': True})
    save(ROOT / 'authorization.template.json', {'approved': False, 'allow_judge_inference': False, 'allow_GPU_submission': False, 'allow_candidate_generation': False, 'allow_model_download': False, 'scope': 'formal_276_fixed_candidates_judge_only_v1', 'stage_B_sha256': None, 'task_ids_sha256': None, 'maximum_submission_slots': 24, 'maximum_seconds_per_slot': 28800, 'maximum_physical_GPU_seconds': 691200, 'automatic_submission': False, 'user_approval_text': '', 'instructions': 'Template only. A future explicit user approval must bind the exact manifest and task hashes from approval_request.json; never reuse old smoke/candidate approval.'})
    print(json.dumps({'reports_written': 5, 'new_GPU_jobs': 0}))
if __name__ == '__main__':
    main()
