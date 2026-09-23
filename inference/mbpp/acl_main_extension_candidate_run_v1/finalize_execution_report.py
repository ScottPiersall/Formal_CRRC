# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Write exclusive completion reports from audited receipts; no remote or model calls."""
import argparse
import json
from pathlib import Path
import platform
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from common import read, save, sha, now

def write_text(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if not snapshot.is_relative_to((ROOT / 'snapshots').resolve()):
        raise ValueError('Use a fetched snapshot in this package')
    result = ROOT / 'results' / snapshot.name
    s = read(result / 'generation_summary.json')
    audit = read(result / 'completion_integrity_audit.json')
    if s['status'] != 'GENERATION_COMPLETE' or audit['status'] != 'PASS':
        raise RuntimeError('Completed generation and integrity audit are required')
    if (ROOT / 'execution_completion_manifest.sha256.json').exists():
        raise FileExistsError('Completion package already recorded')
    science = read(ROOT / 'stage_a/scientific_config.json')
    static = read(result / 'candidate_static_audit.json')
    errors = [r['task_id'] for r in static if r['syntax_error']]
    save(result / 'static_audit_environment.json', {'Python': sys.version, 'platform': platform.platform(), 'method': 'ast.parse only; candidate code was never executed', 'execution_truth_authoritative_environment': science['truth']['image'], 'syntax_check_does_not_assign_pass_bits_or_z': True})
    env = {'observed_at_utc': now(), 'scope': 'completed formal candidate generation only', 'directly_verified': {'candidate_model_load_events': audit['model_load_events'], 'candidate_execution_manifest_sha256': s['execution_manifest_sha256'], 'Stage_A_manifest_sha256': s['stage_a_manifest_sha256'], 'completed_candidates': s['completed_candidate_receipts'], 'scheduler_accounting': s['scheduler'], 'physical_GPU_hours': s['physical_GPU_hours'], 'remote_manifest_files_verified': audit['remote_frozen_files_verified'], 'local_snapshot_transferred_files_verified': s['transferred_files_verified'], 'loaded_model_is_more_than_cache_file_presence': True, 'formal_judge_requests': 0, 'formal_RTS_reasoning_tokens': 0, 'formal_count_requests': 0, 'candidate_truth_executions': 0, 'stage_A_frozen': True, 'stage_B_frozen': False}, 'prior_run_evidence_not_new_formal_execution': {'judge_smoke_job': '814629', 'evidence_package': '../acl_main_extension_smoke_preparation_v1/retry_review_v1', 'fixed_judges': science['judges'], 'reported_engine_versions': science['judging']['engine'], 'interpretation': 'Both fixed judges previously loaded and scored in the completed development smoke. This candidate run loads only Mistral with Transformers, not either formal vLLM judge panel.'}, 'pending': {'candidate_execution_truth': 'two isolated CPU verifiers on frozen eight tests', 'correctness_and_partial_count': None, 'formal_judge_manifests_and_runtime_budget': None, 'current_remaining_shared_account_quota': None, 'monetary_price': None, 'judge_execution_authorization': False}, 'accounting_note': 'AllocTRES billing=10 is a scheduler billing weight; gres/gpu=1 is the physical GPU count. No monetary charge is inferred.'}
    save(ROOT / 'environment_status.execution.json', env)
    handoff = {'prepared_at_utc': now(), 'status': 'CPU_TRUTH_HANDOFF_PREPARED_EXECUTION_PENDING', 'source_generation_summary': f'results/{snapshot.name}/generation_summary.json', 'candidate_handoff_manifest': f'results/{snapshot.name}/truth_handoff_manifest.json', 'candidate_handoff_manifest_sha256': sha(result / 'truth_handoff_manifest.json'), 'next_independent_package_suggested': 'experiments/acl_main_extension_stage_b_preparation_v1', 'new_directory_must_not_overwrite_existing': True, 'reuse_harness_source': '../acl_main_extension_smoke_preparation_v1/execute_truth.py', 'reuse_harness_source_sha256': sha(ROOT.parent / 'acl_main_extension_smoke_preparation_v1/execute_truth.py'), 'required_adaptation': 'Bind all 276 immutable candidates and task tests, replace development-only input paths and gate with a formal CPU-only manifest, retain immutable per-test receipts and infrastructure/mismatch statuses; do not execute the old six-task entry point as if it covered the formal population.', 'frozen_truth_config': science['truth'], 'planned_test_invocations': 276 * 8 * 2, 'sum_per_test_timeouts_CPU_hours_excluding_overhead': 276 * 8 * 2 * 5 / 3600, 'verification_limit': 'Fresh processes, separate call implementations and hash seeds; not independent human or independent algorithmic reference verification.', 'handling': 'Keep all 276 candidates including syntax failures; do not regenerate. Infrastructure missingness and verifier disagreement block reliable z rather than count as wrong. Report whole population and z strata; no partial-count top-up.', 'Stage_B_remaining': science['freeze']['stage_B_pending'], 'future_core_contexts': 276 * 72, 'future_RTS_trace_requests': 276 * 36, 'future_count_requests': 276 * 4, 'formal_judge_submission_authorized': False, 'candidate_execution_started_by_this_handoff': False}
    save(result / 'stage_b_preparation_handoff.json', handoff)
    rel = 'results/' + snapshot.name
    report = 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.'
    write_text(ROOT / 'RUN_STATUS_zh.md', report)
    manifest_path = ROOT / 'execution_completion_manifest.sha256.json'
    files = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file() and p != manifest_path and ('__pycache__' not in p.parts) and (not p.name.endswith('.pyc'))}
    save(manifest_path, {'created_at_utc': now(), 'scope': 'Completed authorized formal candidate generation and immutable evidence; not formal judge execution or Stage B freeze', 'files': files})
    for relative, expected in files.items():
        if sha(ROOT / relative) != expected['sha256']:
            raise RuntimeError('Completion manifest verification failed')
    print(json.dumps({'status': 'CANDIDATE_STAGE_COMPLETE_AND_RECORDED', 'report': str(ROOT / 'RUN_STATUS_zh.md'), 'files_in_completion_manifest': len(files), 'completion_manifest_sha256': sha(manifest_path), 'Stage_B_frozen': False, 'formal_judge_requests': 0}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
