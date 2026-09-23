# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Build CPU-only review artifacts exclusively; never imports an execution runner."""
import copy
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'acl_main_extension_planning_v1'
REVIEW = ROOT.parent / 'acl_main_extension_stage_a_review_v1'
SMOKE = ROOT.parent / 'acl_main_extension_smoke_preparation_v1'
RETRY = SMOKE / 'retry_review_v1'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(rel, value):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def main():
    tasks = read(SMOKE / 'review/main_tasks.reviewed_draft.json')
    old_tasks = {r['task_id']: r for r in read(BASE / 'reference_screening_v2/reference_eligible_manifest.json')}
    old_prompts = {r['task_id']: r for r in read(BASE / 'configs/generation_prompts.draft.json')}
    smoke = read(RETRY / 'configs/smoke.json')
    assert len(tasks) == 276 and (not set(smoke['task_ids']) & {t['task_id'] for t in tasks})
    assert [t['fixed_order'] for t in tasks] == sorted((t['fixed_order'] for t in tasks))
    prompts, changes = ([], [])
    for t in tasks:
        old = old_tasks[t['task_id']]
        for k in ['reference_code', 'inputs_repr', 'expected_repr', 'selected_input_hashes', 'fixed_order']:
            assert t[k] == old[k], (t['task_id'], k)
        p = copy.deepcopy(old_prompts[t['task_id']])
        assert p['user_message'].count(old['problem']) == 1
        p['user_message'] = p['user_message'].replace(old['problem'], t['problem'], 1)
        p['user_sha256'] = hashlib.sha256(p['user_message'].encode()).hexdigest()
        p['template_status'] = 'reviewed_draft_before_formal_Stage_A'
        if p['user_message'] != old_prompts[t['task_id']]['user_message']:
            changes.append({'task_id': t['task_id'], 'old_user_sha256': old_prompts[t['task_id']]['user_sha256'], 'new_user_sha256': p['user_sha256'], 'reason': 'Previously reviewed problem clarification only'})
        prompts.append(p)
    assert {r['task_id'] for r in changes} == {'Mbpp/69', 'Mbpp/237', 'Mbpp/310', 'Mbpp/576'}
    save('data/main_tasks.reviewed_draft.json', tasks)
    save('data/generation_prompts.reviewed_draft.json', prompts)
    save('audit/generation_prompt_changes.json', {'tasks': 276, 'unchanged': 272, 'changed': changes, 'reference_bodies_and_eight_tests_not_added': True, 'new_candidates': 0})
    save('configs/templates.json', read(RETRY / 'configs/templates.json'))
    assert sha(ROOT / 'configs/templates.json') == sha(RETRY / 'configs/templates.json')
    save('configs/protocol_definition.json', {'version': 1, 'scope': 'Prospective formal study; definition chosen after permanently excluded development smoke', 'decision': 'Keep the existing RTS request and retain every technically valid scoring context, including marker-only outputs', 'operational_RTS': 'Append the unchanged corrected_RTS_v2 scaffold, generate to the first natural FINAL: boundary, then apply the same conditional scoring implementation as immediate', 'estimand_interpretation': 'Within-task contrast of requested protocols on a fixed candidate/test population; not the causal effect of performing explicit reasoning', 'visible_prefix_metric': {'marker': 'FINAL:', 'rule': 'bool(text_before_first_literal_marker.strip())', 'whitespace': 'Python str.strip semantics, U+200B counts as non-whitespace', 'valid_scored_context_denominator': True, 'marker_only_rule': 'Natural valid FINAL boundary and no non-whitespace text before it; preserve any extra returned IDs separately', 'missing_marker_value': None, 'nonempty_is_not_reasoning_quality': True}, 'retention': {'valid_marker_only': 'keep in primary analysis', 'valid_nonempty_prefix': 'keep in primary analysis', 'invalid_boundary_or_missing_score': 'technical missing; retain planned denominator and failure reason', 'protocol_failure_regeneration': False, 'prefix_conditioned_primary': False}, 'reporting': {'groups': ['judge', 'template'], 'cluster': 'task; nine thresholds stay together', 'task_rows': 'For each task/judge/template, planned=9, requests observed, valid scoring contexts, marker-only count, visible-prefix count and missing reasons', 'rates': 'Report marker-only / valid contexts and marker-only / all planned contexts together with valid / planned; second rate is observed yield, not a missingness assumption', 'inference': 'Descriptive secondary; no added unadjusted significance tests; no causal claims from post-output subsets', 'old_smoke_diagnostic': 'Post hoc descriptive, not retroactive preregistration or changed engineering acceptance'}, 'cost': {'primary': 'All returned output tokens including marker and extra returned IDs; count and scoring probes separate', 'visible_prefix': 'Characters and whitespace-split words before first marker; not a claim of reasoning quality', 'prefix_token_attribution': 'Do not retokenize to claim exact reasoning-only cost; tokens crossing a text boundary require an explicit convention', 'marker_only_throughput': 'Not a substantive-reasoning throughput benchmark'}, 'templates_sha256': sha(ROOT / 'configs/templates.json'), 'existing_scoring_core_sha256': sha(RETRY / 'runtime/scoring_core.py'), 'old_smoke_acceptance_unchanged': True, 'formal_Stage_A_frozen': False, 'formal_Stage_B_frozen': False})
    c = read(BASE / 'configs/experiment.draft.json')
    c.update(study_id='acl_main_extension_protocol_definition_v1', status='ACCEPTED_PROTOCOL_DEFINITION_FORMAL_FREEZE_PENDING')
    c['source'].update(raw_path='../acl_main_extension_planning_v1/sources/MbppPlus-v0.2.0.jsonl', history_path='../acl_main_extension_planning_v1/task_screening_v2/historical_task_exclusions.csv', task_manifest='data/main_tasks.reviewed_draft.json', task_manifest_sha256=sha(ROOT / 'data/main_tasks.reviewed_draft.json'), semantic_review='25 exclusions and four clarifications recorded before formal candidate generation; final consolidated signoff pending, no independent-human or absence-of-semantic-overlap claim')
    c['selection'].update(main_tasks=276, main_task_ids=[t['task_id'] for t in tasks], development_task_ids=smoke['task_ids'], total_candidate_generation_cap=276, development_generation_in_this_package=0)
    c['generator'].update(prompt_manifest='data/generation_prompts.reviewed_draft.json', native_token_manifest='data/generation_native.reviewed_draft.json', native_token_manifest_sha256=None, native_token_manifest_status='pending_CPU_refresh_for_276_including_four_clarifications', adapter_status='Actual six-task generation verified; formal manifest CPU refresh pending')
    c['judging'].update(template_file='configs/templates.json', engine=smoke['engine'], implementation_status='Shared six-task backend actually validated; formal checkpoint/budget runner pending', protocol_definition='configs/protocol_definition.json')
    c['analysis']['visible_protocol_behavior'] = 'Predefined descriptive diagnostic; all valid marker-only contexts retained; no prefix-conditioned primary estimates'
    c['analysis']['heterogeneity'] = 'Estimate judge/template interactions; no equivalence or stability claim from nonsignificance; no equivalence test planned'
    c['resources'].update(gpu_memory_current_physical_probe='79.18 GiB in completed H100 smoke; allocation availability must be checked at execution', proposed_total_GPU_hour_cap=None, authorized_total_GPU_hour_cap=None, remaining_account_GPU_hours_reported_approx=None, account_snapshot='Historical snapshots in prior package; no current quota query in this CPU definition round')
    c['resources']['budget_reference'] = '../acl_main_extension_smoke_preparation_v1/review/main_budget_update.json'
    c['freeze'] = {'stage_A': False, 'stage_B': False, 'stage_A_pending': ['consolidated final source/attribution and adapter signoff with explicit limitations', 'CPU native prompt refresh and integrity review', 'formal candidate runner with immutable requests, no-resampling recovery and a concrete capped execution proposal', 'Stage A checksum/signoff record before candidate generation'], 'stage_B_pending': ['actual candidate output and extraction audit for complete generated population', 'two candidate execution verifiers and reliable truth bindings', 'candidate-dependent native judge manifests and context-length checks', 'formal runner recovery, allocation plan and analysis implementation verification', 'Stage B checksum/signoff record before judge execution']}
    save('configs/experiment.reviewed_draft.json', c)
    save('decision_record.json', {'user_message': 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'approved': ['Operational requested-RTS definition', 'Keep existing prompts and all valid marker-only results', 'Predefine visible behavior and reporting before formal experiments', 'CPU-only freeze preparation'], 'candidate_generation_authorized': False, 'model_inference_authorized': False, 'GPU_submission_authorized': False, 'formal_stage_A_frozen': False, 'formal_stage_B_frozen': False, 'approval_is_not_a_retroactive_smoke_rule': True})
    save('data/development_exclusions.json', {'task_ids': smoke['task_ids'], 'use': 'Permanently excluded development generation and judge smoke, completed jobs 813840/813849/814629', 'old_history_exclusions_path': c['source']['history_path'], 'formal_overlap': []})
    inputs = [BASE / 'configs/experiment.draft.json', BASE / 'configs/generation_prompts.draft.json', BASE / 'configs/generation_native_token_manifest.draft.json', BASE / 'reference_screening_v2/reference_eligible_manifest.json', REVIEW / 'review/reference_review_summary.json', REVIEW / 'review/semantic_exclusions.json', SMOKE / 'review/source_decisions.json', SMOKE / 'review/main_tasks.reviewed_draft.json', SMOKE / 'review/main_budget_update.json', RETRY / 'configs/smoke.json', RETRY / 'configs/templates.json', RETRY / 'runtime/scoring_core.py', RETRY / 'work/smoke_acceptance.json', RETRY / 'work/trace_fidelity.json', RETRY / 'work/final_execution_summary.json', RETRY / 'work/execution_cost_and_diagnostics.json']
    save('audit/input_bindings.json', {'files': {str(p.relative_to(ROOT.parent)).replace('\\', '/'): sha(p) for p in inputs}, 'read_only_old_material': True})
    print(json.dumps({'formal_tasks': 276, 'candidate_requests_executed': 0, 'core_contexts_planned': 276 * 72, 'RTS_requests_planned': 276 * 36, 'count_requests_planned': 276 * 4, 'four_prompt_clarifications_applied': True}))
if __name__ == '__main__':
    main()
