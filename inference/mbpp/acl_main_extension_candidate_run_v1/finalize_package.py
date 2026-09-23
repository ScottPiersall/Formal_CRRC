# Review adaptation: historical non-English narrative strings omitted; numerical routines retained.
"""Freeze agreed Stage A content and build the reviewable, unauthorized archive.

No network, candidate execution, model import, or job submission. Exclusive final
artifacts; rerunning in the same directory is deliberately refused.
"""
import ast
import collections
import datetime
import hashlib
import json
import subprocess
import tarfile
from prepare import ROOT, PLAN, DEF, read, save, copy_file, sha

def digest(v):
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    if (ROOT / 'stage_a_manifest.json').exists():
        raise FileExistsError('Stage A manifest already exists; do not overwrite a freeze')
    baseline = read(ROOT / 'audit/baseline.json')
    preservation = []
    for rec in baseline['prior_package_manifests']:
        base = ROOT.parent / rec['package']
        mp = base / rec['manifest']
        assert sha(mp) == rec['manifest_sha256']
        files = read(mp)['files']
        for rel, v in files.items():
            assert sha(base / rel) == (v['sha256'] if isinstance(v, dict) else v), (rec['package'], rel)
        preservation.append({'package': rec['package'], 'checked_files': len(files), 'unchanged': True})
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT.parent.parent, capture_output=True, check=True).stdout
    assert hashlib.sha256(diff).hexdigest() == baseline['tracked_diff_sha256']
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'))
    notices = ['mbpp_README.md', 'mbpp_dataset_card.md', 'evalplus_LICENSE']
    for name in notices:
        copy_file(PLAN / 'sources' / name, 'stage_a/source_notices/' + name)
    receipts = read(PLAN / 'sources/fetch_receipts.json') + read(PLAN / 'sources/supplemental_fetch_receipts.json')
    save('stage_a/source_notices/fetch_provenance.json', [r for r in receipts if r['name'] in notices + ['MbppPlus-v0.2.0.jsonl.gz', 'mbpp_release.json']])
    raw = PLAN / 'sources/MbppPlus-v0.2.0.jsonl'
    assert sha(raw) == 'b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad'
    tasks = read(ROOT / 'stage_a/main_tasks.json')
    splits = collections.Counter()
    for t in tasks:
        n = int(t['task_id'].split('/')[1])
        splits['prompting' if n <= 10 else 'test' if n <= 510 else 'validation' if n <= 600 else 'training'] += 1
    assert dict(splits) == {'test': 173, 'training': 75, 'validation': 25, 'prompting': 3}
    save('stage_a/source_split_counts.json', {'classification': 'Original MBPP ID split, descriptive only; not a new inclusion rule', 'counts': dict(splits), 'model_pretraining_exposure_verified': False, 'raw_source_sha256': sha(raw)})
    science = read(ROOT / 'stage_a/scientific_config.json')
    for key in ['planning_only', 'allow_model_load', 'allow_generation', 'allow_job_submission']:
        science.pop(key, None)
    science.update(status='STAGE_A_CONTENT_FROZEN_EXECUTION_AUTHORIZATION_SEPARATE', execution_authorization_record_separate=True)
    science['source']['raw_location_provenance'] = science['source'].pop('raw_path')
    science['source']['raw_file_in_execution_archive'] = False
    science['source']['dataset_license'] = 'Recorded upstream MBPP CC-BY-4.0 and EvalPlus code Apache-2.0; enhanced-test public redistribution attribution pending; internal-use scope recorded in attribution_and_review_zh.md'
    science['generator']['infrastructure_retries'] = 0
    science['generator']['original_planning_retry_ceiling'] = 2
    science['generator']['adapter_status'] = 'Six-task actual generation verified; all 276 native prompt IDs bound in Stage A'
    science['generator']['native_token_manifest_status'] = 'Stage_A_frozen_before_formal_generation'
    science['resources']['candidate_budget_reference'] = 'budget/results.json'
    science['resources']['proposed_candidate_GPU_hours_cap'] = 6
    science['resources']['authorized_candidate_GPU_hours_cap'] = None
    science['freeze']['stage_A'] = True
    science['freeze']['stage_A_pending'] = []
    science['freeze']['stage_A_scope'] = 'Local content freeze, not external preregistration or GPU authorization'
    with (ROOT / 'stage_a/scientific_config.json').open('w', encoding='utf-8', newline='\n') as f:
        json.dump(science, f, ensure_ascii=False, indent=2)
        f.write('\n')
    run_config = read(ROOT / 'configs/candidate_run.json')
    run_config['generator'] = science['generator']
    with (ROOT / 'configs/candidate_run.json').open('w', encoding='utf-8', newline='\n') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
        f.write('\n')
    model = read(ROOT / 'configs/model_inventory.json')
    model = {k: model[k] for k in ['model_id', 'revision', 'path', 'files', 'shards']}
    model['role'] = 'Expected fixed cache metadata. Current CPU observations are in audit/CLUSTER_current.json; actual load evidence is prior development smoke.'
    with (ROOT / 'configs/model_inventory.json').open('w', encoding='utf-8', newline='\n') as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
        f.write('\n')
    save('stage_a/decision_record.json', {'user_approved_direction': 'Keep current RTS prompts and marker-only results; define reporting before formal execution', 'latest_user_instruction': 'Historical narrative omitted from the review export. See numerical outputs and the English reproduction guide.', 'scope_of_latest_instruction': 'Prepare the formal candidate execution package and complete Stage A preparation; no GPU execution approved in this turn', 'source_and_adapter_review': 'Completed consolidated assistant review with prior independent execution-harness evidence; not an independent human semantic review', 'sample_rule': 'Entire fixed 276-task pool, one candidate per task, no partial/effect stopping', 'partial_reference': 80, 'resampling': 'No model-request retries in this execution package; prior at-most-two infrastructure allowance is not exercised', 'scientific_outcome_requirement': 'None: positive, negative and inconclusive outcomes under identical rules', 'public_redistribution': 'Pending enhanced-test attribution; outside this internal execution package', 'formal_stage_B_frozen': False, 'formal_candidates_generated': 0})
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stage_paths = [p for p in (ROOT / 'stage_a').rglob('*') if p.is_file()] + list((ROOT / 'data').glob('*.json')) + [ROOT / 'configs/candidate_run.json', ROOT / 'configs/model_inventory.json']
    stage = {'status': 'FROZEN_BEFORE_FORMAL_CANDIDATE_GENERATION', 'frozen_at_utc': stamp, 'scope': 'Agreed Stage A scientific content and candidate request configuration; not a model execution authorization', 'model_requests_already_executed': 0, 'formal_Stage_B_frozen': False, 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(stage_paths)}}
    save('stage_a_manifest.json', stage)
    c = read(ROOT / 'configs/candidate_run.json')
    auth = {'approved': False, 'allow_candidate_generation': False, 'allow_judge_inference': False, 'scope': c['scope'], 'execution_manifest_sha256': None, 'stage_a_manifest_sha256': sha(ROOT / 'stage_a_manifest.json'), 'task_ids_sha256': digest(c['task_ids']), 'max_submission_slots': 2, 'minutes_per_slot': 180, 'total_physical_GPU_hours_cap': 6, 'maximum_formal_candidates': 276, 'automatic_submission': False, 'user_approval_text': None, 'instructions': 'Only after explicit approval: set approved/allow_candidate_generation true, bind the execution hash from approval_request.json, and record actual user approval text in a new authorization.json. Do not reuse smoke authorization.'}
    save('authorization.template.json', auth)
    paths = stage_paths + [ROOT / 'stage_a_manifest.json', ROOT / 'authorization.template.json', ROOT / 'README_zh.md', ROOT / 'budget.md', ROOT / 'budget/results.json', ROOT / 'plan.py', ROOT / 'submit.py', ROOT / 'control.py']
    paths += list((ROOT / 'runtime').glob('*.py')) + [ROOT / 'runtime/job.sbatch']
    paths += [ROOT / 'audit' / x for x in ['consolidated_review.json', 'CLUSTER_current.json', 'source_bindings.json', 'static_review.json']]
    paths = sorted(set(paths))
    manifest = {'scope': c['scope'], 'stage_a_manifest_sha256': sha(ROOT / 'stage_a_manifest.json'), 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}
    save('execution_manifest.json', manifest)
    execution_hash = sha(ROOT / 'execution_manifest.json')
    archive_name = 'acl_formal_candidates_' + execution_hash[:12] + '.tar.gz'
    archive = ROOT / archive_name
    with tarfile.open(archive, 'x:gz') as tar:
        for p in paths + [ROOT / 'execution_manifest.json']:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
    save('export_receipt.json', {'archive_name': archive_name, 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'remote_directory': '/REDACTED_LOCAL_PATH' + execution_hash[:12], 'files': len(paths) + 1, 'contains_model_weights': False, 'contains_live_authorization': False, 'deployed': False})
    approval = {k: v for k, v in auth.items() if k != 'instructions'}
    approval.update(execution_manifest_sha256=execution_hash, stage_a_manifest_sha256=sha(ROOT / 'stage_a_manifest.json'), status='READY_FOR_EXPLICIT_CANDIDATE_EXECUTION_APPROVAL', candidate_output_token_cap=282624, judge_requests_in_scope=0, archive_sha256=sha(archive))
    save('approval_request.json', approval)
    save('audit/preservation_before_freeze.json', {'checked_at_utc': stamp, 'prior_packages': preservation, 'tracked_user_diff_unchanged': True, 'new_candidates': 0, 'new_inference_requests': 0})
    print(json.dumps({'Stage_A_frozen': True, 'Stage_B_frozen': False, 'stage_a_manifest_sha256': sha(ROOT / 'stage_a_manifest.json'), 'execution_manifest_sha256': execution_hash, 'archive': archive_name, 'archive_bytes': archive.stat().st_size, 'execution_authorized': False, 'prior_files_preserved': sum((p['checked_files'] for p in preservation))}, ensure_ascii=False))
if __name__ == '__main__':
    main()
