"""Audit a completed candidate snapshot and frozen bytes; never execute code/models.

The only remote operation reads this deployment's manifest-listed files and its
authorization. Local outputs are exclusive additions outside the frozen bundle.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from common import read, sha, save, now
from control import remote_path, ssh_python
from gates import verify_bundle, authorize


def verify_manifest(base, manifest):
    files = read(manifest)['files']
    for rel, value in files.items():
        target = (base / rel).resolve()
        expected = value['sha256'] if isinstance(value, dict) else value
        if not target.is_relative_to(base.resolve()) or sha(target) != expected:
            raise RuntimeError('Frozen file changed: ' + str(target))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if not snapshot.is_relative_to((ROOT / 'snapshots').resolve()):
        raise ValueError('Only fetched snapshots inside this package are accepted')
    summary = read(ROOT / 'results' / snapshot.name / 'generation_summary.json')
    if summary['status'] != 'GENERATION_COMPLETE':
        raise RuntimeError('A complete, audited generation is required')
    verify_bundle()
    authorize()
    start = read(ROOT / 'audit/execution_start.json')
    if sha(ROOT / 'authorization.json') != start['authorization_sha256']:
        raise RuntimeError('Execution authorization changed')
    if sha(ROOT / 'completed_package_manifest.sha256.json') != start['prepared_package_manifest_sha256']:
        raise RuntimeError('Prepared package manifest changed')
    prepared_count = verify_manifest(ROOT, ROOT / 'completed_package_manifest.sha256.json')
    preservation = []
    for item in read(ROOT / 'audit/baseline.json')['prior_package_manifests']:
        base = ROOT.parent / item['package']
        manifest = base / item['manifest']
        if sha(manifest) != item['manifest_sha256']:
            raise RuntimeError('Prior package manifest changed')
        preservation.append({'package': item['package'], 'unchanged': True,
                             'files_checked': verify_manifest(base, manifest)})
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT.parent.parent,
                          capture_output=True, check=True).stdout
    if hashlib.sha256(diff).hexdigest() != start['tracked_diff_sha256']:
        raise RuntimeError('Tracked user changes differ from execution start')
    work = snapshot / 'work'
    if list(work.glob('fatal_*.json')) or list(work.rglob('*.writing-*')):
        raise RuntimeError('Failure or partial-write evidence needs review')
    jobs = summary['scheduler']
    checkpoints = [read(p) for p in sorted(work.glob('checkpoint_[12].json'))]
    if len(checkpoints) != len(jobs) or checkpoints[-1]['reason'] != 'all_candidates_completed':
        raise RuntimeError('Final completion checkpoint missing')
    if checkpoints[-1]['pending_count'] != 0 or checkpoints[-1]['completed_count'] != 276:
        raise RuntimeError('Final checkpoint does not cover the complete task population')
    events = [json.loads(line) for line in (work / 'runtime_events.jsonl').read_text(encoding='utf-8').splitlines()]
    loads = [e for e in events if e.get('event') == 'model_loaded']
    if len(loads) != len(jobs) or any(e['revision'] != 'c170c708c41dac9275d15a8fff4eca08d52bab71' for e in loads):
        raise RuntimeError('Loaded-model provenance differs from approved jobs')
    completions = [e for e in events if e.get('event') == 'candidate_completed']
    expected_tasks = [r['task_id'] for r in read(ROOT / 'data/requests.json')]
    if [e['task_id'] for e in completions] != expected_tasks:
        raise RuntimeError('Runtime completion events differ from frozen order')
    expected = dict(read(ROOT / 'execution_manifest.json')['files'])
    expected['execution_manifest.json'] = sha(ROOT / 'execution_manifest.json')
    code = 'REMOTE=' + repr(remote_path()) + '\nEXPECTED=' + repr(expected) + '\n' + r'''
import datetime,hashlib,json,pathlib
root=pathlib.Path(REMOTE)
hashes={}
for rel,expected in EXPECTED.items():
 p=(root/rel).resolve()
 if not p.is_relative_to(root.resolve()):raise RuntimeError('Unexpected remote path')
 hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
 if hashes[rel]!=expected:raise RuntimeError('Remote frozen file differs: '+rel)
auth=(root/'authorization.json').read_bytes()
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'remote_directory':REMOTE,'frozen_files_verified':len(hashes),'hashes':hashes,
 'authorization':json.loads(auth),'authorization_sha256':hashlib.sha256(auth).hexdigest(),
 'remote_mutation':False,'model_import':False}))
'''
    remote = json.loads(ssh_python(code))
    if remote['authorization'] != read(ROOT / 'authorization.json'):
        raise RuntimeError('Remote authorization is not semantically identical')
    source_files = [ROOT / name for name in ['poll_execution.py', 'summarize_candidates.py', 'audit_completed_execution.py']]
    for path in source_files:
        ast.parse(path.read_text(encoding='utf-8'))
    out = ROOT / 'results' / snapshot.name
    save(out / 'remote_frozen_verification.json', remote)
    result = {'checked_at_utc': now(), 'status': 'PASS',
              'prepared_package_files_unchanged': prepared_count,
              'prior_packages': preservation, 'tracked_user_diff_unchanged': True,
              'local_authorization_unchanged': True, 'remote_authorization_parsed_equal': True,
              'authorization_serialization_note': 'Local UTF-8 and remote ASCII escapes can differ in bytes; parsed content equality verified.',
              'remote_frozen_files_verified': remote['frozen_files_verified'],
              'candidate_completion_events_in_exact_frozen_order': len(completions),
              'model_load_events': loads, 'final_checkpoint': checkpoints[-1],
              'physical_GPU_hours': summary['physical_GPU_hours'],
              'candidate_execution_performed': False, 'formal_judge_requests': 0,
              'stage_A_frozen': True, 'stage_B_frozen': False,
              'new_audit_helpers_AST_checked': len(source_files)}
    save(out / 'completion_integrity_audit.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'prepared_package_files_unchanged',
         'tracked_user_diff_unchanged', 'remote_frozen_files_verified',
         'candidate_completion_events_in_exact_frozen_order', 'physical_GPU_hours',
         'candidate_execution_performed', 'formal_judge_requests', 'stage_B_frozen']}, indent=2))


if __name__ == '__main__':
    main()
