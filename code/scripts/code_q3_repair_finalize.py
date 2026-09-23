"""Package the supplement with verifiable, explicit original-study dependency."""
import pathlib,sys,json,subprocess,zipfile,hashlib,concurrent.futures,difflib
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from formalcrrc import code_extension as cp,code_five as f,code_q3_repair as q

def main():
    o=q.out(ROOT);status=cp.load(o/'execution_status.json');q.verify(ROOT)
    assert status['resources']['all_jobs_reconciled'],'Reconcile every GPU allocation before final packaging'
    audit=cp.load(o/'analysis/independent_audit.json')
    assert audit['passed'] and not status['validation_failures']
    assert audit['accepted_scores_independently_checked']==status['models']['qwen3']['main_contexts']+status['models']['qwen3']['smoke_contexts']
    if status['all_authorized_execution_finished']:
        assert status['resources']['forward_count_complete'] and status['resources']['active_job_reserved_gpu_seconds']==0
        assert status['all_continuations_attempted'] and all(r['reason']=='terminal_reasoning_failure' for r in status['missing_score_conditions'])
    old=cp.load(f.out(ROOT)/'handoff_manifest.json')
    def check(item):
        name,digest=item;assert cp.sha((ROOT/name).read_bytes())==digest,name
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,old['files'].items()))
    archive=ROOT.parent/'FormalCRRC_Code_Five_Models_Handoff.zip'
    assert cp.sha(archive.read_bytes())=='c5318a3e97c4de510b9ade2451a65fa5ef5e66153f4f6ff76d0ba314daa8108f'
    before=(o/'operations/initial_tracked_changes.patch').read_bytes()
    after=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    assert before==after,'Pre-existing tracked changes changed during this supplement'
    sources=sorted(set(list((ROOT/'src/formalcrrc').glob('code_q3_repair*.py'))+list((ROOT/'scripts').glob('code_q3_repair*.py'))+
        list((ROOT/'slurm').glob('run_code_q3_repair*.sbatch'))+list((ROOT/'tests').glob('test_code_q3_repair*.py'))))
    patch=''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix())) for p in sources)
    (o/'operations/new_code.patch').write_text(patch)
    text=f"""# Qwen3 continuation/scoring repair supplement

This is a separate post-observation supplement. The delivered v2/v3 evidence and original ZIP are unchanged.

Start with `analysis/REPORT_ZH.md`, `analysis/ROOT_CAUSE_AND_REPAIR_ZH.md`, `analysis/METHODS_RESULTS_EN.md`, `analysis/FIVE_MODEL_SUMMARY.md`, and `execution_status.json`. `analysis/qwen3_repair_audit.png` and its SVG version provide the verified scientific figure.

Dependency: **FormalCRRC_Code_Five_Models_Handoff.zip**, SHA256 `c5318a3e97c4de510b9ade2451a65fa5ef5e66153f4f6ff76d0ba314daa8108f`. It contains the full five-model original evidence including v2. Extract that archive into a workspace, then overlay this supplement. `baseline_dependency_manifest.json` embeds the full original manifest and hashes every directly consumed original input, reasoning trace and score. No model weights, credentials or access tokens are packaged.

The supplement keeps original 8k results distinct from uniformly rescored 8k results and two-stage32k continuation results. Continuation seeds and source hashes are in inputs.json, fixed rules in protocol.json, and pre-inference file hashes in freeze.json. Raw continuation failures remain in continuations/; runtime failures are in failures/. Each score records hashes of its reasoning and shared-prefix probability node receipts.

All authorized execution finished: {status['all_authorized_execution_finished']}. All scoring conditions complete: {status['all_score_conditions_complete']}. All 62 continuation requests attempted: {status['all_continuations_attempted']}. All jobs reconciled: {status['resources']['all_jobs_reconciled']}. See explicit missing conditions in execution_status.json; do not interpret package creation as completion of missing scoring conditions.

Commands from repository root (local WSL scientific environment):

```bash
PYTHONPATH=src .remote/code_pilot_v1_venv/bin/python -m pytest -q tests/test_code_q3_repair.py
.remote/code_pilot_v1_venv/bin/python scripts/code_q3_repair_analyze.py
.remote/code_pilot_v1_venv/bin/python scripts/code_q3_repair_diagnostics.py
.remote/code_pilot_v1_venv/bin/python scripts/code_q3_repair_figures.py
.remote/code_pilot_v1_venv/bin/python scripts/code_q3_repair_finalize.py
python3 scripts/code_q3_repair_CLUSTER status
python3 scripts/code_q3_repair_resume.py --stage main --check
```

The actual submissions and allocated time limits are in submissions.jsonl. The CLUSTER environment is the existing `FormalCRRC_code_extension_v3_five_models/envs/qwen3/bin/python`; the new execution root is `FormalCRRC_code_extension_v4_q3_continuation`. No new model precision audit or allowance was introduced. GPU resume must use only the remaining original budget and must inspect existing receipts first; completed or terminal reasoning must never be resampled. If this execution is complete, no GPU resume is required.

Actual initial GPU commands were `python3 scripts/code_q3_repair_CLUSTER submit --stage smoke --minutes 35` and `python3 scripts/code_q3_repair_CLUSTER submit --stage main --minutes 169`. The existing pending main job804015 was then updated with `scontrol update JobId=804015 MinMemoryNode=98304`. The frozen SBATCH source remains128GiB; the administrative override is recorded separately. The resume helper preserves budget and duplicate-job guards while allowing a unique administrative label such as main_resume_1 to execute the unchanged main runtime. It refuses active/unreconciled jobs and saved terminal failures. Only confirmed infrastructure retries or previously unattempted deferred work qualify; a probability or protocol failure does not authorize resampling.

All26179 original handoff files and the original ZIP SHA256 were reverified at finalization. Existing tracked modifications are byte-identical to the initial binary git diff. New code is included directly and as operations/new_code.patch.
"""
    (o/'HANDOFF_README.md').write_text(text,encoding='utf-8')
    cp.snapshot(o/'final_preservation_audit.json',dict(created_at=cp.now(),original_manifest_files_verified=len(old['files']),
        original_zip_sha256=cp.sha(archive.read_bytes()),preexisting_tracked_binary_diff_unchanged=True,
        preexisting_tracked_diff_sha256=cp.sha(before),new_source_files=len(sources)))
    files=sources+[p for p in o.rglob('*') if p.is_file() and p.name!='handoff_manifest.json' and '__pycache__' not in p.parts]
    manifest=dict(created_at=cp.now(),files=cp.manifest(ROOT,files),dependencies='Original five-model ZIP required; exact hash and full dependency manifest supplied')
    cp.snapshot(o/'handoff_manifest.json',manifest)
    dest=ROOT.parent/'FormalCRRC_Code_Qwen3_32k_Repair_Handoff.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[o/'handoff_manifest.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,digest in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=cp.sha(dest.read_bytes());dest.with_suffix('.zip.sha256').write_text(digest+'  '+dest.name+'\n')
    print(json.dumps(dict(archive=str(dest),sha256=digest,bytes=dest.stat().st_size,files=len(files)+1,all_score_conditions_complete=status['all_score_conditions_complete'])))
if __name__=='__main__':main()
