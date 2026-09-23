"""One-time CPU acceptance, Stage B content freeze and deployable archive creation."""
import argparse
import ast
import contextlib
import io
import platform
import subprocess
import sys
import tarfile
import tempfile
import unittest
from common import *
from plan import validate_content
from control import extract_checked

def preserved():
    previous=ROOT.parent/'acl_main_extension_stage_b_preparation_v1';baseline=read(ROOT/'audit/baseline.json');old=read(previous/'audit/baseline.json')
    records=[]
    sources=[{'package':previous.name,'manifest':'completed_preparation_manifest.sha256.json','manifest_sha256':baseline['preparation_manifest_sha256']},
      {'package':'acl_main_extension_candidate_run_v1','manifest':'execution_completion_manifest.sha256.json','manifest_sha256':old['candidate_manifest_sha256']},*old['prior_packages']]
    for item in sources:
        root=ROOT.parent/item['package'];p=root/item['manifest']
        if sha(p)!=item['manifest_sha256']:raise RuntimeError('Predecessor manifest changed')
        records.append({'package':item['package'],'manifest':item['manifest'],'manifest_sha256':sha(p),'files_checked':verify_manifest(root,p),'unchanged':True})
    diff=subprocess.run(['git','diff','--binary','HEAD'],cwd=ROOT.parent.parent,capture_output=True,check=True).stdout
    if hashlib.sha256(diff).hexdigest()!=baseline['tracked_diff_sha256']:raise RuntimeError('Tracked user changes changed')
    status=subprocess.run(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT.parent.parent,capture_output=True,text=True,check=True).stdout
    return {'packages':records,'files_preserved':sum(r['files_checked'] for r in records),'tracked_diff_sha256':hashlib.sha256(diff).hexdigest(),'tracked_status':status,'tracked_modification_count':len(status.splitlines())}

def local_linux_check():
    sources={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8') for p in ROOT.rglob('*.py')}
    code='import json,sys\nsources='+repr(sources)+'\nfor name,source in sources.items():compile(source,name,"exec")\nprint(json.dumps({"python":sys.version,"compiled_sources":len(sources),"new_inference_requests":0,"model_modules_imported":[]}))\n'
    result=subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','python3','-B','-'],input=code.encode(),capture_output=True,check=True,timeout=30)
    subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','bash','-n'],input=(ROOT/'runtime/job.sbatch').read_bytes(),capture_output=True,check=True,timeout=15)
    return dict(json.loads(result.stdout),platform='local WSL Ubuntu-24.04; not CLUSTER',sbatch_syntax_valid=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*.py')})

def cli_guard(script,args,should_succeed):
    wrapper=r'''
import builtins,json,runpy,subprocess,sys
script=sys.argv[1];sys.argv=sys.argv[1:];events=[];original=builtins.__import__
def guarded_import(name,*a,**kw):
 if name.split('.')[0] in ['torch','transformers','vllm','tokenizers']:
  events.append('model_import:'+name);raise AssertionError('Model import in CPU entry')
 return original(name,*a,**kw)
def forbidden_run(*a,**kw):events.append('subprocess_attempt');raise AssertionError('Network/scheduler/subprocess in CPU entry')
builtins.__import__=guarded_import;subprocess.run=forbidden_run
error=None
try:runpy.run_path(script,run_name='__main__')
except BaseException as e:error={'type':type(e).__name__,'message':str(e)}
print('CPU_GUARD_RESULT='+json.dumps({'error':error,'events':events,'model_modules_imported':[m for m in ['torch','transformers','vllm','tokenizers'] if m in sys.modules]}))
'''
    r=subprocess.run([sys.executable,'-B','-X','utf8','-c',wrapper,script,*args],cwd=ROOT,capture_output=True,text=True,check=True,timeout=60)
    output=json.loads(r.stdout.rsplit('CPU_GUARD_RESULT=',1)[1]);success=output['error'] is None
    if success!=should_succeed or output['events'] or output['model_modules_imported']:raise RuntimeError('CPU CLI guard failed: '+script+' '+str(args)+' '+str(output))
    if not should_succeed and output['error']['type'] not in ['PermissionError','FileNotFoundError']:raise RuntimeError('Guard failed for an unexpected reason')
    return {'script':script,'arguments':args,'expected_success':should_succeed,**output}

def freeze():
    for rel in ['stage_B_manifest.json','authorization.json','work','completed_package_manifest.sha256.json']:
        if (ROOT/rel).exists():raise FileExistsError('Freeze requires an unused final package: '+rel)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    validation=validate_content(require_frozen=False);preservation=preserved();linux=local_linux_check()
    stream=io.StringIO()
    # This process never imports model libraries; tests inject handwritten backends.
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with contextlib.redirect_stdout(stream):result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
    if any(m in sys.modules for m in ['torch','vllm','transformers','tokenizers']):raise RuntimeError('Model module imported by CPU acceptance')
    regression=read(ROOT/'audit/legacy_regression.json')
    if regression['scored_conditions_recomputed']!=432:raise RuntimeError('Historical score regression missing')
    save(ROOT/'audit/CPU_acceptance.json',{'checked_utc':now(),'status':'PASS_LOCAL_CPU_ENGINEERING','tests_run':result.testsRun,'test_output':stream.getvalue(),
      'validation':validation,'preservation':preservation,'Linux_syntax':linux,'python':platform.python_version(),'model_modules_imported':[],
      'new_model_requests':0,'GPU_submissions':0,'new_candidate_generation':0,'historical_regression_sha256':sha(ROOT/'audit/legacy_regression.json'),
      'environmental_exceptions':['CLUSTER SSH control session unavailable; no fresh remote environment verification.','One historical count native decode remains pending restored connection; text-only disagreement retained.']})
    c=read(ROOT/'configs/experiment.json');c['freeze']={'Stage_A':True,'Stage_B':True,'pending':[],'frozen_utc':now()};c['status']='FROZEN_BEFORE_FORMAL_JUDGE_INFERENCE'
    with (ROOT/'configs/experiment.json').open('w',encoding='utf-8',newline='\n') as f:json.dump(c,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    excluded={'__pycache__','execution_audit','snapshots','analyses','work'}
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and not excluded.intersection(p.relative_to(ROOT).parts) and p.suffix!='.pyc'}
    save(ROOT/'stage_B_manifest.json',{'created_utc':now(),'Stage_B_frozen':True,'scope':c['scope'],'independent_tasks':276,'partial_tasks':33,
       'meaning':'Internal content freeze before formal judge inference; not GPU authorization, a completed experiment, or public preregistration.',
       'files':files})
    frozen_hash=sha(ROOT/'stage_B_manifest.json');artifact='FormalCRRC_Stage_B_'+frozen_hash[:12]+'.tar.gz'
    expected=dict(files);expected['stage_B_manifest.json']=frozen_hash
    with tarfile.open(ROOT/artifact,'x:gz',compresslevel=6) as tar:
        for rel in sorted(expected):
            p=ROOT/rel;info=tar.gettarinfo(str(p),arcname=rel);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with p.open('rb') as f:tar.addfile(info,f)
    save(ROOT/'export_receipt.json',{'created_utc':now(),'archive':artifact,'sha256':sha(ROOT/artifact),'bytes':(ROOT/artifact).stat().st_size,
      'uncompressed_file_bytes':sum((ROOT/rel).stat().st_size for rel in expected),'files':len(expected),'stage_B_sha256':frozen_hash,'contains_weights':False,'contains_live_authorization':False})
    save(ROOT/'approval_request.json',{'created_utc':now(),'approved':False,'scope':c['scope'],'stage_B_sha256':frozen_hash,'task_ids_sha256':digest(c['task_ids']),
      'maximum_submission_slots':24,'maximum_seconds_per_slot':28800,'maximum_physical_GPU_seconds':691200,'GPU':'nvidia_h100_pcie:1',
      'automatic_submission':False,'candidate_generation':False,'model_download':False,'conditions':20976,'RTS_requests':9936,'count_requests':1104,
      'authorization_template':'authorization.template.json','must_be_resolved_before_execution':['Restore existing CLUSTER connection and check fixed cache/versions','Review exact scope and record new user GPU authorization'],
      'archive_sha256':sha(ROOT/artifact)})
    guard=[]
    for script,args in [('plan.py',['validate']),('plan.py',['dry-run']),('analyze.py',[]),('control.py',['plan']),('submit.py',['plan']),('environment_probe.py',[])]:guard.append(cli_guard(script,args,True))
    for script,args in [('plan.py',['run']),('plan.py',['generate']),('control.py',['deploy']),('control.py',['submit','--chunk','qwen_01']),('submit.py',['submit','--chunk','qwen_01']),('worker.py',['--slot','1','--chunk','qwen_01'])]:guard.append(cli_guard(script,args,False))
    if (ROOT/'work').exists() or (ROOT/'authorization.json').exists():raise RuntimeError('Acceptance mutated live execution state')
    with tempfile.TemporaryDirectory(prefix='FormalCRRC_Stage_B_archive_CPU_') as d:
        extract_checked(ROOT/artifact,Path(d),expected);verify_manifest(Path(d),Path(d)/'stage_B_manifest.json')
        run=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(d)/'plan.py'),'validate'],cwd=d,capture_output=True,text=True,check=True,timeout=60)
        archive_validation=json.loads(run.stdout)
    validation=validate_content();preservation_final=preserved()
    save(ROOT/'audit/post_freeze_acceptance.json',{'checked_utc':now(),'status':'PASS','Stage_B_frozen':True,'stage_B_sha256':frozen_hash,
      'frozen_files_verified':verify_manifest(ROOT,ROOT/'stage_B_manifest.json'),'archive_files_verified':len(expected),'archive_extracted_validation':archive_validation,
      'live_CLI_guards':guard,'validation':validation,'preservation':preservation_final,'new_model_requests':0,'GPU_submissions':0,'authorization_present':False})
    final_files={p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    save(ROOT/'completed_package_manifest.sha256.json',{'created_utc':now(),'status':'COMPLETE_CPU_ENGINEERING_STAGE_B_FROZEN','formal_judge_experiment_executed':False,'files':final_files})
    print(json.dumps({'status':'STAGE_B_FROZEN','tests_passed':result.testsRun,'frozen_manifest_sha256':frozen_hash,'archive':artifact,'completed_package_files':len(final_files),
      'preserved_previous_files':preservation_final['files_preserved'],'tracked_modifications_preserved':preservation_final['tracked_modification_count'],'GPU_jobs_submitted':0,'current_CLUSTER_connection':'blocked'},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if not a.freeze:print(json.dumps({'mode':'plan','will_not_write_or_run_models':True,'freeze_already_exists':(ROOT/'stage_B_manifest.json').exists()}));return
    freeze()
if __name__=='__main__':main()
