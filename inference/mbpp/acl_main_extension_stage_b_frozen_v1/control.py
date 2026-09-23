"""Explicit transport to the existing CLUSTER connection; default is local plan."""
import argparse
import base64
import io
import subprocess
import tarfile
from common import *
from gates import verify_bundle,authorize
PYTHON='/REDACTED_LOCAL_PATH'
def remote_path():return '/REDACTED_LOCAL_PATH'+sha(ROOT/'stage_B_manifest.json')[:12]
def ssh(code,output=None,timeout=60):
    marker='FORMALCRRC_STAGE_B_TRANSPORT_20260915'
    if marker in code:raise ValueError('Unexpected delimiter')
    script='export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'+PYTHON+" - <<'"+marker+"'\n"+code+'\n'+marker+'\n'
    cmd=['wsl.exe','-d','Ubuntu-24.04','--','ssh','-S','/REDACTED_LOCAL_PATH','-o','BatchMode=yes','-o','ProxyCommand=false','-o','ConnectTimeout=10','ANONYMOUS@CLUSTER','bash','-l','-s']
    r=subprocess.run(cmd,input=script.encode(),stdout=output if output else subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-4000:])
    return r.stdout
def extract_checked(archive,destination,expected):
    destination=Path(destination).resolve();seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            target=(destination/m.name).resolve()
            if m.name in seen or m.name not in expected or not m.isfile() or not target.is_relative_to(destination):raise RuntimeError('Unsafe or unexpected archive member')
            seen.add(m.name);target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with tar.extractfile(m) as src,target.open('xb') as dst:
                while True:
                    block=src.read(1024*1024)
                    if not block:break
                    h.update(block);dst.write(block)
            if h.hexdigest()!=expected[m.name]:raise RuntimeError('Transferred file hash mismatch')
    if seen!=set(expected):raise RuntimeError('Archive file set incomplete')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='plan',choices=['plan','deploy','submit','release-held','status','fetch']);p.add_argument('--chunk');p.add_argument('--slot',type=int);p.add_argument('--recover-score',action='append',default=[]);a=p.parse_args()
    verify_bundle();remote=remote_path()
    if a.action=='plan':print(json.dumps({'remote_directory':remote,'model_imports':[],'network_called':False,'job_submitted':False,'explicit_actions':['deploy','submit --chunk qwen_01','status','fetch']},indent=2));return
    if a.action in ['deploy','submit','release-held']:authorize()
    if a.action=='deploy':
        export=read(ROOT/'export_receipt.json');archive=ROOT/export['archive']
        if sha(archive)!=export['sha256']:raise RuntimeError('Reviewed archive changed')
        expected=dict(read(ROOT/'stage_B_manifest.json')['files']);expected['stage_B_manifest.json']=sha(ROOT/'stage_B_manifest.json')
        code='REMOTE='+repr(remote)+'\nEXPECTED='+repr(expected)+'\nAUTH='+repr(read(ROOT/'authorization.json'))+'\nBLOB='+repr(base64.b64encode(archive.read_bytes()).decode())+'\n'+r'''
import base64,hashlib,io,json,pathlib,subprocess,tarfile
root=pathlib.Path(REMOTE);root.mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BLOB)),mode='r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(EXPECTED) and {m.name for m in members}==set(EXPECTED)
 for m in members:
  p=(root/m.name).resolve();assert m.isfile() and p.is_relative_to(root.resolve())
  data=tar.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==EXPECTED[m.name]
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
  p.chmod(0o700 if m.name.endswith('.sbatch') else 0o600)
with (root/'authorization.json').open('x') as f:json.dump(AUTH,f,indent=2);f.write('\n')
r=subprocess.run(['/REDACTED_LOCAL_PATH','-B','plan.py','validate'],cwd=root,capture_output=True,text=True)
print(json.dumps({'remote_directory':REMOTE,'validation_exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'GPU_submitted':False}));raise SystemExit(r.returncode)
'''
        result=json.loads(ssh(code,timeout=180));save(ROOT/'execution_audit/deployment.json',result);print(json.dumps(result));return
    if a.action in ['submit','release-held']:
        argv=[PYTHON,'-B','submit.py',a.action]
        if a.action=='submit':
            if not a.chunk:raise ValueError('Fixed chunk required')
            argv+=['--chunk',safe_id(a.chunk)]
            for rid in a.recover_score:argv+=['--recover-score',safe_id(rid)]
        else:
            if a.slot is None:raise ValueError('Recorded slot required')
            argv+=['--slot',str(a.slot)]
        code='import subprocess,sys\nr=subprocess.run('+repr(argv)+',cwd='+repr(remote)+',capture_output=True,text=True)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");raise SystemExit(r.returncode)'
        print(ssh(code,timeout=90).decode());return
    code='REMOTE='+repr(remote)+'\n'+r'''
import datetime,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(REMOTE);work=root/'work'
if not root.is_dir():raise RuntimeError('Frozen package not deployed')
rows=[];unknown=[]
for p in sorted((work/'slots').glob('*/reservation.json')):
 sub=p.parent/'submission.json'
 if not sub.exists():unknown.append(str(p.relative_to(root)));continue
 job=json.loads(sub.read_text())['job_id']
 r=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,check=True,timeout=12)
 found=[line.split('|') for line in r.stdout.splitlines() if line.split('|')[0]==job]
 if len(found)!=1:raise RuntimeError('Unknown scheduler state')
 rows.append(found[0])
'''
    if a.action=='status':
        code+="print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scheduler_records':rows,'unknown_reservations':unknown,'completed_condition_files':len(list((work/'results').glob('*.json'))),'planned_conditions':20976,'live_counts_are_non_atomic':True}))"
        print(ssh(code).decode());return
    code+=r'''
terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
if unknown or not rows or any(r[1].split()[0] not in terminal for r in rows):raise RuntimeError('Fetch requires known terminal allocations')
files={}
for p in sorted(work.rglob('*')):
 if p.is_symlink():raise RuntimeError('Symlink in work evidence')
 if p.is_file():files[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scheduler_records':rows,'files':files,'stage_B_sha256':hashlib.sha256((root/'stage_B_manifest.json').read_bytes()).hexdigest()}))
'''
    inventory=json.loads(ssh(code,timeout=180))
    if inventory['stage_B_sha256']!=sha(ROOT/'stage_B_manifest.json'):raise RuntimeError('Remote freeze differs')
    dest=ROOT/'snapshots'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');dest.mkdir(parents=True,exist_ok=False)
    stream='REMOTE='+repr(remote)+'\nFILES='+repr(inventory['files'])+'\n'+r'''
import hashlib,pathlib,sys,tarfile
root=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for rel,expected in FILES.items():
  p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and p.is_file() and not (root/rel).is_symlink()
  assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
  tar.add(p,arcname=rel,recursive=False)
'''
    archive=dest/'work.tar.gz'
    with archive.open('xb') as output:ssh(stream,output=output,timeout=3600)
    extract_checked(archive,dest,inventory['files']);save(dest/'transfer_receipt.json',inventory)
    print(json.dumps({'snapshot':str(dest),'verified_files':len(inventory['files']),'new_inference':False}))
if __name__=='__main__':main()
