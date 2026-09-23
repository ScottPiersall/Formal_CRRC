"""Authorization, frozen content, allocation and global slot accounting gates."""
import os
import re
import subprocess
from common import *
from ledger import inspect_requests
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def verify_bundle(root=ROOT):
    root=Path(root);m=read(root/'stage_B_manifest.json');verify_manifest(root,root/'stage_B_manifest.json')
    c=read(root/'configs/experiment.json')
    if m.get('Stage_B_frozen') is not True or c['freeze']['Stage_B'] is not True or len(c['task_ids'])!=276 or len(c['primary_task_ids'])!=33:raise RuntimeError('Full Stage B freeze required')
    return c
def authorize(root=ROOT):
    root=Path(root);c=verify_bundle(root);a=read(root/'authorization.json')
    expected={'approved':True,'allow_judge_inference':True,'allow_GPU_submission':True,'allow_candidate_generation':False,'allow_model_download':False,
      'scope':c['scope'],'stage_B_sha256':sha(root/'stage_B_manifest.json'),'task_ids_sha256':digest(c['task_ids']),'maximum_submission_slots':24,
      'maximum_seconds_per_slot':28800,'maximum_physical_GPU_seconds':691200,'automatic_submission':False}
    if any(type(a.get(k)) is not type(v) or a.get(k)!=v for k,v in expected.items()):raise PermissionError('A new explicit authorization bound to this exact Stage B and 192-GPU-hour scope is required')
    if not isinstance(a.get('user_approval_text'),str) or not a['user_approval_text'].strip():raise PermissionError('Actual user approval text required')
    return c
def seconds(value):
    days=0
    if '-' in value:days,value=value.split('-',1);days=int(days)
    parts=[int(x) for x in value.split(':')]
    if len(parts)==2:h=0;m,s=parts
    elif len(parts)==3:h,m,s=parts
    else:raise ValueError('Unsupported Slurm duration')
    return days*86400+h*3600+m*60+s
def parse_allocation(text,job):
    f=dict(re.findall(r'(\w+)=([^\s]+)',text));tres=f.get('AllocTRES','');gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
    if f.get('JobId')!=str(job) or f.get('Account')!='<SLURM_ACCOUNT>' or f.get('Partition')!='normal' or f.get('NumNodes')!='1':raise PermissionError('Allocation identity differs')
    if f.get('JobState')!='RUNNING' or f.get('Requeue')!='0' or f.get('Restarts','0')!='0':raise PermissionError('Running non-requeued allocation required')
    if not gpu or int(gpu[1])!=1 or 'nvidia_h100_pcie' not in tres+f.get('TresPerNode',''):raise PermissionError('One H100 PCIe required')
    limit=seconds(f['TimeLimit']);elapsed=seconds(f['RunTime'])
    if not 0<limit<=28800 or elapsed>limit:raise PermissionError('Slot time limit exceeded')
    return f,limit-elapsed
def accounting(work,query=None):
    work=Path(work);reservations=sorted((work/'slots').glob('*/reservation.json'));records=[];elapsed=0
    if len(reservations)>24:raise RuntimeError('More than 24 slot reservations')
    for index,p in enumerate(reservations,1):
        if p.parent.name!=str(index).zfill(2):raise RuntimeError('Slot reservation gap')
        sub=p.parent/'submission.json'
        if not sub.exists():raise RuntimeError('Uncertain prior submission; do not submit again')
        job=read(sub)['job_id']
        if query is None:
            r=subprocess.run(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,check=True,timeout=15)
            rows=[line.split('|') for line in r.stdout.splitlines() if line.split('|')[0]==job]
            if len(rows)!=1:raise RuntimeError('Ambiguous scheduler accounting')
            rec=rows[0]
        else:rec=query(job)
        if rec[0]!=job or rec[1].split()[0] not in TERMINAL:raise RuntimeError('Previous allocation still active or unknown; submissions are sequential')
        gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',rec[4]);time=int(rec[3])
        if not gpu or int(gpu[1])!=1 or time>28800:raise RuntimeError('Prior job exceeded approved allocation')
        elapsed+=time;records.append({'slot':index,'job_id':job,'state':rec[1],'exit_code':rec[2],'elapsed_seconds':time,'AllocTRES':rec[4]})
    if elapsed>691200:raise RuntimeError('Total actual GPU budget exceeded')
    return {'slots_used':len(reservations),'reserved_GPU_seconds':len(reservations)*28800,'actual_GPU_seconds':elapsed,'records':records}
def check_recovery(work,stage_hash,permits):
    if list((Path(work)/'parity_failures').glob('*.json')):raise RuntimeError('Recorded parity failure blocks further allocations under this freeze')
    inspection=inspect_requests(work,stage_hash);allowed=set(permits)
    for item in inspection['unresolved']:
        if item['mode']=='generation' or item['request_id'] not in allowed or item['attempt_index']>=2:raise RuntimeError('Unresolved request blocks submission: '+item['request_id'])
    if allowed!={r['request_id'] for r in inspection['unresolved']}:raise RuntimeError('Score recovery permits must match unresolved deterministic requests exactly')
    return inspection
def claim(slot,chunk,root=ROOT):
    if type(slot) is not int or not 1<=slot<=24:raise PermissionError('Slot must be in the frozen 1..24 range')
    root=Path(root);c=authorize(root);work=root/'work';directory=work/'slots'/str(slot).zfill(2)
    sub=read(directory/'submission.json');reservation=read(directory/'reservation.json');job=os.environ.get('SLURM_JOB_ID','')
    frozen=sha(root/'stage_B_manifest.json')
    if not job.isdigit() or job!=sub['job_id'] or reservation['chunk_id']!=chunk or sub['stage_B_sha256']!=frozen or reservation['stage_B_sha256']!=frozen:raise PermissionError('Job does not match its reserved slot/chunk')
    r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,check=True,timeout=15);fields,remaining=parse_allocation(r.stdout,job)
    check_recovery(work,sha(root/'stage_B_manifest.json'),reservation['score_recovery_request_ids'])
    token={'slot':slot,'chunk_id':chunk,'job_id':job,'stage_B_sha256':sha(root/'stage_B_manifest.json'),'allocation_fields':fields,'claimed_utc':now()}
    save(directory/'claim.json',token);return c,token,remaining,reservation['score_recovery_request_ids']
def check_live_claim(root,token):
    authorize(root)
    if not isinstance(token,dict) or type(token.get('slot')) is not int or not 1<=token['slot']<=24 or token.get('stage_B_sha256')!=sha(root/'stage_B_manifest.json') or read(root/'work/slots'/str(token.get('slot')).zfill(2)/'claim.json')!=token:raise PermissionError('Valid allocation claim required before model imports')
    if os.environ.get('SLURM_JOB_ID')!=token['job_id']:raise PermissionError('Allocation environment differs')
    r=subprocess.run(['scontrol','show','job','-o',token['job_id']],capture_output=True,text=True,check=True,timeout=15);parse_allocation(r.stdout,token['job_id'])
