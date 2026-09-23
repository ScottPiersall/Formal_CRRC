"""Credential-free, network-disabled Docker execution for code_pilot_v1.

Only serialized inputs and this worker enter the container via stdin/argv.
No host directories, sockets, credentials, or environment variables are mounted.
Each test is a fresh Python process. The container supervises a batch of tests;
the host independently performs exact equality on literal outputs.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import subprocess
import time
import uuid

IMAGE = 'sha256:80f5d259a5969c86f6c92145d572de4a68c68e0edd28d4367dec0fb411b42af3'
LIMITS = dict(test_seconds=5, address_space_bytes=512*1024**2,
              container_memory='768m', pids_limit=32, cpus=1, tmpfs='64m',
              timing='5 s wall-clock from immediately before compile/exec through call and output serialization; fresh interpreter startup measured separately; 7 s supervisor deadline')

WORKER = r'''
import ast,contextlib,io,json,os,resource,signal,sys,time
spec=json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
resource.setrlimit(resource.RLIMIT_CPU,(6,6))
os.chdir('/tmp')
class CandidateTimeout(BaseException): pass
def alarm(*a): raise CandidateTimeout()
signal.signal(signal.SIGALRM,alarm)
print('{"worker_ready":true}',flush=True)
signal.setitimer(signal.ITIMER_REAL,5)
start=time.monotonic()
result={}
saved_repr=repr
def finite(v,depth=0):
 if depth>30: return False
 if type(v) in (int,bool,str): return True
 if type(v) in (list,tuple,set,frozenset): return all(finite(x,depth+1) for x in v)
 if type(v) is dict: return all(finite(k,depth+1) and finite(x,depth+1) for k,x in v.items())
 return False
try:
 with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
  ns={}; exec(compile(spec['code'],'candidate.py','exec'),ns)
  if spec['entry_point'] not in ns: raise NameError('undefined entry point')
  value=ns[spec['entry_point']](*spec['args'])
  valid=finite(value)
  output=saved_repr(value)
  if len(output.encode())>65536: output=output[:4096]; valid=False
 result=dict(category='ok',observed_repr=output,supported_output=valid)
except CandidateTimeout: result=dict(category='candidate_timeout',exception='CandidateTimeout')
except MemoryError: result=dict(category='candidate_memory_limit',exception='MemoryError')
except BaseException as e: result=dict(category='candidate_exception',exception=type(e).__name__,message=str(e)[:1000])
finally: signal.setitimer(signal.ITIMER_REAL,0)
result['candidate_seconds']=time.monotonic()-start
print(json.dumps(result,ensure_ascii=True))
'''

SUPERVISOR = r'''
import json,subprocess,sys,time,os
spec=json.load(sys.stdin)
for test in spec['tests']:
 start=time.monotonic()
 try:
  p=subprocess.run([sys.executable,'-I','-c',spec['worker']],input=json.dumps(dict(code=spec['code'],entry_point=spec['entry_point'],args=test['args'])),capture_output=True,text=True,timeout=7)
  if p.returncode==0:
   try: result=json.loads(p.stdout.splitlines()[-1])
   except Exception: result=dict(category='infrastructure_error',reason='invalid worker output',stdout=p.stdout[:2000])
  else: result=dict(category='infrastructure_error',reason='worker exited without classified result',returncode=p.returncode,stderr=p.stderr[:2000])
 except subprocess.TimeoutExpired as e:
  ready=b'"worker_ready":true' in (e.stdout or b'')
  result=dict(category='candidate_timeout' if ready else 'infrastructure_error',reason='supervisor deadline after worker-ready' if ready else 'worker failed to start within supervisor deadline')
 result.update(test_index=test['test_index'],repeat=test['repeat'],process_seconds=time.monotonic()-start)
 print(json.dumps(result),flush=True)
'''

def environment():
    result = subprocess.run(['docker','image','inspect',IMAGE],capture_output=True,text=True,timeout=20,check=True)
    img = json.loads(result.stdout)[0]
    return dict(image_id=img['Id'], repo_digests=img.get('RepoDigests'), limits=LIMITS,
                worker_sha256=hashlib.sha256(WORKER.encode()).hexdigest(),
                supervisor_sha256=hashlib.sha256(SUPERVISOR.encode()).hexdigest(),
                no_host_mounts=True, network='none', read_only_root=True,
                user='65534:65534', capabilities='ALL dropped', no_new_privileges=True)

def execute(code, entry_point, inputs, repeats=2):
    tests=[dict(args=inp,test_index=i,repeat=r) for r in range(repeats) for i,inp in enumerate(inputs)]
    name='fcrrc-cp-'+uuid.uuid4().hex[:16]
    command=['docker','run','--pull=never','--name',name,'--network','none','--read-only',
             '--cap-drop=ALL','--security-opt=no-new-privileges',
             '--user','65534:65534','--pids-limit=32','--memory=768m','--memory-swap=768m',
             '--cpus=1','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','-i',IMAGE,
             'python','-I','-c',SUPERVISOR]
    start=time.monotonic()
    try:
        proc=subprocess.run(command,input=json.dumps(dict(code=code,entry_point=entry_point,tests=tests,worker=WORKER)),
                            capture_output=True,text=True,timeout=30+8*len(tests))
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
        state=json.loads(inspected.stdout)[0]['State'] if inspected.returncode==0 else {}
        if proc.returncode!=0 or state.get('OOMKilled'):
            return dict(status='infrastructure_error',category='container_oom' if state.get('OOMKilled') else 'container_failure',
                        returncode=proc.returncode,stderr=proc.stderr[:4000],state=state,tests=[],elapsed_seconds=time.monotonic()-start)
        rows=[json.loads(x) for x in proc.stdout.splitlines()]
        if len(rows)!=len(tests):
            raise RuntimeError('Incomplete supervisor results')
        return dict(status='complete',tests=rows,elapsed_seconds=time.monotonic()-start)
    except (OSError,subprocess.SubprocessError,ValueError,RuntimeError) as e:
        return dict(status='infrastructure_error',reason=str(e),tests=[],elapsed_seconds=time.monotonic()-start)
    finally:
        # Only the uniquely named container created above is removed.
        try: subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
        except (OSError,subprocess.SubprocessError): pass

def compare(row, expected_repr):
    if row['category']=='infrastructure_error': return None
    if row['category']!='ok': return False
    try:
        observed=ast.literal_eval(row['observed_repr'])
        expected=ast.literal_eval(expected_repr)
        return bool(observed==expected)
    except (ValueError,SyntaxError,TypeError):
        return False

def stable_pair(a,b):
    keys=('category','observed_repr','supported_output','exception','passed')
    return all(a.get(k)==b.get(k) for k in keys)
