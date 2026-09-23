import pathlib,sys,json,hashlib,datetime
O=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=O/'preregistration/freeze.json'
 if p.exists():
  f=json.loads(p.read_text());assert all(sha(O/r)==h for r,h in f['files'].items());print('FREEZE VERIFIED',sha(p));return
 assert (O/'preregistration/cpu_tests.xml').exists()
 assert 'failures="0"' in (O/'preregistration/cpu_tests.xml').read_text()
 files={}
 for folder in ('src','preregistration','operations'):
  for q in (O/folder).rglob('*'):
   if q.is_file() and '__pycache__' not in q.parts and q.suffix not in ('.out','.err'):
    files[q.relative_to(O).as_posix()]=sha(q)
 for q in (O/'data').glob('*.jsonl'):files[q.relative_to(O).as_posix()]=sha(q)
 f=dict(study='code_protocol_followup_v2',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity='Observed v1 original-template negative result; designed after observation and frozen before v2 inference',files=files)
 p.write_text(json.dumps(f,sort_keys=True,indent=2)+'\n');print('FROZEN',sha(p),len(files))
if __name__=='__main__':main()
