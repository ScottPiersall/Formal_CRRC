"""Standard-library helpers; importing this module never executes models or jobs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import uuid
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    temp=p.with_name(p.name+'.writing-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,p);temp.unlink()
def text_file(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def verify_manifest(base,p):
    base=Path(base).resolve();files=read(p)['files']
    for rel,v in files.items():
        q=(base/rel).resolve();expected=v['sha256'] if isinstance(v,dict) else v
        if not q.is_relative_to(base) or sha(q)!=expected:raise RuntimeError('Manifest mismatch: '+rel)
    return len(files)
def safe_id(s):
    if not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9_]+',s):raise ValueError('Unsafe identifier')
    return s
