"""Immutable request ledger and explicit deterministic-score recovery."""
from common import *

class Checkpoint(Exception):pass
class AmbiguousRequest(RuntimeError):pass

def validate_response(response,spec):
    ids=response['generated_token_ids']
    if response['prompt_token_ids']!=spec['context_token_ids']:raise RuntimeError('Engine changed requested context')
    if not all(type(x) is int and x>=0 for x in ids) or len(ids)>spec['params']['max_tokens']:raise RuntimeError('Invalid generated token receipt')
    if response['generated_tokens']!=len(ids):raise RuntimeError('Token count mismatch')
    if not isinstance(response['returned_text'],str):raise RuntimeError('Missing response text')
    import math
    if not math.isfinite(response['elapsed_seconds']) or response['elapsed_seconds']<0:raise RuntimeError('Invalid elapsed time')

class Store:
    def __init__(self,work,stage_hash,job_id,slot,can_start=lambda mode:True,permits=()):
        self.work=Path(work);self.stage_hash=stage_hash;self.job_id=job_id;self.slot=slot;self.can_start=can_start;self.permits=set(permits)
    def request(self,spec,invoke):
        rid=safe_id(spec['request_id']);expected=digest(spec)
        attempts=sorted((self.work/'attempts').glob(rid+'_a*.json'))
        if len(attempts)>3:raise RuntimeError('Technical retry ceiling exceeded')
        missing=[]
        for index,path in enumerate(attempts):
            if path.name!=rid+'_a'+str(index)+'.json':raise RuntimeError('Attempt numbering gap')
            attempt=read(path)
            if attempt['spec_sha256']!=expected or attempt['stage_B_sha256']!=self.stage_hash:raise RuntimeError('Request spec or freeze changed')
            receipt=self.work/'requests'/path.name
            if receipt.exists():
                r=read(receipt)
                if r['spec_sha256']!=expected or r['spec']!=spec or r['stage_B_sha256']!=self.stage_hash or r['attempt_sha256']!=sha(path):raise RuntimeError('Raw request binding changed')
                validate_response(r['response'],spec)
                if index!=len(attempts)-1:raise RuntimeError('A completed request was retried')
                return r['response'],{'path':receipt.relative_to(self.work).as_posix(),'sha256':sha(receipt)}
            missing.append(path)
        if missing and (spec['mode']=='generation' or rid not in self.permits):raise AmbiguousRequest(rid+' began without a complete receipt; no regeneration')
        if missing and len(attempts)>=3:raise AmbiguousRequest('No further deterministic score retries: '+rid)
        if not self.can_start(spec['mode']):raise Checkpoint('Clean time checkpoint before '+rid)
        index=len(attempts);path=self.work/'attempts'/(rid+'_a'+str(index)+'.json')
        save(path,{'request_id':rid,'mode':spec['mode'],'spec_sha256':expected,'stage_B_sha256':self.stage_hash,'job_id':self.job_id,'slot':self.slot,
           'started_utc':now(),'attempt_index':index,'explicit_score_recovery':bool(index),'context_tokens':len(spec['context_token_ids']),'max_output_tokens':spec['params']['max_tokens']})
        response=invoke(spec)
        # Save the returned engine record before validation. An invalid engine response
        # remains evidence and halts processing; it is never silently replaced.
        receipt=self.work/'requests'/path.name
        save(receipt,{'request_id':rid,'spec':spec,'spec_sha256':expected,'stage_B_sha256':self.stage_hash,'attempt_sha256':sha(path),
            'job_id':self.job_id,'slot':self.slot,'response':response,'completed_utc':now()})
        validate_response(response,spec)
        return response,{'path':receipt.relative_to(self.work).as_posix(),'sha256':sha(receipt)}
    def completed(self,key,binding):
        p=self.work/'results'/(safe_id(key)+'.json')
        if not p.exists():return None
        row=read(p)
        if row['stage_B_sha256']!=self.stage_hash or row['binding']!=binding:raise RuntimeError('Condition binding changed')
        for ref in row['request_refs']:
            q=(self.work/ref['path']).resolve()
            if not q.is_relative_to(self.work.resolve()) or sha(q)!=ref['sha256']:raise RuntimeError('Referenced raw request changed')
        return row
    def finish(self,row):save(self.work/'results'/(safe_id(row['result_id'])+'.json'),row)

def inspect_requests(work,stage_hash):
    work=Path(work);completed=[];missing=[]
    if list(work.rglob('*.writing-*')):raise RuntimeError('Uncommitted write remnant; review required')
    for path in sorted((work/'attempts').glob('*.json')):
        a=read(path)
        if a['stage_B_sha256']!=stage_hash:raise RuntimeError('Attempt belongs to another freeze')
        q=work/'requests'/path.name
        if q.exists():
            r=read(q)
            if r['attempt_sha256']!=sha(path) or r['spec_sha256']!=a['spec_sha256'] or digest(r['spec'])!=a['spec_sha256'] or r['stage_B_sha256']!=stage_hash:raise RuntimeError('Raw ledger binding mismatch')
            validate_response(r['response'],r['spec']);completed.append(path.name)
        else:missing.append({'request_id':a['request_id'],'mode':a['mode'],'attempt_index':a['attempt_index'],'attempt_file':path.name,'spec_sha256':a['spec_sha256']})
    for path in (work/'requests').glob('*.json'):
        if not (work/'attempts'/path.name).exists():raise RuntimeError('Response without attempt')
    # Earlier missing deterministic attempts are resolved only by a complete later receipt.
    unresolved=[]
    for m in missing:
        if not any((work/'requests'/(m['request_id']+'_a'+str(i)+'.json')).exists() for i in range(m['attempt_index']+1,3)):unresolved.append(m)
    return {'complete_requests':len(completed),'missing_attempts_retained':missing,'unresolved':unresolved}
