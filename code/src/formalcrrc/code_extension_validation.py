"""Independent in-container Python equality audit, before the main scoring freeze."""
import concurrent.futures
from formalcrrc import code_extension as cp

def equality_source(code,entry,expected):
    # Preserve original code verbatim. The prefix captures bool before candidate
    # definitions; the suffix calls the original entry and compares inside Docker.
    if '__fcrrc_audit_' in code: raise RuntimeError('Reserved audit name collision')
    prefix='from builtins import bool as __fcrrc_audit_bool\n'
    suffix=f'\n__fcrrc_audit_original = {entry}\ndef __fcrrc_audit_check(index, args):\n    expected = [{", ".join(expected)}][index]\n    return __fcrrc_audit_bool(__fcrrc_audit_original(*args) == expected)\n'
    return prefix+code+'\n'+suffix

def verify(root,workers=4):
    out=cp.outdir(root); cp.assert_freeze_a(root)
    def run(item):
        task=item['task_id']; dest=out/'truth'/f'{cp.slug(task)}__direct_equality.json'
        g=cp.load(out/'generation'/f'{cp.slug(task)}.json'); primary=cp.load(out/'truth'/f'{cp.slug(task)}.json')
        if dest.exists():
            record=cp.load(dest)
            if not record['passed'] or record['code_sha256']!=g['code_sha256']: raise RuntimeError('Direct equality resume mismatch')
            return
        logs={}; passed=primary['stable']
        for role,code in [('reference',item['reference_code']),('candidate',g['code'])]:
            wrapper=equality_source(code,item['entry_point'],[t['expected_repr'] for t in item['tests']])
            result=cp.sandbox.execute(wrapper,'__fcrrc_audit_check',[[t['index'],t['args']] for t in item['tests']])
            valid=result['status']=='complete' and len(result['tests'])==16
            bits=[]
            if valid:
                for r in result['tests']: r['passed']=cp.sandbox.compare(r,'True')
                valid=all(cp.sandbox.stable_pair(result['tests'][j],result['tests'][j+8]) and result['tests'][j]['passed'] is not None for j in range(8))
                bits=[int(r['passed']) for r in result['tests'][:8]] if valid else []
                valid=valid and bits==([1]*8 if role=='reference' else primary['pass_bits'])
            logs[role]=dict(wrapper_sha256=cp.sha(wrapper),execution=result,pass_bits=bits,agreement=valid)
            passed &= valid
        cp.immutable(dest,dict(created_at=cp.now(),task_id=task,code_sha256=g['code_sha256'],passed=passed,roles=logs,
            method='Direct Python equality inside the same isolated per-test environment; must agree with both primary repetitions. Original candidate behavior is not repaired.'))
        if not passed: raise RuntimeError('Independent direct equality disagrees: '+task)
        print('DIRECT_EQUALITY_PASS',task,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(run,cp.load(out/'instances.json')))

def smoke(root):
    out=cp.outdir(root); result={}
    fixtures=[('spoofed_repr',"class X:\n def __repr__(self): return '1'\n def __eq__(self, other): return False\ndef f(): return X()",'1',False),
              ('frozenset_equality','def f(): return frozenset([1])','{1}',True),
              ('float_int_equality','def f(): return 1.0','1',True)]
    for name,code,expected,wanted in fixtures:
        r=cp.sandbox.execute(equality_source(code,'f',[expected]),'__fcrrc_audit_check',[[0,[]]])
        ok=r['status']=='complete' and len(r['tests'])==2 and all(cp.sandbox.compare(t,repr(wanted)) is True for t in r['tests'])
        result[name]=dict(passed=ok,execution=r)
    cp.immutable(out/'direct_equality_smoke.json',dict(created_at=cp.now(),passed=all(r['passed'] for r in result.values()),fixtures=result))
    if not all(r['passed'] for r in result.values()): raise RuntimeError('Direct equality smoke failed')
    print('DIRECT_EQUALITY_SMOKE_PASS',flush=True)
